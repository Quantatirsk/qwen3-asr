import io
import os
import subprocess
import sys
import threading
import unittest
from unittest.mock import patch

from deploy import entrypoint as launcher


class SingleContainerTest(unittest.TestCase):
    def service(self, name, code="import time; time.sleep(60)"):
        return launcher.Service(name, [sys.executable, "-c", code], "http://127.0.0.1:9/health", "ready", dict(os.environ))

    def test_one_interpreter_and_local_private_endpoint(self):
        engine, api = launcher.services()
        self.assertEqual(engine.command[0], sys.executable)
        self.assertEqual(api.command[0], sys.executable)
        self.assertEqual(api.env["R2T2_URL"], "http://127.0.0.1:8001")
        self.assertEqual(api.env.get("DEVICE"), os.environ.get("DEVICE"))
        self.assertEqual(api.env.get("QWEN3_ASR_MODEL"), os.environ.get("QWEN3_ASR_MODEL"))
        self.assertEqual(api.env.get("CUDA_VISIBLE_DEVICES"), os.environ.get("CUDA_VISIBLE_DEVICES"))
        self.assertEqual(engine.env["VLLM_PLUGINS"], "")
        self.assertIn("127.0.0.1", engine.command)

    def test_health_validates_payload(self):
        for body, expected in [(b'{"ready":true}', True), (b'{"ready":false}', False), (b'[]', False), (b'not json', False)]:
            response = io.BytesIO(body)
            response.status = 200
            with patch.object(launcher.urllib.request, "urlopen", return_value=response):
                self.assertEqual(launcher.healthy("http://test", "ready"), expected)

    def test_deployment_preserves_explicit_device_and_model_settings(self):
        with patch.dict(os.environ, {"DEVICE": "cpu", "QWEN3_ASR_MODEL": "qwen3-asr-1.7b", "CUDA_VISIBLE_DEVICES": "0"}):
            engine, api = launcher.services()
        self.assertEqual(api.env["DEVICE"], "cpu")
        self.assertEqual(api.env["QWEN3_ASR_MODEL"], "qwen3-asr-1.7b")
        self.assertEqual(api.env["CUDA_VISIBLE_DEVICES"], "0")
        self.assertEqual(engine.env["CUDA_VISIBLE_DEVICES"], "0")

    def test_shutdown_before_start_creates_no_children(self):
        stop = threading.Event()
        stop.set()
        with patch.object(launcher.subprocess, "Popen") as start:
            self.assertEqual(launcher.run([self.service("engine")], stop), 0)
            start.assert_not_called()

    def test_health_sends_api_key_only_to_api(self):
        for url, expected in [(launcher.API_URL, "Bearer test-key"), (launcher.ENGINE_URL, None)]:
            response = io.BytesIO(b'{"ready":true}')
            response.status = 200
            with patch.dict(os.environ, {"API_KEY": "test-key"}):
                with patch.object(launcher.urllib.request, "urlopen", return_value=response) as request:
                    self.assertTrue(launcher.healthy(url, "ready"))
                    self.assertEqual(request.call_args.args[0].get_header("Authorization"), expected)

    def test_failed_start_stops_container_without_starting_api(self):
        original = subprocess.Popen
        with patch.object(launcher.subprocess, "Popen", wraps=original) as start:
            with patch.object(launcher, "healthy", return_value=False):
                with self.assertLogs(launcher.logger, level="ERROR"):
                    result = launcher.run([self.service("engine", "raise SystemExit(7)"), self.service("api")], threading.Event(), startup_timeout=3)
        self.assertEqual(result, 1)
        self.assertEqual(start.call_count, 1)

    def test_startup_timeout_reaps_process(self):
        children = []
        original = subprocess.Popen

        def start(*args, **kwargs):
            process = original(*args, **kwargs)
            children.append(process)
            return process

        with patch.object(launcher.subprocess, "Popen", side_effect=start):
            with patch.object(launcher, "healthy", return_value=False):
                with self.assertLogs(launcher.logger, level="ERROR"):
                    result = launcher.run([self.service("engine")], threading.Event(), startup_timeout=0.01, grace_seconds=1)
        self.assertEqual(result, 1)
        self.assertIsNotNone(children[0].poll())

    def test_graceful_shutdown_stops_api_before_engine(self):
        stop = threading.Event()
        timer = threading.Timer(0.1, stop.set)
        timer.start()
        try:
            with patch.object(launcher, "healthy", return_value=True):
                with patch.object(launcher, "stop_child", wraps=launcher.stop_child) as shutdown:
                    self.assertEqual(launcher.run([self.service("engine"), self.service("api")], stop, grace_seconds=1), 0)
            self.assertEqual([call.args[0] for call in shutdown.call_args_list], ["api", "engine"])
            self.assertTrue(all(call.args[1].poll() is not None for call in shutdown.call_args_list))
        finally:
            timer.cancel()

    def test_engine_failure_after_ready_stops_api(self):
        with patch.object(launcher, "healthy", return_value=True):
            with patch.object(launcher, "stop_child", wraps=launcher.stop_child) as shutdown:
                with self.assertLogs(launcher.logger, level="ERROR"):
                    result = launcher.run([self.service("engine", "import time; time.sleep(0.1)"), self.service("api")], threading.Event(), grace_seconds=1)
        self.assertEqual(result, 1)
        self.assertEqual([call.args[0] for call in shutdown.call_args_list], ["api", "engine"])
        self.assertTrue(all(call.args[1].poll() is not None for call in shutdown.call_args_list))


if __name__ == "__main__":
    unittest.main()
