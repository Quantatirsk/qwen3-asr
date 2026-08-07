from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SingleContainerDeploymentTest(unittest.TestCase):
    def test_image_contract_is_manual_single_container(self) -> None:
        dockerfiles = sorted(path.name for path in PROJECT_ROOT.glob("Dockerfile*"))
        dockerfile = (PROJECT_ROOT / "Dockerfile.ascend").read_text(encoding="utf-8")
        dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

        self.assertEqual(dockerfiles, ["Dockerfile.ascend"])
        self.assertIn('CMD ["/bin/bash"]', dockerfile)
        self.assertNotIn("ENTRYPOINT", dockerfile)
        self.assertIn("UV_PROJECT_ENVIRONMENT=/opt/qwen3-asr-venv", dockerfile)
        self.assertNotIn("download_models", dockerfile)
        self.assertNotIn("--exclude-qwen", dockerfile)
        self.assertNotIn("verify_required_models_integrity", dockerfile)
        self.assertIn("EXPOSE 17003", dockerfile)
        self.assertIn(".venv/", dockerignore)

    def test_stage_command_materializes_huggingface_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "snapshot"
            destination = root / "hf_models" / "Qwen3-ASR-1.7B"
            shard_name = "model-00001-of-00001.safetensors"
            blob = root / "blobs" / shard_name
            source.mkdir()
            blob.parent.mkdir()
            blob.write_bytes(b"weights")
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / shard_name).symlink_to(blob)
            (source / "model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"layer.weight": shard_name}}),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(PROJECT_ROOT / "scripts/stage-qwen-model.sh"),
                    str(source),
                    str(destination),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual((destination / "config.json").read_text(), "{}")
            self.assertEqual((destination / shard_name).read_bytes(), b"weights")
            self.assertFalse((destination / shard_name).is_symlink())
            self.assertIn("Copy finished", completed.stdout)

    def test_stage_command_rejects_incomplete_existing_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "snapshot"
            destination = root / "Qwen3-ASR-1.7B"
            source.mkdir()
            destination.mkdir()
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "model.safetensors").write_bytes(b"weights")
            (destination / "config.json").write_text("{}", encoding="utf-8")

            completed = subprocess.run(
                [
                    "bash",
                    str(PROJECT_ROOT / "scripts/stage-qwen-model.sh"),
                    str(source),
                    str(destination),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("destination exists but is incomplete", completed.stderr)

    def test_stage_command_rejects_index_without_weight_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "snapshot"
            destination = root / "Qwen3-ASR-1.7B"
            source.mkdir()
            destination.mkdir()
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "model.safetensors").write_bytes(b"weights")
            (destination / "config.json").write_text("{}", encoding="utf-8")
            (destination / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.weight": "model-00001-of-00001.safetensors"
                        }
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    "bash",
                    str(PROJECT_ROOT / "scripts/stage-qwen-model.sh"),
                    str(source),
                    str(destination),
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("destination exists but is incomplete", completed.stderr)

    def test_stage_command_rejects_destination_inside_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "snapshot"
            source.mkdir()
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "model.safetensors").write_bytes(b"weights")

            completed = subprocess.run(
                [
                    "bash",
                    str(PROJECT_ROOT / "scripts/stage-qwen-model.sh"),
                    str(source),
                    str(source / "nested"),
                ],
                check=False,
                capture_output=True,
                text=True,
                timeout=3,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("destination must not be inside model source", completed.stderr)

    def test_stage_command_uses_subsecond_copy_rate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "snapshot"
            destination = root / "Qwen3-ASR-1.7B"
            bin_dir = root / "bin"
            source.mkdir()
            bin_dir.mkdir()
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "model.safetensors").write_bytes(b"x" * 1024 * 1024)
            self._write_executable(
                bin_dir / "cp",
                """#!/usr/bin/env bash
/bin/cp "$@"
sleep 0.4
""",
            )
            environment = os.environ.copy()
            environment["PATH"] = f"{bin_dir}:{environment['PATH']}"
            environment["QWEN_COPY_PROGRESS_INTERVAL_SEC"] = "0.2"

            completed = subprocess.run(
                [
                    "bash",
                    str(PROJECT_ROOT / "scripts/stage-qwen-model.sh"),
                    str(source),
                    str(destination),
                ],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            rates = [
                float(value)
                for value in re.findall(
                    r"([0-9]+\.[0-9]+) MB/s",
                    completed.stdout,
                )
            ]
            self.assertTrue(rates, completed.stdout)
            self.assertGreater(max(rates), 2.0)

    def test_start_command_runs_vllm_and_api_in_one_container(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bin_dir = root / "bin"
            model_dir = root / "Qwen3-ASR-1.7B"
            vllm_args = root / "vllm.args"
            api_env = root / "api.env"
            bin_dir.mkdir()
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")

            self._write_executable(
                bin_dir / "vllm",
                """#!/usr/bin/env bash
printf '%s\n' "$@" > "$FAKE_VLLM_ARGS"
trap 'exit 0' TERM INT
while true; do sleep 0.1; done
""",
            )
            self._write_executable(
                bin_dir / "api-python",
                """#!/usr/bin/env bash
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [[ -f "$FAKE_VLLM_ARGS" ]] && break
  sleep 0.01
done
printf '%s\n' \
  "$QWEN_VLLM_BASE_URL" "$HOST" "$PORT" "$DEVICE" \
  "$SPEAKER_DIARIZATION_DEVICE" "$1" > "$FAKE_API_ENV"
exit 0
""",
            )

            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{bin_dir}:{environment['PATH']}",
                    "QWEN_ASCEND_MODEL_PATH": str(model_dir),
                    "QWEN3_ASR_API_PYTHON": str(bin_dir / "api-python"),
                    "QWEN3_ASR_PROJECT_ROOT": str(PROJECT_ROOT),
                    "QWEN_VLLM_STARTUP_TIMEOUT_SEC": "2",
                    "QWEN_VLLM_PORT": "17004",
                    "QWEN3_ASR_API_PORT": "17003",
                    "FAKE_VLLM_ARGS": str(vllm_args),
                    "FAKE_API_ENV": str(api_env),
                }
            )

            completed = subprocess.run(
                ["bash", str(PROJECT_ROOT / "scripts/start-ascend-services.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=5,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            arguments = vllm_args.read_text(encoding="utf-8").splitlines()
            self.assertEqual(arguments[:2], ["serve", str(model_dir)])
            self.assertIn("--served-model-name", arguments)
            self.assertIn("qwen3-asr", arguments)
            self.assertIn("--host", arguments)
            self.assertIn("127.0.0.1", arguments)
            self.assertEqual(
                api_env.read_text(encoding="utf-8").splitlines(),
                [
                    "http://127.0.0.1:17004",
                    "0.0.0.0",
                    "17003",
                    "cpu",
                    "cpu",
                    "1",
                    str(PROJECT_ROOT / "start.py"),
                ],
            )
            self.assertIn("Starting Ascend vLLM", completed.stdout)

    def test_start_command_stops_api_when_ready_vllm_exits(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bin_dir = root / "bin"
            model_dir = root / "Qwen3-ASR-1.7B"
            vllm_ready = root / "vllm.ready"
            api_started = root / "api.started"
            api_stopped = root / "api.stopped"
            bin_dir.mkdir()
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors").write_bytes(b"weights")

            self._write_executable(
                bin_dir / "vllm",
                """#!/usr/bin/env bash
touch "$FAKE_VLLM_READY"
for _ in $(seq 1 50); do
  [[ -f "$FAKE_API_STARTED" ]] && break
  sleep 0.02
done
sleep 0.1
exit 7
""",
            )
            self._write_executable(
                bin_dir / "api-python",
                """#!/usr/bin/env bash
trap 'touch "$FAKE_API_STOPPED"; exit 0' TERM INT
touch "$FAKE_API_STARTED"
for _ in $(seq 1 50); do sleep 0.02; done
exit 0
""",
            )

            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{bin_dir}:{environment['PATH']}",
                    "QWEN_ASCEND_MODEL_PATH": str(model_dir),
                    "QWEN3_ASR_API_PYTHON": str(bin_dir / "api-python"),
                    "QWEN3_ASR_PROJECT_ROOT": str(PROJECT_ROOT),
                    "QWEN_VLLM_STARTUP_TIMEOUT_SEC": "2",
                    "QWEN_PROCESS_POLL_INTERVAL_SEC": "0.02",
                    "FAKE_VLLM_READY": str(vllm_ready),
                    "FAKE_API_STARTED": str(api_started),
                    "FAKE_API_STOPPED": str(api_stopped),
                }
            )

            completed = subprocess.run(
                ["bash", str(PROJECT_ROOT / "scripts/start-ascend-services.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=5,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertTrue(api_stopped.exists())
            self.assertIn("vLLM exited after startup", completed.stderr)

    def test_start_command_rejects_index_without_weight_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            bin_dir = root / "bin"
            model_dir = root / "Qwen3-ASR-1.7B"
            vllm_started = root / "vllm.started"
            bin_dir.mkdir()
            model_dir.mkdir()
            (model_dir / "config.json").write_text("{}", encoding="utf-8")
            (model_dir / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "layer.weight": "model-00001-of-00001.safetensors"
                        }
                    }
                ),
                encoding="utf-8",
            )
            self._write_executable(
                bin_dir / "vllm",
                """#!/usr/bin/env bash
touch "$FAKE_VLLM_STARTED"
exit 1
""",
            )
            self._write_executable(
                bin_dir / "api-python",
                "#!/usr/bin/env bash\nexit 0\n",
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{bin_dir}:{environment['PATH']}",
                    "QWEN_ASCEND_MODEL_PATH": str(model_dir),
                    "QWEN3_ASR_API_PYTHON": str(bin_dir / "api-python"),
                    "QWEN3_ASR_PROJECT_ROOT": str(PROJECT_ROOT),
                    "QWEN_VLLM_STARTUP_TIMEOUT_SEC": "2",
                    "FAKE_VLLM_STARTED": str(vllm_started),
                }
            )

            completed = subprocess.run(
                ["bash", str(PROJECT_ROOT / "scripts/start-ascend-services.sh")],
                check=False,
                capture_output=True,
                text=True,
                env=environment,
                timeout=5,
            )

            self.assertNotEqual(completed.returncode, 0)
            self.assertFalse(vllm_started.exists())
            self.assertIn("staged model is incomplete", completed.stderr)

    @staticmethod
    def _write_executable(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
