from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class SingleContainerDeploymentTest(unittest.TestCase):
    def test_image_contract_is_manual_single_container(self) -> None:
        dockerfiles = sorted(path.name for path in PROJECT_ROOT.glob("Dockerfile*"))
        dockerfile = (PROJECT_ROOT / "Dockerfile.ascend").read_text(encoding="utf-8")

        self.assertEqual(dockerfiles, ["Dockerfile.ascend"])
        self.assertFalse((PROJECT_ROOT / "docker-compose.yml").exists())
        self.assertIn('CMD ["/bin/bash"]', dockerfile)
        self.assertNotIn("ENTRYPOINT", dockerfile)
        self.assertIn("UV_PROJECT_ENVIRONMENT=/opt/qwen3-asr-venv", dockerfile)
        self.assertIn("--exclude-qwen", dockerfile)
        self.assertIn("EXPOSE 17003", dockerfile)

    def test_stage_command_materializes_huggingface_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "snapshot"
            destination = root / "hf_models" / "Qwen3-ASR-1.7B"
            blob = root / "blobs" / "model.safetensors"
            source.mkdir()
            blob.parent.mkdir()
            blob.write_bytes(b"weights")
            (source / "config.json").write_text("{}", encoding="utf-8")
            (source / "model.safetensors").symlink_to(blob)

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
            self.assertEqual((destination / "model.safetensors").read_bytes(), b"weights")
            self.assertFalse((destination / "model.safetensors").is_symlink())
            self.assertIn("Copy finished", completed.stdout)

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
                bin_dir / "curl",
                """#!/usr/bin/env bash
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [[ -f "$FAKE_VLLM_ARGS" ]] && exit 0
  sleep 0.01
done
exit 1
""",
            )
            self._write_executable(
                bin_dir / "api-python",
                """#!/usr/bin/env bash
printf '%s\n' \
  "$QWEN_VLLM_BASE_URL" "$HOST" "$PORT" "$DEVICE" \
  "$SPEAKER_DIARIZATION_DEVICE" "$HF_HUB_OFFLINE" "$1" > "$FAKE_API_ENV"
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
            self.assertIn("vLLM is ready", completed.stdout)

    @staticmethod
    def _write_executable(path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")
        path.chmod(0o755)


if __name__ == "__main__":
    unittest.main()
