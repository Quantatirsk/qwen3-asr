"""The CPU API owns only its independent aligner, never a second ASR model."""

import unittest
from unittest.mock import patch

from app.core.config import settings
from app.services.asr.r2t2_engine import R2T2Engine


class CPUExecutionTest(unittest.TestCase):
    def test_cpu_selects_rust_alignment_without_loading_vllm(self) -> None:
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.asr.r2t2_engine.ForcedAligner") as gpu,
            patch("app.services.asr.r2t2_engine.RustForcedAligner") as cpu,
        ):
            engine = R2T2Engine()
            self.assertIs(engine.aligner, cpu.return_value)
            engine.close()
        gpu.assert_not_called()
        cpu.assert_called_once_with("Qwen/Qwen3-ForcedAligner-0.6B")
        cpu.return_value.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
