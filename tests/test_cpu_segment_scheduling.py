"""A CPU-configured process must never create an offline decoder."""

import unittest
from unittest.mock import patch

from app.core.config import settings
from app.services.asr.r2t2_engine import R2T2Engine


class UnsupportedCPUExecutionTest(unittest.TestCase):
    def test_cpu_is_rejected_before_model_construction(self) -> None:
        with (
            patch.object(settings, "DEVICE", "cpu"),
            patch("app.services.asr.r2t2_engine.ForcedAligner") as backend,
            self.assertRaisesRegex(RuntimeError, "CUDA"),
        ):
            R2T2Engine()
        backend.assert_not_called()


if __name__ == "__main__":
    unittest.main()
