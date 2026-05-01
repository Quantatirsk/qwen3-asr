import unittest
from unittest.mock import patch

from app.services.asr.qwenasr_rust import validate_qwenasr_cpu_features


class QwenASRCpuFeatureTest(unittest.TestCase):
    def test_accepts_x86_avx2_fma_cpu(self) -> None:
        with (
            patch("app.services.asr.qwenasr_rust.platform.machine", return_value="x86_64"),
            patch(
                "app.services.asr.qwenasr_rust._read_linux_cpu_flags",
                return_value={"sse4_2", "avx2", "fma"},
            ),
        ):
            validate_qwenasr_cpu_features()

    def test_rejects_x86_cpu_without_required_features(self) -> None:
        with (
            patch("app.services.asr.qwenasr_rust.platform.machine", return_value="x86_64"),
            patch(
                "app.services.asr.qwenasr_rust._read_linux_cpu_flags",
                return_value={"sse4_2"},
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing: avx2, fma"):
                validate_qwenasr_cpu_features()

    def test_skips_non_x86_cpu(self) -> None:
        with (
            patch("app.services.asr.qwenasr_rust.platform.machine", return_value="aarch64"),
            patch(
                "app.services.asr.qwenasr_rust._read_linux_cpu_flags",
                return_value=set(),
            ),
        ):
            validate_qwenasr_cpu_features()


if __name__ == "__main__":
    unittest.main()
