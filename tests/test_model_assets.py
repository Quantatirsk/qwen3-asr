import unittest
from unittest.mock import patch

from app.services.asr.model_capabilities import QWEN_ASCEND_REVISION
from app.utils.download_models import check_all_models


class ModelAssetPlanTest(unittest.TestCase):
    @patch("app.utils.download_models.check_model_exists", return_value=(False, ""))
    def test_api_preflight_excludes_npu_qwen_weights(self, _exists) -> None:
        runtime_missing = check_all_models(include_qwen=False)
        full_missing = check_all_models(include_qwen=True)

        self.assertFalse(any(source == "huggingface" for _, _, source, _ in runtime_missing))
        qwen_assets = [item for item in full_missing if item[2] == "huggingface"]
        self.assertEqual(
            qwen_assets,
            [
                (
                    "Qwen/Qwen3-ASR-1.7B",
                    "Qwen3-ASR-1.7B for Ascend vLLM",
                    "huggingface",
                    QWEN_ASCEND_REVISION,
                )
            ],
        )


if __name__ == "__main__":
    unittest.main()
