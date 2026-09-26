import unittest
from unittest.mock import patch

from app.core.device import detect_device
from app.core.exceptions import InvalidParameterException
from app.services.asr.manager import get_model_manager
from app.services.asr.model_capabilities import (
    get_download_modelscope_assets,
    get_huggingface_assets,
    get_runtime_required_modelscope_assets,
)
from app.services.asr.model_plan import get_runtime_model_ids
from app.services.realtime.protocol import MODEL_ID, MODEL_REPOSITORY, MODEL_REVISION


class OfflineContractTest(unittest.TestCase):
    def test_one_checkpoint_for_both_modes_with_separate_aligner(self) -> None:
        self.assertEqual(get_runtime_model_ids(), [MODEL_ID])
        assets = get_huggingface_assets()
        self.assertEqual(
            [asset.model_id for asset in assets],
            [MODEL_REPOSITORY, "Qwen/Qwen3-ForcedAligner-0.6B"],
        )
        self.assertEqual(assets[0].revision, MODEL_REVISION)
        models = get_model_manager().list_declared_entries()
        self.assertEqual(len(models), 1)
        self.assertTrue(models[0]["supports_realtime"])
        self.assertEqual(models[0]["offline_model"], models[0]["realtime_model"])
        with self.assertRaises(InvalidParameterException):
            get_model_manager().get_declared_entry_config("unsupported-model")

    def test_speaker_and_vad_assets_retained(self) -> None:
        expected = {
            "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            "iic/speech_campplus_speaker-diarization_common",
            "damo/speech_campplus_sv_zh-cn_16k-common",
            "damo/speech_campplus-transformer_scl_zh-cn_16k-common",
        }
        self.assertEqual(
            {a.model_id for a in get_download_modelscope_assets()}, expected
        )
        self.assertEqual(
            {a.model_id for a in get_runtime_required_modelscope_assets()}, expected
        )

    def test_explicit_cpu_and_cuda_without_fallback(self) -> None:
        with patch("app.core.device.torch.cuda.is_available", return_value=False):
            self.assertEqual(detect_device("cpu"), "cpu")
            with self.assertRaises(RuntimeError):
                detect_device("cuda:0")
        with patch("app.core.device.torch.cuda.is_available", return_value=True):
            for device in ("mps", "npu", "cuda", "cuda:1", "cpu:0"):
                with self.subTest(device=device), self.assertRaises(ValueError):
                    detect_device(device)
            self.assertEqual(detect_device("cuda:0"), "cuda:0")


if __name__ == "__main__":
    unittest.main()
