import unittest
import inspect
import tempfile
from pathlib import Path
from unittest.mock import patch

from app.services.asr.model_capabilities import (
    get_download_modelscope_assets,
    get_runtime_required_modelscope_assets,
    get_enabled_qwen_huggingface_assets,
)
from app.services.asr.model_plan import get_runtime_model_ids
from app.services.asr.manager import get_model_manager
from app.services.asr.qwen3_engine import Qwen3ASREngine
from app.utils.download_models import scoped_assets
from scripts.retire_realtime_models import candidates, RETIRED_MODELS


class OfflineContractTest(unittest.TestCase):
    def test_realtime_asset_scope_never_resolves_offline(self):
        with patch(
            "app.utils.download_models._get_huggingface_assets",
            side_effect=AssertionError("offline resolver called"),
        ):
            ms, hf = scoped_assets("realtime")
        self.assertEqual(ms, [])
        self.assertEqual(hf[0].model_id, "netease-youdao/Confucius4-R2T2")
        self.assertEqual(hf[0].revision, "185ce639118ad1362d049ca0d8ed04b6ec5cd6c9")

    def test_cleanup_allowlist_and_symlink_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp).resolve()
            retained = root / "damo/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            retained.mkdir(parents=True)
            retired = root / RETIRED_MODELS[0]
            retired.mkdir(parents=True)
            self.assertEqual(candidates(root), [retired])
            retired.rmdir()
            retired.symlink_to(retained, target_is_directory=True)
            with self.assertRaises(ValueError):
                candidates(root)

    def test_exact_offline_modelscope_dependencies(self):
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

    def test_qwen_and_aligner_remain_offline_only(self):
        self.assertFalse(inspect.isabstract(Qwen3ASREngine))
        self.assertEqual(len(get_runtime_model_ids()), 1)
        self.assertTrue(
            all(m.startswith("qwen3-asr-") for m in get_runtime_model_ids())
        )
        assets = get_enabled_qwen_huggingface_assets()
        self.assertIn("Qwen/Qwen3-ForcedAligner-0.6B", {a.model_id for a in assets})
        self.assertTrue(
            all(
                not m["supports_realtime"]
                for m in get_model_manager().list_declared_entries()
            )
        )
        self.assertFalse(hasattr(Qwen3ASREngine, "streaming_transcribe"))
