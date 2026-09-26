"""The API process loads the pooling aligner only."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.services.asr.forced_aligner import ForcedAligner, _gpu_memory_utilization
from app.services.asr.r2t2_engine import R2T2Engine


class R2T2GPUConfigTest(unittest.TestCase):
    def test_api_loads_only_forced_aligner_and_no_asr_processor(self) -> None:
        engine = SimpleNamespace(
            llm_engine=SimpleNamespace(
                vllm_config=SimpleNamespace(
                    model_config=SimpleNamespace(
                        hf_config=SimpleNamespace(
                            timestamp_token_id=42, timestamp_segment_time=80
                        )
                    )
                )
            )
        )
        llm = Mock(return_value=engine)
        with (
            patch.dict(os.environ, {"FORCED_ALIGNER_GPU_MEMORY_UTILIZATION": "0.045"}),
            patch(
                "app.services.asr.forced_aligner.resolve_huggingface_snapshot_dir",
                side_effect=lambda path: path,
            ) as resolve,
            patch("app.services.asr.r2t2_engine.detect_device", return_value="cuda:0"),
            patch(
                "app.services.asr.forced_aligner.importlib.import_module",
                return_value=SimpleNamespace(LLM=llm),
            ) as imports,
        ):
            offline = R2T2Engine()
        self.assertIsInstance(offline.aligner, ForcedAligner)
        imports.assert_called_once_with("vllm")
        resolve.assert_called_once_with("Qwen/Qwen3-ForcedAligner-0.6B")
        llm.assert_called_once()
        options = llm.call_args.kwargs
        self.assertEqual(options["model"], "Qwen/Qwen3-ForcedAligner-0.6B")
        self.assertEqual(options["runner"], "pooling")
        self.assertEqual(options["gpu_memory_utilization"], 0.045)
        self.assertTrue(options["enforce_eager"])
        self.assertEqual(options["max_num_seqs"], 1)

    def test_default_memory_budget(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_gpu_memory_utilization(), 0.15)

    def test_invalid_memory_budget_is_rejected(self) -> None:
        for value in ("0", "1.01", "nan", "inf", "invalid"):
            with (
                patch.dict(
                    os.environ, {"FORCED_ALIGNER_GPU_MEMORY_UTILIZATION": value}
                ),
                self.assertRaises(ValueError),
            ):
                _gpu_memory_utilization()


if __name__ == "__main__":
    unittest.main()
