import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from app.services.asr.r2t2_vllm import R2T2VLLMBackend, _shared_gpu_engine_options


class R2T2GPUConfigTest(unittest.TestCase):
    def test_shared_gpu_limits_apply_to_asr_and_aligner(self):
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
        modules = {
            "vllm": SimpleNamespace(LLM=llm, SamplingParams=Mock()),
            "vllm.transformers_utils.processors.qwen3_asr": SimpleNamespace(
                Qwen3ASRProcessor=SimpleNamespace(from_pretrained=Mock())
            ),
        }
        with patch.dict(
            os.environ,
            {
                "R2T2_OFFLINE_MAX_NUM_SEQS": "4",
                "R2T2_OFFLINE_ENFORCE_EAGER": "1",
                "FORCED_ALIGNER_GPU_MEMORY_UTILIZATION": "0.045",
            },
        ):
            with (
                patch(
                    "app.services.asr.r2t2_vllm.resolve_huggingface_snapshot_dir",
                    side_effect=lambda path: path,
                ),
                patch(
                    "app.services.asr.r2t2_vllm._resolve_checkpoint",
                    return_value="pinned-r2t2",
                ),
            ):
                with patch(
                    "app.services.asr.r2t2_vllm.importlib.import_module",
                    side_effect=modules.__getitem__,
                ):
                    backend = R2T2VLLMBackend(
                        "netease-youdao/Confucius4-R2T2",
                        "Qwen/Qwen3-ForcedAligner-0.6B",
                        0.30,
                        4,
                        4096,
                        16384,
                    )
                    backend.ensure_forced_aligner_loaded()
        modules["vllm"].SamplingParams.assert_called_once_with(
            temperature=0, max_tokens=4096, skip_special_tokens=True
        )
        asr, aligner = [call.kwargs for call in llm.call_args_list]
        self.assertEqual(asr["gpu_memory_utilization"], 0.30)
        self.assertEqual(aligner["gpu_memory_utilization"], 0.045)
        for options in (asr, aligner):
            self.assertTrue(options["enforce_eager"])
            self.assertEqual(options["max_num_seqs"], 4)
            self.assertEqual(options["limit_mm_per_prompt"], {"audio": 1})
        self.assertEqual(asr["max_model_len"], 16384)
        self.assertNotIn("max_model_len", aligner)
        self.assertEqual(aligner["runner"], "pooling")

    def test_default_limits_are_bounded(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                _shared_gpu_engine_options(),
                {
                    "max_num_seqs": 4,
                    "limit_mm_per_prompt": {"audio": 1},
                    "enforce_eager": True,
                },
            )

    def test_invalid_concurrency_is_rejected(self):
        for value in ("0", "-1", "invalid"):
            with patch.dict(os.environ, {"R2T2_OFFLINE_MAX_NUM_SEQS": value}):
                with self.assertRaises(ValueError):
                    _shared_gpu_engine_options()


if __name__ == "__main__":
    unittest.main()
