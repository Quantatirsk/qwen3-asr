"""Exercise native offline generation without model weights or a GPU."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from app.services.asr.engines import ASRSegmentResult, WordToken
from app.services.asr.r2t2_engine import R2T2Engine
from app.services.asr.r2t2_vllm import (
    R2T2VLLMBackend,
    _gpu_memory_utilization,
    _resolve_checkpoint,
)
from app.services.realtime.protocol import MODEL_REPOSITORY, MODEL_REVISION


class R2T2OfflineTest(unittest.TestCase):
    def make_backend(self) -> R2T2VLLMBackend:
        backend = R2T2VLLMBackend.__new__(R2T2VLLMBackend)
        backend._processor = SimpleNamespace(
            apply_chat_template=Mock(return_value="full audio prompt")
        )
        backend._llm = SimpleNamespace(generate=Mock())
        backend._sampling_params = object()
        return backend

    def test_complete_audio_is_generated_once_with_fresh_context(self) -> None:
        backend = self.make_backend()
        backend._llm.generate.return_value = [
            SimpleNamespace(
                outputs=[
                    SimpleNamespace(
                        text="language English<asr_text>Fresh offline result.|",
                        finish_reason="stop",
                    )
                ]
            )
        ]
        audio = np.arange(16000, dtype=np.float32)
        result = backend._run_generate([(audio, "Names: Ada", None)])
        self.assertEqual(result[0].text, "Fresh offline result.")
        backend._processor.apply_chat_template.assert_called_once_with(
            [
                {"role": "system", "content": "Names: Ada"},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )
        backend._llm.generate.assert_called_once()
        prompt = backend._llm.generate.call_args.args[0][0]
        self.assertEqual(prompt["prompt"], "full audio prompt")
        self.assertIs(prompt["multi_modal_data"]["audio"][0], audio)

    def test_truncated_or_missing_output_fails_instead_of_returning_partial_text(
        self,
    ) -> None:
        backend = self.make_backend()
        for outputs in (
            [],
            [SimpleNamespace(outputs=[])],
            [
                SimpleNamespace(
                    outputs=[SimpleNamespace(text="partial", finish_reason="length")]
                )
            ],
        ):
            backend._llm.generate.return_value = outputs
            with self.assertRaises((ValueError, RuntimeError)):
                backend._run_generate([(np.zeros(16000), "", None)])

    def test_forced_language_uses_official_prompt_suffix(self) -> None:
        backend = self.make_backend()
        self.assertEqual(
            backend._build_prompt("", "en"),
            "full audio promptlanguage English<asr_text>",
        )

    def test_checkpoint_is_exact_and_local_even_when_main_points_elsewhere(
        self,
    ) -> None:
        with patch(
            "app.services.asr.r2t2_vllm.snapshot_download", return_value="pinned"
        ) as resolve:
            self.assertEqual(_resolve_checkpoint(MODEL_REPOSITORY), "pinned")
        self.assertEqual(resolve.call_args.kwargs["revision"], MODEL_REVISION)
        self.assertTrue(resolve.call_args.kwargs["local_files_only"])
        with tempfile.TemporaryDirectory() as directory:
            with patch("app.services.asr.r2t2_vllm.snapshot_download") as resolve:
                self.assertEqual(
                    _resolve_checkpoint(directory), str(Path(directory).resolve())
                )
                resolve.assert_not_called()

    def test_invalid_memory_budget_is_rejected(self) -> None:
        for value in ("0", "1.01", "nan", "inf", "invalid"):
            with patch.dict(os.environ, {"R2T2_OFFLINE_GPU_MEMORY_UTILIZATION": value}):
                with self.assertRaises(ValueError):
                    _gpu_memory_utilization("R2T2_OFFLINE_GPU_MEMORY_UTILIZATION", 0.30)

    def test_diarized_segments_keep_order_speakers_and_relative_word_times(
        self,
    ) -> None:
        engine = R2T2Engine.__new__(R2T2Engine)
        words = [WordToken("fresh", 0.2, 0.8)]
        engine.model = SimpleNamespace(
            transcribe_batch=Mock(
                return_value=[
                    ASRSegmentResult("fresh", 0, 0, word_tokens=words),
                    ASRSegmentResult("second", 0, 0),
                ]
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            paths = [
                str(Path(directory) / name) for name in ("first.wav", "second.wav")
            ]
            for path in paths:
                Path(path).touch()
            segments = [
                SimpleNamespace(
                    temp_file=paths[0], start_sec=5.0, end_sec=8.0, speaker_id="A"
                ),
                SimpleNamespace(
                    temp_file=paths[1], start_sec=9.0, end_sec=11.0, speaker_id="B"
                ),
            ]
            results = engine.transcribe_segments(segments, word_timestamps=True)
            self.assertEqual([r.text for r in results], ["fresh", "second"])
            self.assertEqual([r.speaker_id for r in results], ["A", "B"])
            self.assertEqual(results[0].start_time, 5.0)
            self.assertIs(results[0].word_tokens, words)
            self.assertEqual(words[0].start_time, 0.2)
            engine.model.transcribe_batch.return_value = []
            with self.assertRaises(ValueError):
                engine.transcribe_segments(segments)
        with self.assertRaises(FileNotFoundError):
            engine.transcribe_segments(segments)


class DiarizedOfflineFlowTest(unittest.TestCase):
    def test_diarization_precedes_independent_offline_generation(self) -> None:
        from app.core.config import settings

        engine = R2T2Engine.__new__(R2T2Engine)
        engine.device = "cuda:0"
        engine.model_id = "confucius4-r2t2"
        stages = []

        def diarize(audio_path: str, output_dir: str) -> list[SimpleNamespace]:
            stages.append("diarize")
            self.assertEqual(Path(audio_path).name, "recording.wav")
            segment = Path(output_dir) / "speaker.wav"
            segment.write_bytes(b"complete speaker audio")
            return [
                SimpleNamespace(
                    temp_file=str(segment), start_sec=2.0, end_sec=5.0, speaker_id="A"
                )
            ]

        def generate(paths: list[str], **kwargs: object) -> list[ASRSegmentResult]:
            stages.append("offline")
            self.assertEqual(Path(paths[0]).read_bytes(), b"complete speaker audio")
            return [ASRSegmentResult("Independent offline text.", 0, 0)]

        engine.model = SimpleNamespace(transcribe_batch=generate)
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(settings, "TEMP_DIR", directory),
            patch("app.services.asr.long_audio.get_audio_duration", return_value=6.0),
            patch("app.utils.speaker_diarizer.SpeakerDiarizer") as diarizer,
            patch("app.utils.audio_splitter.AudioSplitter") as vad,
        ):
            source = Path(directory) / "recording.wav"
            source.write_bytes(b"original")
            diarizer.return_value.split_audio_by_speakers.side_effect = diarize
            result = engine.transcribe_long_audio(str(source))
            self.assertEqual(stages, ["diarize", "offline"])
            self.assertEqual(result.text, "Independent offline text.")
            self.assertEqual(result.segments[0].speaker_id, "A")
            self.assertEqual(list(Path(directory).iterdir()), [source])
            vad.assert_not_called()
            diarizer.return_value.split_audio_by_speakers.side_effect = None
            diarizer.return_value.split_audio_by_speakers.return_value = []
            silent = engine.transcribe_long_audio(str(source))
            self.assertEqual(silent.text, "")
            self.assertEqual(silent.segments, [])
            self.assertEqual(stages, ["diarize", "offline"])
            vad.assert_not_called()

    def test_local_snapshot_resolution_does_not_follow_main(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory)
            repository = cache / "models--netease-youdao--Confucius4-R2T2"
            pinned = repository / "snapshots" / MODEL_REVISION
            pinned.mkdir(parents=True)
            refs = repository / "refs"
            refs.mkdir()
            (refs / "main").write_text("different-revision")
            with patch(
                "app.services.asr.r2t2_vllm.get_huggingface_cache_root",
                return_value=cache,
            ):
                self.assertEqual(Path(_resolve_checkpoint(MODEL_REPOSITORY)), pinned)


if __name__ == "__main__":
    unittest.main()
