"""CUDA-only R2T2 offline recognition after independent speaker segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

from app.core.config import settings
from app.core.device import detect_device
from app.services.realtime.protocol import MODEL_ID, MODEL_REPOSITORY

from .engines import ASRFullResult, ASRSegmentResult
from .long_audio import prepare_long_audio
from .r2t2_vllm import R2T2VLLMBackend, _gpu_memory_utilization

if TYPE_CHECKING:
    from app.utils.audio_splitter import AudioSegment
    from app.utils.speaker_diarizer import SpeakerSegment


class R2T2Engine:
    def __init__(
        self,
        model_path: str = MODEL_REPOSITORY,
        forced_aligner_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
        max_inference_batch_size: int = 4,
        max_new_tokens: int = 4096,
        max_model_len: int | None = None,
    ) -> None:
        self.device = detect_device(settings.DEVICE)
        self.model_id = MODEL_ID
        self.model = R2T2VLLMBackend(
            model_path=model_path,
            forced_aligner_path=forced_aligner_path,
            gpu_memory_utilization=_gpu_memory_utilization(
                "R2T2_OFFLINE_GPU_MEMORY_UTILIZATION", 0.30
            ),
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            max_model_len=max_model_len,
        )
        self.model.ensure_forced_aligner_loaded()

    def transcribe_segments(
        self,
        segments: Sequence[AudioSegment | SpeakerSegment],
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> list[ASRSegmentResult]:
        paths = []
        for segment in segments:
            if not segment.temp_file or not Path(segment.temp_file).is_file():
                raise FileNotFoundError(f"Missing audio segment: {segment.temp_file}")
            paths.append(segment.temp_file)
        if not paths:
            return []
        results = self.model.transcribe_batch(
            paths,
            context=hotwords,
            word_timestamps=word_timestamps,
            enable_itn=enable_itn,
        )
        return [
            ASRSegmentResult(
                text=result.text,
                start_time=segment.start_sec,
                end_time=segment.end_sec,
                speaker_id=segment.speaker_id,
                word_tokens=result.word_tokens if word_timestamps else None,
            )
            for segment, result in zip(segments, results, strict=True)
        ]

    def transcribe_long_audio(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        enable_speaker_diarization: bool = True,
        word_timestamps: bool = False,
        timestamp_scale: float = 1.0,
        task_id: str | None = None,
    ) -> ASRFullResult:
        with prepare_long_audio(
            audio_path,
            self.device,
            enable_speaker_diarization,
            self.model_id,
            task_id,
        ) as audio:
            results = []
            for start in range(0, len(audio.segments), settings.ASR_BATCH_SIZE):
                results.extend(
                    self.transcribe_segments(
                        audio.segments[start : start + settings.ASR_BATCH_SIZE],
                        hotwords=hotwords,
                        enable_punctuation=enable_punctuation,
                        enable_itn=enable_itn,
                        sample_rate=sample_rate,
                        word_timestamps=word_timestamps,
                    )
                )
            return audio.finish(results, timestamp_scale)
