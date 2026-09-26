"""Independent offline recognition through the shared R2T2 inference service."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

from app.core.config import settings
from app.core.device import detect_device
from app.services.realtime.client import transcribe_segment
from app.services.realtime.protocol import MODEL_ID, OFFLINE_CONCURRENCY

from .engines import ASRFullResult, ASRSegmentResult, WordToken
from .forced_aligner import ForcedAligner, _load_audio
from .long_audio import prepare_long_audio
from .rust_backend import RustForcedAligner
from .punctuation import restore_sentence_ending

if TYPE_CHECKING:
    from app.utils.audio_splitter import AudioSegment


class R2T2Engine:
    def __init__(
        self,
        forced_aligner_path: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    ) -> None:
        self.device = detect_device(settings.DEVICE)
        self.model_id = MODEL_ID
        self.aligner = (
            RustForcedAligner(forced_aligner_path)
            if self.device == "cpu"
            else ForcedAligner(forced_aligner_path)
        )

    def close(self) -> None:
        if self.device == "cpu":
            self.aligner.close()

    def transcribe_segments(
        self,
        segments: Sequence[AudioSegment],
        hotwords: str = "",
        enable_punctuation: bool = True,
        sample_rate: int = 16000,
        word_timestamps: bool = False,
    ) -> list[ASRSegmentResult]:
        for segment in segments:
            if not segment.temp_file or not Path(segment.temp_file).is_file():
                raise FileNotFoundError(f"Missing audio segment: {segment.temp_file}")
        audios = [_load_audio(segment.temp_file) for segment in segments]
        # The engine batches concurrent segments; punctuation and alignment stay serial.
        pool = ThreadPoolExecutor(OFFLINE_CONCURRENCY)
        try:
            texts = list(pool.map(lambda audio: transcribe_segment(audio, hotwords), audios))
        finally:
            pool.shutdown(cancel_futures=True)
        results = []
        for segment, audio, text in zip(segments, audios, texts):
            if enable_punctuation:
                text = restore_sentence_ending(text)
            words = None
            if word_timestamps:
                aligned = self.aligner.align_transcript(
                    audio_path=segment.temp_file, text=text, audio=audio
                )
                words = [
                    WordToken(
                        text=str(item["text"]),
                        start_time=round(float(item["start_ms"]) / 1000.0, 3),
                        end_time=round(float(item["end_ms"]) / 1000.0, 3),
                    )
                    for item in aligned
                ]
            results.append(
                ASRSegmentResult(
                    text=text,
                    start_time=segment.start_sec,
                    end_time=segment.end_sec,
                    word_tokens=words or None,
                )
            )
        return results

    def transcribe_long_audio(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
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
            results = self.transcribe_segments(
                audio.segments,
                hotwords=hotwords,
                enable_punctuation=enable_punctuation,
                sample_rate=sample_rate,
                word_timestamps=word_timestamps or enable_speaker_diarization,
            )
            return audio.finish(
                results, timestamp_scale, word_timestamps=word_timestamps
            )
