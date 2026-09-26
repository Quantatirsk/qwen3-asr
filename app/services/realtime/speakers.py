"""Live speaker labels: Nemotron streams beside R2T2 and labels whole utterances.

R2T2 deltas carry no word times, so labels attach to pause-bounded utterances,
never to the newest text. Labels follow text by about Nemotron's 1.04 s buffer.
"""

import asyncio
import logging
from collections import deque

import numpy as np
import torch

from app.utils.speaker_diarizer import SpeakerDiarizer, get_speaker_diarizer

from .protocol import MAX_SECONDS, SAMPLE_RATE

logger = logging.getLogger(__name__)

FRAME_SAMPLES = SAMPLE_RATE // 100  # Nemotron emits one logit frame per 10 ms.


class SpeakerStream:
    """Nemotron streaming state for one connection.

    Every closed utterance gets exactly one label, `None` when unknown or when
    diarization failed; a Nemotron failure never ends the transcription.
    """

    def __init__(self) -> None:
        self.model = self.processor = None
        try:
            self.model, self.processor = get_speaker_diarizer().streaming_parts()
        except Exception:
            logger.exception("Realtime speaker diarization is unavailable")
        self.audio = np.empty(0, dtype=np.float32)
        self.offset = 0  # Absolute sample of self.audio[0].
        self.samples = 0
        self.frames = 0
        self.activity = np.zeros(
            (MAX_SECONDS * SAMPLE_RATE // FRAME_SAMPLES + 200, SpeakerDiarizer.MAX_SPEAKERS),
            dtype=bool,
        )
        self.cache = None
        self.ended = False
        self.finished = self.model is None
        self.pending: deque[tuple[int, int, int]] = deque()
        self.labels: dict[int, str] = {}

    def feed(self, pcm: bytes) -> None:
        if not self.finished:
            audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768
            self.audio = np.concatenate([self.audio, audio])
            self.samples += len(audio)

    def utterance(self, index: int, start_ms: int, end_ms: int) -> None:
        self.pending.append((index, start_ms * 16, end_ms * 16))

    async def advance(self) -> list[dict]:
        """Run every complete chunk, then return labels whose audio is covered.

        Only the audio task calls this, then one final call after the upstream
        `done`; `done` follows `end`, which the audio task sends after its last call.
        """
        try:
            while (chunk := self._next_chunk()) is not None:
                active, self.cache = await asyncio.to_thread(self._infer, *chunk)
                self._accept(active, last=chunk[2])
        except Exception:
            logger.exception("Realtime speaker diarization failed; labels are unknown")
            self.finished, self.model = True, None
        return self.ready()

    def ready(self) -> list[dict]:
        """Pop labels whose audio Nemotron already covers; runs no inference."""
        events = []
        while self.pending and (
            self.finished or self.frames * FRAME_SAMPLES >= self.pending[0][2]
        ):
            index, start, end = self.pending.popleft()
            events.append({"utterance": index, "speaker": self._label(start, end)})
        return events

    def _next_chunk(self):
        if self.finished:
            return None
        first = self.frames == 0
        start = 0 if first else self.processor.audio_chunk_start(self.frames)
        size = (
            self.processor.num_samples_first_audio_chunk
            if first
            else self.processor.num_samples_per_audio_chunk
        )
        last = start + size > self.samples
        if last and not self.ended:
            return None
        chunk = self.audio[start - self.offset :]
        if last and len(chunk) < self.processor.feature_extractor.win_length:
            self.finished = True  # Shorter than one analysis window.
            return None
        return (chunk if last else chunk[:size]).copy(), first, last

    def _infer(self, audio, first, last):
        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=first,
            is_last_audio_chunk=last,
        ).to(self.model.device, dtype=self.model.dtype)
        with torch.inference_mode():
            output = self.model(**inputs, speaker_cache=self.cache)
            active = (output.logits[0].sigmoid() > 0.5).cpu().numpy()
        return active, output.speaker_cache

    def _accept(self, active, last) -> None:
        end = min(len(self.activity), self.frames + len(active))
        self.activity[self.frames : end] = active[: end - self.frames]
        self.frames = end
        self.finished = last
        keep = self.processor.audio_chunk_start(self.frames)
        if keep > self.offset:
            self.audio = self.audio[keep - self.offset :]
            self.offset = keep

    def _label(self, start: int, end: int) -> str | None:
        if self.model is None:
            return None
        counts = self.activity[start // FRAME_SAMPLES : end // FRAME_SAMPLES].sum(axis=0)
        if not counts.any():
            return None
        # ponytail: one dominant speaker per utterance; a turn change without a
        # pause stays with the longer speaker until R2T2 can split at speaker changes.
        column = int(counts.argmax())
        return self.labels.setdefault(column, f"说话人{len(self.labels) + 1}")
