import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from app.services.realtime.speakers import SpeakerStream

STEP = 72  # Nemotron low-latency frames per chunk.


class FakeProcessor:
    """Real chunk geometry; each output frame is speaker 0 if its audio is positive."""

    num_samples_first_audio_chunk = 16680
    num_samples_per_audio_chunk = 17040
    feature_extractor = SimpleNamespace(win_length=400)

    def audio_chunk_start(self, frame):
        return frame * 160 - 256

    def __call__(self, audio, *, is_first_audio_chunk, is_last_audio_chunk, **_):
        start = 0 if is_first_audio_chunk else 256
        frames = len(audio[start:]) // 160 if is_last_audio_chunk else STEP
        signs = audio[start : start + frames * 160 : 160]
        logits = torch.full((1, frames, 8), -9.0)
        logits[0, :, 0] = torch.where(torch.tensor(signs) > 0, 9.0, -9.0)
        logits[0, :, 1] = -logits[0, :, 0]
        return SimpleNamespace(to=lambda *a, **k: {"logits": logits})


class FakeModel:
    device, dtype = "cpu", torch.float32

    def __call__(self, logits, speaker_cache):
        return SimpleNamespace(logits=logits, speaker_cache=(speaker_cache or 0) + 1)


def pcm(value, seconds):
    return (np.full(int(seconds * 16000), value, dtype="<i2")).tobytes()


class SpeakerStreamTest(unittest.IsolatedAsyncioTestCase):
    async def test_labels_wait_for_coverage_and_precede_done(self):
        diarizer = SimpleNamespace(streaming_parts=lambda: (FakeModel(), FakeProcessor()))
        with patch("app.services.realtime.speakers.get_speaker_diarizer", return_value=diarizer):
            stream = SpeakerStream()
        stream.feed(pcm(1000, 3))
        self.assertEqual(await stream.advance(), [])
        stream.utterance(0, 0, 3000)
        self.assertEqual(stream.ready(), [])  # Nemotron has not covered 3 s yet.
        stream.feed(pcm(-1000, 3))
        self.assertEqual(await stream.advance(), [{"utterance": 0, "speaker": "说话人1"}])
        stream.utterance(1, 3000, 6000)
        stream.utterance(2, 5990, 6000)
        stream.ended = True
        self.assertEqual(
            await stream.advance(),
            [{"utterance": 1, "speaker": "说话人2"}, {"utterance": 2, "speaker": "说话人2"}],
        )
        self.assertTrue(stream.finished)
        self.assertEqual(stream.frames, 600)

    async def test_failure_gives_unknown_labels(self):
        diarizer = SimpleNamespace(streaming_parts=lambda: (FakeModel(), None))
        with patch("app.services.realtime.speakers.get_speaker_diarizer", return_value=diarizer):
            stream = SpeakerStream()
        stream.feed(pcm(1000, 2))
        stream.utterance(0, 0, 2000)
        stream.ended = True
        self.assertEqual(await stream.advance(), [{"utterance": 0, "speaker": None}])


if __name__ == "__main__":
    unittest.main()
