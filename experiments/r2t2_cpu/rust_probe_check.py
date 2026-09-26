"""Run with Python directly; checks probe boundaries without model inference."""

import tempfile
import wave
from pathlib import Path

from rust_probe import clean, pcm

assert clean("hello|ignored") == "hello"
assert clean("  hello  ") == "hello"
with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / "silence.wav"
    with wave.open(str(path), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(32000))
    samples, duration = pcm(path)
    assert duration == 1 and len(samples) == 16000
    assert not any(samples)
print("Rust probe checks passed")
