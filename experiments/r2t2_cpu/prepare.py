"""Download public audio fixtures and create a reproducible CPU/GPU test corpus."""

import argparse
import hashlib
import json
import shutil
import subprocess
import urllib.request
import wave
from pathlib import Path

import numpy as np

SOURCES = {
    "zh": "https://raw.githubusercontent.com/netease-youdao/Confucius4-R2T2/26d55a54ce5670cff9947a167d8ed95d569fd4d9/resources/test.wav",
    "en": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
}


def read(path: Path) -> np.ndarray:
    with wave.open(str(path)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (
            16000,
            1,
            2,
        )
        return np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")


def write(path: Path, samples: np.ndarray) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(samples.astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path.home() / ".cache/r2t2-poc/audio"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Copy existing normalized zh.wav/en.wav instead of downloading",
    )
    args = parser.parse_args()
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCES.items():
        target = root / (name + ".wav")
        if args.source_dir:
            source = args.source_dir / target.name
            if source.resolve() != target.resolve():
                shutil.copyfile(source, target)
        elif not target.exists():
            raw = root / (name + "-source.wav")
            with urllib.request.urlopen(url, timeout=60) as response:
                raw.write_bytes(response.read())
            if name == "zh":
                shutil.copyfile(raw, target)
            else:
                subprocess.run(
                    [
                        "ffmpeg",
                        "-v",
                        "error",
                        "-y",
                        "-i",
                        str(raw),
                        "-t",
                        "7",
                        "-ac",
                        "1",
                        "-ar",
                        "16000",
                        "-c:a",
                        "pcm_s16le",
                        str(target),
                    ],
                    check=True,
                )
    zh, en = read(root / "zh.wav"), read(root / "en.wav")
    gap = np.zeros(8000, dtype="<i2")
    write(root / "mixed.wav", np.concatenate([zh, gap, en]))
    write(root / "silence.wav", np.zeros(80000, dtype="<i2"))
    quiet = zh.copy()
    quiet[:24000] = (quiet[:24000].astype(np.float32) * 0.05).astype("<i2")
    write(root / "quiet.wav", quiet)
    write(root / "long.wav", np.tile(np.concatenate([zh, gap, en, gap]), 3))
    manifest = {
        "sample_rate": 16000,
        "sources": SOURCES,
        "reference_kind": "CUDA outputs are cross-backend references, not human ground truth",
        "audio": [],
    }
    for name in ("zh", "en", "mixed", "silence", "quiet", "long"):
        path = root / (name + ".wav")
        data = read(path)
        manifest["audio"].append(
            {
                "name": name,
                "seconds": len(data) / 16000,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "pcm_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
            }
        )
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(root / "manifest.json")


if __name__ == "__main__":
    main()
