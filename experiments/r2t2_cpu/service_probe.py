"""Measure a shared CUDA or Rust service with the same local PoC corpus."""

import argparse
import asyncio
import json
import os
import statistics
import time
import urllib.request
import wave
from pathlib import Path

import numpy as np
import websockets


def headers(internal: bool = False) -> dict[str, str]:
    key = os.getenv("R2T2_INTERNAL_TOKEN" if internal else "API_KEY", "")
    return {"Authorization": "Bearer " + key} if key else {}


def pcm(path: Path) -> bytes:
    with wave.open(str(path)) as audio:
        assert (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) == (
            16000,
            1,
            2,
        )
        return audio.readframes(audio.getnframes())


def offline(path: Path, base: str) -> dict[str, object]:
    audio = np.frombuffer(pcm(path), dtype="<i2").astype("<f4") / 32768
    runs = []
    for _ in range(3):
        request = urllib.request.Request(
            base + "/v1/transcribe",
            data=audio.tobytes(),
            headers={**headers(True), "Content-Type": "application/octet-stream"},
        )
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=150) as response:
            result = json.load(response)
        elapsed = time.perf_counter() - started
        runs.append(
            {
                "text": result["text"],
                "seconds": elapsed,
                "rtf": elapsed / (len(audio) / 16000),
            }
        )
    return {
        "audio": path.stem,
        "mode": "offline",
        "audio_seconds": len(audio) / 16000,
        "runs": runs,
        "median_seconds": statistics.median(r["seconds"] for r in runs),
    }


async def stream(path: Path, base: str) -> dict[str, object]:
    data = pcm(path)
    async with websockets.connect(
        base + "/v1/stream", additional_headers=headers()
    ) as ws:
        await ws.send("{}")
        config = json.loads(await asyncio.wait_for(ws.recv(), 15))
        assert config["ready"], config
        started = time.perf_counter()
        first = None
        deltas, lag = [], []

        async def send() -> None:
            for offset in range(0, len(data), 5120):
                chunk = data[offset : offset + 5120]
                await asyncio.sleep(
                    max(
                        0, started + (offset + len(chunk)) / 32000 - time.perf_counter()
                    )
                )
                await ws.send(chunk)
            await ws.send("end")

        sender = asyncio.create_task(send())
        try:
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), 60))
                if event.get("error"):
                    raise RuntimeError(event)
                if event.get("delta"):
                    first = first or time.perf_counter() - started
                    deltas.append(event["delta"])
                lag.append((time.perf_counter() - started) * 1000 - event["audio_ms"])
                if event["done"]:
                    await sender
                    assert "".join(deltas) == event["text"]
                    return {
                        "audio": path.stem,
                        "mode": "stream",
                        "text": event["text"],
                        "audio_seconds": len(data) / 32000,
                        "wall_seconds": time.perf_counter() - started,
                        "first_text_seconds": first,
                        "updates": len(deltas),
                        "frame_lag_p95_ms": sorted(lag)[
                            min(len(lag) - 1, int(0.95 * len(lag)))
                        ],
                        "final_lag_ms": lag[-1],
                    }
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="vllm-cuda")
    parser.add_argument("--mode", choices=("offline", "stream", "both"), default="both")
    parser.add_argument("--samples", nargs="*")
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--engine-url", default="http://127.0.0.1:8001")
    parser.add_argument("--stream-url", default="ws://127.0.0.1:8000")
    args = parser.parse_args()
    request = urllib.request.Request(
        args.engine_url + "/v1/config", headers=headers(True)
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        config = json.load(response)
    report = {"backend": args.backend, "config": config, "results": []}
    paths = (
        [args.audio_dir / (name + ".wav") for name in args.samples]
        if args.samples
        else sorted(args.audio_dir.glob("*.wav"))
    )
    for path in paths:
        for mode in (("offline", "stream") if args.mode == "both" else (args.mode,)):
            try:
                result = (
                    offline(path, args.engine_url)
                    if mode == "offline"
                    else asyncio.run(stream(path, args.stream_url))
                )
            except Exception as error:
                result = {"audio": path.stem, "mode": mode, "error": str(error)}
            report["results"].append(result)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
