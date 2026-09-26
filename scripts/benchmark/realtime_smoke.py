"""Replay real audio through the public R2T2 API, including concurrent clients.

python -m scripts.benchmark.realtime_smoke zh.wav en.wav --concurrency 4
"""

import argparse
import asyncio
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import websockets


def text_distance(left, right):
    """Normalized character edit distance, ignoring punctuation and casing."""
    left = "".join(c for c in left.casefold() if c.isalnum())
    right = "".join(c for c in right.casefold() if c.isalnum())
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        updated = [i]
        for j, b in enumerate(right, 1):
            updated.append(min(updated[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = updated
    return row[-1] / max(len(left), len(right), 1)


def load_pcm(path, seconds):
    return subprocess.check_output(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-t",
            str(seconds),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "pipe:1",
        ]
    )


async def replay(url, pcm, headers):
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send("{}")
        ready = json.loads(await asyncio.wait_for(ws.recv(), 15))
        assert ready.get("ready"), ready
        start = time.perf_counter()
        latencies, costs, deltas = [], [], []
        first = None

        async def send():
            for offset in range(0, len(pcm), 5120):
                frame = pcm[offset : offset + 5120]
                await asyncio.sleep(
                    max(0, start + (offset + len(frame)) / 32000 - time.perf_counter())
                )
                await ws.send(frame)
            await ws.send("end")

        sender = asyncio.create_task(send())
        try:
            while True:
                event = json.loads(await asyncio.wait_for(ws.recv(), 35))
                assert not event.get("error"), event
                if "speaker" in event:
                    continue  # Speaker labels carry no timing.
                if event.get("delta"):
                    first = first or (time.perf_counter() - start)
                    deltas.append(event["delta"])
                costs.append(event["inference_ms"])
                latencies.append(
                    1000 * (time.perf_counter() - start) - event["audio_ms"]
                )
                if event["done"]:
                    await sender
                    assert "".join(deltas) == event["text"], event
                    assert event["text"], "Empty transcription"
                    assert len(deltas) > 1, "Expected multiple incremental text updates"
                    return {
                        "text": event["text"],
                        "updates": len(deltas),
                        "audio_s": len(pcm) / 32000,
                        "first_text_ms": round(first * 1000),
                        "frame_latency_p95_ms": round(
                            sorted(latencies)[
                                min(len(latencies) - 1, int(len(latencies) * 0.95))
                            ]
                        ),
                        "inference_mean_ms": round(statistics.mean(costs), 1),
                        "final_lag_ms": round(latencies[-1]),
                    }
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)


async def main(args):
    url = (
        args.url.replace("http://", "ws://").replace("https://", "wss://").rstrip("/")
        + "/v1/stream"
    )
    headers = {"Authorization": "Bearer " + args.api_key} if args.api_key else {}
    sources = [load_pcm(path, args.seconds) for path in args.audio]
    report = {"single": [await replay(url, pcm, headers) for pcm in sources]}
    report["concurrent"] = await asyncio.gather(
        *[
            replay(url, sources[i % len(sources)], headers)
            for i in range(args.concurrency)
        ]
    )
    # Greedy decoding is not bitwise invariant to GPU batching. Record small
    # recognition changes, reject material changes and matches to another input.
    report["isolation"] = []
    for i, result in enumerate(report["concurrent"]):
        reference = report["single"][i % len(sources)]["text"]
        distance = text_distance(result["text"], reference)
        assert distance <= 0.05, report
        for other in report["single"]:
            if text_distance(reference, other["text"]) > 0.1:
                assert distance < text_distance(result["text"], other["text"]), report
        report["isolation"].append(
            {
                "exact_match": result["text"] == reference,
                "character_difference": distance,
            }
        )
    if len(sources) >= 2:
        report["switching"] = [
            await replay(url, sources[0] + b"\0" * 16000 + sources[1], headers),
            await replay(url, sources[1] + b"\0" * 16000 + sources[0], headers),
        ]
        for result in report["switching"]:
            assert any("\u4e00" <= c <= "\u9fff" for c in result["text"]), result
            assert sum(c.isascii() and c.isalpha() for c in result["text"]) >= 10, (
                result
            )
        report["long"] = await replay(url, (sources[0] + sources[1]) * 3, headers)
    # Check admission and immediate capacity recovery with idle connections.
    connections = []
    try:
        for _ in range(args.concurrency):
            ws = await websockets.connect(url, additional_headers=headers)
            connections.append(ws)
            await ws.send("{}")
            ready = json.loads(await ws.recv())
            assert ready["ready"], ready
        if ready["max_sessions"] == args.concurrency:
            async with websockets.connect(url, additional_headers=headers) as extra:
                await extra.send("{}")
                result = json.loads(await extra.recv())
                assert result["code"] == "capacity_exceeded", result
                report["capacity_rejection"] = True
    finally:
        await asyncio.gather(*(ws.close() for ws in connections))
    await asyncio.sleep(0.25)
    async with websockets.connect(url, additional_headers=headers) as ws:
        await ws.send("{}")
        result = json.loads(await ws.recv())
        assert result["active_sessions"] == 1, result
        report["disconnect_recovery"] = True
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", nargs="+", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:4174")
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--seconds", type=float, default=7)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--output", type=Path)
    asyncio.run(main(parser.parse_args()))
