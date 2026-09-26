"""Isolated Rust R2T2 probes; no service, model download, or source mutation."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
import wave
from itertools import pairwise
from pathlib import Path


def emit(value: dict[str, object]) -> None:
    print(json.dumps(value, ensure_ascii=False), flush=True)


def rss_mib() -> float:
    units = 1024 * 1024 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / units


def clean(text: str) -> str:
    return text.split("|", 1)[0].split("<asr_text>", 1)[-1].strip()


def pcm(path: Path) -> tuple[ctypes.Array[ctypes.c_float], float]:
    with wave.open(str(path)) as wav:
        assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (
            16000,
            1,
            2,
        )
        raw = wav.readframes(wav.getnframes())
    samples = memoryview(raw).cast("h")
    return (ctypes.c_float * len(samples))(*(value / 32768 for value in samples)), len(
        samples
    ) / 16000


def library(path: Path) -> ctypes.CDLL:
    lib = ctypes.CDLL(str(path))
    ptr, integer, string = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    signatures = {
        "load_model": ([string, integer, integer], ptr),
        "transcribe_file": ([ptr, string], ptr),
        "free_string": ([ptr], None),
        "free": ([ptr], None),
        "stream_new": ([], ptr),
        "stream_free": ([ptr], None),
        "stream_push": (
            [ptr, ptr, ctypes.POINTER(ctypes.c_float), integer, integer],
            ptr,
        ),
        "stream_get_result": ([ptr], ptr),
        "stream_set_chunk_sec": ([ptr, ctypes.c_float], None),
        "force_align_file": ([ptr, string, string, string], ptr),
    }
    for name, (args, result) in signatures.items():
        fn = getattr(lib, "qwen_asr_" + name)
        fn.argtypes, fn.restype = args, result
    return lib


def consume(lib: ctypes.CDLL, address: int | None) -> str:
    if not address:
        return ""
    try:
        return ctypes.string_at(address).decode("utf-8")
    finally:
        lib.qwen_asr_free_string(address)


def worker(args: argparse.Namespace) -> None:
    lib = library(args.library)
    started = time.perf_counter()
    engine = lib.qwen_asr_load_model(str(args.model).encode(), args.threads, 1)
    if not engine:
        raise RuntimeError("Rust model load failed")
    emit(
        {
            "event": "load",
            "seconds": time.perf_counter() - started,
            "peak_rss_mib": rss_mib(),
        }
    )
    try:
        if args.mode == "shared":
            started = time.perf_counter()
            second = lib.qwen_asr_load_model(str(args.model).encode(), args.threads, 1)
            assert second
            emit(
                {
                    "event": "second_context",
                    "seconds": time.perf_counter() - started,
                    "peak_rss_mib": rss_mib(),
                }
            )
            lib.qwen_asr_free(second)
            return
        samples, duration = pcm(args.audio)
        if args.mode == "align":
            started = time.perf_counter()
            raw = consume(
                lib,
                lib.qwen_asr_force_align_file(
                    engine,
                    str(args.audio).encode(),
                    args.text.encode(),
                    args.language.encode(),
                ),
            )
            words = json.loads(raw)
            monotonic = bool(words) and all(
                0 <= word["start_ms"] <= word["end_ms"] <= duration * 1000 + 1
                for word in words
            )
            monotonic = monotonic and all(
                a["end_ms"] <= b["start_ms"] for a, b in pairwise(words)
            )
            emit(
                {
                    "event": "alignment",
                    "seconds": time.perf_counter() - started,
                    "words": words,
                    "monotonic_in_bounds": monotonic,
                    "peak_rss_mib": rss_mib(),
                }
            )
            return
        for repetition in range(args.repeats):
            started = time.perf_counter()
            events: list[dict[str, object]] = []
            if args.mode == "generate":
                from jinja2.sandbox import SandboxedEnvironment
                from tokenizers import Tokenizer

                tokenizer = Tokenizer.from_file(str(args.model / "tokenizer.json"))
                template = json.loads((args.model / "chat_template.json").read_text())[
                    "chat_template"
                ]
                prompt = (
                    SandboxedEnvironment()
                    .from_string(template)
                    .render(
                        messages=[
                            {"role": "system", "content": args.context},
                            {
                                "role": "user",
                                "content": [{"type": "audio", "audio": ""}],
                            },
                        ],
                        add_generation_prompt=True,
                    )
                    + args.assistant_prefix
                )
                ids = tokenizer.encode(prompt, add_special_tokens=False).ids
                assert ids.count(151676) == 1
                split = ids.index(151676)
                before = (ctypes.c_int * split)(*ids[:split])
                after = (ctypes.c_int * (len(ids) - split - 1))(*ids[split + 1 :])
                generate = lib.qwen_asr_generate_pcm
                generate.argtypes = [
                    ctypes.c_void_p,
                    ctypes.POINTER(ctypes.c_float),
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int,
                    ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int,
                    ctypes.c_int,
                ]
                generate.restype = ctypes.c_void_p
                generated = json.loads(
                    consume(
                        lib,
                        generate(
                            engine,
                            samples,
                            len(samples),
                            before,
                            len(before),
                            after,
                            len(after),
                            args.max_tokens,
                        ),
                    )
                )
                raw = tokenizer.decode(
                    generated["token_ids"], skip_special_tokens=False
                )
                events.append(generated)
            elif args.mode == "offline":
                address = lib.qwen_asr_transcribe_file(engine, str(args.audio).encode())
                if not address:
                    raise RuntimeError("Rust transcription failed")
                raw = consume(lib, address)
            else:
                lib.qwen_asr_stream_set_chunk_sec(engine, args.chunk_seconds)
                state = lib.qwen_asr_stream_new()
                try:
                    for offset in range(0, len(samples), 2560):
                        count = min(2560, len(samples) - offset)
                        pointer = ctypes.cast(
                            ctypes.byref(samples, offset * 4),
                            ctypes.POINTER(ctypes.c_float),
                        )
                        call_started = time.perf_counter()
                        delta = consume(
                            lib,
                            lib.qwen_asr_stream_push(engine, state, pointer, count, 0),
                        )
                        events.append(
                            {
                                "audio_seconds": (offset + count) / 16000,
                                "call_seconds": time.perf_counter() - call_started,
                                "elapsed_seconds": time.perf_counter() - started,
                                "delta": delta,
                            }
                        )
                    call_started = time.perf_counter()
                    delta = consume(
                        lib, lib.qwen_asr_stream_push(engine, state, None, 0, 1)
                    )
                    events.append(
                        {
                            "final": True,
                            "call_seconds": time.perf_counter() - call_started,
                            "elapsed_seconds": time.perf_counter() - started,
                            "delta": delta,
                        }
                    )
                    raw = consume(lib, lib.qwen_asr_stream_get_result(state))
                finally:
                    lib.qwen_asr_stream_free(state)
            seconds = time.perf_counter() - started
            emit(
                {
                    "event": "transcription",
                    "repetition": repetition,
                    "raw": raw,
                    "clean": clean(raw),
                    "seconds": seconds,
                    "audio_seconds": duration,
                    "rtf": seconds / duration,
                    "peak_rss_mib": rss_mib(),
                    "stream_events": events,
                }
            )
    finally:
        lib.qwen_asr_free(engine)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audio", type=Path)
    parser.add_argument(
        "--mode",
        choices=["offline", "stream", "align", "shared", "generate"],
        default="offline",
    )
    parser.add_argument("--chunk-seconds", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--text", default="")
    parser.add_argument("--context", default="")
    parser.add_argument("--assistant-prefix", default="")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker"]
    started = time.perf_counter()
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=120, check=False
        )
        stdout, stderr, status = result.stdout, result.stderr, result.returncode
    except subprocess.TimeoutExpired as error:
        stdout = (
            (error.stdout or b"").decode()
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            (error.stderr or b"").decode()
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        status = "timeout_120s"
    events = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    report = {
        "platform": platform.platform(),
        "library": str(args.library.resolve()),
        "library_sha256": hashlib.sha256(args.library.read_bytes()).hexdigest(),
        "model": str(args.model.resolve()),
        "audio": str(args.audio) if args.audio else None,
        "audio_sha256": (
            hashlib.sha256(args.audio.read_bytes()).hexdigest() if args.audio else None
        ),
        "mode": args.mode,
        "context": args.context if args.mode == "generate" else None,
        "assistant_prefix": args.assistant_prefix if args.mode == "generate" else None,
        "max_tokens": args.max_tokens if args.mode == "generate" else None,
        "chunk_seconds": args.chunk_seconds if args.mode == "stream" else None,
        "threads": args.threads,
        "status": status,
        "wall_seconds": time.perf_counter() - started,
        "events": events,
        "stderr": stderr,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    emit(report)
    if status != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
