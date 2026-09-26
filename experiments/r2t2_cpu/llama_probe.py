"""Benchmark the pinned official native R2T2 backend on CPU only."""

from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from r2t2_llama import R2T2LlamaASRModel

import numpy as np
import soundfile as sf


def stream(
    model: R2T2LlamaASRModel, audio: np.ndarray, language: str | None, step_ms: int
) -> dict:
    """Use official state transitions and the example's token-budget policy."""
    step = round(step_ms * 16)
    state = model.init_streaming_state(
        language=language,
        unfixed_chunk_num=0,
        unfixed_token_num=1,
        chunk_size_sec=step_ms / 1000,
    )
    position = 0
    token_budget = max(1, 2 * step // 1280)
    first_budget = token_budget
    token_floor = min(32, max(4, 2 * (step // 1280)))
    last_fixed = ""
    calls = []
    started = time.perf_counter()
    first_text = None
    while position < len(audio):
        size = step * 2 if position == 0 else step
        segment = audio[position : position + size]
        state.chunk_size_samples = size
        state.chunk_size_sec = size / 16000
        position += len(segment)
        # The official adapter only reads max_tokens, without requiring vLLM.
        model.sampling_params = SimpleNamespace(max_tokens=token_budget)
        before = time.perf_counter()
        text, fixed = model.streaming_transcribe(segment, state)
        elapsed = time.perf_counter() - before
        if text and first_text is None:
            first_text = time.perf_counter() - started
        calls.append(
            {
                "audio_end_seconds": position / 16000,
                "seconds": elapsed,
                "text": text,
                "fixed": fixed,
                "max_tokens": token_budget,
            }
        )
        if len(fixed) > len(last_fixed):
            last_fixed = fixed
            token_budget = max(1, step // 1280)
        else:
            token_budget += 1
        token_budget = min(token_floor, token_budget)
    model.sampling_params = SimpleNamespace(max_tokens=first_budget)
    before = time.perf_counter()
    text = model.finish_streaming_transcribe(state).split("|")[0]
    finish_seconds = time.perf_counter() - before
    elapsed = time.perf_counter() - started
    durations = [call["seconds"] for call in calls]
    return {
        "text": text,
        "seconds": elapsed,
        "rtf": elapsed / (len(audio) / 16000),
        "first_text_compute_seconds": first_text,
        "finish_seconds": finish_seconds,
        "step_seconds_p50": float(np.median(durations)),
        "step_seconds_p95": float(np.quantile(durations, 0.95)),
        "calls": calls,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path.home() / ".cache/r2t2-poc")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--samples", nargs="+", default=["zh", "en", "mixed", "silence"]
    )
    parser.add_argument("--mode", choices=["offline", "stream", "both"], default="both")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--language", choices=["auto", "Chinese", "English"])
    parser.add_argument("--step-ms", type=int, default=160)
    args = parser.parse_args()
    if args.step_ms < 1 or args.threads < 1:
        parser.error("--step-ms and --threads must be positive")
    sys.path.insert(0, str(args.cache / "llama-src"))
    from r2t2_llama import R2T2LlamaASRModel
    from r2t2_llama.llama_native_backend import build_asr_prompt

    started = time.perf_counter()
    model = R2T2LlamaASRModel.LlamaNative(
        processor_path=str(args.cache / "llama-processor"),
        gguf_dir=str(args.cache),
        model_gguf_name="llama-decoder-Q8_0.gguf",
        mmproj_gguf_name="llama-mmproj-f16.gguf",
        n_ctx=4096,
        n_batch=2048,
        n_threads=args.threads,
        use_gpu=False,
        n_gpu_layers=0,
        max_new_tokens=512,
    )
    result = {
        "source_revision": "26d55a54ce5670cff9947a167d8ed95d569fd4d9",
        "llama_revision": "ad6c66839af3c5646fba8c6c2e2087a1e4e38948",
        "gguf_revision": "86ff0251cb9f456b63aeef5f80137f104e22869a",
        "device": "cpu",
        "metal_compiled": False,
        "gpu_layers": 0,
        "threads": args.threads,
        "n_ctx": 4096,
        "n_batch": 2048,
        "load_seconds": time.perf_counter() - started,
        "stream_step_ms": args.step_ms,
        "stream_lookahead_ms": args.step_ms,
        "stream_rollback_tokens": 1,
        "samples": [],
    }
    for name in args.samples:
        audio, rate = sf.read(args.cache / "audio" / f"{name}.wav", dtype="float32")
        if rate != 16000 or audio.ndim != 1 or not len(audio):
            raise ValueError("Expected non-empty mono 16 kHz audio")
        language = None if args.language == "auto" else args.language or {"zh": "Chinese", "en": "English"}.get(name)
        sample = {
            "name": name,
            "audio_seconds": len(audio) / rate,
            "forced_language": language,
        }
        if args.mode in {"offline", "both"}:
            before = time.perf_counter()
            output = model.model.backend._generate_with_prompt(
                audio,
                build_asr_prompt(language=language),
                max_tokens=512,
            )
            elapsed = time.perf_counter() - before
            sample["offline"] = {
                "text": output["text"],
                "seconds": elapsed,
                "rtf": elapsed / sample["audio_seconds"],
                "native_output": output,
            }
        if args.mode in {"stream", "both"}:
            sample["stream"] = stream(model, audio, language, args.step_ms)
        sample["process_peak_rss_bytes"] = resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss
        result["samples"].append(sample)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"completed": name, "output": str(args.output)}), flush=True)
    assert result["samples"], "No benchmark samples were run"


if __name__ == "__main__":
    main()
