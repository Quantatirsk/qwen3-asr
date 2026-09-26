"""Native async R2T2 decoding with bounded, per-connection rolling state.

Adapted from NetEase Youdao Confucius4-R2T2 (Apache-2.0), revision
26d55a54ce5670cff9947a167d8ed95d569fd4d9, streaming_transcribe_no_reset.
We use sample positions instead of hardcoded chunk counts, retain English
spaces, and flush withheld tokens even when the input ends on a chunk boundary.
"""

import math
import os
import re
import uuid
from collections import deque

import numpy as np

from app.core.config import settings

from .protocol import (
    CHUNK_SAMPLES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    SAMPLE_RATE,
    StreamConfig,
    OFFLINE_TAIL_SAMPLES,
)

WINDOW_SAMPLES = 16 * SAMPLE_RATE
DISCARD_SAMPLES = 8 * SAMPLE_RATE
TAG = "<asr_text>"


class Session:
    """Only this connection's task may mutate this state."""

    def __init__(self, prompt: str, chunk_samples: int = CHUNK_SAMPLES) -> None:
        self.chunk_samples = chunk_samples
        self.base_budget = max(4, math.ceil(4 * chunk_samples / CHUNK_SAMPLES))
        self.max_budget = max(16, 4 * self.base_budget)
        self.final_budget = max(128, 2 * self.max_budget)
        self.next_decode = 2 * chunk_samples
        self.id = uuid.uuid4().hex
        self.prompt = prompt
        self.audio = np.empty(WINDOW_SAMPLES + 2 * chunk_samples, dtype=np.float32)
        self.size = 0
        self.offset = 0
        self.samples = 0
        self.parts = deque()  # (absolute audio end, committed text delta)
        self.text = ""
        self.header = ""
        self.request_id = None
        self.budget = self.max_budget  # Include room for the language header.
        self.steps = 0
        self.quiet_samples = 0
        self.has_speech = False

    @property
    def next_samples(self) -> int:
        samples = min(CHUNK_SAMPLES, self.chunk_samples)
        return samples if self.size else 2 * samples

    def reset_window(self) -> None:
        """End an utterance without ending the connection or losing its text."""
        self.size = 0
        self.offset = self.samples
        self.parts.clear()
        self.header = ""
        self.budget = self.max_budget
        self.quiet_samples = 0
        self.has_speech = False
        self.next_decode = self.samples + 2 * self.chunk_samples

    def append(self, audio: np.ndarray) -> None:
        if self.size + len(audio) > WINDOW_SAMPLES:
            self.audio[: self.size - DISCARD_SAMPLES] = self.audio[
                DISCARD_SAMPLES : self.size
            ]
            self.size -= DISCARD_SAMPLES
            self.offset += DISCARD_SAMPLES
            while self.parts and self.parts[0][0] <= self.offset:
                self.parts.popleft()
        self.audio[self.size : self.size + len(audio)] = audio
        self.size += len(audio)
        self.samples += len(audio)

    def commit(self, candidate: str, prefix: str) -> str:
        if not candidate.startswith(prefix):
            # A tokenizer boundary must never rewrite already published text.
            return ""
        delta = candidate[len(prefix) :]
        if delta:
            # Do not run two English utterances together after a silence reset.
            separator = (
                " "
                if not self.parts
                and self.offset
                and self.text
                and self.text[-1].isascii()
                and not self.text[-1].isspace()
                and delta[0].isascii()
                and delta[0].isalnum()
                else ""
            )
            self.parts.append((self.samples, delta))
            self.text += separator + delta
            return separator + delta
        return ""


class Model:
    def __init__(self, max_sessions: int) -> None:
        self.device = settings.DEVICE
        seconds = float(
            os.getenv("R2T2_CHUNK_SECONDS", "0.64" if self.device == "cpu" else "0.16")
        )
        if not math.isfinite(seconds) or not 0.08 <= seconds <= 2:
            raise ValueError("R2T2_CHUNK_SECONDS must be between 0.08 and 2 seconds")
        self.chunk_samples = round(seconds * SAMPLE_RATE)
        if self.device == "cpu":
            from .rust_engine import RustAsyncEngine, SamplingParams

            self.engine = RustAsyncEngine(max_sessions)
            self.processor = self.engine.tokenizer
            self.tokenizer = self.engine.tokenizer
        else:
            from vllm import SamplingParams
            from vllm.engine.arg_utils import AsyncEngineArgs
            from vllm.transformers_utils.processors.qwen3_asr import Qwen3ASRProcessor
            from vllm.v1.engine.async_llm import AsyncLLM

            path = os.getenv("R2T2_MODEL_PATH", MODEL_REPOSITORY)
            revision = None if os.path.isdir(path) else MODEL_REVISION
            self.processor = Qwen3ASRProcessor.from_pretrained(
                path, revision=revision, fix_mistral_regex=True
            )
            self.tokenizer = self.processor.tokenizer
            max_model_len = int(os.getenv("R2T2_MAX_MODEL_LEN", "16384"))
            if max_model_len <= 4096:
                raise ValueError(
                    "R2T2_MAX_MODEL_LEN must exceed the offline output budget"
                )
            self.engine = AsyncLLM.from_engine_args(
                AsyncEngineArgs(
                    model=path,
                    revision=revision,
                    gpu_memory_utilization=float(
                        os.getenv("R2T2_GPU_MEMORY_UTILIZATION", "0.30")
                    ),
                    max_model_len=max_model_len,
                    max_num_seqs=max_sessions + 1,
                    max_num_batched_tokens=2048,
                    enable_chunked_prefill=True,
                    scheduling_policy="priority",
                    mm_processor_cache_gb=0,
                    enable_prefix_caching=False,
                    enforce_eager=os.getenv("R2T2_ENFORCE_EAGER", "0") == "1",
                    disable_log_stats=True,
                )
            )
        state = Session("", self.chunk_samples)
        self.sampling = {
            n: SamplingParams(temperature=0, max_tokens=n, skip_special_tokens=True)
            for n in (*range(4, state.max_budget + 1), state.final_budget)
        }
        self.offline_sampling = SamplingParams(
            temperature=0, max_tokens=4096, skip_special_tokens=True
        )

    def _prompt(self, context: str) -> str:
        return self.processor.apply_chat_template(
            [
                {"role": "system", "content": context},
                {"role": "user", "content": [{"type": "audio", "audio": ""}]},
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

    def new_session(self, config: StreamConfig) -> Session:
        # No forced language. The model detects Chinese, English, or mixed speech.
        return Session(self._prompt(config.context), self.chunk_samples)

    async def transcribe(self, audio: np.ndarray, context: str) -> str:
        """Commit a full segment, then resolve its pending tail with end lookahead."""
        prefix = ""
        prompt = self._prompt(context)
        for final in (False, True):
            # Padding is only for ASR; alignment keeps the original audio timeline.
            samples = np.pad(audio, (0, OFFLINE_TAIL_SAMPLES)) if final else audio
            request_id = f"offline-{uuid.uuid4().hex}"
            result = None
            completed = False
            try:
                async for result in self.engine.generate(
                    {
                        "prompt": prompt + prefix,
                        "multi_modal_data": {"audio": [samples]},
                    },
                    self.sampling[128] if final else self.offline_sampling,
                    request_id=request_id,
                    priority=10,
                ):
                    pass
                if result is None or not result.outputs:
                    raise RuntimeError("R2T2 returned no offline output")
                if result.outputs[0].finish_reason == "length":
                    raise RuntimeError(
                        "R2T2 offline decoding exceeded its token budget"
                    )
                prefix = (prefix + result.outputs[0].text).split("|", 1)[0].strip()
                if TAG not in prefix:
                    raise RuntimeError("R2T2 offline output has no language header")
                completed = True
            finally:
                if not completed:
                    await self.engine.abort(request_id)
            if not prefix.split(TAG, 1)[1].strip():
                return ""
        return prefix.split(TAG, 1)[1].strip()

    async def push(
        self, session: Session, audio: np.ndarray, *, final: bool = False
    ) -> str:
        session.append(audio)
        if len(audio):
            # A soft utterance boundary, not a speech filter: every input sample
            # is decoded. A pause releases the previous language/text constraint.
            if float(np.sqrt(np.mean(np.square(audio)))) < 0.004:
                session.quiet_samples += len(audio)
            else:
                session.quiet_samples = 0
                session.has_speech = True
        if not session.size:
            return ""
        if not final and session.samples < session.next_decode:
            # Keep pause detection at 160 ms even when CPU decoding runs less often.
            if session.has_speech and session.quiet_samples >= 2 * CHUNK_SAMPLES:
                delta = await self.push(
                    session, np.empty(0, dtype=np.float32), final=True
                )
                session.reset_window()
                return delta
            return ""
        session.next_decode = session.samples + session.chunk_samples
        prefix = "".join(text for _, text in session.parts)
        header = session.header
        prompt = {
            "prompt": session.prompt + header + prefix,
            "multi_modal_data": {"audio": [session.audio[: session.size].copy()]},
        }
        request_id = f"{session.id}-{session.steps}"
        session.request_id = request_id
        result = None
        completed = False
        try:
            async for result in self.engine.generate(
                prompt,
                self.sampling[session.final_budget if final else session.budget],
                request_id=request_id,
                priority=0,
            ):
                pass
            if result is None or not result.outputs:
                raise RuntimeError("R2T2 returned no output")
            completed = True
        finally:
            try:
                if not completed:
                    await self.engine.abort(request_id)
            finally:
                session.request_id = None
        raw = (header + prefix + result.outputs[0].text).split("|", 1)[0]
        raw = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", raw)
        if TAG not in raw:
            session.steps += 1
            session.budget = session.max_budget
            return ""
        meta, candidate = raw.split(TAG, 1)
        # R2T2 may emit "language None" even with real speech. It is still a
        # complete metadata header: dropping it would make later text unparseable.
        session.header = meta + TAG
        if not final:
            # Keep one token uncommitted, including any incomplete UTF-8 token.
            ids = self.tokenizer.encode(candidate, add_special_tokens=False)
            ids = ids[:-1]
            candidate = self.tokenizer.decode(ids)
            while "\ufffd" in candidate and ids:
                ids.pop()
                candidate = self.tokenizer.decode(ids)
        else:
            candidate = candidate.replace("\ufffd", "")
            if result.outputs[0].finish_reason == "length":
                raise RuntimeError("R2T2 final decoding exceeded its token budget")
        delta = session.commit(candidate, prefix)
        session.budget = (
            session.base_budget
            if delta
            else min(session.max_budget, session.budget + 1)
        )
        session.steps += 1
        if (
            not final
            and session.has_speech
            and session.quiet_samples >= 2 * CHUNK_SAMPLES
        ):
            delta += await self.push(session, np.empty(0, dtype=np.float32), final=True)
            session.reset_window()
        return delta

    async def abort(self, session: Session) -> None:
        if session.request_id is not None:
            await self.engine.abort(session.request_id)

    async def warmup(self) -> None:
        session = self.new_session(StreamConfig())
        await self.push(session, np.zeros(2 * self.chunk_samples, dtype=np.float32))

    async def shutdown(self) -> None:
        if self.device == "cpu":
            await self.engine.shutdown()
        else:
            self.engine.shutdown()
