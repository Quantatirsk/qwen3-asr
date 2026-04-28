# -*- coding: utf-8 -*-
"""Qwen3-ASR websocket streaming service."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import WebSocket, WebSocketDisconnect

from app.core.exceptions import create_error_response
from app.core.executor import run_sync
from app.services.asr.model_selection import validate_realtime_model_id
from app.services.asr.qwen3_engine import Qwen3ASREngine, Qwen3StreamingState
from app.services.asr.runtime import RuntimeEngineLease, get_runtime_router

logger = logging.getLogger(__name__)


def _convert_audio(
    audio_bytes: bytes,
    fmt: str,
    sample_rate: int,
) -> Optional[np.ndarray]:
    try:
        if fmt == "wav" and len(audio_bytes) > 44:
            audio_bytes = audio_bytes[44:]

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        if sample_rate != 16000:
            import scipy.signal

            num = int(len(audio) * 16000 / sample_rate)
            audio = scipy.signal.resample(audio, num)
            if isinstance(audio, tuple):
                audio = audio[0]

        return audio
    except Exception as exc:
        logger.error("Audio conversion failed: %s", exc)
        return None


class ConnectionState(IntEnum):
    READY = 1
    STARTED = 2
    STREAMING = 3


@dataclass
class ConnectionContext:
    state: ConnectionState = ConnectionState.READY
    params: Dict[str, Any] = field(default_factory=dict)
    engine_lease: Optional[RuntimeEngineLease] = None
    engine: Optional[Qwen3ASREngine] = None
    audio_buffer: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    streaming_state: Optional[Qwen3StreamingState] = None
    silence_samples: int = 0
    total_samples: int = 0
    confirmed_segments: List[Dict[str, Any]] = field(default_factory=list)
    segment_index: int = 0

    SILENCE_THRESHOLD = 32000
    MAX_BUFFER = 960000


class Qwen3ASRService:
    async def _ensure_engine(self, ctx: ConnectionContext) -> Qwen3ASREngine:
        if ctx.engine is not None:
            return ctx.engine

        runtime_router = get_runtime_router()
        model = validate_realtime_model_id("qwen3-asr")
        logger.info("Using Qwen3-ASR model: %s", model)

        ctx.engine_lease = await runtime_router.acquire_engine(model)
        engine = ctx.engine_lease.engine
        if not isinstance(engine, Qwen3ASREngine):
            raise RuntimeError("Current model is not Qwen3-ASR")
        if not engine.supports_realtime:
            raise RuntimeError(
                f"Current device {engine.device} does not support Qwen3-ASR realtime streaming; "
                "only CUDA vLLM and CPU Rust paths are supported"
            )

        ctx.engine = engine
        return engine

    def _has_voice(self, audio: np.ndarray) -> bool:
        return np.sqrt(np.mean(audio**2)) >= 0.015

    def _need_truncate(self, ctx: ConnectionContext) -> bool:
        return (
            ctx.total_samples >= ctx.MAX_BUFFER
            or ctx.silence_samples >= ctx.SILENCE_THRESHOLD
        )

    async def _truncate(
        self,
        websocket: WebSocket,
        ctx: ConnectionContext,
        task_id: str,
        reason: str,
    ) -> None:
        try:
            engine = await self._ensure_engine(ctx)

            if len(ctx.audio_buffer) > 0:
                ctx.streaming_state = await run_sync(
                    engine.streaming_transcribe,
                    ctx.audio_buffer,
                    ctx.streaming_state,
                )

            ctx.streaming_state = await run_sync(
                engine.finish_streaming_transcribe,
                ctx.streaming_state,
            )

            segment_text = ctx.streaming_state.last_text or ""
            is_valid = len(segment_text.strip()) >= 3

            if is_valid:
                ctx.confirmed_segments.append(
                    {
                        "index": ctx.segment_index,
                        "text": segment_text,
                        "language": ctx.streaming_state.last_language,
                        "reason": reason,
                    }
                )

                confirmed = "\n".join([segment["text"] for segment in ctx.confirmed_segments])
                full_text = confirmed + "\n" + segment_text if confirmed else segment_text

                await websocket.send_json(
                    {
                        "type": "segment_end",
                        "task_id": task_id,
                        "segment_index": ctx.segment_index,
                        "reason": reason,
                        "result": {
                            "text": full_text,
                            "segment_text": segment_text,
                            "language": ctx.streaming_state.last_language,
                        },
                        "confirmed_texts": [
                            segment["text"] for segment in ctx.confirmed_segments
                        ],
                    }
                )

                ctx.segment_index += 1
                logger.info("[%s] Segment truncated, new segment %s", task_id, ctx.segment_index)
            else:
                logger.debug("[%s] Short segment filtered: %r", task_id, segment_text)

            ctx.audio_buffer = np.array([], dtype=np.float32)
            ctx.silence_samples = 0
            ctx.total_samples = 0
            ctx.streaming_state = engine.init_streaming_state(
                context=ctx.params.get("context", ""),
                language=ctx.params.get("language"),
                chunk_size_sec=ctx.params.get("chunk_size_sec", 2.0),
                unfixed_chunk_num=ctx.params.get("unfixed_chunk_num", 2),
                unfixed_token_num=ctx.params.get("unfixed_token_num", 5),
            )

            if is_valid:
                await websocket.send_json(
                    {
                        "type": "segment_start",
                        "task_id": task_id,
                        "segment_index": ctx.segment_index,
                    }
                )

        except Exception as exc:
            logger.error("[%s] Truncate failed: %s", task_id, exc)
            raise

    async def _send_error(
        self,
        websocket: WebSocket,
        message: str,
        task_id: str,
        code: str = "DEFAULT_SERVER_ERROR",
    ) -> None:
        try:
            error = create_error_response(error_code=code, message=message, task_id=task_id)
            error["type"] = "error"
            await websocket.send_json(error)
        except Exception:
            pass

    async def handle_connection(self, websocket: WebSocket, task_id: str) -> None:
        await websocket.accept()
        logger.info("[%s] Qwen3 websocket connected", task_id)

        ctx = ConnectionContext()

        try:
            while True:
                message = await websocket.receive()

                if "text" in message:
                    data = json.loads(message["text"])
                    msg_type = data.get("type", "")

                    if msg_type == "start":
                        if ctx.state != ConnectionState.READY:
                            await self._send_error(websocket, "识别已在进行中", task_id, "INVALID_STATE")
                            continue

                        payload = data.get("payload", {})
                        ctx.params = {
                            "format": payload.get("format", "pcm"),
                            "sample_rate": payload.get("sample_rate", 16000),
                            "language": payload.get("language"),
                            "context": payload.get("context", ""),
                            "chunk_size_sec": payload.get("chunk_size_sec", 2.0),
                            "unfixed_chunk_num": payload.get("unfixed_chunk_num", 2),
                            "unfixed_token_num": payload.get("unfixed_token_num", 5),
                        }

                        engine = await self._ensure_engine(ctx)
                        ctx.streaming_state = engine.init_streaming_state(
                            context=ctx.params["context"],
                            language=ctx.params["language"],
                            chunk_size_sec=ctx.params["chunk_size_sec"],
                            unfixed_chunk_num=ctx.params["unfixed_chunk_num"],
                            unfixed_token_num=ctx.params["unfixed_token_num"],
                        )

                        await websocket.send_json(
                            {
                                "type": "started",
                                "task_id": task_id,
                                "params": ctx.params,
                            }
                        )
                        ctx.state = ConnectionState.STARTED
                        logger.info("[%s] Recognition started: %s", task_id, ctx.params)

                    elif msg_type == "stop":
                        if ctx.state in (ConnectionState.STARTED, ConnectionState.STREAMING):
                            await self._stop(websocket, ctx, task_id)
                        break

                    else:
                        await self._send_error(
                            websocket,
                            f"未知消息类型: {msg_type}",
                            task_id,
                            "INVALID_MESSAGE",
                        )

                elif "bytes" in message:
                    if ctx.state not in (ConnectionState.STARTED, ConnectionState.STREAMING):
                        await self._send_error(websocket, "请先发送 start", task_id, "INVALID_STATE")
                        continue

                    audio = _convert_audio(
                        message["bytes"],
                        ctx.params["format"],
                        ctx.params["sample_rate"],
                    )
                    if audio is None:
                        continue

                    ctx.total_samples += len(audio)

                    if self._has_voice(audio):
                        ctx.silence_samples = 0
                    else:
                        ctx.silence_samples += len(audio)

                    if self._need_truncate(ctx):
                        reason = (
                            "silence"
                            if ctx.silence_samples >= ctx.SILENCE_THRESHOLD
                            else "max_duration"
                        )
                        await self._truncate(websocket, ctx, task_id, reason)

                    ctx.audio_buffer = np.concatenate([ctx.audio_buffer, audio])

                    chunk_size = int(ctx.params["chunk_size_sec"] * 16000)
                    results = []

                    while len(ctx.audio_buffer) >= chunk_size:
                        chunk = ctx.audio_buffer[:chunk_size]
                        ctx.audio_buffer = ctx.audio_buffer[chunk_size:]

                        engine = await self._ensure_engine(ctx)
                        ctx.streaming_state = await run_sync(
                            engine.streaming_transcribe,
                            chunk,
                            ctx.streaming_state,
                        )

                        current = ctx.streaming_state.last_text or ""
                        confirmed = "\n".join(
                            [segment["text"] for segment in ctx.confirmed_segments]
                        )
                        full = confirmed + "\n" + current if confirmed else current

                        results.append(
                            {
                                "text": full,
                                "current_segment_text": current,
                                "language": ctx.streaming_state.last_language,
                                "chunk_id": ctx.streaming_state.chunk_count,
                                "is_partial": True,
                                "segment_index": ctx.segment_index,
                            }
                        )

                    if results:
                        await websocket.send_json(
                            {
                                "type": "result",
                                "task_id": task_id,
                                "results": results,
                                "segment_index": ctx.segment_index,
                                "confirmed_segments_count": len(ctx.confirmed_segments),
                            }
                        )
                        ctx.state = ConnectionState.STREAMING

        except WebSocketDisconnect:
            logger.info("[%s] WebSocket disconnected", task_id)
        except Exception as exc:
            logger.error("[%s] Connection error: %s", task_id, exc)
            await self._send_error(websocket, str(exc), task_id)
        finally:
            stream_handle = getattr(getattr(ctx, "streaming_state", None), "internal_state", None)
            close_stream = getattr(stream_handle, "close", None)
            if callable(close_stream):
                close_stream()

            if ctx.engine_lease is not None:
                await ctx.engine_lease.close()
            logger.info("[%s] Connection closed", task_id)

    async def _stop(
        self,
        websocket: WebSocket,
        ctx: ConnectionContext,
        task_id: str,
    ) -> None:
        try:
            engine = await self._ensure_engine(ctx)

            if len(ctx.audio_buffer) > 0:
                ctx.streaming_state = await run_sync(
                    engine.streaming_transcribe,
                    ctx.audio_buffer,
                    ctx.streaming_state,
                )

            ctx.streaming_state = await run_sync(
                engine.finish_streaming_transcribe,
                ctx.streaming_state,
            )

            final = ctx.streaming_state.last_text or ""
            if final.strip():
                ctx.confirmed_segments.append(
                    {
                        "index": ctx.segment_index,
                        "text": final,
                        "language": ctx.streaming_state.last_language,
                        "reason": "final",
                    }
                )

            all_texts = [
                segment["text"]
                for segment in ctx.confirmed_segments
                if segment["text"].strip()
            ]
            full_text = "\n".join(all_texts)

            await websocket.send_json(
                {
                    "type": "final",
                    "task_id": task_id,
                    "result": {
                        "text": final,
                        "full_text": full_text,
                        "language": ctx.streaming_state.last_language,
                        "total_chunks": ctx.streaming_state.chunk_count,
                        "total_segments": len(ctx.confirmed_segments),
                        "segments": ctx.confirmed_segments,
                    },
                }
            )

            logger.info("[%s] Recognition completed, segments=%s", task_id, len(ctx.confirmed_segments))

        except Exception as exc:
            logger.error("[%s] Stop failed: %s", task_id, exc)
            await self._send_error(websocket, f"结束识别失败: {exc}", task_id)
