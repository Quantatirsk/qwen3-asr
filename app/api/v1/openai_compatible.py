# -*- coding: utf-8 -*-
"""
OpenAI 兼容 API
实现 OpenAI Audio API 规范，兼容 OpenAI SDK 和第三方客户端
"""

import asyncio
import json
import time
import logging
from typing import AsyncIterator, Optional, List
from enum import Enum
from contextlib import suppress

from fastapi import APIRouter, File, Form, UploadFile, Request, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field
from starlette.types import Receive, Scope, Send

from ...core.config import settings
from ...core.executor import wait_for_completion
from ...services.asr.engines import ASRFullResult
from ...core.security import validate_openai_token
from ...core.exceptions import (
    APIException,
    create_error_response,
    get_http_status_code,
)
from ...services.asr.model_selection import (
    get_offline_model_ids,
)
from ...services.asr.offline_transcription_service import (
    OfflineTranscriptionOptions,
    get_offline_transcription_service,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["OpenAI Compatible"])
HEARTBEAT_INTERVAL_SECONDS = 15.0


# ============= 枚举类型 =============


class ResponseFormat(str, Enum):
    JSON = "json"
    TEXT = "text"
    SRT = "srt"
    VERBOSE_JSON = "verbose_json"
    VTT = "vtt"


# ============= 响应模型 =============


class TranscriptionSegment(BaseModel):
    """转写分段"""

    id: int
    seek: int = 0
    start: float
    end: float
    text: str
    tokens: List[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0
    speaker: Optional[str] = Field(
        default=None, description="Speaker ID; null when unknown"
    )
    speaker_candidates: Optional[List[str]] = Field(
        default=None, description="Candidate speakers for uncertain attribution"
    )


class TranscriptionWord(BaseModel):
    """转写词级别信息"""

    word: str
    start: float
    end: float


class TranscriptionResponse(BaseModel):
    """简单转写响应 (json 格式)"""

    text: str


class SpeakerActivity(BaseModel):
    start: float
    end: float
    speaker: str
    confidence: float


class VerboseTranscriptionResponse(BaseModel):
    """详细转写响应 (verbose_json 格式)"""

    task: str = "transcribe"
    language: str
    duration: float
    text: str
    segments: List[TranscriptionSegment] = Field(default_factory=list)
    words: Optional[List[TranscriptionWord]] = None
    speaker_segments: Optional[List[SpeakerActivity]] = None


class ModelObject(BaseModel):
    """模型对象"""

    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "netease-youdao"


class ModelsResponse(BaseModel):
    """模型列表响应"""

    object: str = "list"
    data: List[ModelObject]


# ============= 辅助函数 =============


def format_timestamp_srt(seconds: float) -> str:
    """格式化时间戳为 SRT 格式 (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """格式化时间戳为 VTT 格式 (HH:MM:SS.mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def generate_srt(segments: List[TranscriptionSegment]) -> str:
    """生成 SRT 字幕格式"""
    lines = []
    for i, seg in enumerate(segments, 1):
        start = format_timestamp_srt(seg.start)
        end = format_timestamp_srt(seg.end)
        lines.append(f"{i}")
        lines.append(f"{start} --> {end}")
        text = seg.text.strip()
        if seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def generate_vtt(segments: List[TranscriptionSegment]) -> str:
    """生成 WebVTT 字幕格式"""
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = format_timestamp_vtt(seg.start)
        end = format_timestamp_vtt(seg.end)
        lines.append(f"{start} --> {end}")
        text = seg.text.strip()
        if seg.speaker:
            text = f"[{seg.speaker}] {text}"
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def detect_language(text: str, language: Optional[str]) -> str:
    """检测识别语言。"""
    if language:
        return language

    import re

    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en"


def build_transcription_payload(
    *,
    response_format: ResponseFormat,
    asr_result,
    audio_duration: float,
    language: Optional[str],
) -> tuple[object, int, int]:
    """构建 OpenAI 转写响应载荷，并返回 segments / words 计数。"""
    segments: List[TranscriptionSegment] = []
    words: List[TranscriptionWord] = []

    for i, seg in enumerate(asr_result.segments):
        segments.append(
            TranscriptionSegment(
                id=i,
                seek=int(seg.start_time * 100),
                start=seg.start_time,
                end=seg.end_time,
                text=seg.text,
                speaker=seg.speaker_id,
                speaker_candidates=seg.speaker_candidates,
            )
        )
        if seg.word_tokens:
            for wt in seg.word_tokens:
                words.append(
                    TranscriptionWord(
                        word=wt.text,
                        start=round(seg.start_time + wt.start_time, 3),
                        end=round(seg.start_time + wt.end_time, 3),
                    )
                )

    detected_language = detect_language(asr_result.text, language)

    if response_format == ResponseFormat.VERBOSE_JSON:
        payload = VerboseTranscriptionResponse(
            task="transcribe",
            language=detected_language,
            duration=audio_duration,
            text=asr_result.text,
            segments=segments,
            words=words if words else None,
            speaker_segments=(
                [
                    SpeakerActivity(
                        start=span.start_sec,
                        end=span.end_sec,
                        speaker=span.speaker_id,
                        confidence=span.confidence,
                    )
                    for span in asr_result.speaker_segments
                ]
                if asr_result.speaker_segments is not None
                else None
            ),
        ).model_dump()
    elif response_format == ResponseFormat.JSON:
        payload = {"text": asr_result.text}
    elif response_format == ResponseFormat.TEXT:
        payload = asr_result.text
    elif response_format == ResponseFormat.SRT:
        if not segments:
            segments = [
                TranscriptionSegment(
                    id=0,
                    start=0,
                    end=audio_duration,
                    text=asr_result.text,
                )
            ]
        payload = generate_srt(segments)
    elif response_format == ResponseFormat.VTT:
        if not segments:
            segments = [
                TranscriptionSegment(
                    id=0,
                    start=0,
                    end=audio_duration,
                    text=asr_result.text,
                )
            ]
        payload = generate_vtt(segments)
    else:
        payload = {"text": asr_result.text}

    return payload, len(segments), len(words)


class TranscriptionStreamingResponse(StreamingResponse):
    """Join inference on every response exit, including a failed ASGI send."""

    def __init__(
        self,
        content: AsyncIterator[bytes],
        inference_task: asyncio.Task[ASRFullResult],
    ) -> None:
        super().__init__(
            content,
            media_type="application/json",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
        self._inference_task = inference_task

    async def stream_response(self, send: Send) -> None:
        started = False
        while True:
            finished = False
            try:
                chunk = await anext(self.body_iterator)
            except StopAsyncIteration:
                chunk, finished = b"", True
            except Exception as exc:
                logger.exception("Transcription response failed")
                if isinstance(exc, APIException):
                    status_code = get_http_status_code(exc.status_code)
                    payload = exc.to_dict()
                else:
                    status_code = (
                        exc.status_code if isinstance(exc, HTTPException) else 500
                    )
                    payload = create_error_response(
                        error_code=(
                            "DEFAULT_CLIENT_ERROR"
                            if status_code < 500
                            else "DEFAULT_SERVER_ERROR"
                        ),
                        message=(
                            exc.detail if isinstance(exc, HTTPException) else str(exc)
                        ),
                    )
                # Heartbeats commit HTTP 200; later errors can only change the body.
                if not started:
                    self.status_code = status_code
                chunk, finished = JSONResponse(payload).body, True
            if not started:
                await send(
                    {
                        "type": "http.response.start",
                        "status": self.status_code,
                        "headers": self.raw_headers,
                    }
                )
                started = True
            if not isinstance(chunk, (bytes, memoryview)):
                chunk = chunk.encode(self.charset)
            await send(
                {"type": "http.response.body", "body": chunk, "more_body": not finished}
            )
            if finished:
                return

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            await super().__call__(scope, receive, send)
        finally:
            if not self._inference_task.done():
                self._inference_task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await wait_for_completion(self._inference_task)


def create_heartbeat_streaming_response(
    *,
    response_format: ResponseFormat,
    inference_task: asyncio.Task[ASRFullResult],
    language: Optional[str],
) -> StreamingResponse:
    """为长耗时 JSON 响应生成带心跳的流式输出。"""

    async def response_stream() -> AsyncIterator[bytes]:
        heartbeat_count = 0

        while True:
            done, _pending = await asyncio.wait(
                {inference_task},
                timeout=HEARTBEAT_INTERVAL_SECONDS,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if inference_task in done:
                break

            heartbeat_count += 1
            logger.info(
                "[OpenAI API] 发送响应心跳: "
                f"format={response_format}, heartbeat_count={heartbeat_count}"
            )
            yield b" \n"

        asr_result = await inference_task
        logger.info(f"[OpenAI API] 识别完成: {len(asr_result.text)} 字符")

        payload, segments_count, words_count = build_transcription_payload(
            response_format=response_format,
            asr_result=asr_result,
            audio_duration=asr_result.duration,
            language=language,
        )
        response_bytes = json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")

        logger.info(
            "[OpenAI API] 准备发送 JSON 响应: "
            f"format={response_format}, "
            f"segments={segments_count}, "
            f"words={words_count}, "
            f"payload_bytes={len(response_bytes)}, "
            f"heartbeat_count={heartbeat_count}"
        )
        yield response_bytes

    return TranscriptionStreamingResponse(response_stream(), inference_task)


# ============= API 端点 =============


@router.get(
    "/models",
    response_model=ModelsResponse,
    summary="列出可用模型",
    description="List Confucius4-R2T2, the CUDA model for offline and realtime transcription.",
)
async def list_models(request: Request):
    """列出可用离线模型 (OpenAI 兼容)"""
    result, _ = validate_openai_token(request)
    if not result:
        response_data = create_error_response(
            error_code="AUTHENTICATION_FAILED",
            message="Invalid authentication",
        )
        return JSONResponse(content=response_data, status_code=401)

    try:
        return ModelsResponse(
            data=[ModelObject(id=model_id) for model_id in get_offline_model_ids()]
        )
    except Exception as e:
        logger.error(f"获取模型列表失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _get_transcription_description() -> str:
    """获取动态的转写端点描述"""
    return f"""将音频文件转写为文本（OpenAI Audio API 文件转写子集，不支持 Realtime 协议）。

**支持的音频格式与常见含音轨视频容器：**
`mp3`, `mp4`, `mpeg`, `mpga`, `m4a`, `wav`, `webm`, `flac`, `ogg`, `amr`, `pcm`, `mov`, `mkv`, `avi`

**音频输入方式：**
1. **文件上传**：通过 `file` 参数上传音频/视频文件（标准 OpenAI 方式）
2. **URL 下载**：通过 `audio_address` 参数提供音频/视频文件 URL（HTTP/HTTPS）

如果同时提供 `file` 和 `audio_address`，服务会优先使用 `file`，并忽略 `audio_address`。

**文件大小限制：**
- 最大支持 {settings.MAX_AUDIO_SIZE // (1024 * 1024)}MB（可通过 `MAX_AUDIO_SIZE` 环境变量配置）
- OpenAI 原生限制为 25MB

**说话人分离：**
- 默认开启 (`enable_speaker_diarization=true`)
- 启用后 `verbose_json` 格式的 segments 会包含 `speaker` 字段（如 "说话人1"）
- 可设置 `enable_speaker_diarization=false` 关闭

**输出格式：**
| 格式 | Content-Type | 说明 |
|------|-------------|------|
| `json` | application/json | 简单 JSON，仅含 text 字段（默认） |
| `text` | text/plain | 纯文本 |
| `verbose_json` | application/json | 详细 JSON，含时间戳、分段和说话人 |
| `srt` | text/plain | SRT 字幕格式 |
| `vtt` | text/vtt | WebVTT 字幕格式 |

**模型选择：**
- Use `confucius4-r2t2` or omit `model` to select the default.
- Other model IDs are rejected. `/v1/models` lists the supported model.

**暂不支持的参数：**
`prompt`、`temperature`、`timestamp_granularities` 参数已保留但暂不生效
"""


@router.post(
    "/audio/transcriptions",
    summary="音频转写",
    description=_get_transcription_description(),
    responses={
        200: {
            "description": "转写成功",
            "content": {
                "application/json": {
                    "example": {"text": "今天天气不错，明天可能会下雨。"}
                },
                "text/plain": {"example": "今天天气不错，明天可能会下雨。"},
            },
        },
        400: {
            "description": "请求错误",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "INVALID_PARAMETER",
                        "message": f"File too large. Maximum size is {settings.MAX_AUDIO_SIZE // (1024 * 1024)}MB",
                        "task_id": "",
                        "timestamp": "2025-01-31T12:00:00Z",
                        "details": {},
                    }
                }
            },
        },
        401: {
            "description": "认证失败",
            "content": {
                "application/json": {
                    "example": {
                        "error_code": "AUTHENTICATION_FAILED",
                        "message": "Invalid API key",
                        "task_id": "",
                        "timestamp": "2025-01-31T12:00:00Z",
                        "details": {},
                    }
                }
            },
        },
    },
)
async def create_transcription(
    request: Request,
    model: Optional[str] = Form(
        None,
        description="Accepted for client compatibility. Any value uses Confucius4-R2T2.",
    ),
    # 1. 音频输入（二选一）
    file: Optional[UploadFile] = File(
        default=None,
        description="要转写的音频/视频文件。若同时提供 audio_address，服务会优先使用这里上传的文件",
    ),
    audio_address: Optional[str] = Form(
        default=None,
        description="音频/视频文件 URL（HTTP/HTTPS）。仅当 file 为空时使用；若同时上传 file，服务会忽略此参数",
        json_schema_extra={"example": "https://media.cdn.vect.one/podcast_demo.mp4"},
    ),
    language: Optional[str] = Form(
        None,
        description="音频语言代码（ISO-639-1），如 zh/en/ja，不填则自动检测",
        examples=["zh", "en", "ja"],
    ),
    # 4. 功能开关
    enable_speaker_diarization: bool = Form(
        True,
        description="是否启用说话人分离（默认开启）。启用后响应 segments 会包含 speaker 字段",
    ),
    word_timestamps: bool = Form(
        False,
        description="Return word timestamps using the forced aligner (disabled by default).",
    ),
    # 5. 输出选项
    response_format: ResponseFormat = Form(
        ResponseFormat.VERBOSE_JSON,
        description="输出格式",
        examples=["verbose_json", "json", "text", "srt", "vtt"],
    ),
    # 6. 兼容性参数（暂不支持）
    prompt: Optional[str] = Form(
        None, description="提示文本（暂不支持，保留兼容）"
    ),  # noqa: ARG001
    temperature: Optional[float] = Form(
        0, description="采样温度（暂不支持，保留兼容）"
    ),  # noqa: ARG001
    timestamp_granularities: Optional[List[str]] = Form(  # noqa: ARG001
        None,
        alias="timestamp_granularities[]",
        description="时间戳粒度（暂不支持，保留兼容）",
    ),
):
    """音频转写 API (OpenAI Audio API 兼容)"""
    # 标记暂不支持的参数（保留以兼容 OpenAI API）
    _ = (model, prompt, temperature, timestamp_granularities)

    logger.info(
        f"[OpenAI API] 收到转写请求: format={response_format}, "
        f"speaker_diarization={enable_speaker_diarization}, word_level={word_timestamps}, "
        f"audio_address={'有' if audio_address else '无'}"
    )

    # 验证输入：至少提供一种输入源；若二者同时存在，优先 file
    if not file and not audio_address:
        response_data = create_error_response(
            error_code="INVALID_PARAMETER",
            message="必须提供 file（上传文件）或 audio_address（音频 URL）其中之一",
        )
        return JSONResponse(content=response_data, status_code=400)

    try:
        result, _ = validate_openai_token(request)
        if not result:
            response_data = create_error_response(
                error_code="AUTHENTICATION_FAILED",
                message="Invalid authentication",
            )
            return JSONResponse(content=response_data, status_code=401)

        transcription_service = get_offline_transcription_service()
        audio_data = await file.read() if file is not None else None
        inference_task = await transcription_service.start_transcription(
            audio_data=audio_data,
            filename=file.filename if file is not None else None,
            audio_address=audio_address,
            options=OfflineTranscriptionOptions(
                sample_rate=16000,
                enable_speaker_diarization=enable_speaker_diarization,
                word_timestamps=word_timestamps,
                task_id=f"openai-{int(time.time() * 1000)}",
            ),
        )
        if response_format in {ResponseFormat.VERBOSE_JSON, ResponseFormat.JSON}:
            return create_heartbeat_streaming_response(
                response_format=response_format,
                inference_task=inference_task,
                language=language,
            )

        asr_result = await inference_task
        payload, _, _ = build_transcription_payload(
            response_format=response_format,
            asr_result=asr_result,
            audio_duration=asr_result.duration,
            language=language,
        )
        if response_format == ResponseFormat.VTT:
            return PlainTextResponse(content=payload, media_type="text/vtt")
        return PlainTextResponse(content=payload)

    except HTTPException as http_exc:
        # 将 HTTPException 转换为标准错误格式
        logger.error(f"[OpenAI API] HTTP异常: {http_exc.detail}")

        response_data = create_error_response(
            error_code=(
                "DEFAULT_CLIENT_ERROR"
                if http_exc.status_code < 500
                else "DEFAULT_SERVER_ERROR"
            ),
            message=http_exc.detail,
        )
        return JSONResponse(content=response_data, status_code=http_exc.status_code)
    except Exception as e:
        logger.error(f"[OpenAI API] 转写失败: {e}")

        # 使用标准错误格式
        response_data = create_error_response(
            error_code="DEFAULT_SERVER_ERROR",
            message=str(e),
        )
        return JSONResponse(content=response_data, status_code=500)
