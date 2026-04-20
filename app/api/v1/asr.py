# -*- coding: utf-8 -*-
"""
ASR API路由
"""

from fastapi import (
    APIRouter,
    Request,
    HTTPException,
    Depends
)
from fastapi.responses import JSONResponse
from typing import Annotated, Optional
import time
import logging

from ...core.config import settings
from ...core.exceptions import (
    AuthenticationException,
    InvalidParameterException,
    InvalidMessageException,
    UnsupportedSampleRateException,
    DefaultServerErrorException,
    get_http_status_code,
)
from ...core.security import validate_token
from ...models.common import SampleRate
from ...models.asr import (
    ASRResponse,
    ASRHealthCheckResponse,
    ASRModelsResponse,
    ASRSuccessResponse,
    ASRErrorResponse,
    ASRQueryParams,
)
from ...utils.common import generate_task_id
from ...services.asr.manager import get_model_manager
from ...services.asr.runtime import OfflineASRRequest, get_runtime_router
from ...services.asr.audio_validation import validate_sample_rate
from ...services.asr.model_selection import resolve_offline_model_id
from ...services.audio import get_audio_service

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/stream/v1", tags=["ASR"])


async def get_asr_params(request: Request) -> ASRQueryParams:
    """从请求中提取并验证ASR参数"""
    # 从URL查询参数中获取
    query_params = dict(request.query_params)

    # 使用统一的验证器验证参数
    try:
        # 验证采样率（转换为整数）
        if "sample_rate" in query_params and query_params["sample_rate"]:
            try:
                sample_rate = int(query_params["sample_rate"])  # type: ignore
                validated_rate = validate_sample_rate(sample_rate)
                query_params["sample_rate"] = str(validated_rate)  # type: ignore
            except ValueError:
                raise InvalidParameterException(
                    f"采样率必须是整数，收到: {query_params['sample_rate']}"
                )

        # 创建ASRQueryParams实例，Pydantic会自动验证和设置默认值
        return ASRQueryParams.model_validate(query_params)
    except InvalidParameterException:
        raise
    except Exception as e:
        raise InvalidParameterException(f"请求参数错误: {str(e)}")


@router.post(
    "/asr",
    response_model=ASRResponse,
    responses={
        200: {
            "description": "识别成功",
            "model": ASRSuccessResponse,
        },
        400: {
            "description": "请求参数错误",
            "model": ASRErrorResponse,
        },
        401: {"description": "认证失败", "model": ASRErrorResponse},
        500: {"description": "服务器内部错误", "model": ASRErrorResponse},
    },
    summary="语音识别（支持长音频）",
    description="""
将音频文件转写为文本，兼容阿里云语音识别 RESTful API。

## 功能特性
- 支持多种音频格式与常见含音轨视频容器：WAV, MP3, M4A, FLAC, OGG, AAC, AMR, PCM, WEBM, MP4, MOV, MKV, AVI 等
- 自动音频格式检测和转换
- 支持长音频自动分段识别（返回带时间戳的分段结果）
- 最大文件大小：{settings.MAX_AUDIO_SIZE // (1024 * 1024)}MB（可通过环境变量 MAX_AUDIO_SIZE 配置）

## 音频输入方式
1. **请求体上传**：将音频/视频二进制数据作为请求体发送
2. **URL 下载**：通过 `audio_address` 参数指定音频/视频文件 URL

如果请求体和 `audio_address` 同时存在，服务会优先使用请求体，并忽略 `audio_address`。

## 注意事项
- 离线路径固定使用服务当前启用的 Qwen3-ASR 模型；`model_id` 仅为兼容参数，传入后会被忽略
- `vocabulary_id` 参数用于传递热词，格式：`热词1 权重1 热词2 权重2`（如：`阿里巴巴 20 腾讯 15`）
- 音频会自动转换为 16kHz 采样率进行识别
""",
    openapi_extra={
        "parameters": [
            # 1. 兼容参数
            {
                "name": "model_id",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 64,
                    "example": "qwen3-asr-0.6b",
                },
                "description": "兼容参数，将被忽略。离线路径固定使用服务当前启用的唯一 Qwen3-ASR 模型",
                "deprecated": True,
            },
            # 2. 输入源
            {
                "name": "audio_address",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 512,
                    "example": "https://media.cdn.vect.one/podcast_demo.mp4",
                },
                "description": "音频/视频文件 URL（HTTP/HTTPS）。仅当请求体为空时使用；若同时上传请求体，服务会忽略此参数",
            },
            # 3. 音频属性
            {
                "name": "sample_rate",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "integer",
                    "enum": SampleRate.get_enums(),
                    "default": 16000,
                    "example": 16000,
                },
                "description": "音频采样率（Hz），实际识别时会自动转换为 16kHz。支持：8000, 16000, 22050, 24000",
            },
            # 4. 功能开关
            {
                "name": "enable_speaker_diarization",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "boolean",
                    "default": True,
                    "example": True,
                },
                "description": "是否启用说话人分离。启用后响应会包含 speaker_id 字段",
            },
            {
                "name": "word_timestamps",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "boolean",
                    "default": False,
                    "example": False,
                },
                "description": "是否返回字词级时间戳（默认关闭；Qwen CUDA vLLM / CPU Rust 在启用时会自动调用 forced aligner）",
            },
            # 5. 增强选项
            {
                "name": "vocabulary_id",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 512,
                    "example": "阿里巴巴 20 腾讯 15",
                },
                "description": "热词字符串，格式：`热词1 权重1 热词2 权重2`。权重范围 1-100，建议 10-30。可提升特定词汇的识别准确率",
            },
            # 6. 认证参数
            {
                "name": "X-NLS-Token",
                "in": "header",
                "required": False,
                "schema": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                    "example": "",
                },
                "description": "访问令牌，用于身份认证。未配置 API_KEY 环境变量时可忽略",
            },
        ],
        "requestBody": {
            "description": "音频/视频文件二进制数据。支持格式：WAV, MP3, M4A, FLAC, OGG, AAC, AMR, PCM, WEBM, MP4, MOV, MKV, AVI 等。若同时提供 audio_address，服务会优先使用这里上传的内容",
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
            "required": False,
        },
    },
)
async def asr_transcribe(
    request: Request, params: Annotated[ASRQueryParams, Depends(get_asr_params)]
) -> JSONResponse:
    """语音识别API端点"""
    task_id = generate_task_id()
    audio_path = None
    normalized_audio_path = None

    # 性能计时
    request_start_time = time.time()

    # 记录请求开始（此时文件已上传完成）
    content_length = request.headers.get("content-length", "unknown")
    ignored_model_id = request.query_params.get("model_id")
    logger.info(f"[{task_id}] 收到ASR请求, content_length={content_length}")
    if ignored_model_id:
        logger.info(f"[{task_id}] 忽略客户端传入的离线 model_id={ignored_model_id}")

    # 获取音频处理服务
    audio_service = get_audio_service()

    try:
        # 验证请求头部（鉴权）
        result, content = validate_token(request, task_id)
        if not result:
            raise AuthenticationException(content, task_id)

        # 使用音频服务处理音频
        target_sample_rate = int(params.sample_rate) if params.sample_rate else 16000
        normalized_audio_path, _audio_duration, audio_path = await audio_service.process_from_request(
            request=request,
            audio_address=params.audio_address,
            task_id=task_id,
            sample_rate=target_sample_rate,
        )

        # 执行语音识别
        logger.info(f"[{task_id}] 正在加载离线ASR模型...")
        runtime_router = get_runtime_router()
        resolved_model_id = resolve_offline_model_id(ignored_model_id)
        logger.info(f"[{task_id}] ASR模型加载完成: {resolved_model_id}")

        # 准备热词（vocabulary_id 参数直接传递热词字符串）
        hotwords = params.vocabulary_id or ""

        # 使用线程池执行模型推理，避免阻塞事件循环
        # 使用长音频识别方法，自动处理超过60秒的音频
        # 默认开启：标点预测、ITN（数字转换）
        logger.info(f"[{task_id}] 开始调用 transcribe_long_audio (enable_speaker_diarization={params.enable_speaker_diarization})...")

        asr_result = await runtime_router.run_offline(
            OfflineASRRequest(
                model_id=resolved_model_id,
                audio_path=normalized_audio_path,
                hotwords=hotwords,
                enable_punctuation=True,
                enable_itn=True,
                sample_rate=int(params.sample_rate or SampleRate.RATE_16000),
                enable_speaker_diarization=(
                    params.enable_speaker_diarization
                    if params.enable_speaker_diarization is not None
                    else True
                ),
                word_timestamps=(
                    params.word_timestamps
                    if params.word_timestamps is not None
                    else False
                ),
                task_id=task_id,
            )
        )

        logger.info(f"[{task_id}] 识别完成，共 {len(asr_result.segments)} 个分段，总字符: {len(asr_result.text)}")

        # 构建分段结果（始终返回 segments，短音频也是 1 个 segment）
        segments_data = []
        for seg in asr_result.segments:
            seg_dict = {
                "text": seg.text,
                "start_time": round(seg.start_time, 2),
                "end_time": round(seg.end_time, 2),
            }
            if seg.speaker_id:
                seg_dict["speaker_id"] = seg.speaker_id
            # 添加字词级时间戳（如果存在）
            if seg.word_tokens:
                seg_dict["word_tokens"] = [
                    {
                        "text": wt.text,
                        "start_time": round(wt.start_time, 3),
                        "end_time": round(wt.end_time, 3),
                    }
                    for wt in seg.word_tokens
                ]
            segments_data.append(seg_dict)

        # 计算请求处理时间
        request_duration = time.time() - request_start_time

        # 返回成功响应（统一数据结构）
        response_data = {
            "task_id": task_id,
            "result": asr_result.text,
            "status": 200,
            "message": "SUCCESS",
            "segments": segments_data,
            "duration": round(asr_result.duration, 2),
            "processing_time": round(request_duration, 3),
        }

        return JSONResponse(content=response_data, headers={"task_id": task_id})

    except (
        AuthenticationException,
        InvalidParameterException,
        InvalidMessageException,
        UnsupportedSampleRateException,
        DefaultServerErrorException,
    ) as e:
        e.task_id = task_id
        logger.error(f"[{task_id}] ASR异常: {e.message}")

        # 使用标准错误格式
        response_data = e.to_dict()
        return JSONResponse(
            content=response_data,
            headers={"task_id": task_id},
            status_code=get_http_status_code(e.status_code),
        )

    except Exception as e:
        logger.error(f"[{task_id}] 未知异常: {str(e)}")

        # 使用标准错误格式
        from ...core.exceptions import create_error_response
        response_data = create_error_response(
            error_code="DEFAULT_SERVER_ERROR",
            message=f"内部服务错误: {str(e)}",
            task_id=task_id,
        )
        return JSONResponse(content=response_data, headers={"task_id": task_id})

    finally:
        # 清理临时文件
        audio_service.cleanup(audio_path, normalized_audio_path)


@router.get(
    "/asr/health",
    response_model=ASRHealthCheckResponse,
    summary="ASR 服务健康检查",
    description="""
检查语音识别服务的运行状态和资源使用情况。

## 返回信息
- **status**: 服务状态（healthy/unhealthy/error）
- **model_loaded**: 默认模型是否已加载
- **device**: 当前推理设备（cuda:0/cpu）
- **loaded_models**: 已加载的模型列表
- **memory_usage**: GPU 显存使用情况（仅 GPU 模式）
""",
)
async def health_check(request: Request):
    """ASR服务健康检查端点"""
    # 鉴权
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "health_check")

    try:
        # 尝试获取默认模型的引擎
        try:
            runtime_router = get_runtime_router()
            default_model = runtime_router.resolve_model_id(None)
            async with await runtime_router.acquire_engine(default_model) as engine:
                model_loaded = True
                device = engine.device
        except Exception:
            model_loaded = False
            device = "unknown"

        runtime_router = get_runtime_router()
        memory_info = runtime_router.get_memory_usage()
        loaded_models = runtime_router.get_loaded_model_ids()

        return {
            "status": "healthy" if model_loaded else "unhealthy",
            "model_loaded": model_loaded,
            "device": device,
            "version": settings.APP_VERSION,
            "message": (
                "ASR service is running normally"
                if model_loaded
                else "ASR model not loaded"
            ),
            "loaded_models": loaded_models,
            "memory_usage": memory_info.get("gpu_memory"),
        }
    except Exception as e:
        return {
            "status": "error",
            "model_loaded": False,
            "device": "unknown",
            "version": settings.APP_VERSION,
            "message": str(e),
        }

@router.get(
    "/asr/models",
    response_model=ASRModelsResponse,
    summary="获取声明条目列表",
    description="""
返回系统声明的离线模型与 realtime capability 信息。

## 条目说明

| ID | 类型 | 说明 |
|----|------|------|
| qwen3-asr-1.7b | model | 离线/实时共用的 Qwen3-ASR 模型条目 |
| qwen3-asr-0.6b | model | 轻量版 Qwen3-ASR 模型条目 |
| paraformer-large | capability | WebSocket realtime capability |

## 返回信息
- **declared_entries**: 声明的模型与 capability 列表
- **declared_count**: 声明项总数
- **runtime**: 运行时加载状态
""",
)
async def list_models(request: Request):
    """获取声明条目列表端点"""
    # 鉴权
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "list_models")

    try:

        model_manager = get_model_manager()
        runtime_router = get_runtime_router()
        loaded_model_ids = runtime_router.get_loaded_model_ids()
        entries = model_manager.list_declared_entries()

        return {
            "declared_entries": entries,
            "declared_count": len(entries),
            "runtime": {
                "loaded_model_ids": loaded_model_ids,
                "loaded_count": len(loaded_model_ids),
                "default_offline_model_id": runtime_router.resolve_model_id(None),
            },
        }
    except Exception as e:
        logger.error(f"获取模型列表时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")
