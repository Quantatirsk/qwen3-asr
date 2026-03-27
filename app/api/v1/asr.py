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
from typing import Annotated
import time
import logging

from ...core.config import settings
from ...core.executor import run_sync
from ...core.exceptions import (
    AuthenticationException,
    InvalidParameterException,
    InvalidMessageException,
    UnsupportedSampleRateException,
    DefaultServerErrorException,
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
from ...services.asr.validators import AudioParamsValidator
from ...services.audio import get_audio_service

# 配置日志
logger = logging.getLogger(__name__)

# 创建路由器
router = APIRouter(prefix="/stream/v1", tags=["ASR"])


def _get_model_schema() -> dict:
    """获取动态的模型 schema（根据显存配置）"""
    from ...services.asr.validators import _get_dynamic_model_list, _get_default_model

    try:
        model_ids = _get_dynamic_model_list()
        default_model = _get_default_model()

        return {
            "type": "string",
            "maxLength": 64,
            "default": default_model,
            "enum": model_ids,
            "example": default_model,
        }
    except Exception as e:
        logger.warning(f"Failed to load dynamic model schema: {e}")

    # Fallback: 使用硬编码的默认值
    return {
        "type": "string",
        "maxLength": 64,
        "default": "qwen3-asr-1.7b",
        "enum": ["qwen3-asr-1.7b", "paraformer-large"],
        "example": "qwen3-asr-1.7b",
    }


def update_openapi_schema():
    """在应用启动时更新 OpenAPI schema，使其包含正确的模型列表"""
    from fastapi.routing import APIRoute
    from ...services.asr.validators import _get_dynamic_model_list, _get_default_model

    model_schema = _get_model_schema()
    default_model = _get_default_model()
    available_models = _get_dynamic_model_list()

    # 构建模型描述
    model_descriptions = {
        "qwen3-asr-1.7b": "Qwen3-ASR 1.7B，52种语言+方言，vLLM高性能（离线）",
        "qwen3-asr-0.6b": "Qwen3-ASR 0.6B，轻量版多语言，适合小显存（离线）",
        "paraformer-large": "高精度中文语音识别，内置VAD+标点（离线/实时）",
    }

    # 找到 asr_transcribe 路由并更新其 openapi_extra
    for route in router.routes:
        if isinstance(route, APIRoute) and route.endpoint.__name__ == "asr_transcribe":
            if route.openapi_extra:
                params = route.openapi_extra.get("parameters", [])
                for param in params:
                    if param.get("name") == "model_id":
                        param["schema"] = model_schema
                        # 构建动态描述
                        model_list_desc = ", ".join(
                            f"{m}（{model_descriptions.get(m, '')}）" for m in available_models
                        )
                        param["description"] = f"ASR 模型 ID。可选值：{model_list_desc}。默认：{default_model}"

                # 更新 description 中的可用模型说明
                # 重新构建模型列表部分
                models_section = "## 可用模型\n"
                for m in available_models:
                    desc = model_descriptions.get(m, "")
                    if m == default_model:
                        models_section += f"- **{m}**（默认）：{desc}\n"
                    else:
                        models_section += f"- **{m}**：{desc}\n"

                # 替换原有的模型说明部分
                import re
                route.description = re.sub(
                    r"## 可用模型\n(- \*\*.+\*\*.+\n)+",
                    models_section,
                    route.description
                )
            break


async def get_asr_params(request: Request) -> ASRQueryParams:
    """从请求中提取并验证ASR参数"""
    # 从URL查询参数中获取
    query_params = dict(request.query_params)

    # 使用统一的验证器验证参数
    try:
        # 验证模型ID
        if "model_id" in query_params:
            query_params["model_id"] = AudioParamsValidator.validate_model_id(
                query_params.get("model_id")
            )

        # 验证采样率（转换为整数）
        if "sample_rate" in query_params and query_params["sample_rate"]:
            try:
                sample_rate = int(query_params["sample_rate"])  # type: ignore
                validated_rate = AudioParamsValidator.validate_sample_rate(sample_rate)
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
- 支持多种音频格式：WAV, MP3, M4A, FLAC, OGG, AAC, AMR 等
- 自动音频格式检测和转换
- 支持长音频自动分段识别（返回带时间戳的分段结果）
- 最大文件大小：{settings.MAX_AUDIO_SIZE // (1024 * 1024)}MB（可通过环境变量 MAX_AUDIO_SIZE 配置）

## 可用模型
- **qwen3-asr-1.7b**（默认）：Qwen3-ASR 1.7B，52种语言+方言，vLLM高性能（离线）
- **qwen3-asr-0.6b**：Qwen3-ASR 0.6B，轻量版多语言，适合小显存（离线）
- **qwen3-asr**：自动路由到当前已启动的 Qwen3-ASR 版本（根据显存或配置自动选择）
- **paraformer-large**：高精度中文语音识别，内置VAD+标点（离线/实时）

## 音频输入方式
1. **请求体上传**：将音频二进制数据作为请求体发送
2. **URL 下载**：通过 `audio_address` 参数指定音频文件 URL

## 注意事项
- `vocabulary_id` 参数用于传递热词，格式：`热词1 权重1 热词2 权重2`（如：`阿里巴巴 20 腾讯 15`）
- 音频会自动转换为 16kHz 采样率进行识别
""",
    openapi_extra={
        "parameters": [
            # 1. 核心参数
            {
                "name": "model_id",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 64,
                    "default": "qwen3-asr-1.7b",
                    "enum": ["qwen3-asr-1.7b", "qwen3-asr-0.6b", "paraformer-large"],
                    "example": "qwen3-asr-1.7b",
                },
                "description": "ASR 模型 ID。可选值：qwen3-asr-1.7b（默认，高性能52语言）、qwen3-asr-0.6b（轻量版）、paraformer-large（高精度中文、支持实时）",
            },
            # 2. 输入源
            {
                "name": "audio_address",
                "in": "query",
                "required": False,
                "schema": {
                    "type": "string",
                    "maxLength": 512,
                    "example": "",
                },
                "description": "音频文件 URL（HTTP/HTTPS）。指定此参数时，将从 URL 下载音频而非读取请求体",
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
                    "default": True,
                    "example": True,
                },
                "description": "是否返回字词级时间戳（仅 Qwen3-ASR 模型支持）",
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
            "description": "音频文件二进制数据。支持格式：WAV, MP3, M4A, FLAC, OGG, AAC, AMR 等。不使用 audio_address 参数时必需",
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
    logger.info(f"[{task_id}] 收到ASR请求, model_id={params.model_id}, content_length={content_length}")

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
        logger.info(f"[{task_id}] 正在加载ASR模型: {params.model_id or '默认'}...")
        import sys
        sys.stdout.flush()

        model_manager = get_model_manager()
        asr_engine = model_manager.get_asr_engine(params.model_id)  # 使用指定模型或默认模型
        logger.info(f"[{task_id}] ASR模型加载完成: {params.model_id or '默认'}")
        sys.stdout.flush()

        # 准备热词（vocabulary_id 参数直接传递热词字符串）
        hotwords = params.vocabulary_id or ""

        # 使用线程池执行模型推理，避免阻塞事件循环
        # 使用长音频识别方法，自动处理超过60秒的音频
        # 默认开启：标点预测、ITN（数字转换）
        logger.info(f"[{task_id}] 开始调用 transcribe_long_audio (enable_speaker_diarization={params.enable_speaker_diarization})...")
        sys.stdout.flush()

        asr_result = await run_sync(
            asr_engine.transcribe_long_audio,
            audio_path=normalized_audio_path,
            hotwords=hotwords,
            enable_punctuation=True,  # 默认开启标点预测
            enable_itn=True,  # 默认开启数字转换
            sample_rate=params.sample_rate,
            enable_speaker_diarization=params.enable_speaker_diarization if params.enable_speaker_diarization is not None else True,
            word_timestamps=params.word_timestamps if params.word_timestamps is not None else False,
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
        return JSONResponse(content=response_data, headers={"task_id": task_id})

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
- **asr_model_mode**: 当前模型加载模式（offline/realtime/all）
""",
)
async def health_check(request: Request):
    """ASR服务健康检查端点"""
    # 鉴权
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "health_check")

    try:
        model_manager = get_model_manager()

        # 尝试获取默认模型的引擎
        try:
            asr_engine = model_manager.get_asr_engine()
            model_loaded = asr_engine.is_model_loaded()
            device = asr_engine.device
        except Exception:
            model_loaded = False
            device = "unknown"

        memory_info = model_manager.get_memory_usage()

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
            "loaded_models": memory_info["model_list"],
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
    summary="获取可用模型列表",
    description="""
返回系统中所有可用的 ASR 模型信息。

## 可用模型

| 模型 ID | 名称 | 说明 | 支持实时 |
|---------|------|------|----------|
| qwen3-asr-1.7b | Qwen3-ASR 1.7B | 高性能多语言语音识别，vLLM后端，52种语言+方言 | ❌ |
| qwen3-asr-0.6b | Qwen3-ASR 0.6B | 轻量版多语言ASR，适合小显存环境 | ❌ |
| paraformer-large | Paraformer Large | 高精度中文语音识别 | ✅ |

## 返回信息
- **models**: 模型详细信息列表
- **total**: 可用模型总数
- **loaded_count**: 已加载到内存的模型数量
- **asr_model_mode**: 当前模型加载模式
""",
)
async def list_models(request: Request):
    """获取可用模型列表端点"""
    # 鉴权
    result, content = validate_token(request)
    if not result:
        raise AuthenticationException(content, "list_models")

    try:

        model_manager = get_model_manager()
        models = model_manager.list_models()

        loaded_count = sum(1 for model in models if model["loaded"])

        # 获取默认模型的加载模式作为系统模式
        default_model = next((m for m in models if m.get("default")), None)
        asr_model_mode = default_model.get("asr_model_mode", "all") if default_model else "all"

        return {
            "models": models,
            "total": len(models),
            "loaded_count": loaded_count,
            "asr_model_mode": asr_model_mode,
        }
    except Exception as e:
        logger.error(f"获取模型列表时发生错误: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取模型列表失败: {str(e)}")
