# -*- coding: utf-8 -*-
"""
ASR数据模型
定义语音识别相关的请求和响应模型
"""

from typing import Optional, List, Union
from pydantic import BaseModel, Field

from .common import (
    SampleRate,
    BaseResponse,
    HealthCheckResponse,
    ErrorResponse,
)


# ============= 请求模型 =============


class ASRQueryParams(BaseModel):
    """ASR接口查询参数模型"""

    audio_address: Optional[str] = Field(
        default=None,
        description="音频文件下载链接（HTTP/HTTPS），格式自动识别",
        max_length=512,
    )

    sample_rate: Optional[SampleRate] = Field(
        default=SampleRate.RATE_16000,
        description=f"音频采样率（Hz）。支持: {', '.join(map(str, SampleRate.get_enums()))}",
    )

    enable_speaker_diarization: Optional[bool] = Field(
        default=True,
        description="是否启用说话人分离。启用后响应会包含 speaker_id",
    )

    word_timestamps: Optional[bool] = Field(
        default=False,
        description="是否返回字词级时间戳（默认关闭；Qwen CUDA vLLM / CPU Rust 会在启用时自动调用 forced aligner）",
    )

    vocabulary_id: Optional[str] = Field(
        default=None,
        description="热词字符串，格式：热词1 权重1 热词2 权重2（如：阿里巴巴 20 腾讯 15）",
        max_length=512,
    )


# ============= 响应模型 =============


class WordToken(BaseModel):
    """字词级时间戳信息"""

    text: str = Field(
        ...,
        description="字词文本",
    )
    start_time: float = Field(
        ...,
        description="开始时间（秒）",
    )
    end_time: float = Field(
        ...,
        description="结束时间（秒）",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "今",
                "start_time": 0.0,
                "end_time": 0.15,
            }
        }
    }


class ASRSegment(BaseModel):
    """ASR 识别分段结果"""

    text: str = Field(
        ...,
        description="该段识别文本",
    )
    start_time: float = Field(
        ...,
        description="段落开始时间（秒）",
    )
    end_time: float = Field(
        ...,
        description="段落结束时间（秒）",
    )
    speaker_id: Optional[str] = Field(
        default=None,
        description="说话人ID（如 说话人1），仅启用说话人分离时返回",
    )
    word_tokens: Optional[List[WordToken]] = Field(
        default=None,
        description="字词级时间戳（仅启用 word_timestamps 且模型支持时返回）",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "text": "今天天气不错。",
                "start_time": 0.0,
                "end_time": 2.5,
                "speaker_id": "说话人1",
                "word_tokens": [
                    {"text": "今", "start_time": 0.0, "end_time": 0.15},
                    {"text": "天", "start_time": 0.15, "end_time": 0.35},
                ],
            }
        }
    }


class ASRSuccessResponse(BaseResponse):
    """ASR成功响应模型"""

    result: str = Field(
        ...,
        description="识别结果文本（完整）",
        max_length=100000,
    )

    segments: Optional[List[ASRSegment]] = Field(
        default=None,
        description="分段识别结果（含时间戳），仅长音频分段识别时返回",
    )

    duration: Optional[float] = Field(
        default=None,
        description="音频总时长（秒）",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "cf7b0c5339244ee29cd4e43fb97f1234",
                "result": "今天天气不错。明天可能会下雨。",
                "segments": [
                    {"text": "今天天气不错。", "start_time": 0.0, "end_time": 2.5, "speaker_id": "说话人1"},
                    {"text": "明天可能会下雨。", "start_time": 3.2, "end_time": 5.8, "speaker_id": "说话人2"},
                ],
                "duration": 5.8,
                "status": 200,
                "message": "SUCCESS",
            }
        }
    }


class ASRErrorResponse(ErrorResponse):
    """ASR错误响应模型"""

    result: str = Field(default="", description="识别结果（错误时为空）")

    model_config = {
        "json_schema_extra": {
            "example": {
                "task_id": "8bae3613dfc54ebfa811a17d8a7a1234",
                "result": "",
                "status": 40000001,
                "message": "Gateway:ACCESS_DENIED:The token 'invalid_token' is invalid!",
            }
        }
    }


class ASRHealthCheckResponse(HealthCheckResponse):
    """ASR健康检查响应模型"""

    model_config = {
        "protected_namespaces": (),
        "json_schema_extra": {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "device": "cuda:0",
                "version": "1.0.0",
                "message": "ASR service is running normally",
                "loaded_models": ["qwen3-asr-1.7b"],
                "memory_usage": {
                    "gpu_memory_used": "2.1GB",
                    "gpu_memory_total": "8.0GB",
                },
            },
        },
    }

    model_loaded: bool = Field(..., description="模型是否已加载")
    device: str = Field(..., description="推理设备")
    loaded_models: Optional[List[str]] = Field(default=[], description="已加载的模型列表")
    memory_usage: Optional[dict] = Field(default=None, description="内存使用情况")


# ============= 模型相关 =============


class ASRDeclaredEntryInfo(BaseModel):
    """声明式 ASR 条目信息，可表示离线模型或 realtime capability。"""

    id: str = Field(..., description="模型id")
    kind: str = Field(..., description="条目类型：model 或 capability")
    name: str = Field(..., description="模型名称")
    engine: str = Field(..., description="引擎类型")
    description: str = Field(..., description="模型描述")
    languages: List[str] = Field(..., description="支持的语言列表")
    default: bool = Field(default=False, description="是否为默认模型")
    supports_realtime: bool = Field(default=False, description="是否支持实时识别")
    offline_model: Optional[dict] = Field(default=None, description="离线模型信息")
    realtime_model: Optional[dict] = Field(default=None, description="实时模型信息")

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "qwen3-asr-1.7b",
                "kind": "model",
                "name": "Qwen3-ASR-1.7B",
                "engine": "qwen3",
                "description": "多语言离线语音识别模型",
                "languages": ["zh", "en"],
                "default": True,
                "supports_realtime": True,
                "offline_model": {
                    "path": "Qwen/Qwen3-ASR-1.7B",
                    "exists": True,
                },
                "realtime_model": None,
            }
        }
    }


class ASRRuntimeInfo(BaseModel):
    """运行时视角的模型加载状态。"""

    loaded_model_ids: List[str] = Field(default_factory=list, description="当前已加载模型 ID 列表")
    loaded_count: int = Field(..., description="已加载模型数量")
    default_offline_model_id: Optional[str] = Field(default=None, description="当前默认离线模型 ID")

    model_config = {
        "json_schema_extra": {
            "example": {
                "loaded_model_ids": ["qwen3-asr-1.7b"],
                "loaded_count": 1,
                "default_offline_model_id": "qwen3-asr-1.7b",
            }
        }
    }


class ASRModelsResponse(BaseModel):
    """ASR 模型列表响应，分离声明视角与运行时视角。"""

    declared_entries: List[ASRDeclaredEntryInfo] = Field(..., description="声明的模型与 capability 列表")
    declared_count: int = Field(..., description="声明条目总数")
    runtime: ASRRuntimeInfo = Field(..., description="运行时加载状态")

    model_config = {
        "json_schema_extra": {
            "example": {
                "declared_entries": [
                    {
                        "id": "qwen3-asr-1.7b",
                        "kind": "model",
                        "name": "Qwen3-ASR-1.7B",
                        "engine": "qwen3",
                        "description": "多语言离线语音识别模型",
                        "languages": ["zh", "en"],
                        "default": True,
                        "supports_realtime": True,
                        "offline_model": {
                            "path": "Qwen/Qwen3-ASR-1.7B",
                            "exists": True,
                        },
                        "realtime_model": None,
                    },
                    {
                        "id": "paraformer-large",
                        "kind": "capability",
                        "name": "Paraformer Large",
                        "engine": "funasr",
                        "description": "中文 WebSocket 实时识别能力",
                        "languages": ["zh"],
                        "default": False,
                        "supports_realtime": True,
                        "offline_model": None,
                        "realtime_model": {
                            "path": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online",
                            "exists": True,
                        },
                    }
                ],
                "declared_count": 3,
                "runtime": {
                    "loaded_model_ids": ["qwen3-asr-1.7b"],
                    "loaded_count": 1,
                    "default_offline_model_id": "qwen3-asr-1.7b",
                },
            }
        }
    }


# ============= 联合响应类型 =============

ASRResponse = Union[ASRSuccessResponse, ASRErrorResponse]
