# -*- coding: utf-8 -*-
"""
ASR引擎模块
支持多种ASR引擎实现
"""

# 基础类和数据类
from .base import (
    BaseASREngine,
    RealTimeASREngine,
    ModelType,
    WordToken,
    ASRSegmentResult,
    ASRFullResult,
    ASRRawResult,
)

# FunASR引擎
from .funasr import FunASREngine

# 全局模型管理
from .global_models import (
    get_global_vad_model,
    get_global_punc_model,
    get_global_punc_realtime_model,
    get_punc_inference_lock,
    get_punc_realtime_inference_lock,
)

__all__ = [
    # 基础类
    "BaseASREngine",
    "RealTimeASREngine",
    "ModelType",
    # 数据类
    "WordToken",
    "ASRSegmentResult",
    "ASRFullResult",
    "ASRRawResult",
    # 引擎实现
    "FunASREngine",
    # 全局模型管理
    "get_global_vad_model",
    "get_global_punc_model",
    "get_global_punc_realtime_model",
    "get_punc_inference_lock",
    "get_punc_realtime_inference_lock",
]
