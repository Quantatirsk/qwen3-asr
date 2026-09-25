# -*- coding: utf-8 -*-
"""
ASR引擎模块
支持多种ASR引擎实现
"""

# 基础类和数据类
from .base import (
    BaseASREngine,
    WordToken,
    ASRSegmentResult,
    ASRFullResult,
    ASRRawResult,
)

# 全局模型管理
from .global_models import (
    get_global_vad_model,
)

__all__ = [
    # 基础类
    "BaseASREngine",
    # 数据类
    "WordToken",
    "ASRSegmentResult",
    "ASRFullResult",
    "ASRRawResult",
    # 全局模型管理
    "get_global_vad_model",
]
