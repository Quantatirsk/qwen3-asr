# -*- coding: utf-8 -*-
"""API v1版本路由"""

from fastapi import APIRouter
from .asr import router as asr_router
from .websocket_asr import router as websocket_asr_router
from .openai_compatible import router as openai_router

api_router = APIRouter()

# 原有 API (阿里云兼容)
api_router.include_router(asr_router)

# WebSocket ASR 端点（包含阿里云协议和 Qwen3 流式协议）
api_router.include_router(websocket_asr_router)

# OpenAI 兼容 API
api_router.include_router(openai_router)
