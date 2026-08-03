# -*- coding: utf-8 -*-
"""API v1版本路由"""

from fastapi import APIRouter
from .asr import router as asr_router
from .openai_compatible import router as openai_router

api_router = APIRouter()

# 原有 API (阿里云兼容)
api_router.include_router(asr_router)

# OpenAI 兼容 API
api_router.include_router(openai_router)
