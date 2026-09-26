# -*- coding: utf-8 -*-
"""
FastAPI应用创建和配置
"""

import warnings
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi_offline import FastAPIOffline
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse

from .core.config import settings
from .core.exceptions import (
    APIException,
    api_exception_handler,
    general_exception_handler,
)
from .core.logging import setup_logging
from .core.executor import shutdown_executor
from .api.v1 import api_router

# 忽略 Pydantic V2 兼容性警告
warnings.filterwarnings("ignore", message="Valid config keys have changed in V2")
warnings.filterwarnings("ignore", message=".*has conflict with protected namespace.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

logger = logging.getLogger(__name__)


def cleanup_temp_directory():
    """清理临时目录中的旧文件"""
    import time

    temp_dir = settings.TEMP_DIR
    if not os.path.exists(temp_dir):
        return

    # 清理超过 1 小时的临时文件
    max_age_seconds = 3600
    current_time = time.time()
    cleaned_count = 0

    try:
        for filename in os.listdir(temp_dir):
            filepath = os.path.join(temp_dir, filename)
            if os.path.isfile(filepath):
                file_age = current_time - os.path.getmtime(filepath)
                if file_age > max_age_seconds:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                    except Exception:
                        pass

        if cleaned_count > 0:
            logger.info(f"已清理 {cleaned_count} 个过期临时文件")
    except Exception as e:
        logger.warning(f"清理临时目录时出错: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Require every offline component before accepting traffic."""
    from .utils.model_loader import preload_models, verify_required_models_integrity

    cleanup_temp_directory()
    integrity = verify_required_models_integrity()
    if integrity["invalid_models"]:
        raise RuntimeError("Required model integrity check failed")
    preload_models()
    logger.info("Shared R2T2 connection, forced aligner, VAD and speaker models ready")
    try:
        yield
    finally:
        shutdown_executor()
        from .services.asr.runtime import get_runtime_router

        get_runtime_router().close()
        from .utils.speaker_diarizer import close_speaker_diarizer

        close_speaker_diarizer()


def create_app() -> FastAPI:
    """创建FastAPI应用"""

    # 设置日志
    setup_logging()

    app = FastAPIOffline(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        docs_url=settings.docs_url,
        redoc_url=settings.redoc_url,
        lifespan=lifespan,  # 添加生命周期管理
    )

    # 添加CORS中间件
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 注册异常处理器
    app.add_exception_handler(APIException, api_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # 注册静态文件服务（用于临时文件）
    app.mount("/tmp", StaticFiles(directory=settings.TEMP_DIR), name="temp_files")

    # 注册API路由
    app.include_router(api_router)

    # 根路径
    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse("realtime")

    return app


# 创建全局应用实例
app = create_app()
