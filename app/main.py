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

from .core.config import settings
from .core.exceptions import (
    APIException,
    api_exception_handler,
    general_exception_handler,
)
from .core.logging import setup_logging, get_worker_id
from .core.executor import shutdown_executor
from .api.v1 import api_router
from .utils.boot_events import emit_boot_event

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
    """应用生命周期管理"""
    workers = int(os.getenv("WORKERS", "1"))
    worker_id = get_worker_id()

    # 启动时
    logger.info(f"Worker [{worker_id}] 启动中...")
    emit_boot_event("phase_start", phase="worker", total=1, message=f"Worker [{worker_id}] 启动中")

    # 清理旧的临时文件（仅主 Worker 执行）
    if worker_id == 0:
        cleanup_temp_directory()

    # 所有 Worker 都需要预加载模型
    try:
        from .utils.model_loader import (
            preload_models,
            verify_required_models_integrity,
        )

        integrity_result = verify_required_models_integrity()
        if integrity_result["invalid_models"]:
            emit_boot_event("error", phase="integrity", message="required model integrity check failed")
            raise RuntimeError("required model integrity check failed")

        logger.info(f"Worker [{worker_id}] 正在加载模型...")
        preload_result = preload_models()

        # 记录加载结果
        asr_results = preload_result.get("asr_models", {})
        loaded_count = sum(1 for r in asr_results.values() if r.get("loaded"))
        total_count = len(asr_results)
        logger.info(f"Worker [{worker_id}] 模型加载完成: {loaded_count}/{total_count}")
        failed_asr_models = {
            model_id: status.get("error")
            for model_id, status in asr_results.items()
            if not status.get("loaded") and status.get("error")
        }
        if failed_asr_models:
            logger.error(f"Worker [{worker_id}] ASR模型预加载失败详情: {failed_asr_models}")
            emit_boot_event("error", phase="preload", message=f"ASR模型预加载失败详情: {failed_asr_models}")

    except Exception as e:
        logger.error(f"Worker [{worker_id}] 模型预加载失败: {e}")
        logger.warning(f"Worker [{worker_id}] 模型将在首次使用时加载")
        emit_boot_event("error", phase="worker", message=f"模型预加载失败: {e}")

    logger.info(f"Worker [{worker_id}] 已就绪")
    emit_boot_event("ready", phase="worker", message=f"Worker [{worker_id}] 已就绪")

    yield

    # 关闭时
    logger.info(f"Worker [{worker_id}] 正在关闭推理线程池...")
    shutdown_executor()
    logger.info(f"Worker [{worker_id}] 已关闭")


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
    @app.get("/", summary="根路径", description="API服务根路径")
    async def root():
        return {
            "message": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "description": settings.APP_DESCRIPTION,
            "endpoints": {
                # 阿里云兼容 API
                "asr": "/stream/v1/asr",
                "asr_models": "/stream/v1/asr/models",
                "asr_health": "/stream/v1/asr/health",
                "ws_asr": "/ws/v1/asr",
                # Qwen3-ASR 专用 WebSocket 流式 (POC)
                "ws_qwen3_asr": "/ws/v1/qwen3/asr",
                # OpenAI 兼容 API
                "openai_models": "/v1/models",
                "openai_transcriptions": "/v1/audio/transcriptions",
                # 文档
                "docs": settings.docs_url or "禁用",
            },
        }

    return app


# 创建全局应用实例
app = create_app()
