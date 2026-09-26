# -*- coding: utf-8 -*-
"""Run blocking inference while retaining resources until cancellation drains."""

import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar, Optional, ParamSpec
from functools import partial

from anyio import CancelScope

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar("T")
P = ParamSpec("P")

# 全局线程池执行器
# 默认线程数：max(4, CPU核心数)，可通过环境变量覆盖
_DEFAULT_WORKERS = max(4, os.cpu_count() or 4)
_MAX_WORKERS = int(os.getenv("INFERENCE_THREAD_POOL_SIZE", str(_DEFAULT_WORKERS)))

_executor: Optional[ThreadPoolExecutor] = None


def get_executor() -> ThreadPoolExecutor:
    """获取全局线程池执行器（懒加载）"""
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(
            max_workers=_MAX_WORKERS, thread_name_prefix="inference_worker"
        )
        logger.info(f"推理线程池已创建，最大工作线程数: {_MAX_WORKERS}")
    return _executor


def shutdown_executor() -> None:
    """关闭线程池执行器"""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("推理线程池已关闭")


async def wait_for_completion(future: asyncio.Future[T]) -> T:
    """Delay cancellation until owned work finishes, including repeated cancellation."""
    cancellation: Optional[asyncio.CancelledError] = None
    # Starlette uses level cancellation; shielding avoids a busy cancellation loop.
    with CancelScope(shield=True):
        while not future.done():
            try:
                await asyncio.shield(future)
            except asyncio.CancelledError as exc:
                if future.cancelled():
                    raise
                cancellation = exc
            except Exception:
                break
    if cancellation is not None:
        if not future.cancelled():
            future.exception()
        raise cancellation
    return future.result()


async def run_sync(func: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    """Run blocking work without releasing caller-owned resources on cancellation."""
    future = asyncio.get_running_loop().run_in_executor(
        get_executor(), partial(func, *args, **kwargs)
    )
    return await wait_for_completion(future)
