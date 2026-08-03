# -*- coding: utf-8 -*-
"""
异步执行器模块

用于将同步的模型推理调用放入线程池执行，避免阻塞事件循环，
实现真正的多路并发处理。

设计要点：
1. 使用 ThreadPoolExecutor 而非 ProcessPoolExecutor
   - 模型已加载在内存中，进程间无法共享
   - GPU操作会自动释放GIL，线程池足以实现并发

2. 线程池用于隔离 CPU 预处理和远程 HTTP 调用，避免阻塞事件循环。
"""

import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar, Optional
from functools import partial

logger = logging.getLogger(__name__)

# 类型变量
T = TypeVar("T")

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
            max_workers=_MAX_WORKERS,
            thread_name_prefix="inference_worker"
        )
        logger.info(f"推理线程池已创建，最大工作线程数: {_MAX_WORKERS}")
    return _executor


def shutdown_executor():
    """关闭线程池执行器"""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=True)
        _executor = None
        logger.info("推理线程池已关闭")


async def run_sync(func: Callable[..., T], *args, **kwargs) -> T:
    """
    在线程池中执行同步函数，不阻塞事件循环

    Args:
        func: 同步函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        函数返回值

    Example:
        result = await run_sync(model.generate, input=audio_array, cache=cache)
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()

    # 使用 partial 绑定参数
    if kwargs:
        func_with_args = partial(func, *args, **kwargs)
    else:
        func_with_args = partial(func, *args) if args else func

    return await loop.run_in_executor(executor, func_with_args)
