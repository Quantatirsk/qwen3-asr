# -*- coding: utf-8 -*-
"""
异步执行器模块

用于将同步的模型推理调用放入线程池执行，避免阻塞事件循环，
实现真正的多路并发处理。

设计要点：
1. 使用 ThreadPoolExecutor 而非 ProcessPoolExecutor
   - 模型已加载在内存中，进程间无法共享
   - GPU操作会自动释放GIL，线程池足以实现并发

2. 对于流式生成器，使用 asyncio.Queue 实现异步迭代

3. 线程池大小根据使用场景配置：
   - CPU推理：受GIL限制，多线程并发收益有限，但可以让I/O不阻塞
   - GPU推理：CUDA操作释放GIL，可以实现真正并发
"""

import os
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar, Generator, AsyncGenerator, Optional, ParamSpec
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


async def run_sync_generator(
    generator_func: Callable[..., Generator[T, None, None]],
    *args,
    **kwargs
) -> AsyncGenerator[T, None]:
    """
    将同步生成器转换为异步生成器，在线程池中执行

    用于流式处理等需要逐步产出结果的场景。

    Args:
        generator_func: 返回生成器的同步函数
        *args: 位置参数
        **kwargs: 关键字参数

    Yields:
        生成器的每个产出值

    Example:
        async for chunk in run_sync_generator(model.inference_sft, text, voice, stream=True):
            await websocket.send_bytes(chunk)
    """
    loop = asyncio.get_running_loop()
    executor = get_executor()
    queue: asyncio.Queue = asyncio.Queue()

    # 标记生成器结束的哨兵值
    _SENTINEL = object()

    def producer():
        """在线程中运行生成器，将结果放入队列"""
        try:
            gen = generator_func(*args, **kwargs)
            for item in gen:
                # 使用 call_soon_threadsafe 安全地将结果放入队列
                loop.call_soon_threadsafe(queue.put_nowait, item)
        except BaseException as e:
            # 发生异常时，记录日志并将异常放入队列
            logger.error(f"生成器执行异常: {type(e).__name__}: {e}")
            loop.call_soon_threadsafe(queue.put_nowait, e)
        finally:
            # 发送结束标记
            loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

    # 在线程池中启动生产者
    future = executor.submit(producer)

    try:
        while True:
            item = await queue.get()

            if item is _SENTINEL:
                break

            # 使用 BaseException 捕获所有异常类型（包括 KeyboardInterrupt 等）
            if isinstance(item, BaseException):
                raise item

            yield item
    finally:
        # 确保检查线程是否有未捕获的异常
        if future.done():
            try:
                # 如果线程已完成，检查是否有异常
                future.result()
            except Exception as e:
                logger.error(f"生成器线程异常: {type(e).__name__}: {e}")
        elif not future.cancelled():
            future.cancel()


class AsyncInferenceWrapper:
    """
    异步推理包装器

    将同步的模型推理方法包装为异步方法，方便复用。

    Example:
        wrapper = AsyncInferenceWrapper(asr_engine.realtime_model)
        result = await wrapper.generate(input=audio_array, cache=cache)
    """

    def __init__(self, model):
        self._model = model

    async def generate(self, *args, **kwargs):
        """异步调用模型的 generate 方法"""
        return await run_sync(self._model.generate, *args, **kwargs)

    async def inference_sft(self, *args, **kwargs):
        """异步流式调用模型的 inference_sft 方法"""
        async for item in run_sync_generator(self._model.inference_sft, *args, **kwargs):
            yield item

    async def inference_zero_shot(self, *args, **kwargs):
        """异步流式调用模型的 inference_zero_shot 方法"""
        async for item in run_sync_generator(self._model.inference_zero_shot, *args, **kwargs):
            yield item
