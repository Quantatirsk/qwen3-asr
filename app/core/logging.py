# -*- coding: utf-8 -*-
"""
日志配置模块
统一的日志配置和管理，支持多 Worker 模式
"""

import logging
import logging.handlers
import sys
import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from pathlib import Path
from .config import settings


class StructuredLogFormatter(logging.Formatter):
    """结构化JSON日志格式化器

    将日志记录格式化为JSON格式，支持extra字段传递结构化数据。

    输出示例:
    {
        "timestamp": "2025-01-31T12:00:00Z",
        "level": "INFO",
        "logger": "app.services.asr",
        "message": "推理完成",
        "task_id": "xxx",
        "duration_ms": 1234,
        "audio_duration_sec": 60,
        "rtf": 0.02,
        "model_id": "qwen3-asr-1.7b"
    }
    """

    def __init__(self, include_extra: bool = True):
        """
        Args:
            include_extra: 是否包含extra字段中的结构化数据
        """
        super().__init__()
        self.include_extra = include_extra
        self._reserved_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
            'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'message', 'asctime'
        }

    def format(self, record: logging.LogRecord) -> str:
        """将日志记录格式化为JSON"""
        log_data: Dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # 添加worker_id（多worker模式下）
        workers = int(os.getenv("WORKERS", "1"))
        if workers > 1:
            log_data["worker_id"] = f"worker-{os.getpid()}"

        # 添加异常信息（如果有）
        if record.exc_info:
            exc_type = record.exc_info[0]
            exc_value = record.exc_info[1]
            if exc_type and exc_value:
                log_data["exception"] = {
                    "type": exc_type.__name__,
                    "message": str(exc_value)
                }

        # 添加extra字段中的结构化数据
        if self.include_extra:
            extra_data = self._extract_extra_data(record)
            if extra_data:
                log_data.update(extra_data)

        return json.dumps(log_data, ensure_ascii=False, default=str)

    def _extract_extra_data(self, record: logging.LogRecord) -> Dict[str, Any]:
        """从日志记录中提取extra数据"""
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in self._reserved_attrs and not key.startswith('_'):
                extra_data[key] = value
        return extra_data


class HybridLogFormatter(logging.Formatter):
    """混合日志格式化器

    根据日志内容自动选择格式：
    - 包含结构化数据（extra字段）的日志使用JSON格式
    - 普通日志使用文本格式

    这允许在代码中逐步迁移到结构化日志，同时保持可读性。
    """

    def __init__(
        self,
        text_format: Optional[str] = None,
        json_formatter: Optional[StructuredLogFormatter] = None,
    ):
        super().__init__()
        self.text_format = text_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        self.json_formatter = json_formatter or StructuredLogFormatter()
        self._reserved_attrs = {
            'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
            'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
            'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
            'thread', 'threadName', 'processName', 'process', 'getMessage',
            'message', 'asctime'
        }

    def format(self, record: logging.LogRecord) -> str:
        """根据内容选择格式"""
        # 检查是否有extra数据
        has_extra = self._has_extra_data(record)

        if has_extra:
            return self.json_formatter.format(record)
        else:
            # 使用文本格式
            text_formatter = logging.Formatter(self.text_format)
            return text_formatter.format(record)

    def _has_extra_data(self, record: logging.LogRecord) -> bool:
        """检查日志记录是否包含extra数据"""
        for key in record.__dict__.keys():
            if key not in self._reserved_attrs and not key.startswith('_'):
                return True
        return False


def get_structured_logger(name: str) -> logging.Logger:
    """获取支持结构化日志的记录器

    这是一个便捷函数，返回一个配置好的日志记录器，
    可以直接使用 extra 参数记录结构化数据。

    示例:
        logger = get_structured_logger(__name__)
        logger.info(
            "推理完成",
            extra={
                "duration_ms": 1234,
                "audio_duration_sec": 60,
                "rtf": 0.02,
                "model_id": "qwen3-asr-1.7b"
            }
        )

    Args:
        name: 记录器名称

    Returns:
        配置好的日志记录器
    """
    return logging.getLogger(name)


def log_inference_metrics(
    logger: logging.Logger,
    message: str,
    task_id: Optional[str] = None,
    duration_ms: Optional[float] = None,
    audio_duration_sec: Optional[float] = None,
    model_id: Optional[str] = None,
    status: str = "success",
    **kwargs
) -> None:
    """记录推理性能指标

    这是一个辅助函数，用于统一记录推理性能指标。

    Args:
        logger: 日志记录器
        message: 日志消息
        task_id: 任务ID
        duration_ms: 推理耗时（毫秒）
        audio_duration_sec: 音频时长（秒）
        model_id: 模型ID
        status: 状态（success/error）
        **kwargs: 其他结构化数据
    """
    extra: Dict[str, Any] = {
        "status": status,
    }

    if task_id:
        extra["task_id"] = task_id
    if duration_ms is not None:
        extra["duration_ms"] = round(duration_ms, 2)
    if audio_duration_sec is not None:
        extra["audio_duration_sec"] = round(audio_duration_sec, 2)
    if model_id:
        extra["model_id"] = model_id

    # 计算RTF（实时率）
    if duration_ms is not None and audio_duration_sec is not None and audio_duration_sec > 0:
        rtf = (duration_ms / 1000) / audio_duration_sec
        extra["rtf"] = round(rtf, 4)

    # 添加其他数据
    extra.update(kwargs)

    logger.info(message, extra=extra)


def get_worker_id() -> str:
    """获取当前 Worker ID

    Returns:
        Worker 标识符，格式为 'worker-{pid}' 或 'main'
    """
    # 检查是否在多 worker 模式下
    workers = int(os.getenv("WORKERS", "1"))
    if workers > 1:
        return f"worker-{os.getpid()}"
    return "main"


def setup_logging(
    level: Optional[str] = None,
    log_file: Optional[str] = None,
    format_string: Optional[str] = None,
    max_bytes: Optional[int] = None,
    backup_count: Optional[int] = None,
    worker_id: Optional[str] = None,
    use_structured: bool = False,
) -> None:
    """设置应用日志配置

    Args:
        level: 日志级别
        log_file: 日志文件路径
        format_string: 日志格式字符串
        max_bytes: 单个日志文件最大大小（字节）
        backup_count: 保留的备份文件数量
        worker_id: Worker 标识符（多 Worker 模式下使用）
        use_structured: 是否使用结构化JSON日志格式
    """
    # 使用传入的参数或配置文件中的设置
    log_level = level or settings.LOG_LEVEL
    log_file_path = log_file or settings.LOG_FILE
    max_file_size = max_bytes or settings.LOG_MAX_BYTES
    backup_files = backup_count or settings.LOG_BACKUP_COUNT

    # 获取 Worker ID
    current_worker_id = worker_id or get_worker_id()
    workers = int(os.getenv("WORKERS", "1"))
    worker_log_path: Optional[Path] = None

    # 确定日志格式
    if use_structured:
        # 使用结构化日志格式
        formatter: logging.Formatter = StructuredLogFormatter()
    else:
        # 使用混合格式（普通日志文本，带extra的JSON）
        if workers > 1:
            text_format = format_string or f"%(asctime)s - [{current_worker_id}] - %(name)s - %(levelname)s - %(message)s"
        else:
            text_format = format_string or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = HybridLogFormatter(text_format=text_format)

    # 创建处理器列表
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers: list[logging.Handler] = [stream_handler]

    # 确定日志文件路径
    if log_file_path:
        log_path = Path(log_file_path)
    else:
        log_path = Path("logs/qwen3-asr.log")

    # 确保日志目录存在
    log_dir = log_path.parent
    log_dir.mkdir(parents=True, exist_ok=True)

    # 多 Worker 模式下，每个 Worker 使用独立的日志文件
    if workers > 1:
        # Example: qwen3-asr.log -> qwen3-asr.worker-12345.log
        worker_log_path = log_dir / f"{log_path.stem}.{current_worker_id}{log_path.suffix}"

        # Worker 专属日志文件
        worker_file_handler = logging.handlers.RotatingFileHandler(
            worker_log_path,
            maxBytes=max_file_size,
            backupCount=backup_files,
            encoding="utf-8",
        )
        worker_file_handler.setFormatter(formatter)
        handlers.append(worker_file_handler)

        # 同时也写入主日志文件（汇总所有 Worker 的日志）
        main_file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_files,
            encoding="utf-8",
        )
        main_file_handler.setFormatter(formatter)
        handlers.append(main_file_handler)
    else:
        # 单 Worker 模式，只写入主日志文件
        file_handler = logging.handlers.RotatingFileHandler(
            log_path,
            maxBytes=max_file_size,
            backupCount=backup_files,
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # 配置根日志记录器
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        handlers=handlers,
        force=True,  # 强制重新配置
    )

    # 设置第三方库的日志级别（由LOG_LEVEL控制）
    third_party_level = getattr(logging, log_level.upper())
    logging.getLogger("urllib3").setLevel(third_party_level)
    logging.getLogger("requests").setLevel(third_party_level)
    logging.getLogger("httpx").setLevel(third_party_level)
    logging.getLogger("httpcore").setLevel(third_party_level)

    # 始终禁用噪音特别大的库
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("numba.core").setLevel(logging.WARNING)
    logging.getLogger("numba.core.ssa").setLevel(logging.WARNING)

    # 多 Worker 模式下记录启动日志
    if workers > 1 and worker_log_path:
        logger = logging.getLogger(__name__)
        logger.info(f"Worker {current_worker_id} 日志系统已初始化，日志文件: {worker_log_path}")
