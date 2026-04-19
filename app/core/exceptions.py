# -*- coding: utf-8 -*-
"""
统一异常处理模块
定义所有自定义异常类和错误处理函数
"""

from datetime import datetime, timezone
from fastapi import Request
from fastapi.responses import JSONResponse
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def get_iso_timestamp() -> str:
    """获取ISO 8601格式的UTC时间戳"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def create_error_response(
    error_code: str,
    message: str,
    task_id: str = "",
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    创建标准错误响应格式

    Args:
        error_code: 错误代码（如 INVALID_PARAMETER）
        message: 人类可读的错误信息
        task_id: 任务ID（可选）
        details: 额外详细信息（可选）

    Returns:
        标准错误响应字典
    """
    return {
        "error_code": error_code,
        "message": message,
        "task_id": task_id or "",
        "timestamp": get_iso_timestamp(),
        "details": details or {},
    }


def get_http_status_code(status_code: int) -> int:
    """Map internal API status code to HTTP status code."""
    return 500 if status_code >= 50000000 else 400


class APIException(Exception):
    """API基础异常类"""

    def __init__(
        self,
        status_code: int,
        message: str,
        task_id: str = "",
        error_code: str = "",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.status_code = status_code
        self.message = message
        self.task_id = task_id
        self.error_code = error_code or self._get_error_code(status_code)
        self.details = details or {}
        super().__init__(self.message)

    def _get_error_code(self, status_code: int) -> str:
        """根据状态码获取错误代码"""
        code_mapping = {
            20000000: "SUCCESS",
            40000000: "DEFAULT_CLIENT_ERROR",
            40000001: "AUTHENTICATION_FAILED",
            40000002: "INVALID_MESSAGE",
            40000003: "INVALID_PARAMETER",
            40000004: "IDLE_TIMEOUT",
            40000005: "TOO_MANY_REQUESTS",
            40000010: "TRIAL_EXPIRED",
            41010101: "UNSUPPORTED_SAMPLE_RATE",
            50000000: "DEFAULT_SERVER_ERROR",
            50000001: "INTERNAL_GRPC_ERROR",
        }
        return code_mapping.get(status_code, "UNKNOWN_ERROR")

    def to_dict(self) -> Dict[str, Any]:
        """
        将异常转换为标准错误响应字典

        Returns:
            标准错误响应字典，包含 error_code, message, task_id, timestamp, details
        """
        return create_error_response(
            error_code=self.error_code,
            message=self.message,
            task_id=self.task_id,
            details=self.details,
        )


# 标准异常类
class AuthenticationException(APIException):
    """身份认证异常"""

    def __init__(self, message: str, task_id: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(40000001, message, task_id, details=details)


class InvalidMessageException(APIException):
    """无效消息异常"""

    def __init__(self, message: str, task_id: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(40000002, message, task_id, details=details)


class InvalidParameterException(APIException):
    """无效参数异常"""

    def __init__(self, message: str, task_id: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(40000003, message, task_id, details=details)


class UnsupportedSampleRateException(APIException):
    """不支持的采样率异常"""

    def __init__(self, message: str, task_id: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(41010101, message, task_id, details=details)


class DefaultServerErrorException(APIException):
    """默认服务端错误异常"""

    def __init__(self, message: str, task_id: str = "", details: Optional[Dict[str, Any]] = None):
        super().__init__(50000000, message, task_id, details=details)


# 异常处理器
async def api_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """API异常处理器"""
    # FastAPI 只会在抛出 APIException 时调用此处理器
    api_exc = exc if isinstance(exc, APIException) else APIException(50000000, str(exc))
    logger.error(f"[{api_exc.task_id}] API异常: {api_exc.message}")

    # 使用标准错误格式
    response_data = api_exc.to_dict()

    # 确定HTTP状态码
    http_status_code = get_http_status_code(api_exc.status_code)

    return JSONResponse(
        content=response_data,
        headers={"task_id": api_exc.task_id} if api_exc.task_id else {},
        status_code=http_status_code,
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """通用异常处理器"""
    logger.error(f"未处理的异常: {str(exc)}", exc_info=True)

    # 使用标准错误格式
    response_data = create_error_response(
        error_code="DEFAULT_SERVER_ERROR",
        message=f"内部服务错误: {str(exc)}",
    )

    return JSONResponse(content=response_data, status_code=500)
