# -*- coding: utf-8 -*-
"""
安全相关功能
包含鉴权、token验证等安全功能
"""

from typing import Optional
from fastapi import Request
from .config import settings

TOKEN_HEADER_NAME = "X-NLS-Token"
AUTH_OPTIONAL_PLACEHOLDER = "optional"


def normalize_token(token: Optional[str]) -> Optional[str]:
    """将 token 归一化为非空字符串或 None。"""
    if token is None:
        return None

    normalized = token.strip()
    return normalized or None


def get_expected_api_key(expected_token: Optional[str] = None) -> Optional[str]:
    """获取归一化后的期望 API_KEY。"""
    if expected_token is not None:
        return normalize_token(expected_token)
    return normalize_token(settings.API_KEY)


def mask_sensitive_data(
    data: str, mask_char: str = "*", keep_prefix: int = 4, keep_suffix: int = 4
) -> str:
    """遮盖敏感数据

    Args:
        data: 需要遮盖的数据
        mask_char: 遮盖字符
        keep_prefix: 保留前缀字符数
        keep_suffix: 保留后缀字符数

    Returns:
        遮盖后的数据
    """
    if not data or len(data) <= keep_prefix + keep_suffix:
        return data

    prefix = data[:keep_prefix]
    suffix = data[-keep_suffix:] if keep_suffix > 0 else ""
    mask_length = len(data) - keep_prefix - keep_suffix
    mask = mask_char * mask_length

    return f"{prefix}{mask}{suffix}"


def validate_token_value(token: Optional[str], expected_token: Optional[str] = None) -> bool:
    """验证访问令牌

    Args:
        token: 客户端提供的token
        expected_token: 期望的token值（从环境变量读取），如果为None则鉴权可选

    Returns:
        bool: 验证结果
    """
    normalized_expected_token = get_expected_api_key(expected_token)
    if not normalized_expected_token:
        return True

    normalized_token = normalize_token(token)
    if not normalized_token:
        return False

    # 简单的token格式验证（长度检查）
    if len(normalized_token) < 10:
        return False

    # 验证token是否匹配
    if normalized_token != normalized_expected_token:
        return False

    return True


def extract_header_token(request: Request) -> Optional[str]:
    """从标准头部提取 token。"""
    return normalize_token(request.headers.get(TOKEN_HEADER_NAME))


def extract_bearer_token(request: Request) -> Optional[str]:
    """从 Authorization: Bearer 提取 token。"""
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        return None

    scheme, _, value = auth_header.partition(" ")
    if scheme.lower() != "bearer":
        return None
    return normalize_token(value)


def extract_openai_token(request: Request) -> Optional[str]:
    """OpenAI 兼容接口鉴权：优先 Bearer，其次 X-NLS-Token。"""
    return extract_bearer_token(request) or extract_header_token(request)


def _validate_resolved_token(
    token: Optional[str],
    missing_message: str,
    expected_token: Optional[str] = None,
) -> tuple[bool, str]:
    """统一 token 校验逻辑。"""
    expected = get_expected_api_key(expected_token)
    normalized_token = normalize_token(token)

    if not expected:
        return True, normalized_token or AUTH_OPTIONAL_PLACEHOLDER

    if not normalized_token:
        return False, missing_message

    if not validate_token_value(normalized_token, expected):
        masked_token = mask_sensitive_data(normalized_token)
        return False, f"Gateway:ACCESS_DENIED:The token '{masked_token}' is invalid!"

    return True, normalized_token


def validate_token(request: Request, task_id: str = "") -> tuple[bool, str]:
    """验证X-NLS-Token头部"""
    _ = task_id
    token = extract_header_token(request)
    return _validate_resolved_token(token, "缺少X-NLS-Token头部")


def validate_openai_token(request: Request, task_id: str = "") -> tuple[bool, str]:
    """验证 OpenAI 兼容接口 token（Bearer/X-NLS-Token）。"""
    _ = task_id
    token = extract_openai_token(request)
    return _validate_resolved_token(token, "缺少Authorization Bearer或X-NLS-Token头部")
