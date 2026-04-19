# -*- coding: utf-8 -*-
"""ASR model metadata and engine factory."""

import json
import threading
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

from typing import Callable
from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException, InvalidParameterException
from .engines import BaseASREngine
from .model_plan import get_default_model_id

logger = logging.getLogger(__name__)

# 引擎注册表（使用Any避免循环导入问题）
_ENGINE_REGISTRY: Dict[str, Callable[[Any], BaseASREngine]] = {}


def _supports_qwen_realtime_on_device(configured_device: str) -> bool:
    """Resolve whether Qwen realtime mode is available on the active device."""
    from app.core.device import detect_device
    from .qwen3_mlx import is_mlx_qwen_available
    from .qwenasr_rust import is_qwenasr_rust_available

    device = detect_device(configured_device)
    if device.startswith("cuda"):
        return True
    if device == "mps":
        return is_mlx_qwen_available()
    if device == "cpu":
        return is_qwenasr_rust_available()
    return False


def register_engine(engine_type: str, factory: Callable[[Any], BaseASREngine]):
    """注册ASR引擎工厂函数"""
    _ENGINE_REGISTRY[engine_type] = factory
    logger.info(f"注册引擎类型: {engine_type}")


class DeclaredEntryConfig:
    """声明条目配置，可表示模型或 capability。"""

    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.name = config["name"]
        self.kind = config.get("kind", "model")
        self.engine = config["engine"]
        self.description = config.get("description", "")
        self.languages = config.get("languages", [])
        self.supports_realtime = config.get("supports_realtime", False)

        # 模型路径结构
        self.models = config.get("models", {})
        self.offline_model_path = self.models.get("offline")
        self.realtime_model_path = self.models.get("realtime")

        # 额外参数（如 trust_remote_code 等）
        self.extra_kwargs = config.get("extra_kwargs", {})

    @property
    def has_offline_model(self) -> bool:
        """是否有离线模型"""
        return bool(self.offline_model_path)

    @property
    def has_realtime_model(self) -> bool:
        """是否有实时模型"""
        return bool(self.realtime_model_path)

    def get_model_path(self, model_type: str = "offline") -> Optional[str]:
        """根据模型类型获取模型路径"""
        if model_type == "offline":
            return self.offline_model_path
        elif model_type == "realtime":
            return self.realtime_model_path
        return None


class ModelManager:
    """Static model metadata plus engine construction."""

    def __init__(self):
        self._declared_entry_configs: Dict[str, DeclaredEntryConfig] = {}
        self._default_model_id: Optional[str] = None
        self._load_models_config()

    def _load_models_config(self) -> None:
        """加载模型配置文件"""
        models_file = Path(settings.models_config_path)
        if not models_file.exists():
            raise DefaultServerErrorException("models.json 配置文件不存在")

        try:
            with open(models_file, "r", encoding="utf-8") as f:
                config = json.load(f)

            for model_id, model_config in config["models"].items():
                self._declared_entry_configs[model_id] = DeclaredEntryConfig(model_id, model_config)
            self._default_model_id = get_default_model_id(
                all_model_ids=list(self._declared_entry_configs.keys()),
            )

            if not self._default_model_id and self._declared_entry_configs:
                self._default_model_id = list(self._declared_entry_configs.keys())[0]

        except (json.JSONDecodeError, KeyError) as e:
            raise DefaultServerErrorException(f"模型配置文件格式错误: {str(e)}")

    def get_declared_entry_config(self, model_id: Optional[str] = None) -> DeclaredEntryConfig:
        """获取声明条目配置。"""
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        if model_id not in self._declared_entry_configs:
            available_models = ", ".join(self._declared_entry_configs.keys())
            raise InvalidParameterException(
                f"未知的模型: {model_id}，可用模型: {available_models}"
            )

        return self._declared_entry_configs[model_id]

    def list_declared_entries(self) -> List[Dict[str, Any]]:
        """列出声明的模型与 capability 元数据。"""
        entries = []
        for model_id, config in self._declared_entry_configs.items():
            offline_path_exists = False
            realtime_path_exists = False

            if config.offline_model_path:
                offline_model_path = (
                    Path(settings.MODELSCOPE_PATH) / config.offline_model_path
                )
                offline_path_exists = offline_model_path.exists()

            if config.realtime_model_path:
                realtime_model_path = (
                    Path(settings.MODELSCOPE_PATH) / config.realtime_model_path
                )
                realtime_path_exists = realtime_model_path.exists()

            supports_realtime = config.supports_realtime
            if config.engine == "qwen3":
                supports_realtime = _supports_qwen_realtime_on_device(settings.DEVICE)

            entries.append(
                {
                    "id": model_id,
                    "kind": config.kind,
                    "name": config.name,
                    "engine": config.engine,
                    "description": config.description,
                    "languages": config.languages,
                    "default": model_id == self._default_model_id,
                    "supports_realtime": supports_realtime,
                    "offline_model": (
                        {
                            "path": config.offline_model_path,
                            "exists": offline_path_exists,
                        }
                        if config.offline_model_path
                        else None
                    ),
                    "realtime_model": (
                        {
                            "path": config.realtime_model_path,
                            "exists": realtime_path_exists,
                        }
                        if config.realtime_model_path
                        else None
                    ),
                }
            )

        return entries

    def _create_engine(self, config: DeclaredEntryConfig) -> BaseASREngine:
        """创建ASR引擎实例"""
        engine_type = config.engine.lower()
        factory = _ENGINE_REGISTRY.get(engine_type)
        if not factory:
            raise InvalidParameterException(
                f"不支持的引擎类型: {config.engine}"
            )
        return factory(config)

    def create_engine(self, model_id: Optional[str] = None) -> BaseASREngine:
        """Create a fresh engine instance."""
        config = self.get_declared_entry_config(model_id)
        return self._create_engine(config)

# 全局模型管理器实例
_model_manager: Optional[ModelManager] = None
_model_manager_lock = threading.Lock()


def get_model_manager() -> ModelManager:
    """获取全局模型管理器实例（线程安全）"""
    global _model_manager
    if _model_manager is None:
        with _model_manager_lock:
            if _model_manager is None:
                _model_manager = ModelManager()
    return _model_manager


# 注册内置引擎
def _register_builtin_engines():
    """注册内置的ASR引擎"""
    # 导入引擎模块并注册
    try:
        from .engines import FunASREngine  # noqa: F401
        from .engines.funasr import _register_funasr_engine
        _register_funasr_engine(register_engine, DeclaredEntryConfig)
    except ImportError as e:
        logger.warning(f"FunASR引擎不可用: {e}")

    try:
        from .qwen3_engine import Qwen3ASREngine  # noqa: F401
        from .qwen3_engine import _register_qwen3_engine
        _register_qwen3_engine(register_engine, DeclaredEntryConfig)
    except ImportError as e:
        logger.warning(f"Qwen3引擎不可用: {e}")


# 模块加载时自动注册内置引擎
_register_builtin_engines()
