# -*- coding: utf-8 -*-
"""
ASR模型管理器
预加载所有模型，无LRU淘汰机制
"""

import json
import threading
import logging
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path

from typing import Callable
from ...core.config import settings
from ...core.exceptions import DefaultServerErrorException, InvalidParameterException
from .engines import BaseASREngine

logger = logging.getLogger(__name__)

# 引擎注册表（使用Any避免循环导入问题）
_ENGINE_REGISTRY: Dict[str, Callable[[Any], BaseASREngine]] = {}


def register_engine(engine_type: str, factory: Callable[[Any], BaseASREngine]):
    """注册ASR引擎工厂函数"""
    _ENGINE_REGISTRY[engine_type] = factory
    logger.info(f"注册引擎类型: {engine_type}")


def get_registered_engine_types() -> List[str]:
    """获取所有已注册的引擎类型"""
    return list(_ENGINE_REGISTRY.keys())


class ModelCache:
    """简单模型缓存，无淘汰机制"""

    def __init__(self):
        self._cache: Dict[str, BaseASREngine] = {}
        self._lock = threading.RLock()

    def get(self, key: str) -> Optional[BaseASREngine]:
        """获取模型"""
        with self._lock:
            return self._cache.get(key)

    def put(self, key: str, value: BaseASREngine) -> None:
        """添加模型"""
        with self._lock:
            self._cache[key] = value

    def remove(self, key: str) -> bool:
        """移除指定模型"""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear_all(self) -> None:
        """清空所有模型"""
        with self._lock:
            self._cache.clear()

    def keys_list(self) -> List[str]:
        """获取所有模型ID列表"""
        with self._lock:
            return list(self._cache.keys())

    def __contains__(self, key: object) -> bool:
        """检查是否包含指定模型"""
        with self._lock:
            return key in self._cache

    def __len__(self) -> int:
        """获取缓存中的模型数量"""
        with self._lock:
            return len(self._cache)


class ModelConfig:
    """模型配置类"""

    def __init__(self, model_id: str, config: Dict[str, Any]):
        self.model_id = model_id
        self.name = config["name"]
        self.engine = config["engine"]
        self.description = config.get("description", "")
        self.languages = config.get("languages", [])
        self.is_default = config.get("default", False)
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
    """模型管理器，预加载所有模型，无淘汰机制"""

    def __init__(self):
        self._models_config: Dict[str, ModelConfig] = {}
        self._loaded_engines = ModelCache()
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
                self._models_config[model_id] = ModelConfig(model_id, model_config)
                if model_config.get("default", False):
                    self._default_model_id = model_id

            # 根据 ENABLED_MODELS 选择默认模型
            self._default_model_id = self._select_default_model()

            if not self._default_model_id and self._models_config:
                # 如果没有指定默认模型，选择第一个
                self._default_model_id = list(self._models_config.keys())[0]

        except (json.JSONDecodeError, KeyError) as e:
            raise DefaultServerErrorException(f"模型配置文件格式错误: {str(e)}")

    def _select_default_model(self) -> Optional[str]:
        """根据 ENABLED_MODELS 选择默认模型

        Returns:
            选择的模型ID，基于 ENABLED_MODELS 配置
        """
        enabled_models = settings.ENABLED_MODELS.strip().lower()

        # all: 优先使用 1.7b，否则 0.6b，否则 paraformer
        if enabled_models == "all":
            for model_id in ["qwen3-asr-1.7b", "qwen3-asr-0.6b", "paraformer-large"]:
                if model_id in self._models_config:
                    return model_id
            return self._default_model_id

        # auto: 根据显存自动选择
        if enabled_models == "auto":
            selected = self._auto_select_by_vram()
            if selected:
                return selected
            # 显存选择失败，回退到配置中的第一个
            return self._default_model_id

        # 其他: 解析逗号分隔列表，优先使用 Qwen 模型，其次是 Paraformer
        requested = [m.strip() for m in settings.ENABLED_MODELS.split(",") if m.strip()]
        for model in requested:
            model_lower = model.lower()
            if model_lower in self._models_config:
                return model_lower

        return self._default_model_id

    def _auto_select_by_vram(self) -> Optional[str]:
        """根据显存/内存大小自动选择模型

        CUDA/MPS: < 32GB 用 0.6B, >= 32GB 用 1.7B
        CPU: 使用 paraformer-large
        """
        from app.core.device import has_gpu, get_vram_gb

        try:
            if not has_gpu():
                return "paraformer-large" if "paraformer-large" in self._models_config else None

            vram = get_vram_gb()
            if vram >= 32:
                if "qwen3-asr-1.7b" in self._models_config:
                    return "qwen3-asr-1.7b"
            else:
                if "qwen3-asr-0.6b" in self._models_config:
                    return "qwen3-asr-0.6b"

        except Exception as e:
            logger.warning(f"显存检测失败: {e}，使用默认模型")

        return None



    def get_model_config(self, model_id: Optional[str] = None) -> ModelConfig:
        """获取模型配置"""
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        if model_id not in self._models_config:
            available_models = ", ".join(self._models_config.keys())
            raise InvalidParameterException(
                f"未知的模型: {model_id}，可用模型: {available_models}"
            )

        return self._models_config[model_id]

    def list_models(self) -> List[Dict[str, Any]]:
        """列出所有可用模型"""
        models = []
        for model_id, config in self._models_config.items():
            # 检查模型文件是否存在
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

            # 检查模型是否已加载
            loaded = model_id in self._loaded_engines

            # 判断模型加载模式
            if config.offline_model_path and config.realtime_model_path:
                asr_model_mode = "all"
            elif config.offline_model_path:
                asr_model_mode = "offline"
            elif config.realtime_model_path:
                asr_model_mode = "realtime"
            else:
                asr_model_mode = "offline"  # 默认

            models.append(
                {
                    "id": model_id,
                    "name": config.name,
                    "engine": config.engine,
                    "description": config.description,
                    "languages": config.languages,
                    "default": config.is_default,
                    "loaded": loaded,
                    "supports_realtime": config.supports_realtime,
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
                    "asr_model_mode": asr_model_mode,
                }
            )

        return models

    def get_asr_engine(self, model_id: Optional[str] = None, streaming: bool = False) -> BaseASREngine:
        """
        获取ASR引擎

        Args:
            model_id: 模型ID
            streaming: 是否用于流式识别（为Qwen3-ASR创建独立实例）

        Returns:
            ASR引擎实例
        """
        if model_id is None:
            model_id = self._default_model_id

        if not model_id:
            raise InvalidParameterException("未指定模型且没有默认模型")

        # 流式模式使用独立的引擎实例（避免状态干扰）
        engine_key = model_id
        if streaming and model_id in ["qwen3-asr-1.7b", "qwen3-asr-0.6b"]:
            engine_key = f"{model_id}-streaming"
            logger.debug(f"使用流式专用引擎: {engine_key}")

        # 检查是否已加载
        engine = self._loaded_engines.get(engine_key)
        if engine is not None:
            logger.debug(f"从缓存获取模型: {engine_key}")
            return engine

        # 加载新模型
        logger.info(f"加载模型: {engine_key}")
        config = self.get_model_config(model_id)
        engine = self._create_engine(config)

        # 缓存引擎
        self._loaded_engines.put(engine_key, engine)

        return engine

    def _create_engine(self, config: ModelConfig) -> BaseASREngine:
        """创建ASR引擎实例"""
        engine_type = config.engine.lower()
        factory = _ENGINE_REGISTRY.get(engine_type)
        if not factory:
            raise InvalidParameterException(
                f"不支持的引擎类型: {config.engine}"
            )
        return factory(config)

    def unload_model(self, model_id: str) -> bool:
        """卸载指定模型"""
        return self._loaded_engines.remove(model_id)

    def get_memory_usage(self) -> Dict[str, Any]:
        """获取内存使用情况"""
        memory_info = {
            "model_list": self._loaded_engines.keys_list(),
            "loaded_count": len(self._loaded_engines),
        }

        if torch.cuda.is_available():
            memory_info["gpu_memory"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1024**3:.2f}GB",
                "cached": f"{torch.cuda.memory_reserved() / 1024**3:.2f}GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1024**3:.2f}GB",
            }

        return memory_info

    def clear_cache(self) -> None:
        """清空模型缓存"""
        self._loaded_engines.clear_all()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def preload_all_models(self) -> Dict[str, Any]:
        """
        预加载所有配置的模型

        Returns:
            预加载结果统计
        """
        logger.info("开始预加载所有模型...")
        results: dict[str, list[str]] = {
            "success": [],
            "failed": [],
            "skipped": [],
        }

        for model_id, _config in self._models_config.items():
            try:
                logger.info(f"预加载模型: {model_id}")
                engine = self.get_asr_engine(model_id)
                if engine.is_model_loaded():
                    results["success"].append(model_id)
                    logger.info(f"模型 {model_id} 预加载成功")
                else:
                    results["failed"].append(f"{model_id} (未正确加载)")
                    logger.error(f"模型 {model_id} 预加载失败: 未正确加载")
            except Exception as e:
                results["failed"].append(f"{model_id} ({str(e)})")
                logger.error(f"模型 {model_id} 预加载失败: {e}")

        logger.info(
            f"模型预加载完成: 成功 {len(results['success'])}, "
            f"失败 {len(results['failed'])}, 跳过 {len(results['skipped'])}"
        )
        return results

    def validate_model_config(self, model_id: str) -> Dict[str, Any]:
        """验证模型配置是否有效"""
        config = self.get_model_config(model_id)

        errors = []
        if not config.has_offline_model and not config.has_realtime_model:
            errors.append(f"模型 {model_id} 既没有离线版本也没有实时版本")

        return {
            "model_id": model_id,
            "errors": errors,
            "valid": len(errors) == 0,
        }


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
        _register_funasr_engine(register_engine, ModelConfig)
    except ImportError as e:
        logger.warning(f"FunASR引擎不可用: {e}")

    try:
        from .qwen3_engine import Qwen3ASREngine  # noqa: F401
        from .qwen3_engine import _register_qwen3_engine
        _register_qwen3_engine(register_engine, ModelConfig)
    except ImportError as e:
        logger.warning(f"Qwen3引擎不可用: {e}")


# 模块加载时自动注册内置引擎
_register_builtin_engines()
