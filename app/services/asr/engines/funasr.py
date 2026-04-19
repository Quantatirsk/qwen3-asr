# -*- coding: utf-8 -*-
"""FunASR realtime engine for Paraformer websocket recognition."""

import logging
from typing import Optional, Dict, Any, NoReturn
from funasr import AutoModel

from app.core.config import settings
from app.core.exceptions import DefaultServerErrorException
from app.infrastructure import resolve_model_path
from app.services.asr.engines.base import RealTimeASREngine, ASRRawResult

logger = logging.getLogger(__name__)


class FunASREngine(RealTimeASREngine):
    """FunASR realtime-only engine used by websocket protocols."""

    OFFLINE_DISABLED_MESSAGE = "paraformer-large only supports realtime websocket recognition"

    def __init__(
        self,
        realtime_model_path: Optional[str] = None,
        device: str = "auto",
    ):
        self.realtime_model: Optional[AutoModel] = None
        self._device: str = self._detect_device(device)
        self.realtime_model_path = realtime_model_path
        self._load_realtime_model()

    def _load_realtime_model(self) -> None:
        """Load the realtime Paraformer model."""
        try:
            if not self.realtime_model_path:
                raise DefaultServerErrorException("未提供实时模型路径")
            resolved_model_path = resolve_model_path(self.realtime_model_path)
            logger.info(f"正在加载实时FunASR模型: {resolved_model_path}")

            model_kwargs = {
                "model": resolved_model_path,
                "device": self._device,
                **settings.FUNASR_AUTOMODEL_KWARGS,
            }

            self.realtime_model = AutoModel(**model_kwargs)
            logger.info("实时FunASR模型加载成功（PUNC将按需使用全局实例）")

        except Exception as e:
            raise DefaultServerErrorException(f"实时FunASR模型加载失败: {str(e)}")

    def _raise_offline_disabled(self) -> NoReturn:
        raise DefaultServerErrorException(self.OFFLINE_DISABLED_MESSAGE)

    def transcribe_file(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = False,
        enable_itn: bool = False,
        enable_vad: bool = False,
        sample_rate: int = 16000,
        language: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        _ = (audio_path, hotwords, enable_punctuation, enable_itn, enable_vad, sample_rate, language, task_id)
        self._raise_offline_disabled()

    def transcribe_file_with_vad(
        self,
        audio_path: str,
        hotwords: str = "",
        enable_punctuation: bool = True,
        enable_itn: bool = True,
        sample_rate: int = 16000,
        **kwargs,
    ) -> ASRRawResult:
        _ = (audio_path, hotwords, enable_punctuation, enable_itn, sample_rate, kwargs)
        self._raise_offline_disabled()

    def transcribe_websocket(
        self,
        audio_chunk: bytes,
        cache: Optional[Dict] = None,
        is_final: bool = False,
        **kwargs: Any,
    ) -> str:
        """This engine is routed through websocket_asr.py, not through generic engine calls."""
        _ = (audio_chunk, cache, is_final, kwargs)
        raise DefaultServerErrorException(
            "FunASR realtime websocket is handled by websocket_asr.py, not by FunASREngine.transcribe_websocket"
        )

    def is_model_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self.realtime_model is not None

    @property
    def device(self) -> str:
        """获取设备信息"""
        return self._device

    @property
    def model_id(self) -> str:
        """获取模型ID"""
        return "paraformer-realtime"


# 自动注册 FunASR 引擎（由 manager.py 显式触发）
def _register_funasr_engine(register_func, _declared_entry_cls):
    """注册 FunASR 引擎到引擎注册表

    Args:
        register_func: register_engine 函数
        declared_entry_cls: DeclaredEntryConfig 类
    """
    from app.core.config import settings

    def _create_funasr_engine(config) -> "FunASREngine":
        return FunASREngine(
            realtime_model_path=config.models.get("realtime"),
            device=settings.DEVICE,
        )

    register_func("funasr", _create_funasr_engine)
