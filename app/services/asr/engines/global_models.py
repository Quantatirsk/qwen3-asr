# -*- coding: utf-8 -*-
"""
全局VAD/PUNC模型管理模块
提供线程安全的全局模型实例管理
"""

import torch
import logging
import threading
from funasr import AutoModel

from app.core.config import settings
from app.infrastructure import resolve_model_path


logger = logging.getLogger(__name__)


# 全局语音活动检测(VAD)模型缓存（避免重复加载）
_global_vad_model = None
_vad_model_lock = threading.Lock()
_vad_inference_lock = threading.Lock()  # 推理互斥锁，防止并发状态混乱

# 全局标点符号模型缓存（避免重复加载）
_global_punc_model = None
_punc_model_lock = threading.Lock()
_punc_inference_lock = threading.Lock()  # 推理互斥锁，防止并发状态混乱

# 全局实时标点符号模型缓存（避免重复加载）
_global_punc_realtime_model = None
_punc_realtime_model_lock = threading.Lock()
_punc_realtime_inference_lock = threading.Lock()  # 推理互斥锁，防止并发状态混乱

# 全局主ASR推理锁（临时串行化主模型推理，避免并发状态串扰）
_main_asr_inference_lock = threading.Lock()


def _resolve_device(device: str) -> str:
    """解析设备字符串，将 auto 转换为实际的设备"""
    from app.core.device import detect_device

    return detect_device(device)


def get_global_vad_model(device: str):
    """获取全局语音活动检测(VAD)模型实例（线程安全，双重检查锁定）"""
    global _global_vad_model

    if _global_vad_model is None:
        with _vad_model_lock:
            if _global_vad_model is None:
                try:
                    # 解析模型路径：优先使用本地缓存
                    resolved_vad_path = resolve_model_path(settings.VAD_MODEL)
                    logger.info(f"正在加载全局语音活动检测(VAD)模型: {resolved_vad_path}")

                    # 解析 auto 设备
                    resolved_device = _resolve_device(device)

                    _global_vad_model = AutoModel(
                        model=resolved_vad_path,
                        device=resolved_device,
                        speech_noise_thres=0.6,  # VAD 语音噪声阈值（FunASR默认0.6，设为0.7稍微严格一些，分段更碎）
                        **settings.FUNASR_AUTOMODEL_KWARGS,
                    )
                    logger.info("全局语音活动检测(VAD)模型加载成功 (speech_noise_thres=0.6)")
                except Exception as e:
                    logger.error(f"全局语音活动检测(VAD)模型加载失败: {str(e)}")
                    _global_vad_model = None
                    raise

    return _global_vad_model


def get_vad_inference_lock():
    """获取VAD模型推理锁（线程安全）"""
    return _vad_inference_lock


def clear_global_vad_model():
    """清理全局语音活动检测(VAD)模型缓存"""
    global _global_vad_model

    with _vad_model_lock:
        if _global_vad_model is not None:
            del _global_vad_model
            _global_vad_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局语音活动检测(VAD)模型缓存已清理")


def get_global_punc_model(device: str):
    """获取全局标点符号模型实例（离线版，线程安全，双重检查锁定）"""
    global _global_punc_model

    if _global_punc_model is None:
        with _punc_model_lock:
            if _global_punc_model is None:
                try:
                    # 解析模型路径：优先使用本地缓存
                    resolved_punc_path = resolve_model_path(settings.PUNC_MODEL)
                    logger.info(f"正在加载全局标点符号模型（离线）: {resolved_punc_path}")

                    # 解析 auto 设备
                    resolved_device = _resolve_device(device)

                    _global_punc_model = AutoModel(
                        model=resolved_punc_path,
                        device=resolved_device,
                        **settings.FUNASR_AUTOMODEL_KWARGS,
                    )
                    logger.info("全局标点符号模型（离线）加载成功")
                except Exception as e:
                    logger.error(f"全局标点符号模型（离线）加载失败: {str(e)}")
                    _global_punc_model = None
                    raise

    return _global_punc_model


def get_punc_inference_lock():
    """获取PUNC模型推理锁（线程安全）"""
    return _punc_inference_lock


def clear_global_punc_model():
    """清理全局标点符号模型缓存"""
    global _global_punc_model

    with _punc_model_lock:
        if _global_punc_model is not None:
            del _global_punc_model
            _global_punc_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局标点符号模型（离线）缓存已清理")


def get_global_punc_realtime_model(device: str):
    """获取全局实时标点符号模型实例（线程安全，双重检查锁定）"""
    global _global_punc_realtime_model

    if _global_punc_realtime_model is None:
        with _punc_realtime_model_lock:
            if _global_punc_realtime_model is None:
                try:
                    # 解析模型路径：优先使用本地缓存
                    resolved_punc_realtime_path = resolve_model_path(settings.PUNC_REALTIME_MODEL)
                    logger.info(f"正在加载全局标点符号模型（实时）: {resolved_punc_realtime_path}")

                    # 解析 auto 设备
                    resolved_device = _resolve_device(device)

                    _global_punc_realtime_model = AutoModel(
                        model=resolved_punc_realtime_path,
                        device=resolved_device,
                        **settings.FUNASR_AUTOMODEL_KWARGS,
                    )
                    logger.info("全局标点符号模型（实时）加载成功")
                except Exception as e:
                    logger.error(f"全局标点符号模型（实时）加载失败: {str(e)}")
                    _global_punc_realtime_model = None
                    raise

    return _global_punc_realtime_model


def get_punc_realtime_inference_lock():
    """获取实时PUNC模型推理锁（线程安全）"""
    return _punc_realtime_inference_lock


def get_main_asr_inference_lock():
    """获取主ASR推理锁（线程安全）"""
    return _main_asr_inference_lock


def clear_global_punc_realtime_model():
    """清理全局实时标点符号模型缓存"""
    global _global_punc_realtime_model

    with _punc_realtime_model_lock:
        if _global_punc_realtime_model is not None:
            del _global_punc_realtime_model
            _global_punc_realtime_model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("全局标点符号模型（实时）缓存已清理")
