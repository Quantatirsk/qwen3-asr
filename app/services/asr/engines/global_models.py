"""Process-wide FSMN VAD model used by offline segmentation."""

import logging
import threading

from funasr import AutoModel

from app.core.config import settings
from app.infrastructure import resolve_model_path

logger = logging.getLogger(__name__)
_global_vad_model = None
_vad_model_lock = threading.Lock()
_vad_inference_lock = threading.Lock()


def get_global_vad_model(device: str):
    global _global_vad_model
    if _global_vad_model is None:
        with _vad_model_lock:
            if _global_vad_model is None:
                from app.core.device import detect_device

                resolved_path = resolve_model_path(settings.VAD_MODEL)
                _global_vad_model = AutoModel(
                    model=resolved_path,
                    device=detect_device(device),
                    speech_noise_thres=0.6,
                    **settings.FUNASR_AUTOMODEL_KWARGS,
                )
                logger.info("FSMN VAD model loaded from %s", resolved_path)
    return _global_vad_model


def get_vad_inference_lock():
    return _vad_inference_lock
