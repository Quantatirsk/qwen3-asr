# -*- coding: utf-8 -*-
"""Centralized device detection utility."""

import torch


def detect_device(configured: str = "auto") -> str:
    """Resolve a device configuration string to a concrete PyTorch device.

    Priority for ``"auto"``: CUDA > CPU.

    Args:
        configured: Value from ``settings.DEVICE`` or caller override.
            Accepted: ``"auto"``, ``"cpu"``, ``"cuda:0"``, ``"npu:0"``, etc.

    Returns:
        A device string ready for ``torch.device()`` / FunASR / ModelScope.
    """
    device = configured.strip().lower()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    # Normalize bare "cuda" to "cuda:0"
    if device == "cuda":
        return "cuda:0"

    if device == "mps":
        return "cpu"

    return device


def is_cuda() -> bool:
    """True when CUDA is available."""
    return torch.cuda.is_available()


def has_gpu() -> bool:
    """True when CUDA is available."""
    return is_cuda()


def get_vram_gb() -> float:
    """Return usable GPU memory in GB.

    CUDA: smallest GPU's total memory (multi-GPU takes the min).
    CPU:  0.0
    """
    try:
        if is_cuda():
            return min(
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(torch.cuda.device_count())
            )
    except Exception:
        pass
    return 0.0
