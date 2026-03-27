# -*- coding: utf-8 -*-
"""
Centralized device detection utility.

All device-related decisions across the project should go through
this module to avoid duplicated / inconsistent detection logic.
"""

import torch


def detect_device(configured: str = "auto") -> str:
    """Resolve a device configuration string to a concrete PyTorch device.

    Priority for ``"auto"``: CUDA > MPS > CPU.

    Args:
        configured: Value from ``settings.DEVICE`` or caller override.
            Accepted: ``"auto"``, ``"cpu"``, ``"cuda:0"``, ``"mps"``, ``"npu:0"``, etc.

    Returns:
        A device string ready for ``torch.device()`` / FunASR / ModelScope.
    """
    device = configured.strip().lower()

    if device == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # Normalize bare "cuda" to "cuda:0"
    if device == "cuda":
        return "cuda:0"

    return device


def is_cuda() -> bool:
    """True when CUDA is available."""
    return torch.cuda.is_available()


def is_mps() -> bool:
    """True when Apple MPS is available."""
    return torch.backends.mps.is_available()


def has_gpu() -> bool:
    """True when any GPU accelerator (CUDA or MPS) is available."""
    return is_cuda() or is_mps()


def get_vram_gb() -> float:
    """Return usable GPU memory in GB.

    CUDA: smallest GPU's total memory (multi-GPU takes the min).
    MPS:  total system RAM (Apple Silicon unified memory).
    CPU:  0.0
    """
    try:
        if is_cuda():
            return min(
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(torch.cuda.device_count())
            )
        if is_mps():
            import os
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / (1024**3)
    except Exception:
        pass
    return 0.0
