# -*- coding: utf-8 -*-
"""Centralized accelerator detection utilities."""

import importlib
from typing import Protocol, cast

import torch


class _DeviceProperties(Protocol):
    total_memory: int


class _NPUBackend(Protocol):
    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def get_device_properties(self, index: int) -> _DeviceProperties: ...


def _get_torch_npu() -> _NPUBackend | None:
    backend = getattr(torch, "npu", None)
    if backend is None:
        try:
            importlib.import_module("torch_npu")
        except ImportError:
            return None
        backend = getattr(torch, "npu", None)
    return cast(_NPUBackend | None, backend)


def detect_device(configured: str = "auto") -> str:
    """Resolve a device configuration string to a concrete PyTorch device.

    Priority for ``"auto"``: CUDA > NPU > CPU.

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
        if is_npu():
            return "npu:0"
        return "cpu"

    if device == "cuda":
        return "cuda:0"
    if device == "npu":
        return "npu:0"

    if device == "mps":
        return "cpu"

    return device


def is_cuda() -> bool:
    """True when CUDA is available."""
    return torch.cuda.is_available()


def is_npu() -> bool:
    """Return whether an Ascend NPU is available through torch_npu."""
    backend = _get_torch_npu()
    return backend is not None and backend.is_available()


def has_gpu() -> bool:
    """Return whether a CUDA GPU or Ascend NPU is available."""
    return is_cuda() or is_npu()


def get_vram_gb() -> float:
    """Return usable GPU memory in GB.

    CUDA/NPU: smallest visible device's total memory.
    CPU: 0.0.
    """
    try:
        if is_cuda():
            return min(
                torch.cuda.get_device_properties(i).total_memory / (1024**3)
                for i in range(torch.cuda.device_count())
            )
        npu = _get_torch_npu()
        if npu is not None and npu.is_available():
            return min(
                npu.get_device_properties(i).total_memory / (1024**3)
                for i in range(npu.device_count())
            )
    except Exception:
        pass
    return 0.0
