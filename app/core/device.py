"""CUDA is the only supported inference device."""

import torch


def detect_device(configured: str = "cuda:0") -> str:
    device = torch.device(configured)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("R2T2 requires an NVIDIA CUDA GPU")
    index = device.index if device.index is not None else 0
    if index != 0:
        raise ValueError(
            "Use cuda:0 and select the physical GPU with CUDA_VISIBLE_DEVICES"
        )
    return f"cuda:{index}"
