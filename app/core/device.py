"""Select an explicit CPU or CUDA device without runtime fallback."""

import torch


def detect_device(configured: str = "cuda:0") -> str:
    if configured == "cpu":
        return "cpu"
    if configured != "cuda:0":
        raise ValueError(
            "DEVICE must be cpu or cuda:0; select the GPU with CUDA_VISIBLE_DEVICES"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("DEVICE=cuda:0 requires an available NVIDIA CUDA GPU")
    return "cuda:0"
