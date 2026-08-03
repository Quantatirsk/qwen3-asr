"""Device policy for CPU-side offline support models."""


def detect_device(configured: str = "auto") -> str:
    """Resolve the API-side device; NPU inference lives in the vLLM container."""
    device = configured.strip().lower()
    if device in {"", "auto", "cpu"}:
        return "cpu"
    raise ValueError(
        f"unsupported API-side device '{configured}'; use cpu with remote Ascend vLLM"
    )
