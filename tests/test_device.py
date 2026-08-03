from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.core.device import detect_device, get_vram_gb, has_gpu


class _FakeNPU:
    def is_available(self) -> bool:
        return True

    def device_count(self) -> int:
        return 2

    def get_device_properties(self, index: int) -> SimpleNamespace:
        memory_gb = 64 if index == 0 else 32
        return SimpleNamespace(total_memory=memory_gb * 1024**3)


class DeviceTest(unittest.TestCase):
    def test_auto_detects_npu_when_cuda_is_unavailable(self) -> None:
        with (
            patch("app.core.device.torch.cuda.is_available", return_value=False),
            patch("app.core.device._get_torch_npu", return_value=_FakeNPU()),
        ):
            self.assertEqual(detect_device("auto"), "npu:0")

    def test_normalizes_bare_npu_device(self) -> None:
        self.assertEqual(detect_device("npu"), "npu:0")

    def test_reports_smallest_npu_memory_for_multi_device_hosts(self) -> None:
        with (
            patch("app.core.device.torch.cuda.is_available", return_value=False),
            patch("app.core.device._get_torch_npu", return_value=_FakeNPU()),
        ):
            self.assertTrue(has_gpu())
            self.assertEqual(get_vram_gb(), 32.0)


if __name__ == "__main__":
    unittest.main()
