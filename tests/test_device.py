import unittest

from app.core.device import detect_device


class DeviceTest(unittest.TestCase):
    def test_api_support_models_are_cpu_only(self) -> None:
        self.assertEqual(detect_device("auto"), "cpu")
        self.assertEqual(detect_device("cpu"), "cpu")
        with self.assertRaisesRegex(ValueError, "remote Ascend vLLM"):
            detect_device("npu:0")


if __name__ == "__main__":
    unittest.main()
