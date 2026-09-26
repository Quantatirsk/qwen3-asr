"""Retired Rust runtime must remain unavailable."""

import importlib.util
import unittest


class RetiredRustRuntimeTest(unittest.TestCase):
    def test_rust_backend_is_not_importable(self) -> None:
        self.assertIsNone(importlib.util.find_spec("app.services.asr.qwenasr_rust"))


if __name__ == "__main__":
    unittest.main()
