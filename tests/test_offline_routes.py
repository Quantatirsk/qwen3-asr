import unittest

from app.main import create_app


class OfflineRouteSurfaceTest(unittest.TestCase):
    def test_application_exposes_only_offline_routes(self) -> None:
        app = create_app()
        paths = [route.path for route in app.routes]

        self.assertFalse(any(path.startswith("/ws") for path in paths))
        self.assertIn("/stream/v1/asr", paths)
        self.assertIn("/v1/audio/transcriptions", paths)


if __name__ == "__main__":
    unittest.main()
