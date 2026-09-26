import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.utils.download_models import check_model_exists


class PinnedCheckpointTest(unittest.TestCase):
    def test_single_file_and_complete_shards_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = root / "snapshots" / "revision"
            snapshot.mkdir(parents=True)
            for name in (
                "config.json",
                "preprocessor_config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ):
                (snapshot / name).write_text("{}")
            with patch("app.utils.download_models._get_cache_path", return_value=root):

                def exists() -> bool:
                    return check_model_exists("org/model", "huggingface", "revision")[0]

                self.assertFalse(exists())
                weight = snapshot / "model.safetensors"
                weight.write_bytes(b"weights")
                self.assertTrue(exists())
                weight.write_bytes(b"")
                self.assertFalse(exists())
                weight.unlink()
                (snapshot / "model.safetensors.index.json").write_text(
                    json.dumps(
                        {
                            "weight_map": {
                                "first": "part-1.safetensors",
                                "second": "part-2.safetensors",
                            }
                        }
                    )
                )
                (snapshot / "part-1.safetensors").write_bytes(b"weights")
                self.assertFalse(exists())
                (snapshot / "part-2.safetensors").write_bytes(b"weights")
                self.assertTrue(exists())


if __name__ == "__main__":
    unittest.main()
