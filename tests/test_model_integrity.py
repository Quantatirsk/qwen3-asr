import tempfile
import unittest
from pathlib import Path

from app.utils.model_loader import ModelIntegritySpec, _check_model_integrity_spec


class ModelIntegritySpecTest(unittest.TestCase):
    def test_accepts_single_safetensors_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "snapshots" / "main"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")

            result = _check_model_integrity_spec(_hf_qwen_spec(root))

        self.assertTrue(result["ok"])

    def test_accepts_sharded_safetensors_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "snapshots" / "main"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors.index.json").write_text("{}", encoding="utf-8")
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"weights")

            result = _check_model_integrity_spec(_hf_qwen_spec(root))

        self.assertTrue(result["ok"])

    def test_rejects_missing_safetensors_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "snapshots" / "main"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")

            result = _check_model_integrity_spec(_hf_qwen_spec(root))

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "required_files_missing")


def _hf_qwen_spec(root: Path) -> ModelIntegritySpec:
    return ModelIntegritySpec(
        description="Qwen test",
        path=root,
        required_patterns=("snapshots/*/config.json",),
        alternative_required_patterns=(
            ("snapshots/*/model.safetensors",),
            (
                "snapshots/*/model.safetensors.index.json",
                "snapshots/*/model-*.safetensors",
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
