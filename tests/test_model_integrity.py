import json
import tempfile
import unittest
from pathlib import Path

from app.utils.model_loader import ModelIntegritySpec, _check_model_integrity_spec


class ModelIntegritySpecTest(unittest.TestCase):
    def test_local_revision_requires_matching_metadata_for_every_required_file(
        self,
    ) -> None:
        revision = "f667ed73aee57d40cc39428eb768b4fd87a0a29e"
        names = ("config.json", "preprocessor_config.json", "model.safetensors")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = ModelIntegritySpec(
                description="Pinned local Nemotron model",
                path=root,
                required_patterns=names,
                expected_revision=revision,
            )
            for name in names:
                (root / name).write_bytes(b"model data")
            metadata = root / ".cache/huggingface/download"
            metadata.mkdir(parents=True)
            for state in ("missing", "wrong", "partial", "correct"):
                with self.subTest(state=state):
                    if state != "missing":
                        for index, name in enumerate(names):
                            commit = (
                                revision
                                if state == "correct"
                                or (state == "partial" and index == 0)
                                else "0" * 40
                            )
                            (metadata / f"{name}.metadata").write_text(
                                f"{commit}\netag\n1234567890.0\n"
                            )
                    result = _check_model_integrity_spec(spec)
                    self.assertEqual(result["ok"], state == "correct")
                    if state != "correct":
                        self.assertEqual(result["reason"], "required_files_missing")
                        self.assertTrue(
                            any(
                                "expected revision" in item
                                for item in result["missing_patterns"]
                            )
                        )
            (root / "config.json").unlink()
            result = _check_model_integrity_spec(spec)
            self.assertFalse(result["ok"])
            self.assertIn("config.json", result["missing_patterns"])

    def test_accepts_single_safetensors_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors").write_bytes(b"weights")

            result = _check_model_integrity_spec(_hf_spec(snapshot))

        self.assertTrue(result["ok"])

    def test_accepts_sharded_safetensors_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")
            (snapshot / "model.safetensors.index.json").write_text(
                json.dumps(
                    {"weight_map": {"weight": "model-00001-of-00002.safetensors"}}
                ),
                encoding="utf-8",
            )
            (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"weights")

            result = _check_model_integrity_spec(_hf_spec(snapshot))

        self.assertTrue(result["ok"])

    def test_rejects_missing_indexed_shard(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {
                            "first": "model-1.safetensors",
                            "second": "model-2.safetensors",
                        }
                    }
                )
            )
            (root / "model-1.safetensors").write_bytes(b"weights")
            result = _check_model_integrity_spec(_hf_spec(root))
        self.assertFalse(result["ok"])
        self.assertIn("model-2.safetensors", result["missing_patterns"])

    def test_rejects_missing_safetensors_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            snapshot = root / "snapshot"
            snapshot.mkdir(parents=True)
            (snapshot / "config.json").write_text("{}", encoding="utf-8")

            result = _check_model_integrity_spec(_hf_spec(snapshot))

        self.assertFalse(result["ok"])
        self.assertEqual(result["reason"], "required_files_missing")


def _hf_spec(root: Path) -> ModelIntegritySpec:
    return ModelIntegritySpec(
        description="Checkpoint test",
        path=root,
        required_patterns=("config.json",),
        alternative_required_patterns=(
            ("model.safetensors",),
            (
                "model.safetensors.index.json",
                "model-*.safetensors",
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
