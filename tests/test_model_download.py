import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.services.asr.model_capabilities import ModelAsset
from app.utils.download_models import download_models


class ModelDownloadTest(unittest.TestCase):
    def test_nemotron_download_pins_revision_and_exports_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            metadata = source / ".cache/huggingface/download/config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text("pinned-revision\netag\n123\n")
            (source / "config.json").write_text("{}")
            asset = ModelAsset(
                "huggingface",
                "nvidia/Nemotron-3-Diarization",
                "Nemotron",
                revision="pinned-revision",
                local_dir=str(source),
            )
            with (
                patch(
                    "app.utils.download_models.check_all_models",
                    side_effect=[
                        [
                            (
                                asset.model_id,
                                asset.description,
                                asset.source,
                                asset.revision,
                            )
                        ],
                        [],
                    ],
                ),
                patch(
                    "app.utils.download_models.is_huggingface_offline",
                    return_value=False,
                ),
                patch(
                    "app.utils.download_models.get_download_modelscope_assets",
                    return_value=[],
                ),
                patch(
                    "app.utils.download_models.get_huggingface_assets",
                    return_value=[asset],
                ),
                patch("app.utils.download_models.hf_snapshot_download") as download,
            ):
                self.assertTrue(download_models(export_dir=str(root / "export")))
            download.assert_called_once_with(
                asset.model_id, revision="pinned-revision", local_dir=str(source)
            )
            exported = root / "export/nemotron-3-diarization"
            self.assertEqual(
                (exported / metadata.relative_to(source)).read_text(),
                metadata.read_text(),
            )

    def test_offline_missing_assets_fail_without_network(self) -> None:
        with (
            patch(
                "app.utils.download_models.check_all_models",
                return_value=[
                    (
                        "nvidia/Nemotron-3-Diarization",
                        "Nemotron",
                        "huggingface",
                        "revision",
                    )
                ],
            ),
            patch(
                "app.utils.download_models.is_huggingface_offline", return_value=True
            ),
            patch("app.utils.download_models.hf_snapshot_download") as download,
        ):
            self.assertFalse(download_models())
        download.assert_not_called()
