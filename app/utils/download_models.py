"""Download, verify and export the single supported model set."""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download as hf_snapshot_download

from app.core.config import settings
from app.infrastructure import (
    get_huggingface_cache_root,
    get_huggingface_model_cache_dir,
    is_huggingface_offline,
)
from app.services.asr.model_capabilities import (
    get_download_modelscope_assets,
    get_huggingface_assets,
)


def _get_cache_path(model_id: str, source: str = "modelscope") -> Path:
    if source == "huggingface":
        return get_huggingface_model_cache_dir(model_id)
    return Path(settings.MODELSCOPE_PATH) / model_id


def check_all_models() -> list[tuple[str, str, str, str | None]]:
    from app.utils.model_loader import (
        _build_required_model_integrity_specs,
        _check_model_integrity_spec,
    )

    assets = [*get_download_modelscope_assets(), *get_huggingface_assets()]
    specs = _build_required_model_integrity_specs()
    return [
        (asset.model_id, asset.description, asset.source, asset.revision)
        for asset, spec in zip(assets, specs, strict=True)
        if not _check_model_integrity_spec(spec)["ok"]
    ]


def download_models(auto_mode: bool = False, export_dir: str | None = None) -> bool:
    missing = check_all_models()
    if missing and is_huggingface_offline():
        print(
            "Offline mode: required model files are missing:",
            [item[0] for item in missing],
        )
        return False
    assets = [*get_download_modelscope_assets(), *get_huggingface_assets()]
    missing_ids = {item[0] for item in missing}
    for asset in assets:
        if asset.model_id not in missing_ids:
            continue
        try:
            if asset.source == "huggingface":
                hf_snapshot_download(
                    asset.model_id, revision=asset.revision, local_dir=asset.local_dir
                )
            else:
                from modelscope.hub.snapshot_download import snapshot_download

                snapshot_download(asset.model_id, revision=asset.revision)
            print("Downloaded", asset.model_id)
        except Exception as error:
            print("Model download failed:", asset.model_id, str(error))
            return False
    if check_all_models():
        print("Model integrity check failed after download")
        return False
    if export_dir:
        destination = Path(export_dir)
        for asset in assets:
            source = (
                Path(asset.local_dir)
                if asset.local_dir
                else _get_cache_path(asset.model_id, asset.source)
            )
            if asset.local_dir:
                target = destination / "nemotron-3-diarization"
            elif asset.source == "modelscope":
                target = destination / "modelscope/hub/models" / asset.model_id
            else:
                target = (
                    destination
                    / "huggingface/hub"
                    / source.relative_to(get_huggingface_cache_root())
                )
            if source.resolve() != target.resolve():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, dirs_exist_ok=True)
        print("Exported model assets to", destination)
    if not auto_mode:
        print("All required models are ready")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir")
    parser.add_argument("--auto-mode", action="store_true")
    args = parser.parse_args()
    return 0 if download_models(args.auto_mode, args.export_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
