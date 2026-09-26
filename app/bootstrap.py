"""Prepare all required model caches before spawning inference processes."""

import logging

logger = logging.getLogger(__name__)


def ensure_models_downloaded() -> bool:
    from app.infrastructure import is_huggingface_offline
    from app.utils.download_models import check_all_models, download_models

    try:
        missing = check_all_models()
        if not missing:
            return True
        if is_huggingface_offline():
            logger.error(
                "Required models are missing in offline mode: %s. "
                "Run scripts/prepare-models.sh with network access first.",
                missing,
            )
            return False
        return download_models(auto_mode=True)
    except Exception:
        logger.exception("Model preparation failed")
        return False
