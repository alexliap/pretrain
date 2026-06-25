"""Download datasets from the Hugging Face Hub."""

import logging
from pathlib import Path

from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def download_dataset(
    repo_id: str,
    local_dir: str | Path,
    repo_type: str = "dataset",
    revision: str | None = None,
    allow_patterns: list[str] | None = None,
) -> Path:
    """Download a dataset from the Hugging Face Hub into a local directory.

    Args:
        repo_id: The Hub repository id, e.g. ``alexliap/high-quality-gr-text``.
        local_dir: Directory the repository files are downloaded into.
        repo_type: Type of the Hub repository (``dataset``, ``model`` or ``space``).
        revision: Optional git revision (branch, tag or commit sha) to download.
        allow_patterns: Optional glob patterns; only matching files are downloaded.

    Returns:
        The path to the directory containing the downloaded files.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading %s '%s' into %s", repo_type, repo_id, local_dir)
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=str(local_dir),
        allow_patterns=allow_patterns,
    )
    logger.info("Finished downloading '%s' into %s", repo_id, path)

    return Path(path)
