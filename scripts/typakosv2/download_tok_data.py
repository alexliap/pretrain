import logging
from fnmatch import fnmatch
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi

from pretrain.data import download_dataset

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
# huggingface_hub logs every HTTP request through httpx at INFO
logging.getLogger("httpx").setLevel(logging.ERROR)

RAW_DIR = Path("data/raw")

# `n_files`: the first N sorted shards, just enough to cover the docs
# train_tokenizer.py reads (10M en, 10M el, 2.2M math). The full subsets
# (~157 GB) don't fit on disk.
SOURCES: dict[str, dict] = {
    "fineweb": {
        "repo_id": "HuggingFaceFW/fineweb",
        "prefix": "sample/10BT/",
        "n_files": 10,  # 10.44M docs, 21.5 GB
    },
    "greek-cc": {
        "repo_id": "alexliap/greek-cc",
        "prefix": "CC-MAIN-*/",
        "n_files": 54,  # 10.14M docs, 17.8 GB
    },
    "finemath": {
        "repo_id": "HuggingFaceTB/finemath",
        "prefix": "finemath-3plus/",
        "n_files": 14,  # 2.34M docs, 7.1 GB
    },
}


def list_shards(source: str) -> list[str]:
    """Repo-relative parquet paths belonging to `source`, sorted.

    Sorting is what makes "the first N shards" a stable, reproducible selection.
    """
    spec = SOURCES[source]
    if not spec["repo_id"]:
        raise ValueError(
            f"Source '{source}' has no repo id configured. Set CC_GREEK_REPO in "
            ".env to download it (already-downloaded shards under data/raw/ are "
            "usable without it)."
        )
    files = HfApi().list_repo_files(spec["repo_id"], repo_type="dataset")
    # fnmatch rather than startswith: prefixes may be globs (e.g. "CC-MAIN-*/")
    return sorted(f for f in files if fnmatch(f, f"{spec['prefix']}*.parquet"))


if __name__ == "__main__":
    for name, source in SOURCES.items():
        download_dataset(
            source["repo_id"],
            local_dir=RAW_DIR / name,
            # exact paths, so the selection is the same on every run
            allow_patterns=list_shards(name)[: source["n_files"]],
        )
