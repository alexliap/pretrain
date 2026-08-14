"""Download the bilingual (English/Greek) pretraining corpus from the Hugging Face Hub.

Every source is declared once in ``SOURCES``. Shard file lists are enumerated
from the Hub at runtime rather than hardcoded, and shards are taken in sorted
order, so a given file count always resolves to the same set of files - runs are
reproducible and interrupted runs resume for free (``snapshot_download`` skips
files already present).

Four modes (run from the repo root):

    # small balanced sample - enough to train the tokenizer and measure
    # bytes-per-token per source (uses each source's probe_files)
    python scripts/typakos_140m/download_data.py --probe

    # every shard of every source, ~459 GB
    python scripts/typakos_140m/download_data.py --full

    # a sized subset, e.g. as written by measure_token_rates.py
    python scripts/typakos_140m/download_data.py --plan data/download_plan.json

All modes are incremental: shards already on disk are skipped, so a plan can be
topped up to --full later without re-fetching anything.

Files land under ``data/raw/<source>/``, which ``concat_data.py`` reads.
"""

import argparse
import json
import logging
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

RAW_DIR = Path("data/raw")

# Source manifest. `prefix` selects this source's shards out of the repo file
# listing; `lang` drives the 50/50 English/Greek token split; `probe_files` is
# how many shards the --probe stage pulls (None = all of them).
#
# Deliberately absent:
#   high-quality-gr-text/finepdfs_el  - OCR'd PDFs, noisy
SOURCES: dict[str, dict] = {
    "fineweb_edu": {
        "repo_id": "HuggingFaceFW/fineweb-edu",
        "prefix": "sample/10BT/",
        "lang": "en",
        "probe_files": None,
    },
    "fineweb_hq_el": {
        "repo_id": "alexliap/high-quality-gr-text",
        "prefix": "fineweb_hq_el/",
        "lang": "el",
        "probe_files": None,
    },
    "finewiki_el": {
        "repo_id": "alexliap/high-quality-gr-text",
        "prefix": "finewiki_el/",
        "lang": "el",
        "probe_files": None,
    },
    "wikipedia_el": {
        "repo_id": "alexliap/high-quality-gr-text",
        "prefix": "wikipedia_el/",
        "lang": "el",
        "probe_files": None,
    },
    "synth_faq": {
        "repo_id": "alexliap/greek-synth-v1",
        "prefix": "faq/",
        "lang": "el",
        "probe_files": None,
    },
    "synth_math": {
        "repo_id": "alexliap/greek-synth-v1",
        "prefix": "math/",
        "lang": "el",
        "probe_files": None,
    },
    "synth_table": {
        "repo_id": "alexliap/greek-synth-v1",
        "prefix": "table/",
        "lang": "el",
        "probe_files": None,
    },
    "synth_tutorial": {
        "repo_id": "alexliap/greek-synth-v1",
        "prefix": "tutorial/",
        "lang": "el",
        "probe_files": None,
    },
}

SYNTH_SOURCES = [name for name in SOURCES if name.startswith("synth_")]


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
    return sorted(
        f for f in files if f.startswith(spec["prefix"]) and f.endswith(".parquet")
    )


def download_source(source: str, n_files: int | None) -> Path:
    """Download the first `n_files` shards of `source` (None = all)."""
    shards = list_shards(source)
    if not shards:
        raise ValueError(f"No parquet shards found for source '{source}'.")

    selected = shards if n_files is None else shards[:n_files]
    if n_files is not None and len(selected) < n_files:
        logger.warning(
            "Source '%s' only has %d shards, %d requested.",
            source,
            len(shards),
            n_files,
        )

    logger.info(
        "%s: %d/%d shards from %s",
        source,
        len(selected),
        len(shards),
        SOURCES[source]["repo_id"],
    )
    # allow_patterns takes the exact paths, so the selection stays deterministic
    # instead of relying on a glob to happen to match the same set.
    return download_dataset(
        repo_id=SOURCES[source]["repo_id"],
        local_dir=RAW_DIR / source,
        allow_patterns=selected,
    )


def downloaded_files(source: str) -> list[Path]:
    """Parquet files already on disk for `source`.

    ``snapshot_download`` preserves the repo's directory layout inside the target
    directory, so the files sit one or more levels down - hence the recursive
    glob rather than a fixed depth.
    """
    return sorted((RAW_DIR / source).rglob("*.parquet"))


def _report(sources: list[str]) -> None:
    total_bytes = 0
    logger.info("%-16s %7s  %10s", "source", "files", "size")
    for source in sources:
        files = downloaded_files(source)
        size = sum(f.stat().st_size for f in files)
        total_bytes += size
        logger.info("%-16s %7d  %9.2f GB", source, len(files), size / 1e9)
    logger.info("%-16s %7s  %9.2f GB", "TOTAL", "", total_bytes / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the bilingual pretraining corpus from the Hub."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--probe",
        action="store_true",
        help="Download the small balanced sample used to train the tokenizer "
        "and measure bytes-per-token (uses each source's probe_files).",
    )
    group.add_argument(
        "--full",
        action="store_true",
        help="Download every shard of every source.",
    )
    group.add_argument(
        "--plan",
        type=str,
        help="Path to a download plan JSON ({source: n_files}) as written by "
        "measure_token_rates.py.",
    )
    group.add_argument(
        "--source",
        choices=sorted(SOURCES),
        help="Download a single source (combine with --n-files).",
    )
    parser.add_argument(
        "--n-files",
        type=int,
        default=None,
        help="Number of shards for --source (default: all).",
    )
    args = parser.parse_args()

    if args.probe:
        plan = {name: spec["probe_files"] for name, spec in SOURCES.items()}
    elif args.full:
        plan = dict.fromkeys(SOURCES)  # None per source == every shard
    elif args.plan:
        with open(args.plan) as f:
            plan = json.load(f)
        unknown = set(plan) - set(SOURCES)
        if unknown:
            raise ValueError(f"Unknown sources in plan: {sorted(unknown)}")
    else:
        plan = {args.source: args.n_files}

    for source, n_files in plan.items():
        download_source(source, n_files)

    _report(list(plan))


if __name__ == "__main__":
    main()
