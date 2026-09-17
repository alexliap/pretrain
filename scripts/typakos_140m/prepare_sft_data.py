"""Download and filter Hub instruction datasets into a token-bounded SFT pool.

Pulls a handful of multilingual instruction sources, keeps only samples that
form a well-formed conversation (an optional single leading system turn, then
alternating user/assistant turns, ending on assistant), and further keeps only
those whose combined token count is <= ``--max-tokens`` (the model's context
length). Turn count is not restricted: a 6-message conversation is kept
alongside a 2-message one as long as it fits the token budget. Nothing is
merged or packed here: each source is saved separately under ``--output-dir``
as a `messages` column (matching TRL's ``dataset_format: "conversational"``)
for a later merge step, and a report table shows how many rows survive each
filter.

    python prepare_sft_data.py
    python prepare_sft_data.py --sources aya_el aya_en --max-tokens 1024

Run from the repo root.
"""

import argparse
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from datasets import Dataset, Features, LargeList, Value

from pretrain.data.download import download_dataset
from pretrain.data.tokenize import load_tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw_sft")
OUTPUT_DIR = Path("data/sft_raw")
# typakos_sft_model/ carries the tokenizer that will actually run the SFT
# (typakos_model plus chat-template special tokens), so token counts here
# match what training will see.
TOKENIZER_PATH = "typakos_sft_model"
MAX_TOKENS = 2048

# `kind="messages"`: rows carry a `messages: [{role, content}]` column.
# `kind="aya"`: rows carry `inputs`/`targets`/`language` columns instead;
# `language` selects the row-level subset (aya_dataset has no per-language
# directory, unlike the other sources).
SOURCES: dict[str, dict] = {
    "dolci_el": {
        "repo_id": "openeurollm/Dolci-Instruct-SFT-translated",
        "allow_patterns": ["el/*.parquet"],
        "kind": "messages",
    },
    "eu_instruct_el": {
        "repo_id": "openeurollm/EU-Instruct-Synthetic",
        "allow_patterns": ["el/train.parquet"],
        "kind": "messages",
    },
    "aya_el": {
        "repo_id": "CohereLabs/aya_dataset",
        "allow_patterns": ["data/train-*.parquet"],
        "kind": "aya",
        "language": "Greek",
    },
    "aya_en": {
        "repo_id": "CohereLabs/aya_dataset",
        "allow_patterns": ["data/train-*.parquet"],
        "kind": "aya",
        "language": "English",
    },
    "smol_constraints": {
        "repo_id": "HuggingFaceTB/smoltalk",
        "allow_patterns": ["data/smol-constraints/train-*.parquet"],
        "kind": "messages",
    },
    "smol_magpie_ultra": {
        "repo_id": "HuggingFaceTB/smoltalk",
        "allow_patterns": ["data/smol-magpie-ultra/train-*.parquet"],
        "kind": "messages",
    },
    "smol_rewrite": {
        "repo_id": "HuggingFaceTB/smoltalk",
        "allow_patterns": ["data/smol-rewrite/train-*.parquet"],
        "kind": "messages",
    },
}


def _fetch_dir(spec: dict, raw_dir: Path) -> Path:
    """Local directory a (repo_id, allow_patterns) download lands in.

    Keyed by repo id *and* allow_patterns (not by source name): aya_el/aya_en
    share the exact same file and so share one directory, but the three
    smoltalk subsets share a repo_id with different allow_patterns and must
    NOT collide, since `read_table` globs every parquet file under this dir.
    """
    repo_part = spec["repo_id"].replace("/", "_")
    patterns_part = "_".join(
        p.removesuffix(".parquet").replace("/", "_").replace("*", "").rstrip("-_")
        for p in spec["allow_patterns"]
    )
    return raw_dir / f"{repo_part}__{patterns_part}"


def read_table(local_dir: Path) -> pa.Table:
    """Concatenate every parquet file already downloaded under `local_dir`."""
    files = sorted(local_dir.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files under {local_dir}")
    return pa.concat_tables([pq.read_table(f) for f in files])


def _clean_conversation(messages: list[dict]) -> list[dict] | None:
    """Return a validated `[{role, content}, ...]` list, or None if malformed.

    Valid shape: an optional single leading system turn, then one or more
    strictly alternating user/assistant turns, ending on assistant. Any
    deviation (non-alternating roles, empty content, a conversation that
    doesn't end on assistant, an extra/misplaced system turn) is dropped.
    Turn count itself is not restricted.
    """
    msgs = list(messages)
    cleaned = []

    if msgs and msgs[0]["role"] == "system":
        content = msgs[0]["content"]
        if not (content and content.strip()):
            return None
        cleaned.append({"role": "system", "content": content})
        msgs = msgs[1:]

    if not msgs:
        return None

    expected_role = "user"
    for msg in msgs:
        if msg["role"] != expected_role:
            return None
        content = msg["content"]
        if not (content and content.strip()):
            return None
        cleaned.append({"role": expected_role, "content": content})
        expected_role = "assistant" if expected_role == "user" else "user"

    if cleaned[-1]["role"] != "assistant":
        return None

    return cleaned


def normalize_messages(table: pa.Table) -> tuple[int, list[list[dict]]]:
    """Validated conversations from a `messages` column."""
    rows = []
    for messages in table.column("messages").to_pylist():
        cleaned = _clean_conversation(messages)
        if cleaned is not None:
            rows.append(cleaned)
    return table.num_rows, rows


def normalize_aya(table: pa.Table, language: str) -> tuple[int, list[list[dict]]]:
    """Validated (inherently 2-turn) conversations for one aya_dataset language."""
    filtered = table.filter(pc.equal(table.column("language"), language))
    rows = []
    for inputs, targets in zip(
        filtered.column("inputs").to_pylist(),
        filtered.column("targets").to_pylist(),
        strict=True,
    ):
        cleaned = _clean_conversation(
            [
                {"role": "user", "content": inputs},
                {"role": "assistant", "content": targets},
            ]
        )
        if cleaned is not None:
            rows.append(cleaned)
    return filtered.num_rows, rows


def filter_by_length(
    rows: list[list[dict]], tokenizer, max_tokens: int
) -> list[tuple[list[dict], int]]:
    """Keep conversations whose total token count is <= `max_tokens`."""
    kept = []
    for messages in rows:
        n_tokens = sum(
            len(tokenizer(msg["content"], add_special_tokens=False)["input_ids"])
            for msg in messages
        )
        if n_tokens <= max_tokens:
            kept.append((messages, n_tokens))
    return kept


# Large-multi-turn sources (e.g. smol_magpie_ultra) can push the `messages`
# column's total string data past the 2GiB limit of pyarrow's default int32
# offsets, raising "offset overflow while concatenating arrays" when `Dataset`
# fingerprints the table. `large_string`/`LargeList` use int64 offsets instead.
SAVE_FEATURES = Features(
    {
        "messages": LargeList(
            {"role": Value("large_string"), "content": Value("large_string")}
        ),
        "num_tokens": Value("int64"),
        "num_turns": Value("int64"),
        "source": Value("large_string"),
    }
)


def save_source(
    name: str, kept: list[tuple[list[dict], int]], output_dir: Path
) -> Path:
    dataset = Dataset.from_dict(
        {
            "messages": [r[0] for r in kept],
            "num_tokens": [r[1] for r in kept],
            # Turn count excluding a leading system message -- useful to slice
            # e.g. single-turn-only later without re-parsing `messages`.
            "num_turns": [
                len(r[0]) - (1 if r[0][0]["role"] == "system" else 0) for r in kept
            ],
            "source": [name] * len(kept),
        },
        features=SAVE_FEATURES,
    )
    out_dir = output_dir / name
    dataset.save_to_disk(str(out_dir))
    return out_dir


def _print_report(report: list[tuple[str, int, int, int]]) -> None:
    logger.info("")
    logger.info("%-20s %10s %14s %10s", "source", "raw", "well-formed", "final")
    for name, raw, well_formed, final in report:
        logger.info("%-20s %10d %14d %10d", name, raw, well_formed, final)
    logger.info(
        "TOTAL %d raw, %d well-formed, %d final",
        sum(r[1] for r in report),
        sum(r[2] for r in report),
        sum(r[3] for r in report),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download and filter SFT sources into a token-length-bounded pool."
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=sorted(SOURCES),
        default=None,
        help="Subset of sources to process (default: all).",
    )
    parser.add_argument("--tokenizer", default=TOKENIZER_PATH)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    selected = args.sources or sorted(SOURCES)
    tokenizer = load_tokenizer(args.tokenizer)

    tables_by_dir: dict[Path, pa.Table] = {}
    report = []
    for name in selected:
        spec = SOURCES[name]
        fetch_dir = _fetch_dir(spec, args.raw_dir)
        if fetch_dir not in tables_by_dir:
            download_dataset(
                spec["repo_id"], fetch_dir, allow_patterns=spec["allow_patterns"]
            )
            tables_by_dir[fetch_dir] = read_table(fetch_dir)
        table = tables_by_dir[fetch_dir]

        if spec["kind"] == "messages":
            raw_rows, rows = normalize_messages(table)
        else:
            raw_rows, rows = normalize_aya(table, spec["language"])

        well_formed = len(rows)
        kept = filter_by_length(rows, tokenizer, args.max_tokens)
        out_dir = save_source(name, kept, args.output_dir)
        logger.info("%s: %d rows -> %s", name, len(kept), out_dir)

        report.append((name, raw_rows, well_formed, len(kept)))

    _print_report(report)


if __name__ == "__main__":
    main()
