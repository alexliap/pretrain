"""Mix the filtered sources into proportionally-mixed, shuffled shards.

Reads ``data/mix_plan.json`` (written by ``measure_token_rates.py``) and emits
``data/mixed_dataset/shard_NN.parquet``, each carrying the same source
proportions as the corpus as a whole and internally shuffled.

Two properties matter downstream:

* **Every shard is a miniature of the whole mix.** The training dataloader reads
  with ``shuffle=False`` on purpose -- resume correctness depends on a stable
  order -- so the on-disk order *is* the training order. A shard that was all
  Greek followed by a shard that was all English would train very differently
  from a 50/50 corpus.

* **Sharding bounds the working set.** Tokenising and packing 30B tokens in one
  piece would need far more scratch space than the disk has; one shard at a time
  keeps intermediates to roughly a tenth of that.

    python concat_data.py --n-shards 10
"""

import argparse
import json
import logging
from pathlib import Path

import polars as pl

from download_data import downloaded_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SEED = 0
N_SHARDS = 20
PLAN_PATH = Path("data/mix_plan.json")
OUT_DIR = Path("data/mixed_dataset")


def _chunks(
    source: str, target_rows: int, n_shards: int
) -> list[tuple[list[Path], int, int]]:
    """Per-output-shard ``(files, row_offset, row_limit)`` for `source`.

    When a source has at least one input file per output shard, the files are
    dealt out to shards with a stride and each group is read from its start: one
    pass over the data, and no offset to scan past. Sources with fewer files than
    shards must serve several shards from the same file, so their filtered stream
    is sliced by row instead -- those sources are small (under 4 GB), so the
    repeated scan costs little.
    """
    files = downloaded_files(source)
    per_shard = target_rows // n_shards

    if len(files) >= n_shards:
        return [(files[i::n_shards], 0, per_shard) for i in range(n_shards)]

    return [(files, i * per_shard, per_shard) for i in range(n_shards)]


def _read_chunk(
    source: str, files: list[Path], offset: int, limit: int
) -> pl.DataFrame:
    """Filtered ``(text, dataset)`` rows for one source within one shard."""
    frame = pl.scan_parquet(files).select(["text"])

    # Empty and null documents would survive tokenisation as a bare bos/eos pair
    # and dilute the mix with sequences carrying no text.
    frame = frame.filter(
        pl.col("text").is_not_null() & (pl.col("text").str.len_bytes() > 0)
    )

    return (
        frame.slice(offset, limit)
        .select(pl.col("text"), pl.lit(source).alias("dataset"))
        .collect()
    )


def build_shards(plan: dict[str, dict], n_shards: int, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks = {
        source: _chunks(source, entry["target_rows"], n_shards)
        for source, entry in plan.items()
    }

    # Sources with fewer files than shards cannot be split by file, and slicing
    # their filtered stream once per shard would rescan them n_shards times. They
    # are small in total (a few tens of GB), so read each once up front and split
    # in memory instead; the large sources stay streamed per shard.
    cached: dict[str, list[pl.DataFrame]] = {}
    for source, entry in plan.items():
        if len(downloaded_files(source)) >= n_shards:
            continue
        full = _read_chunk(source, downloaded_files(source), 0, entry["target_rows"])
        per_shard = -(-full.height // n_shards)  # ceil, so no rows are dropped
        cached[source] = [full.slice(i * per_shard, per_shard) for i in range(n_shards)]
        logger.info(
            "cached %-16s %d rows in memory (%d files < %d shards)",
            source,
            full.height,
            len(downloaded_files(source)),
            n_shards,
        )

    realised: dict[str, dict[str, int]] = {
        source: {"rows": 0, "bytes": 0} for source in plan
    }

    for i in range(n_shards):
        parts = []
        for source in plan:
            if source in cached:
                part = cached[source][i]
            else:
                files, offset, limit = chunks[source][i]
                part = _read_chunk(source, files, offset, limit)
            realised[source]["rows"] += part.height
            realised[source]["bytes"] += int(
                part.select(pl.col("text").str.len_bytes().cast(pl.Int64).sum()).item()
                or 0
            )
            parts.append(part)

        # Shuffle across sources so a shard interleaves them rather than
        # concatenating them in blocks.
        shard = pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=SEED + i)
        path = out_dir / f"shard_{i:02d}.parquet"
        shard.write_parquet(path, compression="zstd")

        logger.info(
            "shard %02d: %d rows, %.2f GB text -> %s (%.2f GB on disk)",
            i,
            shard.height,
            shard.select(pl.col("text").str.len_bytes().cast(pl.Int64).sum()).item()
            / 1e9,
            path.name,
            path.stat().st_size / 1e9,
        )
        del shard, parts

    return realised


def _report(plan: dict, realised: dict, n_shards: int, out_dir: Path) -> dict:
    total_rows = sum(v["rows"] for v in realised.values())
    total_bytes = sum(v["bytes"] for v in realised.values())

    logger.info("")
    logger.info(
        "%-16s %5s %12s %12s %9s %8s",
        "source",
        "lang",
        "rows",
        "planned",
        "text_GB",
        "share",
    )
    by_lang = {"en": 0.0, "el": 0.0}
    for source, entry in plan.items():
        got = realised[source]
        logger.info(
            "%-16s %5s %12d %12d %8.2fG %7.1f%%",
            source,
            entry["lang"],
            got["rows"],
            entry["target_rows"],
            got["bytes"] / 1e9,
            100 * got["rows"] / total_rows if total_rows else 0,
        )
        short = 1 - got["rows"] / entry["target_rows"] if entry["target_rows"] else 0
        if short > 0.02:
            logger.warning(
                "'%s' delivered %d of %d planned rows (%.0f%% short) -- its pool is "
                "smaller than the plan assumed.",
                source,
                got["rows"],
                entry["target_rows"],
                100 * short,
            )

    # Token shares use the plan's measured tokens/row, since the true count is
    # only known after tokenisation (tokenized_data/token_distribution.json).
    est_tokens = {
        s: realised[s]["rows"]
        * plan[s]["target_tokens"]
        / max(1, plan[s]["target_rows"])
        for s in plan
    }
    for source, entry in plan.items():
        by_lang[entry["lang"]] += est_tokens[source]
    est_total = sum(est_tokens.values())

    logger.info(
        "TOTAL %d rows, %.1f GB text, ~%.2fB tokens -- %.1f%% EN / %.1f%% EL",
        total_rows,
        total_bytes / 1e9,
        est_total / 1e9,
        100 * by_lang["en"] / est_total if est_total else 0,
        100 * by_lang["el"] / est_total if est_total else 0,
    )

    distribution = {
        "seed": SEED,
        "n_shards": n_shards,
        "total_rows": total_rows,
        "total_text_bytes": total_bytes,
        "estimated_tokens": int(est_total),
        "per_source": {
            source: {
                "lang": plan[source]["lang"],
                "rows": realised[source]["rows"],
                "planned_rows": plan[source]["target_rows"],
                "text_bytes": realised[source]["bytes"],
                "estimated_tokens": int(est_tokens[source]),
            }
            for source in plan
        },
    }
    (out_dir / "data_distribution.json").write_text(json.dumps(distribution, indent=2))
    logger.info("Wrote %s", out_dir / "data_distribution.json")
    return distribution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mix the sources into proportionally-mixed, shuffled shards."
    )
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--n-shards", type=int, default=N_SHARDS)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    spec = json.loads(args.plan.read_text())
    plan = spec["sources"]
    logger.info(
        "Mixing %.1fB tokens across %d shards (plan: %s)",
        spec["config"]["budget"] / 1e9,
        args.n_shards,
        args.plan,
    )

    realised = build_shards(plan, args.n_shards, args.output_dir)
    _report(plan, realised, args.n_shards, args.output_dir)


if __name__ == "__main__":
    main()
