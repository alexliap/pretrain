"""Measure per-source token rates and derive the corpus mix plan.

Everything downstream is sized in *tokens*, but parquet can only be sliced by
*rows*, and the two are related by a ratio that varies almost 2x across these
sources (Greek costs ~7.2 bytes/token, English ~4.5, and mean document length
ranges from 2.4 KB to 15 KB). This script measures that ratio with the real
tokenizer and writes:

    data/token_rates.json  -- bytes/token and tokens/row per source, plus the
                              token pool each source can supply
    data/mix_plan.json     -- rows to take from each source to hit the target
                              budget at the target English/Greek split

``concat_data.py`` consumes the mix plan. Row counts come from parquet footers
(exact, cheap) and tokens/row from a sample spread across shards, so the pool
estimate does not depend on reading 433 GB of text.

    python measure_token_rates.py --budget 50e9 --en-share 0.5
"""

import argparse
import json
import logging
from pathlib import Path

import polars as pl
from transformers import AutoTokenizer

from data_filters import filter_columns, source_filter
from download_data import SOURCES, downloaded_files

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TOKENIZER_PATH = "models/bilingual_el_en_50k"
OUT_DIR = Path("data")

# Shards sampled per source, and documents per sampled shard. Spread across the
# shard list rather than taken from the front: these sources are ordered (by
# CommonCrawl dump for fineweb-edu/cc-greek), so the head is not representative.
SAMPLE_SHARDS = 10
SAMPLE_DOCS = 10_000

# Greek synthetic data carries generation artifacts (foreign-script characters
# spliced into Greek words), so its share of the Greek half is capped.
SYNTH_CAP = 0.15


def _sample_docs(source: str) -> list[str]:
    """Documents that survive the source's filter, spread across its shards."""
    files = downloaded_files(source)
    if not files:
        raise FileNotFoundError(f"No parquet files for '{source}'.")

    predicate = source_filter(source)
    columns = ["text"] + filter_columns(source)

    n_picks = min(SAMPLE_SHARDS, len(files))
    picks = {int(i * len(files) / n_picks) for i in range(n_picks)}
    docs: list[str] = []
    for i in sorted(picks):
        frame = pl.scan_parquet(files[i]).select(columns).head(SAMPLE_DOCS * 4)
        if predicate is not None:
            frame = frame.filter(predicate)
        docs.extend(t for t in frame.head(SAMPLE_DOCS).collect()["text"] if t)
    return docs


def _row_count(source: str) -> tuple[int, int]:
    """(rows, rows_after_filter) across a source's shards.

    Unfiltered counts come from parquet footers; filtered counts need the filter
    columns read, which for cc_greek is one small integer column but for the
    synth sources means scanning their text.
    """
    files = downloaded_files(source)
    rows = pl.scan_parquet(files).select(pl.len()).collect().item()

    predicate = source_filter(source)
    if predicate is None:
        return rows, rows

    kept = (
        pl.scan_parquet(files)
        .select(filter_columns(source) or ["text"])
        .filter(predicate)
        .select(pl.len())
        .collect()
        .item()
    )
    return rows, kept


def measure(tokenizer) -> dict[str, dict]:
    rates: dict[str, dict] = {}
    logger.info(
        "%-16s %5s %10s %7s %9s %8s %10s",
        "source",
        "lang",
        "kept_rows",
        "kept",
        "B/token",
        "tok/row",
        "pool_tokens",
    )
    for source in SOURCES:
        docs = _sample_docs(source)
        n_bytes = sum(len(d.encode()) for d in docs)
        n_tokens = sum(
            len(ids) for ids in tokenizer(docs, add_special_tokens=False).input_ids
        )
        rows, kept_rows = _row_count(source)

        bytes_per_token = n_bytes / n_tokens
        # +2 for the bos/eos the post-processor adds to every document, which are
        # real tokens in the packed stream and must be budgeted for.
        tokens_per_row = n_tokens / len(docs) + 2
        # Pool is sized on rows that survive the filter, since those are the only
        # ones the mixing stage can draw on.
        pool = int(tokens_per_row * kept_rows)

        rates[source] = {
            "lang": SOURCES[source]["lang"],
            "rows": rows,
            "kept_rows": kept_rows,
            "kept_fraction": round(kept_rows / rows, 4) if rows else 0,
            "bytes_per_token": round(bytes_per_token, 3),
            "tokens_per_row": round(tokens_per_row, 1),
            "pool_tokens": pool,
            "sampled_docs": len(docs),
        }
        logger.info(
            "%-16s %5s %10d %6.0f%% %9.2f %8.0f %10s",
            source,
            SOURCES[source]["lang"],
            kept_rows,
            100 * kept_rows / rows if rows else 0,
            bytes_per_token,
            tokens_per_row,
            f"{pool / 1e9:.2f}B",
        )
    return rates


def plan_mix(
    rates: dict[str, dict], budget: float, en_share: float, synth_cap: float
) -> dict[str, dict]:
    """Allocate the token budget across sources, then convert to row counts.

    Greek is allocated by priority rather than proportionally: the curated
    sources and the capped synthetic set are taken in full first, and cc_greek --
    the only source deep enough to absorb whatever is left -- fills the
    remainder. That keeps the highest-quality Greek text at 100% inclusion
    instead of subsampling it to hit a ratio.
    """
    en_target = int(budget * en_share)
    el_target = int(budget - en_target)

    en_sources = [s for s in rates if rates[s]["lang"] == "en"]
    synth = [s for s in rates if s.startswith("synth_")]
    curated_el = [
        s
        for s in rates
        if rates[s]["lang"] == "el" and s not in synth and s != "cc_greek"
    ]

    alloc: dict[str, int] = {}

    # English: fineweb_edu is the only source and is far deeper than any budget.
    for source in en_sources:
        alloc[source] = min(en_target, rates[source]["pool_tokens"])

    # Greek. Curated sources go in whole, synth is reserved at its cap, and
    # cc_greek absorbs the rest.
    curated_total = 0
    for source in curated_el:
        take = min(rates[source]["pool_tokens"], el_target - curated_total)
        alloc[source] = take
        curated_total += take

    synth_pool = sum(rates[s]["pool_tokens"] for s in synth)
    synth_budget = min(int(el_target * synth_cap), synth_pool)

    cc_pool = rates["cc_greek"]["pool_tokens"] if "cc_greek" in rates else 0
    cc_take = max(0, min(cc_pool, el_target - curated_total - synth_budget))
    non_synth = curated_total + cc_take

    # If the non-synthetic sources cannot reach el_target, let synth grow to take
    # up the slack -- but the cap must bind against the total we actually reach,
    # not the one we asked for. Solving EL = non_synth + cap*EL for EL gives
    # EL = non_synth / (1 - cap); holding synth at cap*el_target instead would
    # push its realised share above the cap.
    if non_synth + synth_budget < el_target:
        reachable_el = min(non_synth / (1 - synth_cap), non_synth + synth_pool)
        synth_budget = min(int(reachable_el - non_synth), synth_pool)

    if "cc_greek" in rates:
        alloc["cc_greek"] = cc_take

    for source in synth:
        # Split the synth budget across its four configs in proportion to what
        # each can supply, so no single config dominates.
        share = rates[source]["pool_tokens"] / synth_pool if synth_pool else 0
        alloc[source] = min(rates[source]["pool_tokens"], int(synth_budget * share))

    remaining = el_target - sum(
        tokens for s, tokens in alloc.items() if rates[s]["lang"] == "el"
    )

    plan = {}
    for source, tokens in alloc.items():
        rate = rates[source]
        plan[source] = {
            "lang": rate["lang"],
            "target_tokens": tokens,
            "target_rows": min(rate["rows"], int(tokens / rate["tokens_per_row"])),
            "pool_tokens": rate["pool_tokens"],
            "fraction_of_pool": round(tokens / rate["pool_tokens"], 4)
            if rate["pool_tokens"]
            else 0,
        }

    _report_plan(plan, budget, en_target, el_target, remaining, synth_cap)
    return plan


def _report_plan(plan, budget, en_target, el_target, shortfall, synth_cap) -> None:
    logger.info("")
    logger.info(
        "MIX PLAN  budget %.1fB tokens (%.0f%% EN / %.0f%% EL)",
        budget / 1e9,
        100 * en_target / budget,
        100 * el_target / budget,
    )
    logger.info(
        "%-16s %5s %12s %12s %9s", "source", "lang", "tokens", "rows", "of pool"
    )
    for source, entry in plan.items():
        logger.info(
            "%-16s %5s %11.2fB %12d %8.0f%%",
            source,
            entry["lang"],
            entry["target_tokens"] / 1e9,
            entry["target_rows"],
            100 * entry["fraction_of_pool"],
        )

    got_en = sum(e["target_tokens"] for e in plan.values() if e["lang"] == "en")
    got_el = sum(e["target_tokens"] for e in plan.values() if e["lang"] == "el")
    total = got_en + got_el
    got_synth = sum(
        e["target_tokens"] for s, e in plan.items() if s.startswith("synth_")
    )
    logger.info(
        "TOTAL %.2fB tokens -- %.1f%% EN / %.1f%% EL, synth %.1f%% of Greek (cap %.0f%%)",
        total / 1e9,
        100 * got_en / total,
        100 * got_el / total,
        100 * got_synth / got_el if got_el else 0,
        synth_cap * 100,
    )

    # Tolerate the integer-rounding residue left by the per-source allocations;
    # only a real shortfall is worth a warning.
    if shortfall > 0.005 * el_target:
        logger.warning(
            "Greek is %.2fB tokens SHORT of its %.2fB target -- the Greek pool is "
            "exhausted. Lower --budget, lower --en-share, or raise --synth-cap.",
            shortfall / 1e9,
            el_target / 1e9,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure token rates and derive the corpus mix plan."
    )
    parser.add_argument("--tokenizer-path", default=TOKENIZER_PATH)
    parser.add_argument(
        "--budget",
        type=float,
        default=30e9,
        help="Target corpus size in tokens. 30B keeps cc_greek at ~80%% of its "
        "post-dedup pool, so measurement drift cannot cause a shortfall.",
    )
    parser.add_argument(
        "--en-share", type=float, default=0.5, help="English fraction of the budget."
    )
    parser.add_argument(
        "--synth-cap",
        type=float,
        default=SYNTH_CAP,
        help="Maximum synthetic-Greek share of the Greek half.",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    logger.info(
        "Measuring token rates with %s (vocab %d) ...",
        args.tokenizer_path,
        len(tokenizer),
    )
    rates = measure(tokenizer)

    en_pool = sum(r["pool_tokens"] for r in rates.values() if r["lang"] == "en")
    el_pool = sum(r["pool_tokens"] for r in rates.values() if r["lang"] == "el")
    logger.info("")
    logger.info(
        "POOL (post-filter): EN %.1fB tokens | EL %.1fB tokens | TOTAL %.1fB",
        en_pool / 1e9,
        el_pool / 1e9,
        (en_pool + el_pool) / 1e9,
    )

    # The synth cap binds against the Greek total, so the achievable Greek total
    # is not simply the sum of the pools: solving EL = other + min(synth, cap*EL)
    # gives EL = other/(1-cap) while the cap is the binding term.
    synth_pool = sum(v["pool_tokens"] for s, v in rates.items() if s.startswith("synth_"))
    other_el = el_pool - synth_pool
    el_max = min(other_el / (1 - args.synth_cap), other_el + synth_pool)
    budget_max = min(el_max / (1 - args.en_share), en_pool / args.en_share)
    logger.info(
        "Largest feasible corpus at %.0f%% EN and a %.0f%% synth cap: %.1fB tokens "
        "(Greek tops out at %.1fB)",
        args.en_share * 100,
        args.synth_cap * 100,
        budget_max / 1e9,
        el_max / 1e9,
    )
    if args.budget > budget_max:
        logger.warning(
            "Requested budget %.1fB exceeds the feasible %.1fB -- the plan below "
            "will come up short.",
            args.budget / 1e9,
            budget_max / 1e9,
        )

    plan = plan_mix(rates, args.budget, args.en_share, args.synth_cap)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "token_rates.json").write_text(json.dumps(rates, indent=2))
    (OUT_DIR / "mix_plan.json").write_text(
        json.dumps(
            {
                "config": {
                    "budget": args.budget,
                    "en_share": args.en_share,
                    "synth_cap": args.synth_cap,
                    "tokenizer": args.tokenizer_path,
                },
                "sources": plan,
            },
            indent=2,
        )
    )
    logger.info(
        "Wrote %s and %s", OUT_DIR / "token_rates.json", OUT_DIR / "mix_plan.json"
    )


if __name__ == "__main__":
    main()
