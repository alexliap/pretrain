"""Per-source quality filters, shared by the measuring and mixing stages.

Kept in one place so that the token pools reported by ``measure_token_rates.py``
are computed with exactly the predicates ``concat_data.py`` will apply -- if the
two drifted, the mix plan would be sized against a pool that does not exist.
"""

import polars as pl

# cc-greek-ds ships a `minhash_cluster_size` column: the size of the
# near-duplicate cluster each document belongs to (median 2, max 515,250). There
# is no cluster id, so one-representative-per-cluster is not expressible -- any
# threshold above 1 keeps whole clusters and therefore admits duplicates.
# Keeping only size-1 clusters means every retained document is unique by
# construction, at the cost of 66% of the source.
MINHASH_MAX = 1

# greek-synth-v1 text carries generation artifacts: characters from other
# scripts spliced into Greek words (e.g. "Τοτε主义", or "υπολείπται" with a
# Cyrillic п). Mean incidence is 0.09-0.26% of characters, so a 1% per-document
# ceiling removes the visibly broken tail (0.8-6.5% of documents by config)
# without touching the bulk.
FOREIGN_CHAR_MAX = 0.01

# Anything outside Greek, Latin, numbers, punctuation, symbols and whitespace.
# Note \p{Greek}/\p{Latin} are script classes, so this keeps accented Greek and
# Latin alike while catching CJK, Cyrillic, Arabic and friends.
_FOREIGN_CHAR = r"[^\p{Greek}\p{Latin}\p{N}\p{P}\p{S}\s]"


def source_filter(source: str) -> pl.Expr | None:
    """Row predicate for `source`, or None when the source is taken as-is."""
    if source == "cc_greek":
        return pl.col("minhash_cluster_size") <= MINHASH_MAX

    if source.startswith("synth_"):
        n_chars = pl.col("text").str.len_chars()
        foreign = pl.col("text").str.count_matches(_FOREIGN_CHAR)
        # Guard the divide: a zero-length document would otherwise yield null and
        # be dropped by the filter for the wrong reason.
        return (foreign / pl.when(n_chars > 0).then(n_chars).otherwise(1)) <= (
            FOREIGN_CHAR_MAX
        )

    return None


def filter_columns(source: str) -> list[str]:
    """Columns a source's filter needs, beyond `text`."""
    return ["minhash_cluster_size"] if source == "cc_greek" else []
