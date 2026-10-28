from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset


def read_parquet(paths: Path | list[Path], columns: list[str]) -> Dataset:
    """Load a mixed shard into memory as a ``(text, dataset)`` Dataset.

    Read through pyarrow rather than ``load_dataset`` so the shard never lands in
    the HF datasets cache: that cache would be a second 61 GB copy of the text,
    and it outlives ``cleanup_cache_files()``.
    """
    table = pq.read_table(paths, columns=columns)
    # `string` has 32-bit offsets (2 GB per array); a whole crawl overflows them
    schema = pa.schema(
        [
            f.with_type(pa.large_string()) if f.type == pa.string() else f
            for f in table.schema
        ]
    )
    return Dataset(table.cast(schema))


def parquet_paths(dir: Path) -> list[Path]:
    paths = sorted(dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files under {dir}.")
    return paths
