"""Tokenize mixed text shards into int32 ``input_ids``.

Three things this deliberately does *not* do, each of which used to cost a full
copy of the corpus on disk or an extra pass over it:

* **No EOS-append pass.** The tokenizer's ``TemplateProcessing`` post-processor
  wraps every document in BOS/EOS at encode time (see ``train_tokenizer.py``),
  so there is no need to rewrite the text column just to concatenate a string.
* **No ``attention_mask``.** It is written by the old code but never read:
  ``dataloader.py`` drops it on both paths and rebuilds the validation mask from
  real sequence lengths inside ``collate_fn``. Dropping it halves the store.
* **No cast pass.** ``features=`` hands Arrow the int32 type up front, so the
  ids are written narrow rather than written as int64 and cast afterwards. Vocab
  50,258 needs 17 bits; int32 leaves plenty of room.
"""

import json
import logging
import os
from collections import defaultdict

import datasets
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Documents longer than the model's context are split across several packed
# sequences rather than truncated, so `text` is encoded whole. The tokenizer
# carries model_max_length=2048 and would otherwise log a "sequence length is
# longer than the specified maximum" warning per worker per long document.
_NO_TRUNCATION = int(1e12)

# Arrow schema of a tokenized shard. `dataset` and `num_tokens` exist only to
# compute the token distribution; they are dropped before anything is saved.
TOKENIZED_FEATURES = datasets.Features(
    {
        "dataset": datasets.Value("string"),
        "input_ids": datasets.List(datasets.Value("int32")),
        "num_tokens": datasets.Value("int32"),
    }
)


def _tokenize_fn(examples, tokenizer: AutoTokenizer):
    tokenized = tokenizer(examples["text"], return_attention_mask=False)
    tokenized["num_tokens"] = [len(ids) for ids in tokenized["input_ids"]]
    return tokenized


def load_tokenizer(tokenizer_repo_id: str) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo_id)
    tokenizer.model_max_length = _NO_TRUNCATION
    return tokenizer


def tokenize_shard(
    dataset: Dataset,
    tokenizer: AutoTokenizer,
    num_proc: int = 48,
    batch_size: int = 1000,
    desc: str = "Tokenizing",
) -> Dataset:
    """Encode `dataset`'s ``text`` column, returning ``TOKENIZED_FEATURES``."""
    return dataset.map(
        _tokenize_fn,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["text"],
        features=TOKENIZED_FEATURES,
        desc=desc,
    )


def count_tokens_per_source(dataset: Dataset) -> dict[str, int]:
    """Tokens contributed by each source in `dataset`.

    Read through the two small columns only; touching ``input_ids`` here would
    pull the whole shard through Python for a number Arrow already knows.
    """
    tokens_per_source: dict[str, int] = defaultdict(int)
    columns = dataset.select_columns(["dataset", "num_tokens"])
    for batch in columns.iter(batch_size=100_000):
        for source, n_tokens in zip(batch["dataset"], batch["num_tokens"]):
            tokens_per_source[source] += n_tokens
    return dict(tokens_per_source)


def save_token_distribution(
    tokens_per_source: dict[str, int], output_path: str
) -> dict:
    """Write ``token_distribution.json`` as ``task.py`` expects to read it."""
    total_tokens = sum(tokens_per_source.values())
    distribution = {
        "total_tokens": total_tokens,
        "tokens_per_source": dict(sorted(tokens_per_source.items())),
        "percentages": {
            source: round(n / total_tokens, 4)
            for source, n in sorted(tokens_per_source.items())
        },
    }

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "token_distribution.json"), "w") as f:
        json.dump(distribution, f, indent=2)

    logger.info(
        "Token distribution (%d total tokens): %s",
        total_tokens,
        distribution["percentages"],
    )
    return distribution


def tokenize_dataset(
    tokenizer_repo_id: str,
    data_path: str,
    test_size: float = 0.005,
    output_path: str = "tokenized_data/",
    num_proc: int = 48,
):
    """Tokenize parquet text into ``output_path`` as train/test splits.

    Single-shot path, kept for the ``pretrain-data tokenize`` CLI and small
    datasets. The 30B corpus goes through ``prepare_shards.py`` instead, which
    packs each shard as it is tokenized rather than persisting the unpacked
    intermediate.

    ``test_size=0`` skips the split and writes a single ``train`` split.
    """
    tokenizer = load_tokenizer(tokenizer_repo_id)

    dataset = load_dataset("parquet", data_files=data_path)["train"]
    tokenized = tokenize_shard(dataset, tokenizer, num_proc=num_proc)

    save_token_distribution(count_tokens_per_source(tokenized), output_path)

    tokenized = tokenized.remove_columns(["dataset", "num_tokens"])
    if test_size:
        splits = tokenized.train_test_split(test_size=test_size, shuffle=True, seed=0)
    else:
        splits = datasets.DatasetDict({"train": tokenized})

    splits.save_to_disk(output_path, num_proc=16)
    dataset.cleanup_cache_files()
    tokenized.cleanup_cache_files()
