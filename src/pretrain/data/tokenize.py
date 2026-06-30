import json
import logging
import os
from collections import defaultdict

from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _formatting_prompts_func(examples, eos_token: str):
    return {"text": [example + eos_token for example in examples["text"]]}


def _tokenize_fn(examples, tokenizer: AutoTokenizer):
    texts = [text for text in examples["text"]]
    tokenized = tokenizer(texts)
    tokenized["num_tokens"] = [len(ids) for ids in tokenized["input_ids"]]
    return tokenized


def _save_token_distribution(dataset, output_path: str):
    """Aggregate tokens per source and write the distribution as JSON."""
    tokens_per_source: dict[str, int] = defaultdict(int)
    for batch in dataset.select_columns(["dataset", "num_tokens"]).iter(batch_size=10_000):
        for source, n_tokens in zip(batch["dataset"], batch["num_tokens"]):
            tokens_per_source[source] += n_tokens

    total_tokens = sum(tokens_per_source.values())
    distribution = {
        "total_tokens": total_tokens,
        "tokens_per_source": dict(tokens_per_source),
        "percentages": {
            source: round(n / total_tokens, 4)
            for source, n in tokens_per_source.items()
        },
    }

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "token_distribution.json"), "w") as f:
        json.dump(distribution, f, indent=2)

    logger.info("Token distribution (%d total tokens): %s", total_tokens, distribution["percentages"])


def tokenize_dataset(
    tokenizer_repo_id: str,
    data_path: str,
    test_size: float = 0.1,
    output_path: str = "tokenized_data/",
):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_repo_id)
    EOS_TOKEN = tokenizer.eos_token

    dataset = load_dataset("parquet", data_files=data_path)
    dataset = dataset.map(
        _formatting_prompts_func,
        fn_kwargs={"eos_token": EOS_TOKEN},
        batched=True,
        num_proc=8,
        desc="Appending EOS token",
    )

    tokenized_dataset = dataset.map(
        _tokenize_fn,
        fn_kwargs={"tokenizer": tokenizer},
        batched=True,
        num_proc=16,
        desc="Tokenizing dataset",
    )

    # Report tokens per source / percentage of total before splitting.
    _save_token_distribution(tokenized_dataset["train"], output_path)

    tokenized_dataset = tokenized_dataset["train"].train_test_split(
        test_size=test_size,
        shuffle=True,
        seed=0,
    )

    # Drop the helper column so it is not persisted in the tokenized dataset.
    tokenized_dataset = tokenized_dataset.remove_columns(["num_tokens"])

    tokenized_dataset.save_to_disk(
        output_path,
        num_proc=16,
    )
