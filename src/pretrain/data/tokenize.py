import logging

from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _formatting_prompts_func(examples, eos_token: str):
    return {"text": [example + eos_token for example in examples["text"]]}


def _tokenize_fn(examples, tokenizer: AutoTokenizer):
    texts = [text for text in examples["text"]]
    return tokenizer(texts)


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

    tokenized_dataset = tokenized_dataset["train"].train_test_split(
        test_size=test_size,
        shuffle=True,
        seed=0,
    )

    tokenized_dataset.save_to_disk(
        output_path,
        num_proc=16,
    )
