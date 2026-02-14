import logging

from datasets import load_dataset
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def formatting_prompts_func(examples):
    return {"text": [example + EOS_TOKEN for example in examples["text"]]}


def tokenize_fn(examples):
    texts = [text for text in examples["text"]]
    return tokenizer(texts)


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("tokenizer/")
    EOS_TOKEN = tokenizer.eos_token

    dataset = load_dataset("parquet", data_files="tinystories.parquet")
    dataset = dataset.map(
        formatting_prompts_func, batched=True, num_proc=8, desc="Appending EOS token"
    )

    tokenized_dataset = dataset.map(
        tokenize_fn, batched=True, num_proc=16, desc="Tokenizing dataset"
    )

    tokenized_dataset = tokenized_dataset["train"].train_test_split(
        test_size=0.1,
        shuffle=True,
        seed=0,
    )

    tokenized_dataset.save_to_disk(
        "tokenized_data/train_data/",
        num_proc=16,
    )
