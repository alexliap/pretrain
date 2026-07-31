"""Train a new tokenizer on the prepared tinystories-gr corpus.

Starts from the Llama 3.2 tokenizer's byte-level BPE pipeline (regex
pre-tokenizer, ByteLevel encoding/decoding -- this is exactly what gives the
256-entry byte alphabet) but trains a fresh vocab/merges on our Greek corpus,
with only the two special tokens this pretraining setup actually uses (bos,
eos) -- not Llama's own ~254 unused reserved/fine-tuning placeholder tokens,
which would otherwise be dragged along unused and untrue for this tokenizer.
"""

import json
import logging

import polars as pl
from dotenv import load_dotenv
from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from transformers import AutoTokenizer, PreTrainedTokenizerFast

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BASE_TOKENIZER = "meta-llama/Llama-3.2-1B"
VOCAB_SIZE = 16640  # 16384 + 256-entry byte-level alphabet
MAX_SEQ_LENGTH = 256  # matches configs/train.yaml's data.max_seq_length
DATA_PATH = "data/mixed_dataset/data.parquet"
OUTPUT_DIR = "models/tinystories_gr_tokenizer"
BATCH_SIZE = 1000
BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"


def text_iterator(texts: list[str]):
    for start in range(0, len(texts), BATCH_SIZE):
        yield texts[start : start + BATCH_SIZE]


if __name__ == "__main__":
    logger.info("Loading text column from %s ...", DATA_PATH)
    texts = pl.read_parquet(DATA_PATH, columns=["text"])["text"].to_list()
    logger.info("Loaded %d rows.", len(texts))

    logger.info("Loading base tokenizer %s ...", BASE_TOKENIZER)
    base_tokenizer = AutoTokenizer.from_pretrained(BASE_TOKENIZER)

    # Reuse Llama's tokenization pipeline (pre_tokenizer/decoder/BPE model
    # settings -- the byte-level machinery that gives the 256-entry byte
    # alphabet) but strip its vocab/merges/added_tokens/post_processor so
    # training starts from a clean slate instead of dragging along ~254 unused
    # reserved/fine-tuning special tokens that don't apply to this tokenizer.
    base_json = json.loads(base_tokenizer._tokenizer.to_str())
    base_json["added_tokens"] = []
    base_json["post_processor"] = None
    base_json["model"]["vocab"] = {}
    base_json["model"]["merges"] = []
    raw_tokenizer = Tokenizer.from_str(json.dumps(base_json))

    logger.info("Training new tokenizer (vocab_size=%d) ...", VOCAB_SIZE)
    trainer = BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=[BOS_TOKEN, EOS_TOKEN])
    raw_tokenizer.train_from_iterator(text_iterator(texts), trainer=trainer)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        model_max_length=MAX_SEQ_LENGTH,
    )

    logger.info("Final vocab size: %d", len(tokenizer))
    tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Saved tokenizer to %s", OUTPUT_DIR)
