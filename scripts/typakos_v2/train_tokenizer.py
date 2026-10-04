import json
import logging
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from download_tok_data import SOURCES
from tokenizers import Tokenizer, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


BASE_TOKENIZER = "Qwen/Qwen3.5-9B"

N_LEARNED = 64 * 1024
N_BYTES = 256
BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN]
VOCAB_SIZE = N_LEARNED + N_BYTES + len(SPECIAL_TOKENS)

# Matches configs/train.yaml's data.max_seq_length.
MAX_SEQ_LENGTH = 4096
OUTPUT_DIR = "models/typakosv2_tokenizer"

BATCH_SIZE = 1000
CHUNK_ROWS = 50_000

RAW_DIR = Path("data/raw")


def build_tokenizer() -> Tokenizer:
    """Tokenization pipeline with an emptied vocabulary.

    Keeping the pipeline as data off the real artifact - rather than
    transcribing its regex here - means the pre-tokenizer cannot silently drift
    from the reference implementation.
    """
    base = AutoTokenizer.from_pretrained(BASE_TOKENIZER, token=os.environ["HF_TOKEN"])
    spec = json.loads(base.backend_tokenizer.to_str())

    spec["added_tokens"] = []
    spec["post_processor"] = None  # ours is installed after training
    spec["model"]["vocab"] = {}
    spec["model"]["merges"] = []

    pre = json.dumps(spec["pre_tokenizer"])
    logger.info("Borrowed pre_tokenizer from %s: %s", BASE_TOKENIZER, pre)
    assert "Regex" in pre, "Base pre_tokenizer lost its regex Split stage"

    return Tokenizer.from_str(json.dumps(spec))


def _read_text(path: Path, n_rows: int) -> list[str]:
    frame = pl.read_parquet(path, columns=["text"], n_rows=n_rows)
    return [text for text in frame["text"] if text]


def batch_iterator(texts: pl.DataFrame):
    # Slicing a DataFrame yields a DataFrame, not strings: hand the trainer
    # plain lists, one batch at a time, so the full corpus is never a list.
    for start in range(0, len(texts), BATCH_SIZE):
        yield texts["text"].slice(start, BATCH_SIZE).to_list()


def verify(tokenizer: PreTrainedTokenizerFast, vocab_size: int = VOCAB_SIZE) -> None:
    """Fail loudly on the properties this tokenizer is supposed to guarantee."""
    assert len(tokenizer) == vocab_size, (
        f"Expected vocab {vocab_size}, got {len(tokenizer)}"
    )

    missing = set(pre_tokenizers.ByteLevel.alphabet()) - set(tokenizer.get_vocab())
    assert not missing, f"{len(missing)} byte-alphabet tokens missing from vocab"

    bos_id, eos_id = tokenizer.bos_token_id, tokenizer.eos_token_id
    assert bos_id == 0 and eos_id == 1, f"Unexpected special ids: {bos_id}, {eos_id}"

    # Specials are added exactly once, at the edges.
    ids = tokenizer("δοκιμή").input_ids
    assert ids[0] == bos_id and ids[-1] == eos_id, "BOS/EOS not wrapped correctly"
    assert ids.count(bos_id) == 1 and ids.count(eos_id) == 1, "Specials duplicated"

    # Byte-level round-trip: Greek, English, mixed, emoji, raw control/high bytes.
    for probe in [
        "Καλημέρα κόσμε! Πώς είσαι σήμερα;",
        "The quick brown fox jumps over the lazy dog.",
        "Μαθηματικά: 2 + 2 = 4, ποσοστό 15%",
        "emoji 🇬🇷 🎉 and \x00\x01\xff raw bytes",
        "mixed Ελληνικά and English on one line",
    ]:
        decoded = tokenizer.decode(tokenizer(probe).input_ids, skip_special_tokens=True)
        assert decoded == probe, (
            f"Round-trip failed:\n  in : {probe!r}\n  out: {decoded!r}"
        )

    logger.info("All tokenizer assertions passed.")


def downloaded_files(source: str) -> list[Path]:
    """Parquet files already on disk for `source`.

    ``snapshot_download`` preserves the repo's directory layout inside the target
    directory, so the files sit one or more levels down - hence the recursive
    glob rather than a fixed depth.
    """
    return sorted((RAW_DIR / source).rglob("*.parquet"))


def report_fertility(tokenizer: PreTrainedTokenizerFast) -> dict[str, float]:
    logger.info("Fertility (higher bytes/token = more text per token):")
    rates: dict[str, float] = {}
    for source in SOURCES:
        files = downloaded_files(source)
        if not files:
            continue
        texts = _read_text(files[0], 2000)
        n_bytes = sum(len(t.encode()) for t in texts)
        n_chars = sum(len(t) for t in texts)
        n_words = sum(len(t.split()) for t in texts)
        n_tokens = sum(
            len(ids) for ids in tokenizer(texts, add_special_tokens=False).input_ids
        )
        rates[source] = n_bytes / n_tokens
        logger.info(
            "  %-16s %.2f bytes/token  %.2f chars/token %.2f token/words (%d docs)",
            source,
            n_bytes / n_tokens,
            n_chars / n_tokens,
            n_tokens / n_words,
            len(texts),
        )
    return rates


def sample_texts() -> list[str]:
    texts = None
    for name in SOURCES:
        files = downloaded_files(name)
        if name == "finemath":
            size = int(2.2 * 10**6)
        else:
            size = 10**7

        if texts is None:
            texts = pl.scan_parquet(files).select("text").slice(0, size).collect()
        else:
            texts = pl.concat(
                [texts, pl.scan_parquet(files).select("text").slice(0, size).collect()]
            )

    return texts


if __name__ == "__main__":
    logger.info("Loading training corpus ...")
    texts = sample_texts()

    tokenizer = build_tokenizer()
    logger.info(
        "Training byte-level BPE: vocab_size=%d (%d learned + %d bytes + %d special)",
        VOCAB_SIZE,
        VOCAB_SIZE - N_BYTES - len(SPECIAL_TOKENS),
        N_BYTES,
        len(SPECIAL_TOKENS),
    )
    tokenizer.train_from_iterator(
        batch_iterator(texts),
        trainer=trainers.BpeTrainer(
            vocab_size=VOCAB_SIZE,
            special_tokens=SPECIAL_TOKENS,
            # The guarantee that all 256 UTF-8 bytes stay representable.
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        ),
        length=len(texts),
    )
    logger.info("Trained vocab size: %d", tokenizer.get_vocab_size())

    tokenizer.post_processor = processors.Sequence(
        [
            processors.ByteLevel(trim_offsets=False),
            processors.TemplateProcessing(
                single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
                pair=f"{BOS_TOKEN} $A {EOS_TOKEN} {BOS_TOKEN} $B {EOS_TOKEN}",
                special_tokens=[(BOS_TOKEN, 0), (EOS_TOKEN, 1)],
            ),
        ]
    )

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token=BOS_TOKEN,
        eos_token=EOS_TOKEN,
        # No dedicated pad entry (it would cost a vocab slot); reusing eos keeps
        # task.py from stamping pad_token_id=None onto the model config.
        pad_token=EOS_TOKEN,
        model_max_length=MAX_SEQ_LENGTH,
    )

    verify(fast_tokenizer, VOCAB_SIZE)
    report_fertility(fast_tokenizer)

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    fast_tokenizer.save_pretrained(OUTPUT_DIR)
    logger.info("Saved tokenizer to %s", OUTPUT_DIR)
