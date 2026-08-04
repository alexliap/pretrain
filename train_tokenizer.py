import argparse
import json
import logging
import os
from pathlib import Path

import polars as pl
from dotenv import load_dotenv
from tokenizers import Tokenizer, pre_tokenizers, processors, trainers
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from download_data import SOURCES, downloaded_files

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Any Llama 3.x model carries the same 128,256-entry tokenizer and the same
# pre-tokenization regex, so which size we borrow the pipeline from is
# immaterial -- only its structure is used, never its vocabulary.
BASE_TOKENIZER = "meta-llama/Llama-3.2-1B"

N_LEARNED = 50_000
N_BYTES = 256
BOS_TOKEN = "<|begin_of_text|>"
EOS_TOKEN = "<|end_of_text|>"
SPECIAL_TOKENS = [BOS_TOKEN, EOS_TOKEN]
VOCAB_SIZE = N_LEARNED + N_BYTES + len(SPECIAL_TOKENS)  # 50,258

# Matches configs/train.yaml's data.max_seq_length.
MAX_SEQ_LENGTH = 2048
OUTPUT_DIR = "models/bilingual_el_en_50k"

# BPE quality saturates well before the full corpus; what matters is that the
# sample is balanced across languages and sources, not that it is large.
SAMPLE_GB = 20.0
BATCH_SIZE = 1000
# Rows pulled per read while filling a source's byte budget.
CHUNK_ROWS = 50_000

# How the sample budget splits across sources. English takes half; the Greek half
# is weighted towards the cleanest Greek text available.
CORPUS_WEIGHTS = {
    "fineweb_edu": 0.50,
    "fineweb_hq_el": 0.25,
    "finewiki_el": 0.05,
    "wikipedia_el": 0.05,
    "synth_math": 0.05,
    "synth_faq": 0.05,
    "synth_table": 0.03,
    "synth_turotial": 0.02,
}


def build_tokenizer() -> Tokenizer:
    """Llama 3.2's tokenization pipeline with an emptied vocabulary.

    Keeping the pipeline as data off the real artifact -- rather than
    transcribing its regex here -- means the pre-tokenizer cannot silently drift
    from the reference implementation.
    """
    base = AutoTokenizer.from_pretrained(
        BASE_TOKENIZER, token=os.environ["GATED_TOKEN"]
    )
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


def _take_bytes(path: Path, budget: int, out: list[str]) -> int:
    """Append documents from `path` to `out` until `budget` bytes are taken.

    Reads in row chunks and stops as soon as the budget is met, so only the row
    groups actually needed are touched -- a 2 GB shard never lands in memory
    whole. Counting real bytes as we go, rather than extrapolating from the first
    N rows, matters because document length is heavily skewed in these sources
    (the head of a shard is not representative of its mean).
    """
    taken = 0
    offset = 0
    while taken < budget:
        frame = pl.scan_parquet(path).select("text").slice(offset, CHUNK_ROWS).collect()
        if frame.height == 0:
            break
        for text in frame["text"]:
            if not text:
                continue
            out.append(text)
            taken += len(text.encode())
            if taken >= budget:
                break
        offset += CHUNK_ROWS
    return taken


def sample_texts(sample_gb: float) -> list[str]:
    """Read a balanced English/Greek text sample from data/raw/.

    A source that cannot fill its share of the budget is reported rather than
    quietly padded around: an under-filled source shifts the realised
    English/Greek ratio, which is exactly what the vocabulary split depends on.
    """
    budget = int(sample_gb * 1e9)
    texts: list[str] = []
    delivered: dict[str, int] = {}

    for source, weight in CORPUS_WEIGHTS.items():
        files = downloaded_files(source)
        if not files:
            raise FileNotFoundError(
                f"No parquet files for '{source}' under data/raw/{source}. "
                "Run `python download_data.py --probe` first."
            )

        requested = int(budget * weight)
        per_file = max(1, requested // len(files))
        got = sum(_take_bytes(path, per_file, texts) for path in files)
        delivered[source] = got

        short = 1 - got / requested if requested else 0
        logger.info(
            "  %-16s %6.2f / %6.2f GB requested from %d shard(s)%s",
            source,
            got / 1e9,
            requested / 1e9,
            len(files),
            f"  SHORT by {short:.0%}" if short > 0.02 else "",
        )
        if short > 0.02:
            logger.warning(
                "'%s' ran out of downloaded text (%.2f of %.2f GB). Download more "
                "shards to honour its %.0f%% weight.",
                source,
                got / 1e9,
                requested / 1e9,
                weight * 100,
            )

    total = sum(delivered.values())
    by_lang = {"en": 0, "el": 0}
    for source, got in delivered.items():
        by_lang[SOURCES[source]["lang"]] += got
    realised_en = by_lang["en"] / total
    target_en = sum(w for s, w in CORPUS_WEIGHTS.items() if SOURCES[s]["lang"] == "en")

    logger.info(
        "Sample: %.2f / %.2f GB of text in %d documents",
        total / 1e9,
        sample_gb,
        len(texts),
    )
    logger.info(
        "Realised mix: %.0f%% EN / %.0f%% EL (target %.0f%% / %.0f%%)",
        realised_en * 100,
        (1 - realised_en) * 100,
        target_en * 100,
        (1 - target_en) * 100,
    )
    if abs(realised_en - target_en) > 0.05:
        logger.warning(
            "Realised English share is %.0f%% against a %.0f%% target -- the "
            "vocabulary will be skewed towards the over-represented language.",
            realised_en * 100,
            target_en * 100,
        )
    return texts


def batch_iterator(texts: list[str]):
    for start in range(0, len(texts), BATCH_SIZE):
        yield texts[start : start + BATCH_SIZE]


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


def report_fertility(tokenizer: PreTrainedTokenizerFast) -> dict[str, float]:
    """Log bytes-per-token per source -- the number that sizes the corpus.

    Also compares against the Llama 3.2 base on Greek: Greek characters cost two
    UTF-8 bytes each, so a tokenizer that has not learned Greek subwords drifts
    towards one character per token.
    """
    logger.info("Fertility (higher bytes/token = more text per token):")
    rates: dict[str, float] = {}
    for source in CORPUS_WEIGHTS:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the bilingual EN/EL tokenizer.")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--sample-gb", type=float, default=SAMPLE_GB)
    parser.add_argument("--vocab-size", type=int, default=VOCAB_SIZE)
    args = parser.parse_args()

    logger.info("Sampling training corpus (target %.1f GB) ...", args.sample_gb)
    texts = sample_texts(args.sample_gb)

    tokenizer = build_tokenizer()
    logger.info(
        "Training byte-level BPE: vocab_size=%d (%d learned + %d bytes + %d special)",
        args.vocab_size,
        args.vocab_size - N_BYTES - len(SPECIAL_TOKENS),
        N_BYTES,
        len(SPECIAL_TOKENS),
    )
    tokenizer.train_from_iterator(
        batch_iterator(texts),
        trainer=trainers.BpeTrainer(
            vocab_size=args.vocab_size,
            special_tokens=SPECIAL_TOKENS,
            # The guarantee that all 256 UTF-8 bytes stay representable.
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        ),
        length=len(texts),
    )
    logger.info("Trained vocab size: %d", tokenizer.get_vocab_size())

    # Wrap every document in bos/eos at encode time. This is what lets the
    # tokenize stage skip a separate pass that would otherwise rewrite the whole
    # corpus just to append an EOS string. The ByteLevel stage mirrors Llama's
    # own post-processor and keeps offset mappings correct (SFT's
    # assistant-only loss relies on them).
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

    verify(fast_tokenizer, args.vocab_size)
    report_fertility(fast_tokenizer)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fast_tokenizer.save_pretrained(args.output_dir)
    logger.info("Saved tokenizer to %s", args.output_dir)


if __name__ == "__main__":
    main()
