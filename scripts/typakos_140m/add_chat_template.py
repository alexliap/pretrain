"""Add chat-template special tokens to typakos_model and resize its embeddings.

typakos_model/ is a base-model checkpoint: its tokenizer carries only BOS/EOS
special tokens and no chat_template. SFT needs a way to mark role/turn
boundaries and, for assistant_only_loss, a template with {% generation %}
markers. This adds three new special tokens (<|start_header_id|>,
<|end_header_id|>, <|eot_id|>), points the tokenizer's chat_template at
chat_template.jinja, resizes the model's embedding matrix to match the grown
vocab, and writes tokenizer + model to a new output directory so the original
checkpoint is left untouched.

    python add_chat_template.py
    python add_chat_template.py --model-dir typakos_model --output-dir typakos_sft_model

Run from the repo root.
"""

import argparse
import logging
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_DIR = "typakos_model"
OUTPUT_DIR = "typakos_sft_model"
CHAT_TEMPLATE_PATH = Path(__file__).parent / "chat_template.jinja"
NEW_SPECIAL_TOKENS = ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model-dir", default=MODEL_DIR)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--chat-template", type=Path, default=CHAT_TEMPLATE_PATH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    logger.info(
        "Loaded tokenizer from %s (vocab_size=%d)", args.model_dir, len(tokenizer)
    )

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": NEW_SPECIAL_TOKENS}
    )
    logger.info("Added %d special tokens: %s", added, NEW_SPECIAL_TOKENS)

    tokenizer.chat_template = args.chat_template.read_text()

    model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    old_size = model.get_input_embeddings().weight.shape[0]
    model.resize_token_embeddings(len(tokenizer))
    new_size = model.get_input_embeddings().weight.shape[0]
    logger.info("Resized embeddings: %d -> %d", old_size, new_size)

    tokenizer.save_pretrained(args.output_dir)
    model.save_pretrained(args.output_dir)
    logger.info("Saved tokenizer and model to %s", args.output_dir)


if __name__ == "__main__":
    main()
