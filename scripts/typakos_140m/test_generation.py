"""Smoke-test a pretrained checkpoint by generating completions for a handful
of English and Greek prompts.

This is a base (non-chat) causal LM, so prompts are plain-text continuations
rather than chat turns. Run from the repo root - ``--model-path`` defaults to
a checkpoint dir there.

Example:
    python scripts/typakos_140m/test_generation.py
    python scripts/typakos_140m/test_generation.py --model-path typakos_model --max-new-tokens 80
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ENGLISH_PROMPTS = [
    "The capital of France is",
    "Once upon a time, in a small village,",
    "The most important thing to remember about machine learning is",
    "def fibonacci(n):",
]

GREEK_PROMPTS = [
    "Η πρωτεύουσα της Ελλάδας είναι",
    "Μια φορά κι έναν καιρό, σε ένα μικρό χωριό,",
    "Η τεχνητή νοημοσύνη είναι μια τεχνολογία που",
    "Ο καιρός σήμερα στην Αθήνα είναι",
]


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate(
    model,
    tokenizer,
    prompt: str,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
) -> str:
    # The tokenizer's post-processing template wraps every encoded sequence in
    # bos ... eos - deliberate (see train_tokenizer.py) so the data-prep
    # pipeline gets free EOS-wrapping at tokenize time, without a separate pass
    # over the corpus to append it. That default is wrong for a generation
    # prompt though: a trailing eos tells the model the document is finished,
    # so it starts hallucinating a new, unrelated one after it. Bypass the
    # template here instead of fighting it: encode with no special tokens and
    # prepend bos ourselves, matching training's start-of-document token
    # without training's end-of-document one.
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    input_ids = torch.tensor([[tokenizer.bos_token_id, *ids]], device=device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids),
    }
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output_ids[0], skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model-path", default="typakos_model", help="Model / checkpoint directory."
    )
    parser.add_argument("--max-new-tokens", type=int, default=60)
    parser.add_argument(
        "--do-sample", action="store_true", help="Sample instead of greedy decoding."
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = _select_device()
    print(f"Loading model from {args.model_path} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype).to(
        device
    )
    model.eval()

    with torch.no_grad():
        for label, prompts in [("ENGLISH", ENGLISH_PROMPTS), ("GREEK", GREEK_PROMPTS)]:
            print("\n" + "=" * 60)
            print(label)
            print("=" * 60)
            for prompt in prompts:
                completion = generate(
                    model,
                    tokenizer,
                    prompt,
                    device,
                    args.max_new_tokens,
                    args.do_sample,
                    args.temperature,
                )
                print(f"\nPrompt:     {prompt!r}")
                print(f"Completion: {completion!r}")


if __name__ == "__main__":
    main()
