"""Smoke-test an SFT checkpoint by generating chat completions for a handful
of English and Greek user turns.

Unlike test_generation.py (which continues raw text on the base model), this
renders each prompt through the tokenizer's chat template
(chat_template.jinja, added by add_chat_template.py) with
add_generation_prompt=True, and decodes only the newly generated tokens.
Generation stops at <|eot_id|> (end of the assistant's turn), not the base
tokenizer's <|end_of_text|>. Run from the repo root - ``--model-path``
defaults to the SFT run's output dir.

Example:
    python scripts/typakos_140m/test_chat_generation.py
    python scripts/typakos_140m/test_chat_generation.py --model-path models/typakos_sft_model --max-new-tokens 200
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ENGLISH_PROMPTS = [
    "What is the capital of France?",
    "Write a short story about a small village.",
    "What's the most important thing to remember about machine learning?",
    "Write a Python function to compute the nth Fibonacci number.",
]

GREEK_PROMPTS = [
    "Ποια είναι η πρωτεύουσα της Ελλάδας;",
    "Γράψε μια σύντομη ιστορία για ένα μικρό χωριό.",
    "Τι είναι η τεχνητή νοημοσύνη;",
    "Πώς είναι ο καιρός σήμερα στην Αθήνα;",
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
    eos_token_ids: list[int],
) -> str:
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(device)
    input_ids = inputs["input_ids"]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_token_ids,
    )
    completion_ids = output_ids[0, input_ids.shape[1] :]
    return tokenizer.decode(completion_ids, skip_special_tokens=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--model-path",
        default="alexliap/typakos_140m_it",
        help="SFT checkpoint directory.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument(
        "--do-sample", action="store_true", help="Sample instead of greedy decoding."
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    device = _select_device()
    print(f"Loading model from {args.model_path} on {device} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=dtype).to(
        device
    )
    model.eval()

    eot_token_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
    eos_token_ids = [tokenizer.eos_token_id]
    if eot_token_id is not None and eot_token_id != tokenizer.unk_token_id:
        eos_token_ids.append(eot_token_id)

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
                    eos_token_ids,
                )
                print(f"\nUser:      {prompt!r}")
                print(f"Assistant: {completion!r}")


if __name__ == "__main__":
    main()
