"""Merge a LoRA adapter into its base model to produce a standalone HF model.

One-time prerequisite for SFT-ing from a continued-pretraining (CPT) checkpoint:
the CPT checkpoints under ``checkpoints/.../last/`` hold only the adapter
(``adapter_config.json`` + ``adapter_model.safetensors``). This merges the
adapter weights into the base and writes a full model dir that ``sft.py`` can
load directly via ``model_name_or_path``.

Example:
    python merge_adapter.py \
        --adapter-path checkpoints/lfm2_5_base_r64_a64/v5_10b_r64_a64/last \
        --output-dir models/lfm2_5_cpt_merged

The base model is read from the adapter's ``base_model_name_or_path`` unless
``--base-model`` is given explicitly.
"""

import argparse
import json
import os

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _resolve_base_model(adapter_path: str, base_model: str | None) -> str:
    if base_model is not None:
        return base_model
    with open(os.path.join(adapter_path, "adapter_config.json")) as f:
        return json.load(f)["base_model_name_or_path"]


def merge_adapter(adapter_path: str, output_dir: str, base_model: str | None) -> None:
    base = _resolve_base_model(adapter_path, base_model)
    print(f"Loading base model from {base} ...")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16)

    print(f"Applying adapter from {adapter_path} and merging ...")
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    print(f"Saving merged model to {output_dir} ...")
    model.save_pretrained(output_dir, safe_serialization=True)

    # Save the tokenizer alongside so the merged dir is self-contained. Prefer
    # the adapter's tokenizer (it travels with the checkpoint), else the base's.
    tokenizer_src = (
        adapter_path
        if os.path.exists(os.path.join(adapter_path, "tokenizer.json"))
        else base
    )
    AutoTokenizer.from_pretrained(tokenizer_src).save_pretrained(output_dir)
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--adapter-path", required=True, help="Directory with the LoRA adapter.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the merged model into.")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model path/id (default: read from adapter_config.json).",
    )
    args = parser.parse_args()
    merge_adapter(args.adapter_path, args.output_dir, args.base_model)


if __name__ == "__main__":
    main()
