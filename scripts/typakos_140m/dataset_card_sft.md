---
language:
- en
- el
license: apache-2.0
pretty_name: Typakos SFT Dataset
tags:
- sft
- instruction-tuning
- conversational
- bilingual
- greek
task_categories:
- text-generation
datasets:
- openeurollm/Dolci-Instruct-SFT-translated
- openeurollm/EU-Instruct-Synthetic
- CohereLabs/aya_dataset
- HuggingFaceTB/smoltalk
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: validation
    path: data/validation-*
---

# Typakos SFT Dataset

A bilingual (Greek/English) instruction-tuning dataset used to supervised-fine-tune [`alexliap/typakos-140m-base`](https://huggingface.co/alexliap/typakos-140m-base) into `alexliap/typakos-140m-it`. It mixes 7 filtered subsets drawn from 4 upstream Hub sources into a single shuffled pool of 936,491 train and 104,055 validation conversations, each token-bounded to fit the model's 2048-token context length.

The full construction pipeline, code, and configs live in [`scripts/typakos_140m/`](https://github.com/alexliap/pretrain/tree/llama120_gr/scripts/typakos_140m) on GitHub.

## Dataset Structure

Conversational format, matching TRL's `dataset_format: "conversational"`:

| Column | Type | Description |
|---|---|---|
| `messages` | `list[{role, content}]` | An optional single leading `system` turn, then alternating `user`/`assistant` turns ending on `assistant` |
| `num_tokens` | `int64` | Total token count of the conversation, tokenized with the `typakos_sft_model` tokenizer (`add_special_tokens=False`) |
| `num_turns` | `int64` | Turn count excluding a leading system message |
| `source` | `string` | Which of the 7 named subsets below the row came from |

## Source Composition

| source | upstream dataset | language | raw | well-formed | final (<=2048 tok) |
|---|---|---|---:|---:|---:|
| dolci_el | openeurollm/Dolci-Instruct-SFT-translated | el | 494,661 | 467,948 | 448,718 |
| eu_instruct_el | openeurollm/EU-Instruct-Synthetic | el | 138,048 | 138,048 | 137,988 |
| aya_el | CohereLabs/aya_dataset (Greek subset) | el | 623 | 623 | 623 |
| aya_en | CohereLabs/aya_dataset (English subset) | en | 3,944 | 3,941 | 3,938 |
| smol_constraints | HuggingFaceTB/smoltalk (smol-constraints) | en | 34,424 | 34,424 | 34,423 |
| smol_magpie_ultra | HuggingFaceTB/smoltalk (smol-magpie-ultra) | en | 409,537 | 409,537 | 361,514 |
| smol_rewrite | HuggingFaceTB/smoltalk (smol-rewrite) | en | 53,342 | 53,342 | 53,342 |
| **TOTAL** | | | **1,134,579** | **1,107,863** | **1,040,546** |

Roughly 901M tokens total across the kept rows (~394M Greek, ~507M English).

## Splits

The 1,040,546 kept rows were shuffled (seed 0) and split, with 10% held out from the back of that shuffled order as validation:

| split | rows |
|---|---:|
| train | 936,491 |
| validation | 104,055 |

## Construction

Built in two steps:

1. **Filter** ([`prepare_sft_data.py`](https://github.com/alexliap/pretrain/blob/llama120_gr/scripts/typakos_140m/prepare_sft_data.py)): downloads each upstream source, keeps only well-formed conversations (an optional single leading `system` turn, then strictly alternating `user`/`assistant` turns ending on `assistant`; empty content or any other shape is dropped), then keeps only conversations whose total token count is <=2048. Turn count itself is not restricted, a long multi-turn conversation is kept whole as long as it fits the token budget.
2. **Merge and split** ([`concat_sft_data.py`](https://github.com/alexliap/pretrain/blob/llama120_gr/scripts/typakos_140m/concat_sft_data.py)): concatenates the 7 filtered pools (identical schema throughout), shuffles the combined pool, and splits off validation as described above.

## Licensing

Declared as `Apache-2.0` for this repackaged/filtered pool. Upstream source licenses: `openeurollm/Dolci-Instruct-SFT-translated` (apache-2.0), `openeurollm/EU-Instruct-Synthetic` (apache-2.0), `CohereLabs/aya_dataset` (apache-2.0), `HuggingFaceTB/smoltalk` (no license declared upstream at time of writing). Check each upstream source directly before redistribution.
