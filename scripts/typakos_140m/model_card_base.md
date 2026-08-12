---
language:
- en
- el
license: mit
library_name: transformers
base_model: alexliap/typakos-140m-base
pipeline_tag: text-generation
tags:
- llama
- bilingual
- greek
- pretraining
datasets:
- HuggingFaceFW/fineweb-edu
- alexliap/high-quality-gr-text
- alexliap/greek-synth-v1
---

# Typakos-140M-Base

The name "Typakos" comes from the Greek "Τυπάκος," meaning "small dude," a nod to the model's small (140M) parameter count.

Typakos-140M-Base is a 140M-parameter bilingual (Greek/English) base language model with a Llama-family architecture and a 2048-token context length. It was trained from scratch (no warm start), full-parameter, on approximately 9.33B tokens split roughly 50/50 between English and Greek. This is a **base, completion-only checkpoint**: it has no chat template and has not undergone any instruction tuning, so it should be prompted as a plain text continuation model, not as a chat assistant.

The full training pipeline, code, and configs live in [`scripts/typakos_140m/`](https://github.com/alexliap/pretrain/tree/llama120_gr/scripts/typakos_140m) on GitHub.

## Model Details

| Model Configuration | Value |
|---|---|
| Layers | 18 |
| Hidden size | 768 |
| Intermediate size | 1792 |
| Attention heads | 12 (query) / 6 (KV, grouped-query attention) |
| Head dim | 64 |
| Context length | 2048 |
| Vocab size | 50,258 |
| Tied embeddings | yes |
| Precision | bf16 |
| Parameters | ~145M total, ~106M non-embedding |

## Tokenizer

[`alexliap/bilingual_el_en_50k`](https://huggingface.co/alexliap/bilingual_el_en_50k): a byte-level BPE tokenizer with 50,000 learned merges plus 256 byte tokens and 2 special tokens (`<|begin_of_text|>`, id 0; `<|end_of_text|>`, id 1, which also doubles as the pad token). The pre-tokenizer regex pipeline is structurally borrowed from Llama 3.2, but the vocabulary itself was trained from scratch on the bilingual corpus below.

The tokenizer's post-processor wraps every encode in `<|begin_of_text|> ... <|end_of_text|>` by default. This is convenient for training but wrong for a generation prompt (a trailing EOS tells the model the prompt is already a finished document). See the usage example below for how to bypass it.

## Training Data

Sourced with a 50/50 English/Greek token target:

| Source | Repo | Language |
|---|---|---|
| fineweb_edu | HuggingFaceFW/fineweb-edu | en |
| fineweb_hq_el | alexliap/high-quality-gr-text | el |
| finewiki_el | alexliap/high-quality-gr-text | el |
| wikipedia_el | alexliap/high-quality-gr-text | el |
| synth_faq | alexliap/greek-synth-v1 | el |
| synth_math | alexliap/greek-synth-v1 | el |
| synth_table | alexliap/greek-synth-v1 | el |
| synth_tutorial | alexliap/greek-synth-v1 | el |

## Training Procedure

| Training Configuration | Value |
|---|---|
| GPUs | 8 |
| Batch size | 16 sequences/GPU x 2048 tokens = 262,144 tokens/step |
| Gradient accumulation | 1 |
| Optimizer | AdamW, lr 2e-4, betas (0.9, 0.95), eps 1e-10, weight_decay 0.001 |
| LR schedule | Linear warmup, 2000 steps |
| Tokens seen | ~9.33B (1 epoch, run stopped at corpus exhaustion) |
| Precision | bf16 |

### Reproduction

All code, Hydra configs, and step-by-step reproduction commands (data download, tokenizer training, shard packing, and the training launch script) are in [`scripts/typakos_140m/`](https://github.com/alexliap/pretrain/tree/llama120_gr/scripts/typakos_140m) in the `pretrain` GitHub repo, under the [`README.md`](https://github.com/alexliap/pretrain/blob/llama120_gr/scripts/typakos_140m/README.md) there:

1. Environment setup and initial data pull (`setup.sh`)
2. Train the bilingual BPE tokenizer (`train_tokenizer.py`)
3. Measure bytes/token per source and derive the token mix plan (`measure_token_rates.py`)
4. Download the full corpus per that plan (`download_data.py`)
5. Mix sources into shuffled, proportional shards (`concat_data.py`)
6. Tokenize and pack into 2048-token sequences (`prepare_shards.py`)
7. Train (`train.sh`, pointed at [`configs/train.yaml`](https://github.com/alexliap/pretrain/blob/llama120_gr/scripts/typakos_140m/configs/train.yaml))

## How to Use

This is a base model, so prompt it with plain text and let it continue. Because the tokenizer auto-wraps every encode in BOS/EOS, use `add_special_tokens=False` and prepend BOS manually to avoid signaling a "finished document" to the model:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "alexliap/typakos-140m-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

prompt = "The history of Athens begins"
input_ids = tokenizer(
    tokenizer.bos_token + prompt, add_special_tokens=False, return_tensors="pt"
).input_ids

output = model.generate(input_ids, max_new_tokens=100, do_sample=True, temperature=0.7)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```

## Evaluation

Evaluated 0-shot with [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), except GSM8K at 5-shot, matching the shot conventions stated on the SmolLM2 model card.

| Benchmark | Typakos-140M-Base | SmolLM2-135M-8k |
|---|---:|---:|
| HellaSwag (acc_norm) | 26.5 | 42.1 |
| ARC (avg acc_norm) | 28.3 | 43.9 |
| PIQA (acc_norm) | 56.6 | 68.4 |
| MMLU cloze (acc) | 23.4 | 31.5 |
| CommonsenseQA (acc) | 19.6 | 33.9 |
| TriviaQA (exact_match) | 0.0 | 4.1 |
| Winogrande (acc) | 49.6 | 51.3 |
| OpenBookQA (acc_norm) | 33.2 | 34.6 |
| GSM8K 5-shot (exact_match) | 0.2 | 1.4 |

> The gap to SmolLM2-135M-8k below is largely a training-budget gap, not an architecture gap: per its [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M), SmolLM2-135M was pretrained on about 2T tokens, roughly 200 times the ~9.33B tokens Typakos-140M-Base has seen, on a curated English-only mix (FineWeb-Edu, DCLM, The Stack, plus additional filtered sources). Typakos-140M-Base is a single-stage, from-scratch run on a much smaller, bilingual (EN/EL) corpus, so it is comparable in size and architecture but not yet in data scale.

## Limitations

- It is a base, completion-only model with no instruction tuning or chat capability.
- It is best suited for continued pretraining, fine-tuning, or research experimentation, not for direct deployment as an assistant.

## License

MIT
