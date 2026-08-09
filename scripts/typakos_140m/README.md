# typakos-140m

The pretraining stage of **typakos**, a 140M-parameter bilingual (Greek /
English) base language model. Trained from scratch (no warm start) on ~9.33B tokens, full-parameter. `typakos_model/` itself is a **base** checkpoint: no chat template, no instruction tuning. This directory also holds the post-training steps that build on it, see [Post-training](#post-training) below.

Every script here runs from the **repo root** (paths inside them are
CWD-relative, e.g. `data/raw`, `tokenized_data`, `models/...`), so invoke them
as `python scripts/typakos_140m/<script>.py`, not from inside this directory.

## Contents

- [Architecture](#architecture)
- [Tokenizer](#tokenizer)
- [Corpus](#corpus)
- [Training setup](#training-setup)
- [Reproduce](#reproduce)
- [Post-training](#post-training)
  - [SFT data pool](#sft-data-pool)
  - [Chat template](#chat-template)
  - [Not done yet](#not-done-yet)

## Architecture

Llama-family, from [`configs/model/llama_140m.yaml`](configs/model/llama_140m.yaml)
via `AutoConfig.from_pretrained("meta-llama/Llama-3.2-1B")` as a structure
template only; see `PretrainTask._init_model_and_tokenizer`.

| | |
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

[`alexliap/bilingual_el_en_50k`](https://huggingface.co/alexliap/bilingual_el_en_50k),
trained by [`train_tokenizer.py`](train_tokenizer.py): byte-level BPE, 50,000
learned merges + 256 byte tokens + 2 special tokens (`<|begin_of_text|>` id 0,
`<|end_of_text|>` id 1, doubling as pad). The pre-tokenizer regex is borrowed
structurally from Llama 3.2 (`meta-llama/Llama-3.2-1B`), only its pipeline,
never its vocabulary.

The tokenizer's `post_processor` wraps **every** encode in
`<|begin_of_text|> … <|end_of_text|>` (a `TemplateProcessing` stage, verified
by `train_tokenizer.py`'s `verify()`). This is deliberate: it lets
`prepare_shards.py` skip a separate corpus-rewrite pass just to append EOS.
It is wrong for a generation prompt, though: a trailing EOS tells the model
the prompt is a finished document, and it starts hallucinating a new one.
Bypass it for generation with `add_special_tokens=False` plus a manual BOS
prepend, as [`test_generation.py`](test_generation.py) does.

## Corpus

Sourced by [`download_data.py`](download_data.py), targeting a 50/50 English /
Greek token split (`measure_token_rates.py --en-share 0.5`, the default):

| Source | Repo | Language |
|---|---|---|
| `fineweb_edu` | `HuggingFaceFW/fineweb-edu` | en |
| `fineweb_hq_el` | `alexliap/high-quality-gr-text` | el |
| `finewiki_el` | `alexliap/high-quality-gr-text` | el |
| `wikipedia_el` | `alexliap/high-quality-gr-text` | el |
| `synth_faq` | `alexliap/greek-synth-v1` | el |
| `synth_math` | `alexliap/greek-synth-v1` | el |
| `synth_table` | `alexliap/greek-synth-v1` | el |
| `synth_tutorial` | `alexliap/greek-synth-v1` | el |

`concat_data.py` mixes the filtered sources into 20 proportional, internally
shuffled `shard_NN.parquet` files (deterministic order: the training
dataloader reads with `shuffle=False`, so on-disk order *is* training order).
`prepare_shards.py` tokenizes each shard, holds out 500,000 documents from the
tail of the last shard as a variable-length validation split, and packs the
rest into 2048-token sequences with zero padding.

## Training setup

Recovered directly from the released checkpoint's own Accelerate state
(`typakos_model/state/{optimizer.bin,scheduler.bin,custom_checkpoint_0.pkl,
random_states_*.pkl}`) rather than from git history: the committed
`configs/train.yaml` has since moved on to unrelated experiments and no
longer reflects this run. The full reconstruction, including exactly which
fields are checkpoint-verified vs. carried over, is documented in
[`configs/train.yaml`](configs/train.yaml)'s header comments.

| | |
|---|---|
| GPUs | 8 |
| Batch size | 16 sequences/GPU × 2048 tokens = 262,144 tokens/step |
| Gradient accumulation | 1 |
| Optimizer | AdamW, lr 2e-4, betas (0.9, 0.95), eps 1e-10, weight_decay 0.001 |
| LR schedule | Linear warmup, 2000 steps |
| Tokens seen | ~ 9,3B |
| Precision | bf16 (Accelerate mixed precision) |

`train.sh` sets `NCCL_P2P_DISABLE=1`: cross-NUMA GPU links without NVLink hang
NCCL's P2P handshake indefinitely inside Docker before the first step.

## Reproduce

All commands from the repo root:

```bash
# 0. Environment + a first data pull (bilingual tokenizer + prior packed data,
#    if you're continuing rather than starting from raw text)
./scripts/typakos_140m/setup.sh

# 1. Probe sample for tokenizer training
python scripts/typakos_140m/download_data.py --probe

# 2. Train the bilingual BPE tokenizer -> models/bilingual_el_en_50k/
python scripts/typakos_140m/train_tokenizer.py

# 3. Measure bytes/token per source and derive the 10B-token mix plan
python scripts/typakos_140m/measure_token_rates.py --budget 10e9 --en-share 0.5

# 4. Download the full plan
python scripts/typakos_140m/download_data.py --plan data/download_plan.json

# 5. Mix into 20 shuffled, proportional shards
python scripts/typakos_140m/concat_data.py --n-shards 20

# 6. Tokenize + pack -> tokenized_data/
python scripts/typakos_140m/prepare_shards.py

# 7. Train
./scripts/typakos_140m/train.sh
```

`train.sh` points Hydra at [`configs/train.yaml`](configs/train.yaml) in this directory (`-cp scripts/typakos_140m/configs -cn train`) rather than the root `configs/train.yaml`, which is a different, unrelated experiment.

## Post-training

The steps below turn the base `typakos_model/` checkpoint into
`typakos_sft_model/`: a token-bounded SFT data pool and a chat template with new special tokens. The SFT run itself (`trl.SFTTrainer` via the repo root's `sft.py`) hasn't happened yet.

### SFT data pool

[`prepare_sft_data.py`](prepare_sft_data.py) pulls five Hub instruction
sources, keeps only well-formed conversations (an optional single leading
`system` turn, then alternating `user`/`assistant` turns ending on
`assistant`), and further keeps only those whose combined token count is
<= 2048 (tokenized with `typakos_model`'s own tokenizer, `add_special_tokens=False`).
Multi-turn conversations are kept whole as long as they fit that budget,
turn count itself isn't restricted. Each source is saved separately under
`data/sft_raw/<name>/` as a `datasets.Dataset` with `messages`/`num_tokens`/
`num_turns`/`source` columns, not yet merged into one training-ready dataset:

| source | raw | well-formed | final (<= 2048 tok) |
|---|---:|---:|---:|
| dolci_el | 494,661 | 467,948 | 448,718 |
| eu_instruct_el | 138,048 | 138,048 | 137,988 |
| aya_el | 623 | 623 | 623 |
| aya_en | 3,944 | 3,941 | 3,938 |
| smol_constraints | 34,424 | 34,424 | 34,423 |
| smol_magpie_ultra | 409,537 | 409,537 | 361,514 |
| smol_rewrite | 53,342 | 53,342 | 53,342 |
| **TOTAL** | **1,134,579** | **1,107,863** | **1,040,546** |

~901M tokens total across the kept rows (~394M Greek, ~507M English).

```bash
python scripts/typakos_140m/prepare_sft_data.py
```

### Chat template

`typakos_model`'s tokenizer carries only BOS/EOS (`<|begin_of_text|>` id 0,
`<|end_of_text|>` id 1) and no `chat_template`. [`add_chat_template.py`](add_chat_template.py)
adds three Llama-3-style special tokens (`<|start_header_id|>`,
`<|end_header_id|>`, `<|eot_id|>`), resizes the tied embedding matrix to
match (new rows are mean/covariance-initialized, not random), points the
tokenizer's `chat_template` at [`chat_template.jinja`](chat_template.jinja),
and saves the result to `typakos_sft_model/`, leaving `typakos_model/`
untouched. The template is adapted from TRL's `llama3_training.jinja`: it
renders each message as
`<|start_header_id|>ROLE<|end_header_id|>\n\nCONTENT<|eot_id|>` and wraps
assistant content in `{% generation %}`/`{% endgeneration %}` markers so
`assistant_only_loss=True` in `trl.SFTConfig` can mask the loss to assistant
turns only.

```bash
python scripts/typakos_140m/add_chat_template.py
```

`typakos_sft_model/` is uploaded to the private Hub repo
[`alexliap/llama_140m_sft`](https://huggingface.co/alexliap/llama_140m_sft)
via [`upload_checkpoint.sh`](../../upload_checkpoint.sh).

### Not done yet

Merging the 7 SFT pools above into one training-ready dataset, and the
actual SFT run, are still open.
