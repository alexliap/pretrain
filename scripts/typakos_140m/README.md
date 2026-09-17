# typakos-140m

![Typakos](tyapkos.png)

The full training pipeline for **typakos**, a 140M-parameter bilingual (Greek / English) language
model: pretraining a base checkpoint from scratch, then post-training it into an instruction-tuned,
preference-aligned chat model. `train.sh` (below) writes that **base** checkpoint locally to
`typakos_model/` -- gitignored, not part of this repo -- and it's published to the Hub as
[`alexliap/typakos-140m-base`](https://huggingface.co/alexliap/typakos-140m-base): trained from
scratch, no warm start, on ~9.33B tokens, full-parameter; no chat template, no instruction tuning,
prompted as plain-text continuation only. [Post-training](#post-training) below covers the two
stages built on top of it, SFT then DPO, that turn it into `typakos-140m-it`.

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
  - [Merge and split](#merge-and-split)
  - [Chat template](#chat-template)
  - [SFT config](#sft-config)
  - [DPO data](#dpo-data)
  - [DPO config](#dpo-config)
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

Trained by [`train_tokenizer.py`](train_tokenizer.py): byte-level BPE, 50,000
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

The steps below turn the base `typakos_model/` checkpoint into the instruction-tuned, DPO-aligned
`typakos-140m-it` model, in two stages: SFT (`trl.SFTTrainer`) for chat formatting and
instruction-following, then DPO (`trl.DPOTrainer`) for preference alignment on top of the SFT
result. Both stages are full-parameter (no LoRA).

### SFT data pool

[`prepare_sft_data.py`](prepare_sft_data.py) pulls five Hub instruction
sources, keeps only well-formed conversations (an optional single leading
`system` turn, then alternating `user`/`assistant` turns ending on
`assistant`), and further keeps only those whose combined token count is
<= 2048 (tokenized with `typakos_sft_model`'s tokenizer, `add_special_tokens=False`,
identical token counts to `typakos_model`'s since the chat-template special
tokens it adds never appear in raw content). Multi-turn conversations are
kept whole as long as they fit that budget, turn count itself isn't
restricted. Each source is saved separately under `data/sft_raw/<name>/` as a
`datasets.Dataset` with `messages`/`num_tokens`/`num_turns`/`source` columns:

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

### Merge and split

[`concat_sft_data.py`](concat_sft_data.py) concatenates the 7 pools above
(same schema throughout, so no casting needed), shuffles the combined pool,
and splits off 10% as validation:

| split | rows |
|---|---:|
| train | 936,491 |
| validation | 104,055 |

Saved as a single `datasets.DatasetDict` to `data/sft_pool/` (`save_to_disk`
format) and pushed to the private Hub dataset repo
[`alexliap/typakos_sft_dataset`](https://huggingface.co/datasets/alexliap/typakos_sft_dataset)
(`DatasetDict.push_to_hub`, both splits). `setup_sft.sh` pulls that Hub repo
back down as a raw parquet snapshot to `data/typakos_sft_dataset/` rather than
regenerating the local `data/sft_pool/` copy, so `configs/sft.yaml`'s
`dataset.dataset_id` points at `data/typakos_sft_dataset` instead --
`pretrain.sft.task.SFTTask._load_split` detects the missing
`dataset_dict.json`/`dataset_info.json` `save_to_disk` markers there and falls
back to `datasets.load_dataset`, which auto-discovers the `train-*`/
`validation-*` parquet split pattern the same way it would for a Hub id.

```bash
python scripts/typakos_140m/concat_sft_data.py
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

### SFT config

[`configs/sft.yaml`](configs/sft.yaml) points `trl.SFTTrainer` (via the repo
root's `sft.py`) at `models/typakos_sft_model` and `data/typakos_sft_dataset`,
same pattern as [`configs/train.yaml`](configs/train.yaml): a directory-local
snapshot rather than the root `configs/sft.yaml`, which is a different,
unrelated experiment. Full-parameter SFT (no LoRA, `typakos_140m` is small
enough not to need it), `dataset_format: "conversational"`,
`assistant_only_loss: true`. No `chat_template_path` override needed,
`typakos_sft_model`'s tokenizer already carries `chat_template.jinja`.
`adam_beta1`/`adam_beta2`/`adam_epsilon` are set to match the pretraining
run's optimizer (0.9 / 0.95 / 1e-10, see [Training setup](#training-setup));
`warmup_ratio` is left `null` since the installed `transformers` (5.x) dropped
that field from `TrainingArguments` entirely, so only `warmup_steps` is
honored (`SFTTask._build_sft_config` only forwards `warmup_ratio` when set,
to avoid a `TypeError`). `dataset_num_proc: 16` parallelizes `SFTTrainer`'s
upfront tokenize/label-building pass across CPU cores -- without it, that
single-process pass over the full ~936K-row train split can outlast
`accelerate`'s NCCL barrier timeout while the other 7 ranks wait on an 8-GPU
launch. The `sft:` block's batch size / epochs / eval cadence are otherwise
still untuned starting points, not a recovered or validated run like
`train.yaml`.

```bash
./scripts/typakos_140m/sft.sh
```

`sft.sh` points Hydra at `configs/sft.yaml` in this directory (`-cp
scripts/typakos_140m/configs -cn sft`), mirroring `train.sh`.

### DPO data

[`prepare_dpo_data.py`](prepare_dpo_data.py) pulls the `el`/`en` configs of
`openeurollm/Dolci-Instruct-DPO-translated` (already shaped as
`prompt`/`chosen`/`rejected` conversational triples, no reshaping needed
beyond dropping the `id` column), concatenates both languages, shuffles
(seed 0), and splits off 10% as validation:

| split | rows |
|---|---:|
| train | 423,954 |
| validation | 47,107 |

Saved as a `datasets.DatasetDict` via `save_to_disk` to
`data/typakos_dpo_dataset/`, the local-directory format
`pretrain.dpo.task.DPOTask._load_split` expects.

```bash
python scripts/typakos_140m/prepare_dpo_data.py
```

### DPO config

[`configs/dpo.yaml`](configs/dpo.yaml) points `trl.DPOTrainer` (via the repo
root's `dpo.py`) at `alexliap/typakos-140m-it` -- the SFT result, still the
pre-DPO variant as of writing -- and `data/typakos_dpo_dataset`. Full-parameter
DPO (no LoRA, no separate `ref_model`; TRL derives the reference model
internally), `dataset_format: "conversational"`, sigmoid loss with `beta:
0.1`. `adam_beta1`/`adam_beta2`/`adam_epsilon` again match the
pretraining/SFT runs' optimizer settings. `lr_scheduler_type:
"constant_with_warmup"` with `warmup_steps: 500` ramps the learning rate
before holding it flat -- plain `"constant"` ignores warmup entirely in
transformers 5.x, regardless of `warmup_steps`. The learning rate itself has
gone through a few iterations during tuning (`2e-7` -> `1e-6` -> `5e-6`);
DPO's un-length-normalized sigmoid loss produces much larger raw gradients
than SFT/pretraining's per-token cross-entropy, so `grad_norm` running in the
tens-to-hundreds (occasionally spiking higher, absorbed by
`max_grad_norm: 1.0` clipping) is expected here, not a sign of instability.

```bash
./scripts/typakos_140m/dpo.sh
```

`dpo.sh` points Hydra at `configs/dpo.yaml` in this directory, mirroring
`train.sh`/`sft.sh`.

### Not done yet

The DPO run is in progress; evaluation of the resulting `typakos-140m-it`
checkpoint hasn't happened yet.
