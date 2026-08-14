# Pretrain Experiment

A lightweight language model **pretraining / continued-pretraining / supervised fine-tuning** framework built on vanilla PyTorch and HuggingFace Accelerate for the (continued-)pretraining loop, and TRL's `SFTTrainer` for the SFT loop. Configuration is driven by Hydra, experiment tracking by trackio.

## Features

- **Vanilla PyTorch**: Pure PyTorch pretraining loop without heavy frameworks
- **HuggingFace Accelerate**: Seamless distributed training support (multi-GPU, mixed precision)
- **Hydra Configuration**: YAML-based configuration with composable model presets and CLI overrides
- **Continued Pretraining (CPT)**: Resume/continue training from any checkpoint or pretrained model directory, optionally with a LoRA adapter
- **Supervised Fine-Tuning (SFT)**: TRL-based SFT with conversational data and assistant-only loss
- **LoRA / PEFT**: Train adapters instead of full parameters for both CPT and SFT (via `peft`)
- **Adapter Merging**: Merge a trained LoRA adapter back into its base model to bridge CPT → SFT
- **Token-Budget Stopping**: Stop training after a target number of tokens (`total_tokens`)
- **Automatic Logging**: GPU metrics and training stats via trackio, plus a write-enabled dashboard
- **Custom Tokenizers**: Support for custom tokenizer training and usage

## From a checkpoint or pretrained directory (any HF causal LM)

When `saved_checkpoint_path` (training) or `model_name_or_path` (SFT) points at a checkpoint or a saved model directory, the model is loaded via `AutoConfig`/`AutoModelForCausalLM`, so **any HuggingFace causal-LM architecture works**: the size presets are bypassed.

## Configuration

Training is configured via Hydra YAML files. The main (continued-)pretraining config is `configs/train.yaml`, which composes a model preset via `defaults`. Settings are grouped into per-concern sections (`data`, `optimizer`, `scheduler`, `accelerate`, `validation`, `logging`, `checkpoint`, `evaluation`, and an optional `lora`), with run-level fields kept at the top level:

```yaml
defaults:
  - model: qwen_small   # or qwen_tiny, qwen_medium, qwen_large, qwen_xlarge
  - _self_

# Run-level (top-level)
saved_checkpoint_path: null    # continue from a checkpoint / pretrained dir; null = build from preset
compile: "torch"               # torch.compile; set null (recommended with LoRA)
num_epochs: 1
total_steps: null
total_tokens: null             # token-budget stop (highest priority when set)

# LoRA adapter (optional). Omit / null for full-parameter training.
lora:
  r: 32
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj"]   # null lets peft auto-infer

# Data
data:
  tokenizer_path: "./models/lfm2_5_base/"
  max_seq_length: 2048
  use_packed_data: true
  num_workers: 4
  train_batch_size: 8
  val_batch_size: 8

# Optimizer (incl. gradient clipping)
optimizer:
  learning_rate: 8e-5
  weight_decay: 0.01
  betas: [0.9, 0.95]
  eps: 1e-10
  max_grad_norm: 1.0

# Scheduler
scheduler:
  warmup_steps: 1000

# Accelerate
accelerate:
  mixed_precision: "bf16"
  gradient_accumulation_steps: 1

# Validation
validation:
  val_check_interval: 4000   # steps
  val_size: 80000            # held-out validation samples

# Logging
logging:
  project_name: "my-cpt-run"
  auto_log_gpu: true
  log_every_n: 10

# Checkpointing
checkpoint:
  save_dir: "checkpoints"
  experiment_name: null      # subdir under save_dir; falls back to model.name when null
  run_name: null             # nested run subdir; defaults to a timestamp when null
  save_top_k: 1
  save_every_n_steps: 4000
  save_last: true
  max_shard_size: "5GB"
```

Override any parameter from the command line using its section path:

```bash
python main.py data.train_batch_size=32 optimizer.learning_rate=3e-4 model=qwen_medium
```

## Usage

### 1. Download Data

Download raw parquet data from the Hub (writes per-dataset files under `data/`) with the CLI:

```bash
uv run pretrain-data download <repo_id> <local_dir>
```

### 2. Mix / Consolidate Data

Mix the downloaded, filtered sources into proportionally-mixed, shuffled parquet shards (`data/mixed_dataset/shard_NN.parquet`), reading the per-source row targets from a mix plan JSON:

```bash
python scripts/typakos_140m/concat_data.py --n-shards 20
```

This particular `concat_data.py` is part of the bilingual EL/EN corpus pipeline used for the typakos_140m run (see [scripts/typakos_140m/README.md](scripts/typakos_140m/README.md) for the full, reproducible sequence, including how the mix plan itself is derived); a from-scratch pretraining run with a different corpus would replace this step with its own mixing logic.

### 3. Tokenize Data

Tokenize the parquet dataset into train/test splits using the `pretrain-data` CLI:

```bash
uv run pretrain-data tokenize \
  --tokenizer-repo-id <tokenizer-id-or-path> \
  --data-path "data/mixed_dataset/shard_*.parquet" \
  --output-path tokenized_data/
```

Use `--test-size` to change the held-out fraction (default `0.1`).

### 4. Pack Data

Pack tokenized sequences into fixed-length blocks for efficient training (eliminates padding waste), using the `pretrain-data` CLI:

```bash
uv run pretrain-data pack \
  --input-dir tokenized_data/train \
  --output-dir tokenized_data/packed_train_data_2048 \
  --max-seq-length 2048
```

### 5. Train

Run (continued-)pretraining with your configuration:

```bash
# Single GPU
python main.py

# Multi-GPU with Accelerate
accelerate launch main.py

# With config overrides
python main.py model=qwen_large optimizer.learning_rate=3e-4

# Configure accelerate (first time)
accelerate config

# Point at a different config directory/file (e.g. a reference run's own snapshot)
uv run accelerate launch main.py -cp scripts/typakos_140m/configs -cn train
```

#### Continued Pretraining from a Checkpoint

Set `saved_checkpoint_path` to a checkpoint or pretrained model directory to continue training an existing model instead of building one from a preset:

```bash
python main.py \
  saved_checkpoint_path=./checkpoints/my_experiment/my_run/last \
  total_tokens=1_000_000_000
```

- Add a `lora` section to train a LoRA adapter on top of the frozen base (recommended for large-model CPT). With LoRA, only the adapter is saved into each checkpoint.
- When resuming an **interrupted** run (a checkpoint that contains a `state/` directory), the full training state (global step, epoch, tokens seen, and dataloader position) is restored automatically so no samples are seen twice or skipped.
- When using LoRA, set `compile: null`.

#### Distributed Training

```bash
# Launch on all available GPUs
accelerate launch --multi_gpu main.py

# Launch on a specific number of GPUs
accelerate launch --num_processes 4 main.py
```

### 6. Merge Adapter (CPT → SFT bridge)

CPT checkpoints trained with LoRA store only the adapter. Before SFT-ing from such a checkpoint, merge the adapter into its base to produce a standalone model directory:

```bash
python merge_adapter.py \
  --adapter-path checkpoints/my_experiment/my_run/last \
  --output-dir models/my_cpt_merged
```

The base model is read from the adapter's `adapter_config.json` unless `--base-model` is given explicitly. The merged directory (with tokenizer) can then be loaded by `sft.py` via `model_name_or_path`.

### 7. Supervised Fine-Tuning (SFT)

SFT is configured via `configs/sft.yaml` and run through TRL's `SFTTrainer`:

```bash
# Single GPU
python sft.py

# Multi-GPU with Accelerate
accelerate launch sft.py
```

Key configuration:

- `model_name_or_path`: the merged-CPT directory (or any base model) to fine-tune.
- `dataset`: `dataset_id` (HF hub id or local `save_to_disk` dir), splits, and `dataset_format`, one of `conversational` (rows of `{"messages": [...]}`), `prompt_completion`, or `text`.
- `chat_template_path`: a ChatML template with `{% generation %}` markers (`configs/lfm2_chatml_gen.jinja`) so **assistant-only loss** works on conversational data; set to `null` to keep the tokenizer's own template.
- `sft`: knobs forwarded to `trl.SFTConfig` (`output_dir`, `learning_rate`, `per_device_train_batch_size`, `max_length`, `assistant_only_loss`, `packing`, eval/save intervals, `bf16`, …).
- `lora`: optional LoRA section for adapter-based SFT (omit for full-parameter).

trackio logging is wired automatically through a custom callback (`pretrain.sft.task.TrackioCallback`): no `report_to` setting is needed.

## Training Features

### Automatic Mixed Precision
Training uses bfloat16 by default for faster computation and lower memory usage.

### Gradient Accumulation
Simulate larger batch sizes without OOM errors:
```yaml
accelerate:
  gradient_accumulation_steps: 4  # Effective batch size = data.train_batch_size * 4
```

### Learning Rate Warmup
Linear warmup scheduler for a stable training start:
```yaml
scheduler:
  warmup_steps: 1000
```

### Periodic Validation
Automatic validation runs during training:
```yaml
validation:
  val_check_interval: 4000  # Run validation every 4000 steps
```

### Dataset Packing
Combines multiple examples into fixed-length sequences to achieve near 100% token utilization (no padding waste). Enable with `use_packed_data: true`.

### LoRA / PEFT
Add a `lora` section to train a LoRA adapter (via HuggingFace `peft`) instead of full-parameter training, for both continued pretraining (`train.yaml`) and SFT (`sft.yaml`). When `lora` is absent (the default), training is full-parameter.

```yaml
lora:
  r: 32
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: ["q_proj", "v_proj"]   # default; null lets peft auto-infer instead
```

For CPT, only the adapter is trained (the base is frozen) and each checkpoint stores the adapter **separately**: `adapter_config.json` + `adapter_model.safetensors`, not a full model copy. The base model is taken from `saved_checkpoint_path`. When using LoRA it's recommended to set `compile: null`.

### Crash-Safe Resume
When a run is interrupted, resuming from a checkpoint that contains a `state/` directory restores the full training state (step, epoch, tokens seen) and the dataloader position, so training continues exactly where it left off.

### Checkpoint Management
Automatically saves the top-K best checkpoints based on validation loss (older/worse ones are pruned), plus a fixed `last/` checkpoint when `save_last: true`. Checkpoints are organized as `checkpoints/<experiment_name>/<run_name>/...`.

### GPU Monitoring
Automatic GPU utilization, memory, and power logging via trackio.

## Evaluation

The framework includes built-in evaluation benchmarks, configurable in the training YAML config (`evaluation` section, `enabled` master switch off by default):

- **HumanEval**: Code generation benchmark
- **IFEVAL**: Instruction-following evaluation
- **MMLU**: Multiple-choice knowledge assessment

Each benchmark supports configurable sample count, temperature, and max generation tokens. Results are saved to `eval_results/`.

## Monitoring

Training metrics are logged using [trackio](https://github.com/gradio-app/trackio), which provides:
- Training loss, learning rate, tokens processed
- Validation loss at regular intervals
- GPU utilization, memory usage, and power consumption
- Web-based dashboard for visualization

Metrics logged:
- `train_loss`: Training loss per batch
- `val_loss`: Validation loss
- `learning_rate`: Current learning rate
- `tokens_passed`: Total tokens processed
- GPU metrics (when `auto_log_gpu: true`)

To open a **write-enabled** dashboard (so you can delete/rename runs from the browser):

```bash
uv run dashboard.py               # all projects
uv run dashboard.py <project>     # open a specific project
```

## Data Format

The project expects tokenized datasets in HuggingFace Datasets format with:
- `input_ids`: Tokenized sequences
- Train/test splits saved by `pretrain-data tokenize` under `tokenized_data/train` and `tokenized_data/test`
- Packed data stored at `tokenized_data/packed_train_data_<max_seq_length>` (e.g. `..._2048`), selected automatically when `use_packed_data: true`

## Optimization Details

Defaults are config-driven; the values below reflect the current `configs/train.yaml`:

- **Optimizer**: AdamW with betas=(0.9, 0.95), weight_decay=0.01, eps=1e-10
- **Scheduler**: Linear warmup (`scheduler.warmup_steps`)
- **Gradient Clipping**: Max norm of 1.0
- **Sequence Length**: Configurable via `data.max_seq_length` (default 2048)

## Reference Runs

Self-contained records of specific pretraining runs (pipeline scripts, the reconstructed config that produced the released checkpoint, and architecture/corpus/hyperparameter details) live under `scripts/<run_name>/`:

- [`scripts/typakos_140m/`](scripts/typakos_140m/README.md): the 140M-parameter bilingual Greek/English base model released as [`alexliap/typakos-140m-base`](https://huggingface.co/alexliap/typakos-140m-base).

## Requirements

- Python **>= 3.12**
- Managed with [`uv`](https://github.com/astral-sh/uv) (`uv.lock` committed). Key dependencies: `accelerate`, `transformers`, `trl`, `peft`, `datasets`, `hydra-core`, `trackio`, `polars`.

## Testing

```bash
uv run pytest
```

The test suite (under `tests/`) is CPU-only and focused on the crash-resume correctness logic: training-state round-trips and deterministic dataloader skip-on-resume.
