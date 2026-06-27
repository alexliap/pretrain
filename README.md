# Pretrain Experiment

A lightweight language model pretraining framework using vanilla PyTorch and HuggingFace Accelerate for distributed training.

## Features

- **Vanilla PyTorch**: Pure PyTorch implementation without heavy frameworks
- **HuggingFace Accelerate**: Seamless distributed training support (multi-GPU, mixed precision)
- **Hydra Configuration**: YAML-based configuration with composable model presets
- **Mixed Precision Training**: Built-in bf16 support for faster training
- **Gradient Accumulation**: Efficient training with large effective batch sizes
- **Dataset Packing**: Efficient sequence packing to maximize token utilization
- **Automatic Logging**: GPU metrics and training stats via trackio
- **Custom Tokenizers**: Support for custom tokenizer training and usage
- **Checkpoint Management**: Top-K checkpoint saving based on validation loss
- **Evaluation Benchmarks**: Built-in support for HumanEval, IFEVAL, and MMLU

## Architecture

The model configuration currently only supports the **Qwen3** LLM architecture (for now). The framework provides predefined model size presets under `configs/model/`:

| Preset | Layers | Hidden Size | FFN Size | Head Dim | Heads |
|--------|--------|-------------|----------|----------|-------|
| `qwen_tiny` | 4 | 768 | 1024 | 64 | 4 |
| `qwen_small` | 8 | 768 | 1024 | 64 | 8 |
| `qwen_medium` | 15 | 768 | 1024 | 64 | 8 |
| `qwen_large` | 20 | 768 | 1024 | 128 | 8 |
| `qwen_xlarge` | 20 | 768 | 1024 | 64 | 8 |

All presets use `Qwen/Qwen3-0.6B` as the base model config reference and support customizable vocabulary size via custom tokenizers.

## Project Structure

```
pretrain/
├── configs/
│   ├── train.yaml             # Main training config (Hydra)
│   └── model/                 # Model size presets
│       ├── qwen_tiny.yaml
│       ├── qwen_small.yaml
│       ├── qwen_medium.yaml
│       ├── qwen_large.yaml
│       └── qwen_xlarge.yaml
├── src/pretrain/
│   ├── config.py              # Training configuration dataclasses (nested sections)
│   ├── task.py                # PretrainTask: end-to-end training workflow + orchestrator
│   ├── dataloader.py          # Dataset loading and packed/padded collation
│   ├── cli.py                 # `pretrain-data` CLI (download, tokenize)
│   ├── data/                  # Data download & tokenization helpers
│   ├── checkpoint/            # Checkpoint management (top-K + last)
│   └── evaluation/            # Benchmark evaluation (HumanEval, IFEVAL, MMLU)
├── main.py                    # Entry point for training (Hydra)
├── get_data.py                # Download raw parquet data from the Hub
├── concat_data.py             # Concatenate raw files into a single parquet
├── pack_data.py               # Dataset packing script
├── dashboard.py               # Launch the trackio dashboard (write access)
├── launch.sh / train.sh       # Accelerate launch helpers
├── data/                      # Raw data storage
└── tokenized_data/            # Tokenized and packed datasets
```

## Configuration

Training is configured via Hydra YAML files. The main config is `configs/train.yaml`, which composes a model preset via defaults:

Settings are grouped into per-concern sections (`data`, `optimizer`, `scheduler`,
`accelerate`, `validation`, `logging`, `checkpoint`, `evaluation`), with run-level
fields kept at the top level:

```yaml
defaults:
  - model: qwen_small  # or qwen_tiny, qwen_medium, qwen_large, qwen_xlarge

# Run-level (top-level)
num_epochs: 1
total_steps: null
total_tokens: null

# Data
data:
  tokenizer_path: "tokenizer/"
  max_seq_length: 512
  use_packed_data: true
  batch_size: 16

# Optimizer (incl. gradient clipping)
optimizer:
  learning_rate: 1e-4
  max_grad_norm: 1.0

# Scheduler
scheduler:
  warmup_steps: 500

# Accelerate
accelerate:
  mixed_precision: "bf16"
  gradient_accumulation_steps: 1

# Validation
validation:
  val_check_interval: 1000
  val_size: 15000

# Checkpointing
checkpoint:
  save_top_k: 3
  save_every_n_steps: 500

# Logging
logging:
  project_name: "scaling-laws"
  auto_log_gpu: true
```

You can override any parameter from the command line using its section path:
```bash
python main.py data.batch_size=32 optimizer.learning_rate=3e-4 model=qwen_medium
```

## Usage

### 1. Download Data

Download bilingual Greek-English text data from the HuggingFace Hub:

```bash
python get_data.py
```

Alternatively, fetch any Hub dataset with the CLI: `uv run pretrain-data download <repo_id> <local_dir>`.

### 2. Consolidate Data

Concatenate the downloaded files into a single parquet file (`data/concat_dataset/data.parquet`):

```bash
python concat_data.py
```

### 3. Tokenize Data

Tokenize the parquet dataset into train/test splits using the `pretrain-data` CLI:

```bash
uv run pretrain-data tokenize \
  --tokenizer-repo-id <tokenizer-id-or-path> \
  --data-path data/concat_dataset/data.parquet \
  --output-path tokenized_data/
```

Use `--test-size` to change the held-out fraction (default `0.1`).

### 4. Pack Data

Pack tokenized sequences into fixed-length blocks for efficient training (eliminates padding waste):

```bash
python pack_data.py --max_seq_length 2048 \
  --input_dir tokenized_data/train \
  --output_dir tokenized_data/packed_train_data_2048
```

### 5. Train

Run training with your configuration:

```bash
# Single GPU
python main.py

# Multi-GPU with Accelerate
accelerate launch main.py

# With config overrides
python main.py model=qwen_large optimizer.learning_rate=3e-4

# Configure accelerate (first time)
accelerate config
```

### Distributed Training

To launch distributed training across multiple GPUs:

```bash
# Launch on all available GPUs
accelerate launch --multi_gpu main.py

# Launch on specific number of GPUs
accelerate launch --num_processes 4 main.py
```

## Training Features

### Automatic Mixed Precision
Training uses bfloat16 by default for faster computation and lower memory usage.

### Gradient Accumulation
Simulate larger batch sizes without OOM errors:
```yaml
accelerate:
  gradient_accumulation_steps: 4  # Effective batch size = data.batch_size * 4
```

### Learning Rate Warmup
Linear warmup scheduler for stable training start:
```yaml
scheduler:
  warmup_steps: 500
```

### Periodic Validation
Automatic validation runs during training:
```yaml
validation:
  val_check_interval: 1000  # Run validation every 1000 steps
```

### Dataset Packing
Combines multiple examples into fixed-length sequences to achieve near 100% token utilization (no padding waste). Enable with `use_packed_data: true`.

### Checkpoint Management
Automatically saves the top-K best checkpoints based on validation loss. Older/worse checkpoints are removed to save disk space.

### GPU Monitoring
Automatic GPU utilization, memory, and power logging via trackio.

## Evaluation

The framework includes built-in evaluation benchmarks, configurable in the YAML config:

- **HumanEval**: Code generation benchmark
- **IFEVAL**: Instruction-following evaluation
- **MMLU**: Multiple-choice knowledge assessment

Each benchmark supports configurable sample count, temperature, and max generation tokens. Results are saved to `eval_results/`.

## Monitoring

Training metrics are logged using [trackio](https://github.com/alexanderthebaptist/trackio), which provides:
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

## Data Format

The project expects tokenized datasets in HuggingFace Datasets format with:
- `input_ids`: Tokenized sequences
- Train/test splits saved by `pretrain-data tokenize` under `tokenized_data/train` and `tokenized_data/test`
- Packed data stored at `tokenized_data/packed_train_data_<max_seq_length>` (e.g. `..._2048`), selected automatically when `use_packed_data: true`

## Optimization Details

- **Optimizer**: AdamW with beta=(0.9, 0.95), weight_decay=0.1, eps=1e-10
- **Scheduler**: Linear warmup
- **Gradient Clipping**: Max norm of 1.0
- **Sequence Length**: Configurable via `data.max_seq_length` (default 2048)
