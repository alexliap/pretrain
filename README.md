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
│   ├── config.py              # Training configuration dataclass
│   ├── trainer.py             # Training loops and utilities
│   ├── model.py               # Data loading and model definitions
│   ├── checkpoint/            # Checkpoint management (top-K saving)
│   └── evaluation/            # Benchmark evaluation (HumanEval, IFEVAL, MMLU)
├── main.py                    # Entry point for training
├── get_data.py                # Data download script
├── tokenize_data.py           # Tokenization script
├── pack_data.py               # Dataset packing script
├── launch.sh                  # Accelerate launch helper
├── data/                      # Raw data storage
├── tokenized_data/            # Preprocessed tokenized datasets
└── tokenizer/                 # Custom tokenizer files
```

## Configuration

Training is configured via Hydra YAML files. The main config is `configs/train.yaml`, which composes a model preset via defaults:

```yaml
defaults:
  - model: qwen_small  # or qwen_tiny, qwen_medium, qwen_large, qwen_xlarge

tokenizer_path: "tokenizer/"

# Training
learning_rate: 1e-4
batch_size: 16
num_epochs: 1
max_grad_norm: 1.0

# Optimizer
warmup_steps: 500

# Validation
val_check_interval: 1000
val_size: 15000

# Data
max_seq_length: 512
use_packed_data: true

# Accelerate
mixed_precision: "bf16"
gradient_accumulation_steps: 1

# Checkpointing
save_top_k: 3
save_every_n_steps: 500

# Logging
project_name: "scaling-laws"
auto_log_gpu: true
```

You can override any parameter from the command line:
```bash
python main.py batch_size=32 learning_rate=3e-4 model=qwen_medium
```

## Usage

### 1. Prepare Data

Download and prepare your training data:

```bash
python get_data.py
```

This downloads bilingual Greek-English text data from HuggingFace Hub.

### 2. Tokenize Data

Tokenize your dataset:

```bash
python tokenize_data.py
```

### 3. Pack Data

Pack tokenized sequences for efficient training (eliminates padding waste):

```bash
python pack_data.py
```

### 4. Train

Run training with your configuration:

```bash
# Single GPU
python main.py

# Multi-GPU with Accelerate
accelerate launch main.py

# With config overrides
python main.py model=qwen_large learning_rate=3e-4

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
```python
gradient_accumulation_steps=4  # Effective batch_size = 16 * 4 = 64
```

### Learning Rate Warmup
Linear warmup scheduler for stable training start:
```python
warmup_steps=500
```

### Periodic Validation
Automatic validation runs during training:
```python
val_check_interval=1000  # Run validation every 1000 steps
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
- Train/test splits stored in `tokenized_data/train_data/`
- Packed data support for configurable sequence lengths (via `max_seq_length`, default 512)

## Optimization Details

- **Optimizer**: AdamW with beta=(0.9, 0.95), weight_decay=0.1, eps=1e-10
- **Scheduler**: Linear warmup
- **Gradient Clipping**: Max norm of 1.0
- **Sequence Length**: Configurable via `max_seq_length` (default 512)
