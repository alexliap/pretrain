# Pretrain Experiment

A lightweight language model pretraining framework using vanilla PyTorch and HuggingFace Accelerate for distributed training.

## Features

- **Vanilla PyTorch**: Pure PyTorch implementation without heavy frameworks
- **HuggingFace Accelerate**: Seamless distributed training support (multi-GPU, mixed precision)
- **Flexible Configuration**: Dataclass-based configuration for easy experimentation
- **Mixed Precision Training**: Built-in bf16 support for faster training
- **Gradient Accumulation**: Efficient training with large effective batch sizes
- **Automatic Logging**: GPU metrics and training stats via trackio
- **Custom Tokenizers**: Support for custom tokenizer training and usage
- **Validation Checkpoints**: Periodic validation during training

## Architecture

The project uses a modified Qwen3-0.6B architecture with customizable:
- Hidden size
- Intermediate (FFN) size
- Vocabulary size
- Custom tokenizers

## Project Structure

```
pretrain_exp/
├── src/pretrain/
│   ├── config.py          # Training configuration dataclass
│   ├── trainer.py         # Training loops and utilities
│   └── model.py           # Data loading and model definitions
├── main.py                # Entry point for training
├── get_data.py            # Data download script
├── tokenize_data.py       # Tokenization script
├── data/                  # Raw data storage
├── tokenized_data/        # Preprocessed tokenized datasets
└── tokenizer/       # Custom tokenizer files
```

## Configuration

Training is configured via the `TrainingConfig` dataclass in [config.py](src/pretrain/config.py):

```python
config = TrainingConfig(
    # Model
    tokenizer_path="path/to/tokenizer",
    base_model="Qwen/Qwen3-0.6B",
    hidden_size=128,
    intermediate_size=1024,

    # Training
    learning_rate=1e-4,
    batch_size=16,
    num_epochs=1,
    gradient_accumulation_steps=1,
    max_grad_norm=1.0,

    # Optimization
    warmup_steps=200,
    mixed_precision="bf16",

    # Validation
    val_check_interval=500,
    val_size=5000,

    # Logging
    project_name="my-experiment",
    auto_log_gpu=True,
)
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

### 3. Train

Run training with your configuration:

```bash
# Single GPU
python main.py

# Multi-GPU with Accelerate
accelerate launch main.py

# Configure accelerate (first time)
accelerate config
```

### Distributed Training

To launch distributed training across multiple GPUs:

```bash
# Launch on all available GPUs
accelerate launch --multi_gpu main.py

# Launch on specific GPUs
accelerate launch --num_processes 4 main.py
```

Configure your accelerate setup:
```bash
accelerate config
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
warmup_steps=200
```

### Periodic Validation
Automatic validation runs during training:
```python
val_check_interval=500  # Run validation every 500 steps
```

### GPU Monitoring
Automatic GPU utilization, memory, and power logging via trackio.

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
- GPU metrics (when `auto_log_gpu=True`)

## Data Format

The project expects tokenized datasets in HuggingFace Datasets format with:
- `input_ids`: Tokenized sequences (automatically padded to 512 tokens)
- Train/test splits stored in `tokenized_data/train_data/`

## Optimization Details

- **Optimizer**: AdamW with β=(0.9, 0.95), weight_decay=0.1, eps=1e-10
- **Scheduler**: Linear warmup
- **Gradient Clipping**: Max norm of 1.0
- **Sequence Length**: Up to 512 tokens (with dynamic padding)
