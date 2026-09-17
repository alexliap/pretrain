#!/usr/bin/env bash
# Launch the typakos_140m pretraining run from the repo root, against this
# dir's own config snapshot rather than the root configs/train.yaml (which has
# since drifted onto a different experiment - see README.md).
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export NCCL_P2P_DISABLE=1
uv run accelerate launch main.py -cp scripts/typakos_140m/configs -cn train
