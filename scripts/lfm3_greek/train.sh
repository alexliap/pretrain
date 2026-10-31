#!/usr/bin/env bash
# Launch LFM3-1.7B CPT on greek-cc from the repo root. A full run keeps several
# hundred GB in the HF datasets cache, so point HF_DATASETS_CACHE at a large disk.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export NCCL_P2P_DISABLE=1
uv run accelerate launch scripts/lfm3_greek/cpt.py "$@"
