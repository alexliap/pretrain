#!/usr/bin/env bash
# Upload a checkpoint directory to a private HuggingFace Hub model repo.
#
# Usage:
#   ./upload_checkpoint.sh <checkpoint-dir> <repo-id> [revision]
#
# Example:
#   ./upload_checkpoint.sh \
#     checkpoints/llama_80m_pretraining/llama_80m_pretraining_v1/last \
#     alexliap/llama_80m_pretraining
#
# Requires the `hf` CLI (huggingface_hub, via `uv run`) and either a prior
# `hf auth login` or an HF_TOKEN env var with write access.

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 <checkpoint-dir> <repo-id> [revision]" >&2
    exit 1
fi

CHECKPOINT_DIR="$1"
REPO_ID="$2"
REVISION="${3:-main}"

if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "Checkpoint directory not found: ${CHECKPOINT_DIR}" >&2
    exit 1
fi

uv run hf upload "${REPO_ID}" "${CHECKPOINT_DIR}" . \
    --repo-type model \
    --private \
    --revision "${REVISION}" \
    --commit-message "Upload checkpoint from ${CHECKPOINT_DIR}"
