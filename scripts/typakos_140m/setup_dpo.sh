#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv -p 3.13
uv sync

# Requires HF auth (run `uv run hf auth login` or set HF_TOKEN) for the model
# download below and the dataset download inside prepare_dpo_data.py.
uv run python scripts/typakos_140m/prepare_dpo_data.py
