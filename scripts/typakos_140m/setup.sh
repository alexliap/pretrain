#!/usr/bin/env bash
# Bootstrap a dev environment: install uv, create a 3.13 venv, sync deps, and
# pull down the bilingual tokenizer and the 10B-token dataset from the Hub.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv -p 3.13
uv sync

# Requires HF auth (run `uv run hf auth login` or set HF_TOKEN) for the downloads below.
uv run hf download alexliap/bilingual_el_en_50k \
    --repo-type model \
    --local-dir models/bilingual_el_en_50k

uv run hf sync hf://buckets/alexliap/el_en_10B_tokens ./tokenized_data
