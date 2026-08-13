#!/usr/bin/env bash
# Bootstrap a dev environment: install uv, create a 3.13 venv, sync deps, and
# pull down the SFT checkpoint DPO starts from. Add the preference-dataset
# download once it's prepared and pushed to the Hub (see
# scripts/typakos_140m/configs/dpo.yaml's dataset.dataset_id).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv -p 3.13
uv sync

uv run hf download alexliap/typakos_140m_it \
    --repo-type model \
    --local-dir models/typakos_140m_it

# uv run hf download <your-preference-dataset-repo> \
#     --repo-type dataset \
#     --local-dir data/typakos_dpo_dataset
