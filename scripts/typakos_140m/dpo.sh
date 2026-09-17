#!/usr/bin/env bash
# Launch the typakos_140m DPO run from the repo root, against this dir's own
# config (scripts/typakos_140m/configs/dpo.yaml) rather than the root
# configs/dpo.yaml, which is a generic template.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

export NCCL_P2P_DISABLE=1
uv run accelerate launch dpo.py -cp scripts/typakos_140m/configs -cn dpo
