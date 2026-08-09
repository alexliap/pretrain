#!/usr/bin/env bash
# Launch the typakos_140m SFT run from the repo root, against this dir's own
# config (scripts/typakos_140m/configs/sft.yaml) rather than the root
# configs/sft.yaml, which is a different, unrelated experiment.
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

uv run accelerate launch sft.py -cp scripts/typakos_140m/configs -cn sft
