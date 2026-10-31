"""Vendored IFEval instruction-checking logic.

Copied from lm-evaluation-harness's `lm_eval/tasks/ifeval/` (Apache License
2.0), which itself vendors Google Research's original IFEval checker. Kept
as-is (only the cross-module imports were rewritten to this package's path)
so scoring stays comparable to eval_results/typakos-140m-it-dpo's offline
lm-eval benchmark run, which used the same reference implementation.

Source: https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/ifeval
"""
