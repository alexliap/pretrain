"""Shared log-likelihood scoring for multiple-choice / cloze-style benchmarks.

Used by MMLU and HellaSwag (and any future benchmark with the same shape:
one context string + N candidate continuations, gold = correct index) so the
teacher-forced scoring math lives in exactly one place.
"""

from dataclasses import dataclass

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer


@dataclass
class LoglikelihoodResult:
    sum_logprob: float
    num_tokens: int

    @property
    def normalized(self) -> float:
        """Length-normalized log-likelihood, used for acc_norm."""
        return self.sum_logprob / max(self.num_tokens, 1)


def score_loglikelihoods(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    pairs: list[tuple[str, str]],
) -> list[LoglikelihoodResult]:
    """Score the teacher-forced log-likelihood of each (context, continuation) pair.

    Tokenizes context + continuation for every pair, right-pads them into one
    batch, runs a single forward pass, and sums each pair's
    continuation-token log-probs (context tokens excluded from the sum - the
    standard loglikelihood-classification scoring lm-evaluation-harness uses
    for multiple-choice tasks). Every pair in `pairs` is scored in one
    forward pass; callers control batch size by chunking `pairs` themselves
    (see score_choices_batched).

    Assumes each `context` is non-empty (so there is always at least one
    preceding token to condition the first continuation token on).
    """
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    full_ids_list = []
    context_lens = []
    for context, continuation in pairs:
        context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(context + continuation, add_special_tokens=False)[
            "input_ids"
        ]
        full_ids_list.append(full_ids)
        context_lens.append(len(context_ids))

    max_len = max(len(ids) for ids in full_ids_list)
    input_ids = torch.full((len(pairs), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(pairs), max_len), dtype=torch.long)
    for i, ids in enumerate(full_ids_list):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    log_probs = logits.log_softmax(dim=-1)

    results = []
    for i, (full_ids, ctx_len) in enumerate(zip(full_ids_list, context_lens)):
        seq_len = len(full_ids)
        cont_len = seq_len - ctx_len
        if cont_len <= 0:
            results.append(
                LoglikelihoodResult(sum_logprob=float("-inf"), num_tokens=0)
            )
            continue
        # log_probs[i, t] is the distribution predicting the token at t+1, so
        # the continuation token at position p (ctx_len <= p < seq_len) is
        # scored by log_probs[i, p - 1].
        token_log_probs = log_probs[i, ctx_len - 1 : seq_len - 1, :]
        target_ids = input_ids[i, ctx_len:seq_len]
        picked = token_log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        results.append(
            LoglikelihoodResult(sum_logprob=picked.sum().item(), num_tokens=cont_len)
        )
    return results


def score_choices_batched(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: list[tuple[str, list[str]]],
    batch_size: int,
) -> list[list[LoglikelihoodResult]]:
    """Score every example's choices, chunking `batch_size` EXAMPLES per
    forward pass (so each forward pass contains up to
    `batch_size * len(choices)` sequences - this is what makes
    EvaluationTaskConfig.batch_size actually control memory/throughput).
    Returns one list of LoglikelihoodResult per example, in the same example
    and choice order as the input.
    """
    all_results: list[list[LoglikelihoodResult]] = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        pairs = [(context, choice) for context, choices in chunk for choice in choices]
        flat_results = score_loglikelihoods(model, tokenizer, pairs)
        offset = 0
        for _, choices in chunk:
            n = len(choices)
            all_results.append(flat_results[offset : offset + n])
            offset += n
    return all_results


def accuracy_from_choices(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    examples: list[dict],
    batch_size: int,
) -> dict[str, float]:
    """Shared acc/acc_norm loop for 4-way (or N-way) multiple-choice tasks.

    Each example must have "context": str, "choices": list[str], and "gold":
    int (index into choices). Returns {"acc": ..., "acc_norm": ...}.
    """
    pairs = [(ex["context"], ex["choices"]) for ex in examples]
    scored = score_choices_batched(model, tokenizer, pairs, max(batch_size, 1))

    correct = correct_norm = 0
    for example, results in zip(examples, scored):
        pred = max(range(len(results)), key=lambda i: results[i].sum_logprob)
        pred_norm = max(range(len(results)), key=lambda i: results[i].normalized)
        correct += int(pred == example["gold"])
        correct_norm += int(pred_norm == example["gold"])

    n = len(examples) or 1
    return {"acc": correct / n, "acc_norm": correct_norm / n}
