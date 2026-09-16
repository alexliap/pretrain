"""score_loglikelihoods must correctly isolate and sum continuation-token
log-probs, excluding context tokens - verified against a fake model with a
fixed, target-independent next-token rule so a "predictable" continuation and
an "unpredictable" one score differently (a model that just echoed back
whichever token it's asked to score would make every continuation look
equally likely and couldn't catch a scoring bug)."""

import torch
import torch.nn as nn

from pretrain.evaluation.scoring import accuracy_from_choices, score_loglikelihoods


class _FakeTokenizer:
    """Maps each character to its ord() as a token id; no special tokens."""

    pad_token_id = 0
    eos_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


class _FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class _FakeModel(nn.Module):
    """Deterministic next-token predictor: whenever the current token is
    ord(c), it confidently predicts ord(c) + 1 as the next token (e.g. 'a' ->
    'b', 'b' -> 'c'), regardless of what the actual next token in the
    sequence is. This makes "ab" -> "b" ... continuations that follow the
    rule score near-zero log-likelihood, and continuations that break the
    rule score very negative - exactly the discrimination needed to verify
    the scorer, unlike a fake model that always "predicts" the true next
    token (which can't distinguish a right choice from a wrong one)."""

    def __init__(self, vocab_size=128):
        super().__init__()
        self.vocab_size = vocab_size
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, attention_mask):
        batch, seq_len = input_ids.shape
        logits = torch.full((batch, seq_len, self.vocab_size), -10.0)
        for b in range(batch):
            for t in range(seq_len - 1):
                predicted_next = input_ids[b, t].item() + 1
                if 0 <= predicted_next < self.vocab_size:
                    logits[b, t, predicted_next] = 10.0
        return _FakeOutput(logits)


def test_score_loglikelihoods_scores_only_continuation_tokens():
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    # Context "za": 'z' -> 'a' breaks the model's rule (predicts '{'), so if
    # the scorer wrongly included context-token predictions in the sum, this
    # would come out very negative. The continuation-only score should stay
    # near zero since 'a' -> 'b' and 'b' -> 'c' both follow the rule.
    results = score_loglikelihoods(model, tokenizer, [("za", "b"), ("za", "bc")])
    assert results[0].sum_logprob > -0.01
    assert results[0].num_tokens == 1
    assert results[1].sum_logprob > -0.01
    assert results[1].num_tokens == 2


def test_accuracy_from_choices_picks_gold_when_matching_continuation_scores_higher():
    model = _FakeModel()
    tokenizer = _FakeTokenizer()
    # Context "za" ends in 'a', which the rule predicts is followed by 'b':
    # choice "b" follows the rule (near-zero log-likelihood), choice "q"
    # breaks it (very negative), so gold=0 ("b") should win.
    examples = [{"context": "za", "choices": ["b", "q"], "gold": 0}]
    metrics = accuracy_from_choices(model, tokenizer, examples, batch_size=2)
    assert metrics["acc"] == 1.0
    assert metrics["acc_norm"] == 1.0
