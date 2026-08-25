"""ModelDraft's persistent KV cache must draft exactly what the stateless
draft drafts (greedy argmax), across rounds with partial acceptance, full
rejection, full acceptance, trim() rollback, a caller that forgets to
trim, and reset(). Uses a tiny randomly-initialised Qwen3 model built
in-process — no downloads."""

import random

import pytest

mx = pytest.importorskip("mlx.core")
pytest.importorskip("mlx_lm")

from mlx_expert_sniper.speculative import ModelDraft  # noqa: E402

VOCAB = 64


def _tiny_model(seed=0):
    from mlx_lm.models.qwen3 import Model, ModelArgs
    mx.random.seed(seed)
    args = ModelArgs(model_type="qwen3", hidden_size=32, num_hidden_layers=2,
                     intermediate_size=64, num_attention_heads=4,
                     rms_norm_eps=1e-6, vocab_size=VOCAB,
                     num_key_value_heads=2, max_position_embeddings=2048,
                     rope_theta=10000.0, head_dim=8, tie_word_embeddings=False)
    m = Model(args)
    mx.eval(m.parameters())
    return m


def stateless_draft(model, context, k):
    """The pre-#2 ModelDraft.__call__, minus the 512-token window."""
    inp = mx.array([list(context)])
    logits = model(inp)
    out = []
    cur = mx.argmax(logits[:, -1, :], axis=-1)
    for _ in range(k):
        out.append(int(cur.item()))
        logits = model(mx.concatenate([inp, mx.array([out])], axis=1))
        cur = mx.argmax(logits[:, -1, :], axis=-1)
    return out


class _Tok:
    vocab_size = VOCAB


@pytest.fixture
def draft():
    model = _tiny_model()
    return ModelDraft(None, _Tok(), model=model, tokenizer=_Tok())


def _run_rounds(draft, rng, k, rounds, call_trim=True):
    """Simulate spec_generate_stream's loop: draft, accept m, trim, append
    accepted + correction. Returns the number of cached-vs-stateless
    comparisons made."""
    context = [rng.randrange(VOCAB) for _ in range(7)]
    checks = 0
    for r in range(rounds):
        got = draft(context, k)
        want = stateless_draft(draft.model, context, k)
        assert got == want, f"round {r}: cached {got} != stateless {want}"
        checks += 1
        # Cache must hold exactly context + drafts[:-1]
        assert draft._cached == context + got[:-1]
        assert all(c.offset == len(draft._cached) for c in draft.cache)
        m = [0, k, k - 1, rng.randrange(k + 1)][r % 4]
        rejected = k - m
        if rejected and call_trim:
            draft.trim(rejected)
            assert draft._cached == context + got[:m]
        correction = (got[m] + 1) % VOCAB if m < k else rng.randrange(VOCAB)
        context = context + got[:m] + [correction]
    return checks


def test_incremental_cached_draft_matches_stateless_with_trim(draft):
    rng = random.Random(1)
    assert _run_rounds(draft, rng, k=4, rounds=12) == 12


def test_cached_draft_self_heals_when_trim_not_called(draft):
    rng = random.Random(2)
    assert _run_rounds(draft, rng, k=3, rounds=8, call_trim=False) == 8


def test_reset_starts_fresh_generation(draft):
    rng = random.Random(3)
    _run_rounds(draft, rng, k=4, rounds=3)
    assert draft._cached
    draft.reset()
    assert draft._cached == [] and all(c.offset == 0 for c in draft.cache)
    context = [rng.randrange(VOCAB) for _ in range(5)]
    assert draft(context, 4) == stateless_draft(draft.model, context, 4)


def test_unrelated_context_after_cache_is_handled(draft):
    rng = random.Random(4)
    _run_rounds(draft, rng, k=4, rounds=2)
    # A shorter, diverging context without reset(): prefix check drops it
    context = [rng.randrange(VOCAB) for _ in range(3)]
    assert draft(context, 5) == stateless_draft(draft.model, context, 5)
    assert draft._cached[:3] == context
