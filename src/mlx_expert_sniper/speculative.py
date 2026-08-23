"""Speculative decoding — verify K drafted tokens in one forward pass.

Why this matters here: both tiers pay a fixed per-forward cost that does
not grow with the number of tokens verified. On the Expert Network the
cost is per-layer round trips (nodes hold experts resident, so the larger
active-expert union of a K-token batch is nearly free); on SSD streaming
it is the read latency the union partially amortizes (the lab measured
K=8 inflating expert bytes 3.34x for a ~2.4x net win). Accepting m drafts
turns one forward into m+1 tokens.

Draft sources:
- prompt-lookup (default): propose the K tokens that followed the longest
  matching suffix n-gram earlier in the context. No second model, no
  tokenizer constraints. Acceptance is text-dependent — high for code,
  lists, and structured text; low for free prose (where it gracefully
  degrades to normal decoding).
- draft model: any tokenizer-compatible smaller model (e.g. Qwen3-0.6B
  drafting for Qwen3-30B). Checked at load: vocab must match.

Greedy acceptance: drafts are accepted while they match the target's
argmax, then the target's own next token is emitted; the KV cache is
trimmed back over rejected positions (KVCache.trim).
"""
import numpy as np


def prompt_lookup_draft(context, k, max_ngram=3):
    """Propose k tokens by matching the context's suffix n-gram earlier in
    the context. Returns [] when nothing matches (normal decode step)."""
    n_ctx = len(context)
    for n in range(max_ngram, 0, -1):
        if n_ctx <= n:
            continue
        suffix = context[-n:]
        # newest match wins
        for start in range(n_ctx - n - 1, -1, -1):
            if context[start:start + n] == suffix:
                cont = context[start + n:start + n + k]
                if cont:
                    return cont
    return []


class ModelDraft:
    """Draft with a smaller tokenizer-compatible model."""

    def __init__(self, path_or_repo, target_tokenizer):
        from mlx_lm import load
        self.model, self.tokenizer = load(path_or_repo)
        if self.tokenizer.vocab_size != target_tokenizer.vocab_size:
            raise ValueError(
                f"draft vocab {self.tokenizer.vocab_size} != target "
                f"{target_tokenizer.vocab_size} — speculation needs a "
                f"tokenizer-compatible draft model")

    def __call__(self, context, k, max_ngram=None):
        import mlx.core as mx
        # Stateless per call: re-prefill the tail of the context. Cheap for
        # a sub-1B draft; a persistent draft KV cache is future work.
        inp = mx.array([context[-512:]])
        logits = self.model(inp)
        out = []
        cur = mx.argmax(logits[:, -1, :], axis=-1)
        for _ in range(k):
            out.append(int(cur.item()))
            logits = self.model(mx.concatenate([inp, mx.array([out])], axis=1))
            cur = mx.argmax(logits[:, -1, :], axis=-1)
        return out


def spec_generate_stream(engine, messages, bias=0.0, max_tokens=200, k=8,
                         draft=None, stats=None):
    """Generator yielding token strings, speculative version of
    generate_stream. `draft` is a callable (context_tokens, k) -> [token]
    (default: prompt-lookup). `stats` (optional dict) receives acceptance
    counters."""
    import mlx.core as mx
    from .generate import make_forward, eos_token_ids, STOP_TOKENS

    if hasattr(engine, "per_expert_scales"):
        raise ValueError("speculative decoding is not wired for the Gemma 4 "
                         "engine yet")
    draft = draft or prompt_lookup_draft
    if stats is None:
        stats = {}
    stats.setdefault("drafted", 0)
    stats.setdefault("accepted", 0)
    stats.setdefault("forwards", 0)

    engine.reset_cache()
    tok = engine.tokenizer
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                       add_generation_prompt=True,
                                       enable_thinking=False)
    except Exception:
        try:
            text = tok.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True)
        except Exception:
            text = messages[-1]["content"]
    context = list(tok.encode(text))
    eos_ids = eos_token_ids(tok)

    forward = make_forward(engine, bias=bias)

    # Prefill everything except the last prompt token; `cur` stays out of
    # the KV cache so each verify step feeds [cur, d1..dk].
    if len(context) > 1:
        logits = forward(mx.array([context[:-1]]))
        mx.eval(logits)
        stats["forwards"] += 1
    cur = context[-1]

    emitted = 0
    while emitted < max_tokens:
        drafts = draft(context, k)
        stats["drafted"] += len(drafts)

        inp = mx.array([[cur] + drafts])
        logits = forward(inp)          # [1, 1+len(drafts), vocab]
        mx.eval(logits)
        stats["forwards"] += 1
        preds = [int(t) for t in
                 np.array(mx.argmax(logits[0], axis=-1))]

        # Accept drafts while they match the target's own predictions
        m = 0
        while m < len(drafts) and drafts[m] == preds[m]:
            m += 1
        stats["accepted"] += m

        # Roll the KV cache back over rejected draft positions
        rejected = len(drafts) - m
        if rejected:
            for c in engine.cache:
                if c is not None:
                    c.trim(rejected)

        out_tokens = drafts[:m] + [preds[m]]
        stop = False
        for i, t in enumerate(out_tokens):
            if t in eos_ids:
                stop = True
                break
            chunk = tok.decode([t])
            if any(st in chunk for st in STOP_TOKENS):
                stop = True
                break
            yield chunk
            emitted += 1
            context.append(t)
            if emitted >= max_tokens:
                stop = True
                i += 1
                break
        if stop:
            break

        # KV now holds [.., cur, d1..dm]; the target's correction token
        # preds[m] is the new `cur`, not yet in the cache.
        cur = out_tokens[-1]
