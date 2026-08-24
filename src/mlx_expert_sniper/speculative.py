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


class RemoteDraft:
    """Draft over the network: any OpenAI-compatible /v1/completions
    endpoint proposes the K tokens — a FreeToken box, a DGX Spark, vLLM,
    or mlx_lm.server. This is the heterogeneous split the network is
    built for: fast dense hardware proposes, the expert mesh verifies a
    whole batch in one forward.

    The draft model MUST share the target's tokenizer family. Drafting is
    text-level (decode tail -> complete -> re-encode), which sidesteps
    protocol differences; the re-encode is anchored on the decoded tail so
    boundary tokens resolve consistently under the target tokenizer.
    """

    def __init__(self, base_url, model, tokenizer, tail_tokens=512,
                 timeout=20, api_key=None):
        import os
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("DRAFT_API_KEY")
        self.tokenizer = tokenizer
        self.tail_tokens = tail_tokens
        self.timeout = timeout
        self.requests = 0
        self.failures = 0

    def __call__(self, context, k, max_ngram=None):
        import json as _json
        import urllib.request
        tail = self.tokenizer.decode(context[-self.tail_tokens:])
        payload = {"prompt": tail, "max_tokens": k + 4, "temperature": 0}
        if self.model:  # some servers 404 on unknown ids; omit when unset
            payload["model"] = self.model
        body = _json.dumps(payload).encode()
        headers = {"Content-Type": "application/json",
                   "User-Agent": "expert-sniper-draft/0.3"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(
            f"{self.base_url}/completions", data=body, headers=headers)
        self.requests += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                cont = _json.loads(r.read())["choices"][0].get("text", "")
        except Exception:
            self.failures += 1
            return []  # graceful: this step decodes normally
        if not cont:
            return []
        base_len = len(self.tokenizer.encode(tail))
        toks = self.tokenizer.encode(tail + cont)
        return toks[base_len:base_len + k]


class DFlashDraft:
    """Block-diffusion drafting (z-lab DFlash) inside the streaming engine.

    The draft head is conditioned on the TARGET's hidden states at a few
    layers, so it must live inside the target's forward: make_forward
    captures the residual stream after each listed layer (all positions);
    this adapter projects them through the head's `fc`, denoises a block of
    [last_token, mask...] against that context in ONE draft pass, and reads
    the proposals out through the target's own final norm + lm_head. One
    pass drafts up to block_size-1 tokens — versus one draft forward per
    token for an autoregressive draft model.

    Requires `dflash-mlx` (pip) and an official head for the target
    (e.g. z-lab/Qwen3-Coder-30B-A3B-DFlash). v1: no draft KV cache —
    context is re-projected each step (fine for chat-length contexts).
    """

    def __init__(self, head_path, engine):
        import json, os
        import mlx.core as mx
        from mlx_lm.utils import load_model
        from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs
        path = self._resolve(head_path)
        cfg_path = os.path.join(path, "config.json")
        cfg = json.load(open(cfg_path))
        # Older z-lab configs predate the mlx port's schema
        changed = False
        if "rope_theta" not in cfg:
            cfg["rope_theta"] = (cfg.get("rope_parameters") or {}).get("rope_theta", 1e7); changed = True
        if "block_size" not in cfg:
            cfg["block_size"] = (cfg.get("dflash_config") or {}).get("block_size", 16); changed = True
        if changed:
            json.dump(cfg, open(cfg_path, "w"), indent=2)
        self.model, _ = load_model(
            path, get_model_classes=lambda c: (DFlashDraftModel, DFlashDraftModelArgs))
        self.engine = engine
        self.layer_ids = list(self.model.target_layer_ids)
        self.block = int(self.model.block_size)
        self.mask_id = int(self.model.mask_token_id)
        if engine.num_layers <= max(self.layer_ids):
            raise ValueError(f"head expects target layers {self.layer_ids}, "
                             f"target has {engine.num_layers}")
        engine.capture_layers = set(self.layer_ids)
        engine.captured = {}
        self.mx = mx

    @staticmethod
    def _resolve(ref):
        import os
        p = os.path.expanduser(ref)
        if os.path.isdir(p):
            return p
        from huggingface_hub import snapshot_download
        return snapshot_download(ref)

    def trim(self, n):
        """Roll captured context back over n rejected positions."""
        if n <= 0:
            return
        for k, v in list(self.engine.captured.items()):
            self.engine.captured[k] = v[:, :-n, :]

    def __call__(self, context, k, max_ngram=None):
        mx = self.mx
        cap = self.engine.captured
        if any((i + 1) not in cap for i in self.layer_ids):
            return []
        feats = mx.concatenate([cap[i + 1] for i in self.layer_ids], axis=-1)
        draft_context = self.model.project_target_hidden(feats)
        block_ids = mx.array([[context[-1]] + [self.mask_id] * (self.block - 1)],
                             dtype=mx.uint32)
        tm = self.engine.model
        noise = tm.model.embed_tokens(block_ids) * self.model.embed_scale
        noise = noise.astype(draft_context.dtype)
        draft_hidden = self.model.forward_projected_context(
            noise_embedding=noise, draft_context=draft_context, cache=None)
        logits = tm.lm_head(tm.model.norm(draft_hidden[:, 1:, :]))
        toks = mx.argmax(logits[0], axis=-1)
        mx.eval(toks)
        return [int(t) for t in toks.tolist()][:k]


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
    if hasattr(engine, "captured"):
        engine.captured = {}
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
            if hasattr(draft, "trim"):
                draft.trim(rejected)

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
