"""Teacher-forced perplexity evaluation.

Measures downstream quality of the streaming system as actually served:
same forward pass as serve/run (cache-aware routing bias, co-activation
prefetch, expert streaming). Used by the calibration bias sweep and the
`mlx-sniper eval` command.

The default corpus is a held-out public-domain text shipped with the
package, so the eval is fully offline and reproducible.
"""
import math
import os

DEFAULT_TEXT = os.path.join(os.path.dirname(__file__), "data", "eval_text.txt")


def perplexity(engine, bias=0.0, text_path=None, seq_len=512, max_chunks=8,
               verbose=False, mode="prefill"):
    """Teacher-forced perplexity over held-out text.

    mode="prefill": each chunk is one batched prefill forward (fast). Note
    that during prefill the expert cache covers few experts, so a routing
    bias barely engages — this mode measures baseline quality well but can
    UNDERSTATE the impact of a bias.
    mode="decode": token-by-token teacher forcing with a growing KV cache —
    the cache-aware bias engages exactly as it does when serving. ~seq_len
    times slower per chunk; use fewer/shorter chunks.

    Returns exp(mean NLL) over all predicted positions.
    """
    import mlx.core as mx

    is_gemma4 = hasattr(engine, "per_expert_scales")
    if is_gemma4:
        if bias > 0:
            raise ValueError("routing bias is not supported on the Gemma 4 "
                             "engine — eval with bias=0")
        forward = engine.forward
    else:
        from .generate import make_forward
        forward = make_forward(engine, bias=bias)

    with open(text_path or DEFAULT_TEXT, encoding="utf-8") as f:
        text = f.read()
    tokens = engine.tokenizer.encode(text)

    needed = max_chunks * seq_len
    if len(tokens) < 2 * seq_len:
        raise ValueError(f"eval text too short: {len(tokens)} tokens "
                         f"< {2 * seq_len}")
    chunks = [tokens[i:i + seq_len]
              for i in range(0, min(len(tokens), needed), seq_len)]
    chunks = [c for c in chunks if len(c) == seq_len][:max_chunks]

    total_nll = 0.0
    total_count = 0
    for ci, chunk in enumerate(chunks):
        engine.reset_cache()
        if mode == "decode":
            chunk_nll = 0.0
            for t in range(len(chunk) - 1):
                logits = forward(mx.array([[chunk[t]]]))
                x = logits[0, -1].astype(mx.float32)
                lse = mx.logsumexp(x)
                chunk_nll += float(lse - x[chunk[t + 1]])
                del logits
            mx.clear_cache()
        else:
            input_ids = mx.array([chunk])
            logits = forward(input_ids)  # [1, seq_len, vocab]
            x = logits[0, :-1].astype(mx.float32)
            logprobs = x - mx.logsumexp(x, axis=-1, keepdims=True)
            targets = mx.array(chunk[1:]).reshape(-1, 1)
            nll = -mx.take_along_axis(logprobs, targets, axis=-1)
            chunk_nll = float(mx.sum(nll))
            del logits, logprobs, nll
            mx.clear_cache()
        total_nll += chunk_nll
        total_count += len(chunk) - 1
        if verbose:
            print(f"  chunk {ci + 1}/{len(chunks)}: "
                  f"ppl={math.exp(chunk_nll / (len(chunk) - 1)):.2f}")

    return math.exp(total_nll / total_count)


def evaluate_model(model_dir, bias=None, text_path=None, seq_len=512,
                   max_chunks=8):
    """CLI entry: load the engine, run perplexity, print reader stats.

    bias=None uses the calibrated bias (0.0 if uncalibrated).
    """
    from .generate import load_engine

    engine, calibrated_bias, model_type = load_engine(model_dir)
    use_bias = calibrated_bias if bias is None else bias
    print(f"Evaluating {model_type} at bias={use_bias} "
          f"({max_chunks} chunks x {seq_len} tokens)")

    ppl = perplexity(engine, bias=use_bias, text_path=text_path,
                     seq_len=seq_len, max_chunks=max_chunks, verbose=True)

    print(f"\nPerplexity: {ppl:.3f}")
    print(f"\nReader stats:\n  {engine.reader.stats()}")
    engine.reader.close()
    return ppl
