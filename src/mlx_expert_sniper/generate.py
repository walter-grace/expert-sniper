"""Shared generation logic for run, chat, and serve."""
import os, sys, time, json
import numpy as np

STOP_TOKENS = {"<|im_end|>", "<|endoftext|>", "<|im_start|>"}


def load_engine(model_dir):
    """Load engine with calibration. Returns (engine, bias, model_type)."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import mlx.core as mx
    from .calibrate import load_calibration, auto_size_cache, _detect_model_type

    cal = load_calibration(model_dir)
    if cal:
        cache_size = cal["cache_size"]
        bias = cal["routing_bias"]
    else:
        cache_size, _, _ = auto_size_cache(model_dir)
        bias = 0.0

    model_type = _detect_model_type(model_dir)
    if "gemma4" in model_type:
        from . import engine_gemma4 as engine_mod
        engine_mod.MODEL_DIR = model_dir
        from .engine_gemma4 import MoESniperEngineGemma4 as EngineClass
    elif "qwen3_next" in model_type:
        from . import engine_next as engine_mod
        engine_mod.MODEL_DIR = model_dir
        from .engine_next import MoESniperEngineNext as EngineClass
    elif "qwen3_5" in model_type:
        from . import engine as engine_mod
        engine_mod.MODEL_DIR = model_dir
        from .engine import MoESniperEngine35B as EngineClass
    else:
        from . import engine_30b as engine_mod
        engine_mod.MODEL_DIR = model_dir
        from .engine_30b import MoESniperEngine30B as EngineClass

    eng = EngineClass(cache_size=cache_size, enable_prediction=True)
    eng.load()
    return eng, bias, model_type


def generate_stream(engine, messages, bias=0.0, max_tokens=200):
    """Generator yielding token strings. Handles Qwen + Gemma 4 architectures."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask

    # Detect model type — Gemma 4 uses its own forward pass
    is_gemma4 = hasattr(engine, 'per_expert_scales')  # Gemma 4 engine has this
    if is_gemma4:
        return _generate_stream_gemma4(engine, messages, max_tokens)

    from .engine import run_expert_ffn
    has_ssm = hasattr(engine.model.model, 'fa_idx')
    num_experts = 256 if has_ssm else 128

    engine.reset_cache()
    tok = engine.tokenizer
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    except Exception:
        try:
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            text = messages[-1]["content"]
    tokens = tok.encode(text)
    input_ids = mx.array([tokens])

    def forward(inp):
        h = engine.model.model.embed_tokens(inp)
        if has_ssm:
            from mlx_lm.models.base import create_ssm_mask
            fa_mask = create_attention_mask(h, engine.cache[engine.model.model.fa_idx])
            ssm_mask = create_ssm_mask(h, engine.cache[engine.model.model.ssm_idx])
        else:
            fa_mask = create_attention_mask(h, engine.cache[0])
            ssm_mask = None

        for i in range(engine.num_layers):
            layer = engine.model.model.layers[i]
            if has_ssm:
                mask = ssm_mask if layer.is_linear else fa_mask
            else:
                mask = fa_mask
            normed = layer.input_layernorm(h)
            if has_ssm and layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=engine.cache[i])
            else:
                attn_out = layer.self_attn(normed, mask=mask, cache=engine.cache[i])
            h = h + attn_out
            mx.eval(h)

            normed = layer.post_attention_layernorm(h)
            raw_logits = layer.mlp.gate(normed)
            if bias > 0 and engine.reader.lru is not None:
                cached_mask = np.zeros(num_experts, dtype=np.float32)
                for eid in range(num_experts):
                    if engine.reader.lru.get(i, eid) is not None:
                        cached_mask[eid] = bias
                raw_logits = raw_logits + mx.array(cached_mask).reshape(1, -1)

            gates = mx.softmax(raw_logits, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)

            active_ids = list(set(int(e) for e in np.array(inds).flatten()))
            engine.coact.record_layer(i, active_ids)
            if engine.coact.ready and i + 1 < engine.num_layers:
                predicted = engine.coact.predict_next_layer(i, active_ids, top_k=6)
                if predicted:
                    to_fetch = [eid for eid in predicted
                                if engine.reader.lru and engine.reader.lru.get(i+1, eid) is None]
                    if to_fetch:
                        engine.reader.prefetch_experts(i+1, to_fetch)
            if i + 1 < engine.num_layers:
                engine.reader.prefetch_experts(i+1, active_ids)

            expert_data = engine.reader.get_experts(i, active_ids)
            expert_out = run_expert_ffn(normed, expert_data, inds, scores)

            if hasattr(layer.mlp, 'shared_expert'):
                shared_out = layer.mlp.shared_expert(normed)
                shared_gate = mx.sigmoid(layer.mlp.shared_expert_gate(normed))
                if shared_gate.ndim < shared_out.ndim:
                    shared_gate = shared_gate[..., None]
                expert_out = expert_out + shared_gate * shared_out

            h = h + expert_out
            mx.eval(h)
            del expert_data, expert_out, normed, attn_out
            mx.clear_cache()

        engine.coact.end_token()
        h = engine.model.model.norm(h)
        return engine.model.lm_head(h)

    logits = forward(input_ids)
    mx.eval(logits)

    eos_ids = {248044, 248045}
    tok_obj = engine.tokenizer

    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tid = token.item()
        if tid in eos_ids:
            break
        chunk = tok_obj.decode([tid])
        if any(st in chunk for st in STOP_TOKENS):
            break
        yield chunk
        logits = forward(token.reshape(1, 1))
        mx.eval(logits)


def _generate_stream_gemma4(engine, messages, max_tokens=200):
    """Generator for Gemma 4 — uses engine's own forward pass."""
    import mlx.core as mx

    engine.reset_cache()
    tok = engine.tokenizer
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True)
    except Exception:
        text = messages[-1]["content"]
    tokens = tok.encode(text)
    input_ids = mx.array([tokens])

    logits = engine.forward(input_ids)
    mx.eval(logits)

    # Gemma 4 EOS tokens
    eos_ids = set()
    if hasattr(tok, 'eos_token_id'):
        if isinstance(tok.eos_token_id, list):
            eos_ids.update(tok.eos_token_id)
        elif tok.eos_token_id is not None:
            eos_ids.add(tok.eos_token_id)
    eos_ids.update({1, 106, 212})  # Gemma 4 EOS tokens

    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tid = token.item()
        if tid in eos_ids:
            break
        chunk = tok.decode([tid])
        if any(st in chunk for st in STOP_TOKENS):
            break
        yield chunk
        logits = engine.forward(token.reshape(1, 1))
        mx.eval(logits)
