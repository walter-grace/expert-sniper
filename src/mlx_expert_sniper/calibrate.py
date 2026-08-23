#!/usr/bin/env python3
"""
mlx-sniper calibrate — one-time calibration for a sniper model.

Runs ~8 min (or ~2 min with --quick). Saves:
  sniper_config.json       — metadata, cache size, bias
  sniper_calibration.npz   — REAP scores, dead mask, co-activation matrix
"""
import json, os, sys, time, platform
from collections import defaultdict, Counter
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

CALIBRATION_PROMPTS = [
    "What is the square root of 69?",
    "Write a Python function to sort a list.",
    "Explain how photosynthesis works.",
]

QUALITY_CHECKS = [
    ("What is the capital of Australia?", "Canberra"),
    ("What is 2+2?", "4"),
]

BIAS_VALUES = [0.5, 1.0, 1.5]


def auto_size_cache(model_dir, ram_gb=None):
    if ram_gb is None:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024**3)

    pinned_path = os.path.join(model_dir, "pinned.safetensors")
    pinned_bytes = os.path.getsize(pinned_path) if os.path.exists(pinned_path) else 2 * 1024**3

    bin_dir = os.path.join(model_dir, "bin")
    layer_file = None
    for f in sorted(os.listdir(bin_dir)):
        if f.endswith(".bin"):
            layer_file = os.path.join(bin_dir, f)
            break
    with open(layer_file, "rb") as f:
        raw = f.read(16384)
    hdr = json.loads(raw.rstrip(b"\x00"))
    expert_block_bytes = hdr["layout"]["expert_block_size"]

    os_overhead = 4 * 1024**3
    headroom = 3 * 1024**3
    available = (ram_gb * 1024**3) - os_overhead - pinned_bytes - headroom
    max_cache = int(available / expert_block_bytes)
    max_cache = max(500, min(max_cache, 10000))
    return max_cache, expert_block_bytes, pinned_bytes


def _detect_model_type(model_dir):
    config = json.load(open(os.path.join(model_dir, "config.json")))
    return config.get("model_type", "qwen3_5_moe")


def _build_engine(model_dir, cache_size):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model_type = _detect_model_type(model_dir)
    if "gemma4" in model_type:
        from .engine_gemma4 import MoESniperEngineGemma4 as EngineClass
        from . import engine_gemma4 as engine_mod
        engine_mod.MODEL_DIR = model_dir
    elif "qwen3_next" in model_type:
        from .engine_next import MoESniperEngineNext as EngineClass
        from . import engine_next as engine_mod
        engine_mod.MODEL_DIR = model_dir
    elif "qwen3_5" in model_type:
        from .engine import MoESniperEngine35B as EngineClass
        from . import engine as engine_mod
        engine_mod.MODEL_DIR = model_dir
    else:
        from .engine_30b import MoESniperEngine30B as EngineClass
        from . import engine_30b as engine_mod
        engine_mod.MODEL_DIR = model_dir
    engine = EngineClass(cache_size=cache_size, enable_prediction=False)
    engine.load()
    return engine


def run_shared_calibration_pass(engine, prompts, tokens_per_prompt=20):
    """Single pass: records REAP scores AND co-activation matrix."""
    import mlx.core as mx
    from .engine import run_expert_ffn
    # Get model_dir from the engine's reader
    model_dir = os.path.dirname(engine.reader.expert_dir)
    config = json.load(open(os.path.join(model_dir, "config.json")))
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_experts"]

    count = np.zeros((num_layers, num_experts), dtype=np.int32)
    gate_sum = np.zeros((num_layers, num_experts), dtype=np.float64)
    coact = np.zeros((num_layers, num_experts, num_experts), dtype=np.float32)
    prev_layer_experts = {}
    total_tokens = 0

    has_ssm = hasattr(engine.model.model, 'fa_idx')

    def instrumented_forward(input_ids):
        nonlocal prev_layer_experts
        from mlx_lm.models.base import create_attention_mask
        h = engine.model.model.embed_tokens(input_ids)
        if has_ssm:
            from mlx_lm.models.base import create_ssm_mask
            fa_mask = create_attention_mask(h, engine.cache[engine.model.model.fa_idx])
            ssm_mask = create_ssm_mask(h, engine.cache[engine.model.model.ssm_idx])
        else:
            fa_mask = create_attention_mask(h, engine.cache[0])
            ssm_mask = None
        prev_layer_experts = {}
        for i in range(num_layers):
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
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)
            active_ids = [int(e) for e in np.array(inds).flatten()]
            gate_weights = [float(s) for s in np.array(scores.astype(mx.float32)).flatten()]
            active_set = list(set(active_ids))
            for eid, gw in zip(active_ids, gate_weights):
                count[i, eid] += 1
                gate_sum[i, eid] += gw
            if i > 0 and (i - 1) in prev_layer_experts:
                for prev_eid in prev_layer_experts[i - 1]:
                    for cur_eid in active_set:
                        coact[i - 1, prev_eid, cur_eid] += 1
            prev_layer_experts[i] = set(active_set)
            if i + 1 < num_layers:
                engine.reader.prefetch_experts(i + 1, active_set)
            expert_data = engine.reader.get_experts(i, active_set)
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
        h = engine.model.model.norm(h)
        return engine.model.lm_head(h)

    tok = engine.tokenizer
    for pi, prompt in enumerate(prompts):
        engine.reset_cache()
        messages = [{"role": "user", "content": prompt}]
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
        logits = instrumented_forward(input_ids)
        mx.eval(logits)
        total_tokens += 1
        for ti in range(tokens_per_prompt):
            token = mx.argmax(logits[:, -1, :], axis=-1)
            mx.eval(token)
            tid = token.item()
            if tid in (248044, 248045): break
            logits = instrumented_forward(token.reshape(1, 1))
            mx.eval(logits)
            total_tokens += 1
        sys.stdout.write(f"\r  Calibration: prompt {pi+1}/{len(prompts)}, {total_tokens} tokens")
        sys.stdout.flush()
    print()

    avg_gate = np.where(count > 0, gate_sum / np.maximum(count, 1), 0.0)
    importance = (count * avg_gate).astype(np.float32)
    for li in range(num_layers):
        mx_val = importance[li].max()
        if mx_val > 0:
            importance[li] /= mx_val
    dead_mask = importance < 0.01
    return importance, dead_mask, coact


def _generate_with_bias(engine, prompt, bias, max_tokens=40):
    """Generate with routing bias on raw logits. No REAP masking."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask
    # Import run_expert_ffn from whichever engine module loaded this engine
    from .engine import run_expert_ffn

    has_ssm = hasattr(engine.model.model, 'fa_idx')
    num_experts_total = 256 if has_ssm else 128  # 35B vs 30B

    engine.reset_cache()
    tok = engine.tokenizer
    messages = [{"role": "user", "content": prompt}]
    try:
        text = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
    except:
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    tokens = tok.encode(text)
    input_ids = mx.array([tokens])

    def biased_forward(input_ids):
        h = engine.model.model.embed_tokens(input_ids)
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
                cached_mask = np.zeros(num_experts_total, dtype=np.float32)
                for eid in range(num_experts_total):
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

    logits = biased_forward(input_ids)
    mx.eval(logits)
    generated = []
    for _ in range(max_tokens):
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        tid = token.item()
        if tid in (248044, 248045): break
        generated.append(tid)
        logits = biased_forward(token.reshape(1, 1))
        mx.eval(logits)
    return generated, tok.decode(generated)


def sweep_routing_bias(model_dir, cache_size, reap_mask, coact_matrix,
                       bias_values=BIAS_VALUES):
    """Find highest bias where quality checks pass. No REAP during sweep."""
    import mlx.core as mx, gc

    best_bias = 0.0
    for bias in bias_values:
        print(f"  Testing bias={bias}...", end=" ", flush=True)
        engine = _build_engine(model_dir, cache_size)
        engine._enable_prediction = True

        all_pass = True
        for prompt, expected in QUALITY_CHECKS:
            _, output = _generate_with_bias(engine, prompt, bias, max_tokens=40)
            if expected.lower() not in output.lower():
                print(f"FAIL ('{expected}' not in: '{output[:60]}')")
                all_pass = False
                break

        if all_pass:
            print("PASS")
            best_bias = bias
        else:
            break

        del engine; gc.collect(); mx.clear_cache()

    return best_bias


def calibrate(model_dir, ram_gb=None, quick=False):
    import mlx.core as mx, gc

    print(f"{'='*60}")
    print(f"mlx-sniper calibrate")
    print(f"Model: {model_dir}")
    print(f"Mode: {'quick' if quick else 'full'}")
    print(f"{'='*60}\n")

    t0 = time.time()

    cache_size, expert_block_bytes, pinned_bytes = auto_size_cache(model_dir, ram_gb)
    print(f"Step 1: Cache sizing")
    print(f"  Recommended cache: {cache_size} experts")

    print(f"\nStep 2: REAP + co-activation (shared pass)")
    engine = _build_engine(model_dir, cache_size)
    importance, dead_mask, coact_cross = run_shared_calibration_pass(
        engine, CALIBRATION_PROMPTS, tokens_per_prompt=20
    )
    dead_pct = np.mean(dead_mask)
    print(f"  Dead experts: {np.sum(dead_mask)}/{dead_mask.size} ({dead_pct:.1%})")

    del engine; gc.collect(); mx.clear_cache()

    if quick:
        best_bias = 0.5
        print(f"\nStep 3: Bias sweep (skipped — quick mode, using {best_bias})")
    else:
        print(f"\nStep 3: Routing bias sweep {BIAS_VALUES}")
        best_bias = sweep_routing_bias(model_dir, cache_size, dead_mask, coact_cross)
        print(f"  Sweet spot: {best_bias}")

    config = {
        "version": 1,
        "model_dir": model_dir,
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": {
            "chip": platform.processor() or platform.machine(),
            "ram_gb": ram_gb or round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1),
            "os": platform.platform(),
        },
        "cache_size": cache_size,
        "routing_bias": best_bias,
        "reap_threshold": 0.01,
        "reap_dead_pct": float(dead_pct),
        "coact_warmup_tokens": 3,
        "num_layers": int(importance.shape[0]),
        "num_experts": int(importance.shape[1]),
        "expert_block_bytes": expert_block_bytes,
        "pinned_bytes": pinned_bytes,
        "quick_mode": quick,
    }

    config_path = os.path.join(model_dir, "sniper_config.json")
    npz_path = os.path.join(model_dir, "sniper_calibration.npz")

    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    np.savez_compressed(npz_path,
        reap_importance=importance,
        reap_dead_mask=dead_mask,
        coact_cross=coact_cross,
    )

    elapsed = time.time() - t0
    npz_size = os.path.getsize(npz_path)

    print(f"\n{'='*60}")
    print(f"Calibration complete ({elapsed:.0f}s)")
    print(f"  Cache: {cache_size} experts")
    print(f"  Dead experts: {dead_pct:.1%}")
    print(f"  Routing bias: {best_bias}")
    print(f"  Config: {config_path}")
    print(f"  Arrays: {npz_path} ({npz_size/1e6:.1f} MB)")
    print(f"{'='*60}")
    return config


def load_calibration(model_dir):
    config_path = os.path.join(model_dir, "sniper_config.json")
    npz_path = os.path.join(model_dir, "sniper_calibration.npz")
    if not os.path.exists(config_path) or not os.path.exists(npz_path):
        return None
    config = json.load(open(config_path))
    arrays = np.load(npz_path)
    return {
        "cache_size": config["cache_size"],
        "routing_bias": config["routing_bias"],
        "reap_threshold": config["reap_threshold"],
        "reap_dead_pct": config["reap_dead_pct"],
        "reap_importance": arrays["reap_importance"],
        "reap_dead_mask": arrays["reap_dead_mask"],
        "coact_cross": arrays["coact_cross"],
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="mlx-sniper calibrate")
    parser.add_argument("model_dir", help="Path to sniper model directory")
    parser.add_argument("--ram", type=float, default=None, help="Override RAM (GB)")
    parser.add_argument("--quick", action="store_true", help="Skip bias sweep, use bias=0.5")
    parser.add_argument("--force", action="store_true", help="Overwrite existing calibration")
    args = parser.parse_args()

    if not args.force:
        existing = load_calibration(args.model_dir)
        if existing:
            print(f"Calibration exists: cache={existing['cache_size']}, "
                  f"bias={existing['routing_bias']}, dead={existing['reap_dead_pct']:.1%}")
            print(f"Use --force to overwrite.")
            sys.exit(0)

    calibrate(args.model_dir, ram_gb=args.ram, quick=args.quick)
