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

CALIBRATION_PROMPTS = [
    "What is the square root of 69?",
    "Write a Python function to sort a list.",
    "Explain how photosynthesis works.",
]

BIAS_VALUES = [0.5, 1.0, 1.5]

# A bias passes if its perplexity stays within this factor of bias=0.
PPL_TOLERANCE = 1.05


def _available_ram_bytes():
    """Reclaimable memory right now (free + inactive + purgeable +
    speculative pages). Total RAM overstates what the cache can use when
    other apps are running — overshooting sends the machine into swap on
    the same SSD the experts stream from, collapsing read throughput."""
    import subprocess
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page_size = 16384
    pages = 0
    for line in out.splitlines():
        for key in ("Pages free", "Pages inactive", "Pages purgeable",
                    "Pages speculative"):
            if line.startswith(key + ":"):
                pages += int(line.split(":")[1].strip().rstrip("."))
    return pages * page_size


def auto_size_cache(model_dir, ram_gb=None):
    measured = None
    if ram_gb is None:
        import subprocess
        result = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        ram_bytes = int(result.stdout.strip())
        ram_gb = ram_bytes / (1024**3)
        try:
            measured = _available_ram_bytes()
        except Exception:
            measured = None

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
    if measured is not None:
        # Don't budget memory other running apps are already using.
        available = min(available, measured - pinned_bytes - 2 * 1024**3)
    max_cache = int(available / expert_block_bytes)
    max_cache = max(500, min(max_cache, 10000))
    return max_cache, expert_block_bytes, pinned_bytes


def _detect_model_type(model_dir):
    config = json.load(open(os.path.join(model_dir, "config.json")))
    return config.get("model_type", "qwen3_5_moe")


def _build_engine(model_dir, cache_size, enable_prediction=True):
    # enable_prediction must match what serve/run use (True), or the
    # calibrated bias is tuned on a different system than the one serving.
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
    engine = EngineClass(cache_size=cache_size, enable_prediction=enable_prediction)
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
    from .generate import eos_token_ids
    eos_ids = eos_token_ids(tok)
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
            if tid in eos_ids: break
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


def sweep_routing_bias(model_dir, cache_size, bias_values=BIAS_VALUES,
                       eval_text=None, ppl_tolerance=PPL_TOLERANCE):
    """Perplexity-gated bias sweep.

    Measures teacher-forced perplexity at bias=0 (baseline) and at each
    candidate bias, on an engine configured exactly as serve/run configure
    it (co-activation prediction ON). Returns the highest bias whose
    perplexity stays within ppl_tolerance of baseline, plus all results.
    """
    import mlx.core as mx, gc
    from .evaluate import perplexity
    from .generate import generate_stream

    results = {}
    for bias in [0.0] + list(bias_values):
        print(f"  Measuring ppl at bias={bias}...", end=" ", flush=True)
        engine = _build_engine(model_dir, cache_size, enable_prediction=True)
        try:
            results[bias] = perplexity(engine, bias=bias)
            print(f"ppl={results[bias]:.3f}")
            # One qualitative generation per bias — eyeball only, not a gate.
            sample = "".join(generate_stream(
                engine, [{"role": "user", "content": "Briefly explain what a mixture-of-experts model is."}],
                bias=bias, max_tokens=40))
            print(f"    sample: {sample[:100]!r}")
        finally:
            engine.reader.close()
            del engine; gc.collect(); mx.clear_cache()

    baseline = results[0.0]
    passing = [b for b in bias_values
               if results.get(b) is not None and results[b] <= baseline * ppl_tolerance]
    best = max(passing) if passing else 0.0
    return best, results


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

    engine.reader.close()
    del engine; gc.collect(); mx.clear_cache()

    if quick:
        best_bias = 0.5
        bias_sweep_ppl = None
        print(f"\nStep 3: Bias sweep (skipped — quick mode, using {best_bias})")
        print("  WARNING: bias not validated — run full calibrate before "
              "trusting quality at this bias.")
    else:
        print(f"\nStep 3: Routing bias sweep {BIAS_VALUES} "
              f"(perplexity gate, tolerance {PPL_TOLERANCE}x)")
        best_bias, bias_sweep_ppl = sweep_routing_bias(model_dir, cache_size)
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
        "bias_sweep_ppl": ({str(k): round(v, 4) for k, v in bias_sweep_ppl.items()}
                           if bias_sweep_ppl else None),
        "ppl_tolerance": PPL_TOLERANCE,
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
