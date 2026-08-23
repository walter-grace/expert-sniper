#!/usr/bin/env python3
"""
Preprocess Gemma 4-26B-A4B into sniper streaming format.

Expert tensor naming (different from Qwen):
  Qwen:   layers.N.mlp.switch_mlp.{gate,up,down}_proj.{weight,scales,biases}
  Gemma4: layers.N.experts.gate_up_proj  (fused, [128, 1408, 2816] bf16)
          layers.N.experts.down_proj     ([128, 2816, 704] bf16)
          layers.N.router.{proj.weight, scale, per_expert_scale}

The experts are stored as bf16 (not quantized at source).
We can optionally quantize during preprocessing for smaller disk footprint.
"""
import os, json, gc, time, re, glob
import numpy as np
import mlx.core as mx

PAGE_SIZE = 16384

# Gemma 4 expert tensors (per layer, shape includes expert dim)
EXPERT_TENSORS = [
    "experts.gate_up_proj",  # [num_experts, 2*moe_inter, hidden]
    "experts.down_proj",     # [num_experts, hidden, moe_inter]
]


def preprocess_gemma4(input_dir, output_dir, quantize_experts=False):
    """Split Gemma 4 into pinned + streaming experts.

    Args:
        input_dir: HuggingFace download directory
        output_dir: sniper streaming format output
        quantize_experts: if True, quantize experts to 4-bit (saves disk)
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "bin"), exist_ok=True)

    config = json.load(open(os.path.join(input_dir, "config.json")))
    tc = config.get("text_config", config)
    NUM_LAYERS = tc["num_hidden_layers"]
    NUM_EXPERTS = tc["num_experts"]
    hidden_size = tc["hidden_size"]
    moe_inter = tc["moe_intermediate_size"]

    shard_files = sorted(glob.glob(os.path.join(input_dir, "model-*.safetensors")))
    print(f"Gemma 4: {NUM_LAYERS} layers, {NUM_EXPERTS} experts, {len(shard_files)} shards")
    print(f"  Hidden: {hidden_size}, MoE inter: {moe_inter}")
    print(f"  Expert storage: bf16 (not quantized)")

    pinned = {}
    expert_keys = {}  # layer -> {name: tensor}
    expert_layers_done = set()
    t0 = time.time()
    total_expert_bytes = 0

    for si, sf in enumerate(shard_files):
        shard_name = os.path.basename(sf)
        print(f"\n  Shard {si+1}/{len(shard_files)}: {shard_name}")
        w = mx.load(sf)

        for k, v in w.items():
            # Strip language_model. prefix
            clean_k = k.replace("model.language_model.", "")

            # Check if this is an expert tensor
            is_expert = False
            for et in EXPERT_TENSORS:
                if et in clean_k:
                    is_expert = True
                    break

            if is_expert:
                m = re.search(r"layers\.(\d+)\.", clean_k)
                if m:
                    layer_idx = int(m.group(1))
                    # Local name: just the part after "layers.N."
                    local_name = clean_k.split(f"layers.{layer_idx}.")[-1]
                    if layer_idx not in expert_keys:
                        expert_keys[layer_idx] = {}
                    expert_keys[layer_idx][local_name] = v
            else:
                # Skip vision tower for pinned
                if "vision_tower" not in k and "embed_vision" not in k:
                    pinned[clean_k] = v

        # Write complete expert layers
        for layer_idx in sorted(expert_keys.keys()):
            if layer_idx in expert_layers_done:
                continue
            if len(expert_keys[layer_idx]) < len(EXPERT_TENSORS):
                continue

            lt = expert_keys[layer_idx]
            _write_expert_layer(output_dir, layer_idx, lt, NUM_EXPERTS, t0)
            total_expert_bytes += os.path.getsize(
                os.path.join(output_dir, "bin", f"moe_layer_{layer_idx:02d}.bin"))
            expert_layers_done.add(layer_idx)
            del expert_keys[layer_idx]

        del w; gc.collect()
        os.remove(sf)
        print(f"    Deleted {shard_name}")

    # Handle any remaining cross-shard layers
    for layer_idx in sorted(expert_keys.keys()):
        if layer_idx in expert_layers_done:
            continue
        lt = expert_keys[layer_idx]
        if len(lt) < len(EXPERT_TENSORS):
            print(f"  WARNING: Layer {layer_idx} incomplete ({len(lt)} tensors)")
            continue
        _write_expert_layer(output_dir, layer_idx, lt, NUM_EXPERTS, t0)
        total_expert_bytes += os.path.getsize(
            os.path.join(output_dir, "bin", f"moe_layer_{layer_idx:02d}.bin"))

    # Save pinned
    pinned_bytes = sum(v.nbytes for v in pinned.values())
    mx.save_safetensors(os.path.join(output_dir, "pinned.safetensors"), pinned)
    print(f"\n  Saved pinned.safetensors: {pinned_bytes/1e9:.2f} GB ({len(pinned)} keys)")
    del pinned; gc.collect()

    # Symlinks
    for i in range(NUM_LAYERS):
        src = f"moe_layer_{i:02d}.bin"
        dst = os.path.join(output_dir, "bin", f"layer_{i:02d}.bin")
        if os.path.exists(os.path.join(output_dir, "bin", src)) and not os.path.exists(dst):
            os.symlink(src, dst)

    # Write config
    stream_config = {
        "model_type": "gemma4",
        "hidden_size": hidden_size,
        "num_hidden_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "top_k_experts": tc["top_k_experts"],
        "moe_intermediate_size": moe_inter,
        "intermediate_size": tc["intermediate_size"],
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "num_global_key_value_heads": tc.get("num_global_key_value_heads", 2),
        "global_head_dim": tc.get("global_head_dim", 512),
        "head_dim": tc.get("head_dim", 256),
        "vocab_size": tc["vocab_size"],
        "rms_norm_eps": tc.get("rms_norm_eps", 1e-6),
        "sliding_window": tc.get("sliding_window", 1024),
        "layer_types": tc.get("layer_types", []),
        "hidden_activation": tc.get("hidden_activation", "gelu_pytorch_tanh"),
        "final_logit_softcapping": tc.get("final_logit_softcapping", 30.0),
        "enable_moe_block": tc.get("enable_moe_block", True),
        "attention_k_eq_v": tc.get("attention_k_eq_v", True),
        "rope_parameters": tc.get("rope_parameters"),
        "max_position_embeddings": tc.get("max_position_embeddings", 262144),
        "tie_word_embeddings": config.get("tie_word_embeddings", True),
        "streaming": {"pinned_file": "pinned.safetensors", "expert_dir": "bin"},
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(stream_config, f, indent=2)

    # Copy tokenizer
    import shutil
    for tf in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "added_tokens.json", "tokenizer.model"]:
        src = os.path.join(input_dir, tf)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, tf))

    # Verify
    layer_count = sum(1 for f in os.listdir(os.path.join(output_dir, "bin"))
                      if f.startswith("moe_layer_") and f.endswith(".bin"))
    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.0f}s!")
    print(f"  Pinned: {pinned_bytes/1e9:.2f} GB, Experts: {total_expert_bytes/1e9:.2f} GB")
    print(f"  Layers: {layer_count}/{NUM_LAYERS}")


def _write_expert_layer(output_dir, layer_idx, layer_tensors, num_experts, t0):
    """Write one layer's experts to a binary file."""
    # Build tensor info and calculate sizes
    tensor_order = ["experts.gate_up_proj", "experts.down_proj"]
    tensor_info = {}
    offset = 0
    for tname in tensor_order:
        t = layer_tensors[tname]
        per_expert_shape = list(t.shape[1:])  # remove expert dim
        per_expert_bytes = int(np.prod(per_expert_shape)) * t.dtype.size
        tensor_info[tname] = {
            "inner_offset": offset,
            "nbytes": per_expert_bytes,
            "shape_per_expert": per_expert_shape,
            "dtype": str(t.dtype),
        }
        offset += per_expert_bytes

    expert_block_size = ((offset + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE

    header = {
        "layer_idx": layer_idx,
        "num_experts": num_experts,
        "layout": {
            "expert_block_size": expert_block_size,
            "data_start": PAGE_SIZE,
            "tensors": tensor_info,
        }
    }
    header_json = json.dumps(header).encode()
    header_padded = header_json + b"\x00" * (PAGE_SIZE - len(header_json))

    layer_path = os.path.join(output_dir, "bin", f"moe_layer_{layer_idx:02d}.bin")
    with open(layer_path, "wb") as f:
        f.write(header_padded)
        for eid in range(num_experts):
            expert_data = bytearray()
            for tname in tensor_order:
                expert_t = layer_tensors[tname][eid]
                mx.eval(expert_t)
                if expert_t.dtype == mx.bfloat16:
                    raw = np.array(expert_t.view(mx.uint16)).tobytes()
                else:
                    raw = np.array(expert_t).tobytes()
                expert_data.extend(raw)
            pad = expert_block_size - len(expert_data)
            if pad > 0:
                expert_data.extend(b"\x00" * pad)
            f.write(bytes(expert_data))

    sym = os.path.join(output_dir, "bin", f"layer_{layer_idx:02d}.bin")
    if not os.path.exists(sym):
        os.symlink(f"moe_layer_{layer_idx:02d}.bin", sym)

    elapsed = time.time() - t0
    layer_bytes = os.path.getsize(layer_path)
    print(f"    Layer {layer_idx:2d}: {layer_bytes/1e6:.1f} MB ({elapsed:.0f}s)")
