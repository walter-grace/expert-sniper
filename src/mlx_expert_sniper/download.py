"""
mlx-sniper download — download, preprocess, and calibrate a model.

Usage:
    mlx-sniper download qwen3.5-35b -o ~/models/qwen35-35b
    mlx-sniper download qwen3.5-35b  # defaults to ~/models/<name>
"""
import os, sys, json, time, re, gc, glob, shutil
import numpy as np

PAGE_SIZE = 16384

# Supported models: name → HuggingFace repo
MODEL_REGISTRY = {
    # 16 GB Macs
    "qwen3.5-35b": {
        "repo": "mlx-community/Qwen3.5-35B-A3B-4bit",
        "default_dir": "qwen35-35b-stream",
        "description": "Qwen3.5-35B-A3B 4-bit (19.5 GB, 256 experts, 5.4 tok/s on M4 16GB)",
    },
    "qwen3-30b": {
        "repo": "mlx-community/Qwen3-30B-A3B-4bit",
        "default_dir": "qwen3-30b-stream",
        "description": "Qwen3-30B-A3B 4-bit (17.2 GB, 128 experts, 4.3 tok/s on M4 16GB)",
    },
    "qwen3-coder-30b": {
        "repo": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "default_dir": "qwen3-coder-30b-stream",
        "description": "Qwen3-Coder-30B-A3B 4-bit (17.2 GB, 128 experts, coding model)",
    },
    # 32 GB Macs
    "qwen3.5-122b": {
        "repo": "mlx-community/Qwen3.5-122B-A10B-4bit",
        "default_dir": "qwen35-122b-stream",
        "description": "Qwen3.5-122B-A10B 4-bit (~65 GB, 256 experts, needs 32+ GB RAM)",
    },
    "qwen3-next-80b": {
        "repo": "mlx-community/Qwen3-Next-80B-A3B-Instruct-4bit",
        "default_dir": "qwen3-next-80b-stream",
        "description": "Qwen3-Next-80B-A3B 4-bit (~40 GB, 512 experts, needs 32+ GB RAM)",
    },
    # 64 GB+ Macs
    "qwen3-235b": {
        "repo": "mlx-community/Qwen3-235B-A22B-Instruct-2507-4bit",
        "default_dir": "qwen3-235b-stream",
        "description": "Qwen3-235B-A22B 4-bit (~130 GB, 128 experts, needs 64+ GB RAM)",
    },
    # Gemma 4 (Google) — NEW ARCHITECTURE
    "gemma4-26b": {
        "repo": "google/gemma-4-26B-A4B-it",
        "default_dir": "gemma4-26b-stream",
        "description": "Gemma 4-26B-A4B bf16 (~50 GB, 128 experts, Google MoE — EXPERIMENTAL)",
        "preprocess": "gemma4",
    },
}

TENSOR_ORDER = [
    "switch_mlp.gate_proj.weight", "switch_mlp.gate_proj.scales", "switch_mlp.gate_proj.biases",
    "switch_mlp.up_proj.weight", "switch_mlp.up_proj.scales", "switch_mlp.up_proj.biases",
    "switch_mlp.down_proj.weight", "switch_mlp.down_proj.scales", "switch_mlp.down_proj.biases",
]


def list_models():
    """Print available models."""
    print("Available models:\n")
    print("  16 GB Macs:")
    for name, info in MODEL_REGISTRY.items():
        if "16GB" in info["description"] or "coding" in info["description"]:
            print(f"    {name:<22} {info['description']}")
    print("\n  32 GB+ Macs:")
    for name, info in MODEL_REGISTRY.items():
        if "32+" in info["description"]:
            print(f"    {name:<22} {info['description']}")
    print("\n  64 GB+ Macs:")
    for name, info in MODEL_REGISTRY.items():
        if "64+" in info["description"]:
            print(f"    {name:<22} {info['description']}")
    print("\n  Experimental (new architectures):")
    for name, info in MODEL_REGISTRY.items():
        if "EXPERIMENTAL" in info["description"]:
            print(f"    {name:<22} {info['description']}")
    print(f"\nUsage: mlx-sniper download <model-name> [-o output_dir]")


def download_model(model_name, output_dir=None, calibrate_quick=True, keep_download=False):
    """Download, preprocess, and calibrate a model."""
    if model_name not in MODEL_REGISTRY:
        print(f"Unknown model: {model_name}")
        list_models()
        return False

    info = MODEL_REGISTRY[model_name]
    repo = info["repo"]

    if output_dir is None:
        output_dir = os.path.expanduser(f"~/models/{info['default_dir']}")

    # Check if already processed
    if os.path.exists(os.path.join(output_dir, "pinned.safetensors")):
        bin_dir = os.path.join(output_dir, "bin")
        if os.path.isdir(bin_dir) and len(os.listdir(bin_dir)) > 0:
            print(f"Model already exists at {output_dir}")
            print(f"Use mlx-sniper calibrate {output_dir} to recalibrate.")
            return True

    download_dir = output_dir + "_download"
    t0 = time.time()

    # Step 1: Download from HuggingFace
    print(f"{'='*60}")
    print(f"mlx-sniper download: {model_name}")
    print(f"  Source: {repo}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}\n")

    print(f"Step 1/3: Downloading from HuggingFace...")
    print(f"  This may take 10-30 minutes depending on your connection.\n")

    from huggingface_hub import snapshot_download
    snapshot_download(repo, local_dir=download_dir)
    print(f"  Download complete.\n")

    # Step 2: Preprocess (split into streaming format)
    print(f"Step 2/3: Preprocessing into sniper streaming format...")
    print(f"  This takes ~5-20 minutes. Shards are deleted after processing to save disk.\n")

    if info.get("preprocess") == "gemma4":
        from .preprocess_gemma4 import preprocess_gemma4
        preprocess_gemma4(download_dir, output_dir)
    else:
        _preprocess(download_dir, output_dir)

    # Clean up download dir
    if not keep_download:
        remaining = glob.glob(os.path.join(download_dir, "*.safetensors"))
        if not remaining:
            # All shards were deleted during preprocessing, clean up the rest
            shutil.rmtree(download_dir, ignore_errors=True)
            print(f"  Cleaned up download directory.\n")

    # Step 3: Calibrate
    print(f"Step 3/3: Calibrating (one-time optimization)...\n")
    from .calibrate import calibrate
    calibrate(output_dir, quick=calibrate_quick)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes!")
    print(f"  Model ready at: {output_dir}")
    print(f"")
    print(f"  Run:       mlx-sniper run {output_dir} -p \"Hello\" -v")
    print(f"  Calibrate: mlx-sniper calibrate {output_dir} --force")
    print(f"{'='*60}")
    return True


def _preprocess(download_dir, output_dir):
    """Split MLX 4-bit model into pinned + streaming experts."""
    import mlx.core as mx

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "bin"), exist_ok=True)

    config = json.load(open(os.path.join(download_dir, "config.json")))
    tc = config.get("text_config", config)
    NUM_LAYERS = tc["num_hidden_layers"]
    NUM_EXPERTS = tc["num_experts"]

    shard_files = sorted(glob.glob(os.path.join(download_dir, "model-*.safetensors")))
    print(f"  Model: {NUM_LAYERS} layers, {NUM_EXPERTS} experts, {len(shard_files)} shards")

    pinned = {}
    expert_layers_done = set()
    expert_keys = {}
    t0 = time.time()
    total_expert_bytes = 0

    for si, sf in enumerate(shard_files):
        shard_name = os.path.basename(sf)
        print(f"  Shard {si+1}/{len(shard_files)}: {shard_name}")
        w = mx.load(sf)

        for k, v in w.items():
            if "switch_mlp" in k:
                m = re.search(r"layers\.(\d+)\.", k)
                layer_idx = int(m.group(1))
                local_name = k.split(f"layers.{layer_idx}.mlp.")[-1]
                if layer_idx not in expert_keys:
                    expert_keys[layer_idx] = {}
                expert_keys[layer_idx][local_name] = v
            else:
                pinned[k] = v

        # Write complete layers
        for layer_idx in sorted(expert_keys.keys()):
            if layer_idx in expert_layers_done:
                continue
            if len(expert_keys[layer_idx]) < len(TENSOR_ORDER):
                continue

            lt = expert_keys[layer_idx]
            tensor_info = {}
            offset = 0
            for tname in TENSOR_ORDER:
                t = lt[tname]
                per_expert_shape = list(t.shape[1:])
                per_expert_bytes = int(np.prod(per_expert_shape)) * t.dtype.size
                tensor_info[tname] = {
                    "inner_offset": offset, "nbytes": per_expert_bytes,
                    "shape_per_expert": per_expert_shape, "dtype": str(t.dtype),
                }
                offset += per_expert_bytes
            expert_block_size = ((offset + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE

            header = {"layer_idx": layer_idx, "num_experts": NUM_EXPERTS,
                      "layout": {"expert_block_size": expert_block_size,
                                 "data_start": PAGE_SIZE, "tensors": tensor_info}}
            header_json = json.dumps(header).encode()
            header_padded = header_json + b"\x00" * (PAGE_SIZE - len(header_json))

            layer_path = os.path.join(output_dir, "bin", f"moe_layer_{layer_idx:02d}.bin")
            layer_bytes = PAGE_SIZE
            with open(layer_path, "wb") as f:
                f.write(header_padded)
                for eid in range(NUM_EXPERTS):
                    expert_data = bytearray()
                    for tname in TENSOR_ORDER:
                        expert_t = lt[tname][eid]
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
                    layer_bytes += expert_block_size

            total_expert_bytes += layer_bytes
            expert_layers_done.add(layer_idx)
            del expert_keys[layer_idx]
            elapsed = time.time() - t0
            print(f"    Layer {layer_idx:2d}/{NUM_LAYERS}: {layer_bytes/1e6:.1f} MB ({elapsed:.0f}s)")

        del w; gc.collect()
        os.remove(sf)
        print(f"    Deleted {shard_name} to free disk")

    # Handle any layers split across shards (incomplete after last shard)
    for layer_idx in sorted(expert_keys.keys()):
        if layer_idx in expert_layers_done:
            continue
        lt = expert_keys[layer_idx]
        if len(lt) < len(TENSOR_ORDER):
            print(f"  WARNING: Layer {layer_idx} incomplete ({len(lt)}/{len(TENSOR_ORDER)} tensors)")
            continue
        # Same write logic as above
        tensor_info = {}
        offset = 0
        for tname in TENSOR_ORDER:
            t = lt[tname]
            per_expert_shape = list(t.shape[1:])
            per_expert_bytes = int(np.prod(per_expert_shape)) * t.dtype.size
            tensor_info[tname] = {
                "inner_offset": offset, "nbytes": per_expert_bytes,
                "shape_per_expert": per_expert_shape, "dtype": str(t.dtype),
            }
            offset += per_expert_bytes
        expert_block_size = ((offset + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        header = {"layer_idx": layer_idx, "num_experts": NUM_EXPERTS,
                  "layout": {"expert_block_size": expert_block_size,
                             "data_start": PAGE_SIZE, "tensors": tensor_info}}
        header_json = json.dumps(header).encode()
        header_padded = header_json + b"\x00" * (PAGE_SIZE - len(header_json))
        layer_path = os.path.join(output_dir, "bin", f"moe_layer_{layer_idx:02d}.bin")
        layer_bytes = PAGE_SIZE
        import mlx.core as mx
        with open(layer_path, "wb") as f:
            f.write(header_padded)
            for eid in range(NUM_EXPERTS):
                ed = bytearray()
                for tname in TENSOR_ORDER:
                    expert_t = lt[tname][eid]
                    mx.eval(expert_t)
                    if expert_t.dtype == mx.bfloat16:
                        raw = np.array(expert_t.view(mx.uint16)).tobytes()
                    else:
                        raw = np.array(expert_t).tobytes()
                    ed.extend(raw)
                pad = expert_block_size - len(ed)
                if pad > 0:
                    ed.extend(b"\x00" * pad)
                f.write(bytes(ed))
                layer_bytes += expert_block_size
        total_expert_bytes += layer_bytes
        print(f"    Layer {layer_idx:2d}/{NUM_LAYERS}: {layer_bytes/1e6:.1f} MB (cross-shard)")

    # Save pinned
    pinned_bytes = sum(v.nbytes for v in pinned.values())
    import mlx.core as mx
    mx.save_safetensors(os.path.join(output_dir, "pinned.safetensors"), pinned)
    print(f"  Saved pinned.safetensors: {pinned_bytes/1e9:.2f} GB ({len(pinned)} keys)")
    del pinned; gc.collect()

    # Symlinks: layer_XX.bin -> moe_layer_XX.bin
    for i in range(NUM_LAYERS):
        src = f"moe_layer_{i:02d}.bin"
        dst = os.path.join(output_dir, "bin", f"layer_{i:02d}.bin")
        if os.path.exists(os.path.join(output_dir, "bin", src)) and not os.path.exists(dst):
            os.symlink(src, dst)

    # Write streaming config
    stream_config = {
        "model_type": tc.get("model_type", "qwen3_5_moe"),
        "hidden_size": tc["hidden_size"],
        "num_hidden_layers": NUM_LAYERS,
        "num_attention_heads": tc["num_attention_heads"],
        "num_key_value_heads": tc["num_key_value_heads"],
        "rms_norm_eps": tc["rms_norm_eps"],
        "vocab_size": tc["vocab_size"],
        "max_position_embeddings": tc.get("max_position_embeddings", 262144),
        "head_dim": tc.get("head_dim"),
        "tie_word_embeddings": config.get("tie_word_embeddings", False),
        "num_experts": tc["num_experts"],
        "num_experts_per_tok": tc["num_experts_per_tok"],
        "shared_expert_intermediate_size": tc.get("shared_expert_intermediate_size"),
        "moe_intermediate_size": tc["moe_intermediate_size"],
        "linear_num_value_heads": tc.get("linear_num_value_heads"),
        "linear_num_key_heads": tc.get("linear_num_key_heads"),
        "linear_key_head_dim": tc.get("linear_key_head_dim"),
        "linear_value_head_dim": tc.get("linear_value_head_dim"),
        "linear_conv_kernel_dim": tc.get("linear_conv_kernel_dim"),
        "full_attention_interval": tc.get("full_attention_interval"),
        "rope_parameters": tc.get("rope_parameters"),
        "quantization": config.get("quantization", {"bits": 4, "group_size": 64}),
        "streaming": {"pinned_file": "pinned.safetensors", "expert_dir": "bin"},
    }
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(stream_config, f, indent=2)

    # Copy tokenizer files
    for tf in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "added_tokens.json", "vocab.json", "merges.txt"]:
        src = os.path.join(download_dir, tf)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(output_dir, tf))

    # Verify all layers
    layer_count = sum(1 for f in os.listdir(os.path.join(output_dir, "bin"))
                      if f.startswith("moe_layer_") and f.endswith(".bin"))
    if layer_count == NUM_LAYERS:
        print(f"\n  All {NUM_LAYERS} layers written. Total experts: {total_expert_bytes/1e9:.2f} GB")
    else:
        print(f"\n  WARNING: Only {layer_count}/{NUM_LAYERS} layers written!")
