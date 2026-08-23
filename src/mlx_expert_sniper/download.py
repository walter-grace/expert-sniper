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
    # Small models — ideal for the Expert Network demo and CI
    "olmoe-1b-7b": {
        "repo": "mlx-community/OLMoE-1B-7B-0125-Instruct-4bit",
        "default_dir": "olmoe-stream",
        "description": "OLMoE-1B-7B 4-bit (3.6 GB, 64 experts, 8 GB Macs / network demo)",
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
    if info.get("preprocess") == "gemma4":
        snapshot_download(repo, local_dir=download_dir)
    else:
        # Weight shards are fetched one at a time during preprocessing and
        # deleted as they're consumed, so peak disk stays near the final
        # model size instead of double it.
        snapshot_download(repo, local_dir=download_dir,
                          ignore_patterns=["model-*.safetensors"])
    print(f"  Download complete.\n")

    # Step 2: Preprocess (split into streaming format)
    print(f"Step 2/3: Preprocessing into sniper streaming format...")
    print(f"  This takes ~5-20 minutes. Shards are fetched and deleted one at a time.\n")

    if info.get("preprocess") == "gemma4":
        from .preprocess_gemma4 import preprocess_gemma4
        preprocess_gemma4(download_dir, output_dir)
    else:
        _preprocess(download_dir, output_dir, repo=repo)

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


def _tensor_bytes(expert_t):
    import mlx.core as mx
    mx.eval(expert_t)
    if expert_t.dtype == mx.bfloat16:
        return np.array(expert_t.view(mx.uint16)).tobytes()
    return np.array(expert_t).tobytes()


def _write_layer(output_dir, layer_idx, lt, num_experts, verify=True):
    """Write one layer's experts as a 16KB-header + fixed-block .bin file.

    With verify=True, re-reads two random expert blocks from disk and
    byte-compares them against the source tensors — the only moment this
    check is possible, since source shards are deleted right after.
    Returns bytes written.
    """
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

    header = {"layer_idx": layer_idx, "num_experts": num_experts,
              "layout": {"expert_block_size": expert_block_size,
                         "data_start": PAGE_SIZE, "tensors": tensor_info}}
    header_json = json.dumps(header).encode()
    header_padded = header_json + b"\x00" * (PAGE_SIZE - len(header_json))

    layer_path = os.path.join(output_dir, "bin", f"moe_layer_{layer_idx:02d}.bin")
    expected_size = PAGE_SIZE + num_experts * expert_block_size
    if os.path.exists(layer_path) and os.path.getsize(layer_path) == expected_size:
        return 0  # already written by a previous (interrupted) run

    layer_bytes = PAGE_SIZE
    with open(layer_path, "wb") as f:
        f.write(header_padded)
        for eid in range(num_experts):
            expert_data = bytearray()
            for tname in TENSOR_ORDER:
                expert_data.extend(_tensor_bytes(lt[tname][eid]))
            pad = expert_block_size - len(expert_data)
            if pad > 0:
                expert_data.extend(b"\x00" * pad)
            f.write(bytes(expert_data))
            layer_bytes += expert_block_size

    if verify:
        import random
        with open(layer_path, "rb") as f:
            for eid in random.sample(range(num_experts), min(2, num_experts)):
                f.seek(PAGE_SIZE + eid * expert_block_size)
                on_disk = f.read(expert_block_size)
                expected = bytearray()
                for tname in TENSOR_ORDER:
                    expected.extend(_tensor_bytes(lt[tname][eid]))
                if on_disk[:len(expected)] != bytes(expected):
                    raise RuntimeError(
                        f"verify FAILED: layer {layer_idx} expert {eid} "
                        f"on-disk bytes differ from source tensors")

    return layer_bytes


def _shard_names(download_dir):
    """Full shard list from the safetensors index (works even when some
    shards were already deleted by an interrupted run), else glob."""
    index_path = os.path.join(download_dir, "model.safetensors.index.json")
    if os.path.exists(index_path):
        index = json.load(open(index_path))
        names = sorted(set(index["weight_map"].values()))
        return names
    names = [os.path.basename(p) for p in
             sorted(glob.glob(os.path.join(download_dir, "model-*.safetensors")))]
    single = os.path.join(download_dir, "model.safetensors")
    if not names and os.path.exists(single):
        names = ["model.safetensors"]  # single-shard checkpoints (e.g. OLMoE)
    return names


def _stack_experts(w, num_layers, num_experts):
    """Per-expert checkpoints (mlp.experts.N.*, e.g. OLMoE) → stacked
    switch_mlp tensors, the format the layer splitter expects. Mirrors
    mlx_lm's Model.sanitize. No-op for already-stacked checkpoints."""
    import mlx.core as mx
    if "model.layers.0.mlp.experts.0.up_proj.weight" not in w:
        return w
    for l in range(num_layers):
        prefix = f"model.layers.{l}"
        for n in ("up_proj", "down_proj", "gate_proj"):
            for k in ("weight", "scales", "biases"):
                if f"{prefix}.mlp.experts.0.{n}.{k}" in w:
                    joined = [w.pop(f"{prefix}.mlp.experts.{e}.{n}.{k}")
                              for e in range(num_experts)]
                    w[f"{prefix}.mlp.switch_mlp.{n}.{k}"] = mx.stack(joined)
    return w


def _free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def _preprocess(download_dir, output_dir, delete_shards=True, repo=None):
    """Split MLX 4-bit model into pinned + streaming experts.

    delete_shards=True frees disk as it goes (download flow); pass False
    when preprocessing a checkpoint the user supplied themselves.

    Disk-safety notes (learned the hard way):
    - mx.load() returns lazy arrays backed by the shard's mmap. Any tensor
      still referenced when the shard file is deleted keeps the file's
      blocks allocated (POSIX), so "delete to free disk" silently frees
      nothing. Therefore pinned tensors are persisted to per-shard part
      files immediately, and incomplete cross-shard expert tensors are
      round-tripped through a carry file, so nothing references a shard's
      mmap by the time it is removed.
    - Layers already fully written by a previous interrupted run are
      skipped, and with `repo` set, missing shards are downloaded one at a
      time (present shards are processed first) — so an interrupted run
      resumes with minimal disk and no re-download of finished work.
    """
    import mlx.core as mx

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "bin"), exist_ok=True)
    parts_dir = os.path.join(output_dir, "pinned_parts")
    os.makedirs(parts_dir, exist_ok=True)

    config = json.load(open(os.path.join(download_dir, "config.json")))
    tc = config.get("text_config", config)
    NUM_LAYERS = tc["num_hidden_layers"]
    NUM_EXPERTS = tc["num_experts"]

    names = _shard_names(download_dir)
    # Process locally-present shards first so missing ones are downloaded
    # only after their disk has been freed.
    local = [n for n in names if os.path.exists(os.path.join(download_dir, n))]
    missing = [n for n in names if n not in local]
    ordered = local + missing
    print(f"  Model: {NUM_LAYERS} layers, {NUM_EXPERTS} experts, "
          f"{len(names)} shards ({len(local)} local, {len(missing)} to fetch)")

    expert_layers_done = set()
    expert_keys = {}
    carry_paths = [os.path.join(output_dir, "carry_a.safetensors"),
                   os.path.join(output_dir, "carry_b.safetensors")]
    carry_gen = 0
    t0 = time.time()
    total_expert_bytes = 0

    def flush_layers(final=False):
        nonlocal total_expert_bytes
        for layer_idx in sorted(expert_keys.keys()):
            if layer_idx in expert_layers_done:
                continue
            lt = expert_keys[layer_idx]
            if len(lt) < len(TENSOR_ORDER):
                if final:
                    print(f"  WARNING: Layer {layer_idx} incomplete "
                          f"({len(lt)}/{len(TENSOR_ORDER)} tensors)")
                continue
            layer_bytes = _write_layer(output_dir, layer_idx, lt, NUM_EXPERTS)
            total_expert_bytes += layer_bytes
            expert_layers_done.add(layer_idx)
            del expert_keys[layer_idx]
            gc.collect()
            elapsed = time.time() - t0
            note = "skipped (exists)" if layer_bytes == 0 else f"{layer_bytes/1e6:.1f} MB"
            print(f"    Layer {layer_idx:2d}/{NUM_LAYERS}: {note} ({elapsed:.0f}s)")

    for si, shard_name in enumerate(ordered):
        sf = os.path.join(download_dir, shard_name)
        part_path = os.path.join(parts_dir, f"part_{shard_name}")
        if not os.path.exists(sf) and os.path.exists(part_path):
            # Shard was fully processed by a previous run (its pinned part is
            # only saved after all its layers are flushed) and then deleted —
            # don't re-download it.
            print(f"  Shard {si+1}/{len(ordered)}: {shard_name} "
                  f"already processed — skipping")
            continue
        if not os.path.exists(sf):
            if repo is None:
                raise FileNotFoundError(f"{sf} missing and no repo to fetch from")
            print(f"  Fetching {shard_name} ({_free_gb(download_dir):.1f} GB free)...")
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo, shard_name, local_dir=download_dir)

        print(f"  Shard {si+1}/{len(ordered)}: {shard_name} "
              f"({_free_gb(download_dir):.1f} GB free)")
        w = _stack_experts(mx.load(sf), NUM_LAYERS, NUM_EXPERTS)

        shard_pinned = {}
        for k, v in w.items():
            if "switch_mlp" in k:
                m = re.search(r"layers\.(\d+)\.", k)
                layer_idx = int(m.group(1))
                local_name = k.split(f"layers.{layer_idx}.mlp.")[-1]
                expert_keys.setdefault(layer_idx, {})[local_name] = v
            else:
                shard_pinned[k] = v

        flush_layers()

        # Persist this shard's pinned tensors NOW so no lazy reference into
        # the shard's mmap survives its deletion.
        if shard_pinned:
            mx.save_safetensors(part_path, shard_pinned)
        del shard_pinned

        # Round-trip incomplete cross-shard expert tensors through a carry
        # file for the same reason.
        pending = {li: lt for li, lt in expert_keys.items()
                   if li not in expert_layers_done}
        if pending and delete_shards:
            flat = {f"L{li}|{name}": t
                    for li, lt in pending.items() for name, t in lt.items()}
            new_carry = carry_paths[carry_gen % 2]
            mx.save_safetensors(new_carry, flat)
            del flat
            reloaded = mx.load(new_carry)
            expert_keys = {}
            for key, t in reloaded.items():
                li_s, name = key.split("|", 1)
                expert_keys.setdefault(int(li_s[1:]), {})[name] = t
            old_carry = carry_paths[(carry_gen + 1) % 2]
            if os.path.exists(old_carry):
                os.remove(old_carry)
            carry_gen += 1

        del w; gc.collect()
        if delete_shards:
            os.remove(sf)
            print(f"    Deleted {shard_name} to free disk "
                  f"({_free_gb(download_dir):.1f} GB free)")

    flush_layers(final=True)
    for cp in carry_paths:
        if os.path.exists(cp):
            os.remove(cp)

    # Merge pinned parts
    pinned = {}
    part_files = sorted(glob.glob(os.path.join(parts_dir, "part_*")))
    for pf in part_files:
        pinned.update(mx.load(pf))
    pinned_bytes = sum(v.nbytes for v in pinned.values())
    mx.save_safetensors(os.path.join(output_dir, "pinned.safetensors"), pinned)
    print(f"  Saved pinned.safetensors: {pinned_bytes/1e9:.2f} GB ({len(pinned)} keys)")
    del pinned; gc.collect()
    shutil.rmtree(parts_dir, ignore_errors=True)

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
        # OLMoE-style models size their experts by intermediate_size
        "moe_intermediate_size": tc.get("moe_intermediate_size",
                                        tc.get("intermediate_size")),
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
    # qwen3_moe-family keys (engine_30b reads these with .get defaults):
    # persist the source model's real values when present, omit otherwise
    # so the engine-side defaults still apply.
    for key in ("intermediate_size", "norm_topk_prob", "decoder_sparse_step",
                "mlp_only_layers", "rope_theta",
                # olmoe-family keys
                "attention_bias", "mlp_bias", "rope_traditional", "rope_scaling"):
        if tc.get(key) is not None:
            stream_config[key] = tc[key]
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
