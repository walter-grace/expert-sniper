#!/usr/bin/env python3
"""Split Qwen3.5-35B-A3B-4bit — delete source shards after processing to save disk."""
import os, json, gc, time, re, glob
import numpy as np
import mlx.core as mx

MLX_MODEL_DIR = "/Users/bigneek/models/qwen35-35b-mlx-4bit"
OUTPUT_DIR = "/Users/bigneek/models/qwen35-35b-stream"
PAGE_SIZE = 16384

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/bin", exist_ok=True)

    config = json.load(open(f"{MLX_MODEL_DIR}/config.json"))
    tc = config.get("text_config", config)
    NUM_LAYERS = tc["num_hidden_layers"]
    NUM_EXPERTS = tc["num_experts"]

    tensor_order = [
        "switch_mlp.gate_proj.weight", "switch_mlp.gate_proj.scales", "switch_mlp.gate_proj.biases",
        "switch_mlp.up_proj.weight", "switch_mlp.up_proj.scales", "switch_mlp.up_proj.biases",
        "switch_mlp.down_proj.weight", "switch_mlp.down_proj.scales", "switch_mlp.down_proj.biases",
    ]

    shard_files = sorted(glob.glob(f"{MLX_MODEL_DIR}/model-*.safetensors"))
    print(f"Model: {NUM_LAYERS} layers, {NUM_EXPERTS} experts, {len(shard_files)} shards")

    # First pass: collect pinned weights, write expert layers per shard
    pinned = {}
    expert_layers_done = set()
    expert_keys = {}  # layer -> [(name, tensor)]
    t0 = time.time()
    total_expert_bytes = 0

    for si, sf in enumerate(shard_files):
        print(f"\nShard {si+1}/{len(shard_files)}: {os.path.basename(sf)}")
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

        # Write any complete layers (all 9 tensors present)
        for layer_idx in sorted(expert_keys.keys()):
            if layer_idx in expert_layers_done:
                continue
            if len(expert_keys[layer_idx]) < len(tensor_order):
                continue

            lt = expert_keys[layer_idx]
            tensor_info = {}
            offset = 0
            for tname in tensor_order:
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

            layer_path = f"{OUTPUT_DIR}/bin/moe_layer_{layer_idx:02d}.bin"
            layer_bytes = PAGE_SIZE
            with open(layer_path, "wb") as f:
                f.write(header_padded)
                for eid in range(NUM_EXPERTS):
                    expert_data = bytearray()
                    for tname in tensor_order:
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
            print(f"  Layer {layer_idx:2d}/{NUM_LAYERS}: {layer_bytes/1e6:.1f} MB ({elapsed:.0f}s)")

        del w; gc.collect()

        # Delete the shard file to free disk
        os.remove(sf)
        print(f"  Deleted {os.path.basename(sf)} to free disk")

    # Save pinned
    pinned_bytes = sum(v.nbytes for v in pinned.values())
    mx.save_safetensors(f"{OUTPUT_DIR}/pinned.safetensors", pinned)
    print(f"\nSaved pinned.safetensors: {pinned_bytes/1e9:.2f} GB ({len(pinned)} keys)")
    del pinned; gc.collect()

    # Symlinks
    for i in range(NUM_LAYERS):
        src = f"moe_layer_{i:02d}.bin"
        dst = f"{OUTPUT_DIR}/bin/layer_{i:02d}.bin"
        if os.path.exists(f"{OUTPUT_DIR}/bin/{src}") and not os.path.exists(dst):
            os.symlink(src, dst)

    # Config
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
    with open(f"{OUTPUT_DIR}/config.json", "w") as f:
        json.dump(stream_config, f, indent=2)

    import shutil
    for tf in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
               "added_tokens.json", "vocab.json", "merges.txt"]:
        src = f"{MLX_MODEL_DIR}/{tf}"
        if os.path.exists(src):
            shutil.copy(src, f"{OUTPUT_DIR}/{tf}")

    print(f"\nDone in {time.time()-t0:.0f}s!")
    print(f"Pinned: {pinned_bytes/1e9:.2f} GB, Experts: {total_expert_bytes/1e9:.2f} GB")

if __name__ == "__main__":
    main()
