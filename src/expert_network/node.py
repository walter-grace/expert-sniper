#!/usr/bin/env python3
"""Expert node — serves an expert partition from this machine's SSD/RAM.

Loads its partition of the sniper streaming format (bin/layer_XX.bin) into
unified memory once, then answers /compute_bin with the partial FFN output
for whichever active experts it owns. Partition can be an explicit range
(--partition 0-63), an id list (--experts 0,3,9), or derived from a shared
roster via rendezvous hashing (--roster a,b,c --me b) so every node computes
the same assignment with no coordination.

Security: binds 127.0.0.1 by default. There is NO authentication — pass
--host 0.0.0.0 only on a network where an open compute endpoint is fine.
"""
import argparse
import fcntl
import gc
import json
import os
import time

import numpy as np

from .protocol import unpack_request, pack_response
from .hrw import partition as hrw_partition

PAGE_SIZE = 16384
F_NOCACHE = 48
BITS = 4
GROUP_SIZE = 64


def parse_layer_header(layer_path):
    with open(layer_path, "rb") as f:
        raw = f.read(PAGE_SIZE)
    return json.loads(raw.rstrip(b"\x00"))


def load_expert_from_bin(fd, expert_id, layout, tensor_layout):
    import mlx.core as mx
    MLX_DTYPES = {
        "uint32": mx.uint32, "float16": mx.float16,
        "float32": mx.float32, "bfloat16": mx.bfloat16,
    }
    offset = layout["data_start"] + expert_id * layout["expert_block_size"]
    raw = os.pread(fd, layout["expert_block_size"], offset)
    result = {}
    for name, info in tensor_layout.items():
        arr_bytes = raw[info["inner_offset"]:info["inner_offset"] + info["nbytes"]]
        dtype = MLX_DTYPES.get(info["dtype"].replace("mlx.core.", ""), mx.float16)
        flat = mx.array(np.frombuffer(arr_bytes, dtype=np.uint8))
        result[name] = flat.view(dtype).reshape(info["shape_per_expert"])
    return result


def load_partition(model_dir, expert_ids, num_layers):
    """Load the partition, materialized in unified memory. F_NOCACHE on the
    load fd — this is a one-shot bulk read; without it a multi-GB partition
    load pollutes the OS page cache for nothing."""
    import mlx.core as mx
    expert_dir = os.path.join(model_dir, "bin")
    experts = {}
    total_bytes = 0
    for layer_idx in range(num_layers):
        layer_path = os.path.join(expert_dir, f"layer_{layer_idx:02d}.bin")
        header = parse_layer_header(layer_path)
        layout = header["layout"]
        fd = os.open(layer_path, os.O_RDONLY)
        fcntl.fcntl(fd, F_NOCACHE, 1)
        try:
            for eid in expert_ids:
                experts[(layer_idx, eid)] = load_expert_from_bin(
                    fd, eid, layout, layout["tensors"])
                total_bytes += layout["expert_block_size"]
        finally:
            os.close(fd)
        arrays = []
        for eid in expert_ids:
            arrays.extend(experts[(layer_idx, eid)].values())
        mx.eval(*arrays)
        print(f"  Layer {layer_idx:2d}: {len(expert_ids)} experts loaded")
    return experts, total_bytes


def compute_expert_ffn(x, expert_data_list, local_indices, top_k_weights):
    """gather_qmm SwiGLU FFN for the experts this node owns."""
    import mlx.core as mx
    import mlx.nn as nn
    active_ids = [eid for eid, _ in expert_data_list]
    data_map = dict(expert_data_list)

    def stack_proj(proj):
        w = mx.stack([data_map[eid][f"switch_mlp.{proj}.weight"] for eid in active_ids])
        s = mx.stack([data_map[eid][f"switch_mlp.{proj}.scales"] for eid in active_ids])
        b = mx.stack([data_map[eid][f"switch_mlp.{proj}.biases"] for eid in active_ids])
        return w, s, b

    gate_w, gate_s, gate_b = stack_proj("gate_proj")
    up_w, up_s, up_b = stack_proj("up_proj")
    down_w, down_s, down_b = stack_proj("down_proj")

    x_exp = mx.expand_dims(x, (-2, -3))
    gate_out = mx.gather_qmm(x_exp, gate_w, scales=gate_s, biases=gate_b,
                             rhs_indices=local_indices, transpose=True,
                             group_size=GROUP_SIZE, bits=BITS)
    up_out = mx.gather_qmm(x_exp, up_w, scales=up_s, biases=up_b,
                           rhs_indices=local_indices, transpose=True,
                           group_size=GROUP_SIZE, bits=BITS)
    hidden = nn.silu(gate_out) * up_out
    down_out = mx.gather_qmm(hidden, down_w, scales=down_s, biases=down_b,
                             rhs_indices=local_indices, transpose=True,
                             group_size=GROUP_SIZE, bits=BITS)
    out = down_out.squeeze(-2)
    return (out * top_k_weights[..., None]).sum(axis=-2)


def create_app(experts, expert_ids, num_layers, num_experts):
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    import mlx.core as mx

    app = FastAPI(title="Expert Network Node")
    partition_set = set(expert_ids)
    partition_arr = np.zeros(num_experts, dtype=bool)
    partition_arr[list(partition_set)] = True
    stats = {"count": 0, "time": 0.0}

    def compute(layer_idx, req_ids, h_np, inds_np, weights_np):
        # Experts this node owns AND actually has loaded for this layer —
        # one set, so local indices can never silently point at the wrong
        # expert.
        my_ids = sorted(eid for eid in req_ids
                        if eid in partition_set and (layer_idx, eid) in experts)
        if not my_ids:
            return 0, np.zeros(h_np.shape, dtype=np.float16)

        # Vectorized lookup table instead of a per-element Python loop.
        lut = np.zeros(num_experts, dtype=np.int32)
        lut[my_ids] = np.arange(len(my_ids), dtype=np.int32)
        local_indices = mx.array(lut[inds_np])
        mask = partition_arr[inds_np].astype(np.float32)
        masked_weights = mx.array(weights_np) * mx.array(mask)

        expert_data_list = [(eid, experts[(layer_idx, eid)]) for eid in my_ids]
        result = compute_expert_ffn(mx.array(h_np), expert_data_list,
                                    local_indices, masked_weights)
        mx.eval(result)
        return len(my_ids), np.array(result.astype(mx.float16))

    @app.post("/compute_bin")
    async def compute_bin(request: Request):
        t0 = time.time()
        layer_idx, req_ids, h_np, inds_np, weights_np = unpack_request(
            await request.body())
        n_computed, out_np = compute(layer_idx, req_ids, h_np, inds_np, weights_np)
        stats["count"] += 1
        stats["time"] += time.time() - t0
        return Response(content=pack_response(n_computed, out_np),
                        media_type="application/octet-stream")

    @app.get("/health")
    async def health():
        avg_ms = (stats["time"] / stats["count"] * 1000) if stats["count"] else 0
        return {
            "status": "ok",
            "experts_per_layer": len(partition_set),
            "expert_ids": sorted(partition_set)[:8] + (
                ["..."] if len(partition_set) > 8 else []),
            "total_experts_loaded": len(experts),
            "num_layers": num_layers,
            "memory_gb": round(mx.get_active_memory() / 1e9, 2),
            "compute_requests": stats["count"],
            "avg_compute_ms": round(avg_ms, 2),
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="Expert Network node")
    parser.add_argument("--model-dir", required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--partition", help="Contiguous range, e.g. '0-63'")
    g.add_argument("--experts", help="Explicit ids, e.g. '0,3,9,12'")
    g.add_argument("--roster", help="Comma-separated node ids; partition is "
                                    "derived by rendezvous hashing (needs --me)")
    parser.add_argument("--me", help="This node's id within --roster")
    parser.add_argument("--port", type=int, default=8301)
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind address (0.0.0.0 exposes an UNAUTHENTICATED "
                             "compute endpoint — trusted networks only)")
    parser.add_argument("--memory-limit-gb", type=float, default=None,
                        help="Metal memory cap (default: partition size + 25%%)")
    args = parser.parse_args()

    with open(os.path.join(os.path.expanduser(args.model_dir), "config.json")) as f:
        config = json.load(f)
    num_layers = config["num_hidden_layers"]
    num_experts = config["num_experts"]

    if args.partition:
        p0, p1 = (int(x) for x in args.partition.split("-"))
        expert_ids = list(range(p0, p1 + 1))
    elif args.experts:
        expert_ids = sorted(int(x) for x in args.experts.split(","))
    else:
        if not args.me:
            parser.error("--roster requires --me")
        expert_ids = hrw_partition(args.roster.split(","), args.me, num_experts)
        print(f"HRW partition for {args.me!r}: {len(expert_ids)} experts")
    assert expert_ids and 0 <= min(expert_ids) and max(expert_ids) < num_experts

    model_dir = os.path.expanduser(args.model_dir)
    hdr = parse_layer_header(os.path.join(model_dir, "bin", "layer_00.bin"))
    block = hdr["layout"]["expert_block_size"]
    est_gb = len(expert_ids) * num_layers * block / 1e9

    import mlx.core as mx
    limit = args.memory_limit_gb or est_gb * 1.25 + 0.5
    mx.set_memory_limit(int(limit * 1024**3))
    mx.set_cache_limit(256 * 1024**2)

    print(f"Expert Network node — {len(expert_ids)} experts/layer x "
          f"{num_layers} layers = {est_gb:.1f} GB est. (limit {limit:.1f} GB)")
    t0 = time.time()
    experts, total_bytes = load_partition(model_dir, expert_ids, num_layers)
    gc.collect(); mx.clear_cache()
    print(f"Loaded {len(experts)} blocks ({total_bytes/1e9:.1f} GB) "
          f"in {time.time()-t0:.1f}s; active {mx.get_active_memory()/1e9:.2f} GB")

    app = create_app(experts, expert_ids, num_layers, num_experts)
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
