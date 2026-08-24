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


def create_app(experts, expert_ids, num_layers, num_experts, model_dir=None):
    from fastapi import FastAPI, Request
    from fastapi.responses import Response
    import mlx.core as mx

    app = FastAPI(title="Expert Network Node")
    expert_dir = os.path.join(model_dir, "bin") if model_dir else None
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

    @app.get("/block/{layer_idx}/{expert_id}")
    async def block(layer_idx: int, expert_id: int):
        """Serve one raw expert block. The Machine Yield coordinator uses
        this as a proof-of-capability challenge: it knows the block's
        sha256 from the model manifest, so a node can't fake ownership or
        bandwidth — it either serves the right bytes fast or it doesn't."""
        if expert_dir is None or expert_id not in partition_set \
                or not 0 <= layer_idx < num_layers:
            return Response(status_code=404)
        layer_path = os.path.join(expert_dir, f"layer_{layer_idx:02d}.bin")
        header = parse_layer_header(layer_path)
        layout = header["layout"]
        fd = os.open(layer_path, os.O_RDONLY)
        try:
            raw = os.pread(fd, layout["expert_block_size"],
                           layout["data_start"]
                           + expert_id * layout["expert_block_size"])
        finally:
            os.close(fd)
        return Response(content=raw, media_type="application/octet-stream")

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


def _open_sealed(link, coordinator):
    """Decrypt a Hero sealed link locally. The #fragment key never reaches the coordinator; the
    plaintext never touches disk. Returns the dict of sealed variables."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except Exception:
        raise SystemExit("--sealed needs the network extra:  pip install 'mlx-expert-sniper[network]'")
    import base64, json as _json, urllib.request
    frag = link.split("#", 1)[1] if "#" in link else ""
    kb = frag[2:] if frag.startswith("k=") else frag
    path = link.split("#", 1)[0]
    sid = path.split("/s/")[1] if "/s/" in path else path.rstrip("/").split("/")[-1]
    raw = base64.urlsafe_b64decode(kb + "=" * (-len(kb) % 4))
    with urllib.request.urlopen(f"{coordinator}/api/seal?id={sid}&consume=1") as r:
        j = _json.load(r)
    if j.get("error"):
        raise SystemExit(f"sealed link: {j['error']}")
    pt = AESGCM(raw).decrypt(base64.b64decode(j["iv"]), base64.b64decode(j["ct"]), None)
    return _json.loads(pt.decode())


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
    parser.add_argument("--join", metavar="API_KEY", default=None,
                        help="Join Machine Yield with your Hero API key")
    parser.add_argument("--sealed", metavar="LINK", default=None,
                        help="Join with a Hero sealed link (/s/<id>#<key>) carrying HERO_RUN_KEY "
                             "instead of putting the key on the command line / in shell history")
    parser.add_argument("--coordinator", default="https://herorunai.com",
                        help="Machine Yield coordinator")
    parser.add_argument("--node-id", default=None,
                        help="Stable node name (default: hostname-port)")
    parser.add_argument("--advertise-url", default=None,
                        help="URL the coordinator can reach this node at "
                             "(default: http://<host>:<port>)")
    args = parser.parse_args()

    # A sealed link keeps the API key out of the command line and shell history: its #fragment key
    # decrypts here (never sent to the coordinator), yielding HERO_RUN_KEY (and optionally an
    # advertise URL). See herorunai.com/seal.
    if args.sealed and not args.join:
        v = _open_sealed(args.sealed, args.coordinator)
        args.join = v.get("HERO_RUN_KEY") or v.get("hero_run_key")
        if not args.advertise_url:
            args.advertise_url = v.get("PUBLIC_URL") or v.get("advertise_url")
        if not args.join:
            parser.error("sealed link did not contain HERO_RUN_KEY")

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

    app = create_app(experts, expert_ids, num_layers, num_experts,
                     model_dir=model_dir)

    if args.join:
        import socket
        from .yield_client import YieldClient
        node_id = args.node_id or f"{socket.gethostname()}-{args.port}"
        advertise = args.advertise_url or f"http://{args.host}:{args.port}"
        yc = YieldClient(args.coordinator, args.join, node_id, advertise, {
            "model_config": {"num_layers": num_layers,
                             "num_experts": num_experts,
                             "expert_block_size": block},
            "experts_per_layer": len(expert_ids),
            "expert_ids": sorted(expert_ids),
            "est_gb": round(est_gb, 2),
        })
        yc.join()
        yc.start_heartbeat(lambda: {
            "memory_gb": round(mx.get_active_memory() / 1e9, 2),
        })

    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
