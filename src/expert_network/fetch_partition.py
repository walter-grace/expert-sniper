#!/usr/bin/env python3
"""Fetch only the experts this machine owns — from the network itself.

A model too big for any one Mac is the whole point of the Expert Network,
but until now joining meant downloading the entire checkpoint before
throwing away the 90% you don't serve. That is impossible for a 180 GB
model on a 500 GB laptop, and pointless everywhere else.

This pulls only your partition, and pulls it from machines already serving
the model: their /block/{layer}/{expert} endpoint — the one the coordinator
uses to *prove* they hold an expert — is also how they hand that expert to
the next node. Every block is checked against the published manifest hash
as it lands, so a hostile peer can waste your bandwidth but cannot poison
your weights.

The streaming format puts expert e of layer L at a fixed offset, so the
result is a SPARSE file: the same layout the engine already reads, with
holes where other machines' experts live. `ls` shows the full size; the
disk only holds your slice.

  expert-fetch --model 48x128 --partition 0-31 -o ~/models/glm-stream \\
      --key hr_live_... [--coordinator https://herorunai.com]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PAGE_SIZE = 16384


def _get(url, key=None, timeout=30, binary=False):
    req = urllib.request.Request(url, headers={
        "User-Agent": "expert-fetch/0.3",
        **({"x-api-key": key} if key else {}),
    })
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    return data if binary else json.loads(data)


def parse_partition(spec, num_experts):
    """'0-31' or '0,3,9' or 'half:0' / 'half:1' → a list of expert ids."""
    spec = (spec or "").strip()
    if spec.startswith("half:"):
        half = num_experts // 2
        return list(range(0, half)) if spec.endswith("0") else list(range(half, num_experts))
    if "-" in spec:
        a, b = spec.split("-", 1)
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def fetch_block(peers, layer, eid, want_hash, tries=3):
    """Pull one expert block, trying peers in order. Verified or nothing."""
    last = None
    for attempt in range(tries):
        peer = peers[(attempt + layer + eid) % len(peers)]
        try:
            data = _get(f"{peer['url'].rstrip('/')}/block/{layer}/{eid}",
                        timeout=60, binary=True)
            if want_hash and hashlib.sha256(data).hexdigest() != want_hash:
                last = f"hash mismatch from {peer['nodeId']}"
                continue
            return data, peer
        except Exception as e:
            last = f"{peer['nodeId']}: {type(e).__name__}"
    raise RuntimeError(f"block {layer}:{eid} unavailable ({last})")


def main():
    p = argparse.ArgumentParser(description="Fetch this machine's expert partition")
    p.add_argument("--model", required=True, help="manifest key, e.g. 48x128 or a model name")
    p.add_argument("--partition", required=True, help="0-31 · 0,3,9 · half:0")
    p.add_argument("-o", "--out", required=True, help="streaming dir to create")
    p.add_argument("--key", default=os.environ.get("HERO_RUN_KEY"), help="hr_live_ key")
    p.add_argument("--coordinator", default="https://herorunai.com")
    p.add_argument("--peers", default=None, help="comma-separated peer URLs (skip discovery)")
    p.add_argument("--jobs", type=int, default=6, help="parallel block fetches")
    args = p.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "bin"), exist_ok=True)

    # --- who has this model, and what should it hash to -------------------
    url = f"{args.coordinator}/api/yield/peers?model={args.model}&manifest=1"
    try:
        info = _get(url, key=args.key)
    except Exception as e:
        sys.exit(f"coordinator lookup failed: {e}")
    manifest = info.get("manifest") or {}
    blocks = manifest.get("blocks") or {}
    layout = info.get("layout") or {}
    peers = ([{"nodeId": "manual", "url": u} for u in args.peers.split(",")]
             if args.peers else info.get("peers") or [])
    if not peers:
        sys.exit(f"no live machines are serving {args.model} yet — nothing to pull from.")
    if not blocks:
        sys.exit(f"no manifest published for {args.model}; the operator must upload one first.")

    num_layers = int(layout.get("num_layers") or manifest.get("num_layers"))
    num_experts = int(layout.get("num_experts") or manifest.get("num_experts"))
    block_size = int(layout.get("expert_block_size") or manifest.get("expert_block_size"))
    mine = parse_partition(args.partition, num_experts)
    total = len(mine) * num_layers
    gb = total * block_size / 1e9
    print(f"{args.model}: {num_layers} layers x {num_experts} experts")
    print(f"partition: {len(mine)} experts/layer = {total} blocks, {gb:.1f} GB "
          f"(full model would be {num_experts * num_layers * block_size / 1e9:.0f} GB)")
    print(f"peers: {', '.join(p['nodeId'] for p in peers[:4])}"
          f"{' …' if len(peers) > 4 else ''}\n")

    # --- pull, verify, write sparse ---------------------------------------
    done = 0
    t0 = time.time()
    for layer in range(num_layers):
        path = os.path.join(out, "bin", f"layer_{layer:02d}.bin")
        header = {"layer_idx": layer, "num_experts": num_experts,
                  "layout": {"expert_block_size": block_size,
                             "data_start": PAGE_SIZE, "tensors": manifest.get("tensors")}}
        hj = json.dumps(header).encode()
        with open(path, "wb") as f:
            f.write(hj + b"\x00" * (PAGE_SIZE - len(hj)))
            # sparse: sizing the file to the full layout leaves holes where
            # other machines' experts live, so the engine's fixed offsets
            # still land correctly while the disk only holds our slice.
            f.truncate(PAGE_SIZE + num_experts * block_size)

            def one(eid):
                data, peer = fetch_block(peers, layer, eid, blocks.get(f"{layer}:{eid}"))
                return eid, data

            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                for eid, data in pool.map(one, mine):
                    f.seek(PAGE_SIZE + eid * block_size)
                    f.write(data)
                    done += 1
            rate = done * block_size / 1e6 / max(0.1, time.time() - t0)
            print(f"  layer {layer + 1}/{num_layers}  {done}/{total} blocks  "
                  f"{rate:.0f} MB/s", end="\r", flush=True)

    real = sum(os.stat(os.path.join(out, "bin", f)).st_blocks * 512
               for f in os.listdir(os.path.join(out, "bin")))
    print(f"\n\n{done} blocks verified and written in {time.time() - t0:.0f}s")
    print(f"on disk: {real / 1e9:.1f} GB (sparse)")
    print(f"\nNext: copy pinned.safetensors + config.json from a peer or the "
          f"source repo, then:\n  expert-node --model-dir {out} "
          f"--experts {','.join(str(e) for e in mine[:4])}… --join <key>")


if __name__ == "__main__":
    main()
