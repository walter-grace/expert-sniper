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
import random
import sys
import threading
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


class PeerRanker:
    """Peers ranked by *measured* throughput, not by the coordinator's order.

    Every successful transfer updates a per-peer MB/s estimate (EWMA), every
    failure halves it. Dispatch draws peers at random weighted by that
    estimate, with a floor so a slow peer still takes a small share of the
    load instead of sitting idle: that keeps the pull spread across the
    swarm, and keeps fresh measurements coming in for every peer."""

    FLOOR = 0.1     # slowest peer gets at least 10% of the fastest's weight
    ALPHA = 0.3     # EWMA weight of the newest measurement

    def __init__(self, peers, seed=None):
        self.peers = list(peers)
        self.speed = {p["nodeId"]: None for p in self.peers}   # MB/s
        self._lock = threading.Lock()
        self._rng = random.Random(seed)

    def probe(self, layer, eid, want_hash, timeout=60):
        """One timed block fetch per peer. Returns {nodeId: (data|None, sec)}.

        A real block is the throughput signal we care about; /health only
        measures latency. Peers that fail or serve a bad hash are ranked
        last (speed 0) but not dropped: they may recover mid-pull."""
        results = {}
        for peer in self.peers:
            t = time.time()
            try:
                data = _get(f"{peer['url'].rstrip('/')}/block/{layer}/{eid}",
                            timeout=timeout, binary=True)
                sec = max(1e-6, time.time() - t)
                if want_hash and hashlib.sha256(data).hexdigest() != want_hash:
                    raise ValueError("hash mismatch")
                self.speed[peer["nodeId"]] = len(data) / 1e6 / sec
                results[peer["nodeId"]] = (data, sec)
            except Exception:
                self.speed[peer["nodeId"]] = 0.0
                results[peer["nodeId"]] = (None, time.time() - t)
        return results

    def record(self, node_id, nbytes, sec):
        mbps = nbytes / 1e6 / max(1e-6, sec)
        with self._lock:
            old = self.speed.get(node_id)
            self.speed[node_id] = mbps if not old else (
                (1 - self.ALPHA) * old + self.ALPHA * mbps)

    def penalize(self, node_id):
        with self._lock:
            self.speed[node_id] = (self.speed.get(node_id) or 0.0) / 2

    def order(self):
        """Peers in dispatch order: a weighted draw without replacement, so
        the first choice is usually the fastest peer and failover walks the
        rest, also fastest-first on average."""
        with self._lock:
            speeds = {n: (s or 0.0) for n, s in self.speed.items()}
            best = max(speeds.values(), default=0.0)
            remaining = list(self.peers)
            if best <= 0:
                self._rng.shuffle(remaining)
                return remaining
            weights = {n: max(s, best * self.FLOOR) for n, s in speeds.items()}
            out = []
            while remaining:
                pick = self._rng.choices(
                    remaining, weights=[weights[p["nodeId"]] for p in remaining])[0]
                remaining.remove(pick)
                out.append(pick)
            return out

    def table(self):
        rows = sorted(self.peers, key=lambda p: -(self.speed[p["nodeId"]] or 0))
        return "\n".join(
            f"  {p['nodeId'][:24]:<24} "
            + (f"{self.speed[p['nodeId']]:8.1f} MB/s"
               if self.speed[p["nodeId"]] else "    failed")
            for p in rows)


def fetch_block(ranker, layer, eid, want_hash, tries=3):
    """Pull one expert block, fastest peers first. Verified or nothing.

    Returns (data, peer) so the caller can credit the peer that served it:
    seeding is paid work, and the receiver is the only party that knows
    what actually arrived intact."""
    last = None
    order = ranker.order()
    for attempt in range(tries):
        peer = order[attempt % len(order)]
        t = time.time()
        try:
            data = _get(f"{peer['url'].rstrip('/')}/block/{layer}/{eid}",
                        timeout=60, binary=True)
            if want_hash and hashlib.sha256(data).hexdigest() != want_hash:
                last = f"hash mismatch from {peer['nodeId']}"
                ranker.penalize(peer["nodeId"])
                continue
            ranker.record(peer["nodeId"], len(data), time.time() - t)
            return data, peer
        except Exception as e:
            last = f"{peer['nodeId']}: {type(e).__name__}"
            ranker.penalize(peer["nodeId"])
    raise RuntimeError(f"block {layer}:{eid} unavailable ({last})")


def _have_block(fd, offset, size, want_hash):
    """True if the bytes already at this block's fixed offset verify.

    Only our own offsets are ever read; the holes belonging to other
    machines' experts are never touched."""
    if not want_hash:
        return False
    try:
        raw = os.pread(fd, size, offset)
    except OSError:
        return False
    return len(raw) == size and hashlib.sha256(raw).hexdigest() == want_hash


def fetch_partition(out, blocks, mine, num_layers, num_experts, block_size,
                    peers, jobs=6, tensors=None, log=print, seed=None):
    """Pull every (layer, expert) in `mine` into sparse layer files under
    `out/bin`, skipping blocks the files already hold intact.

    Returns a stats dict: fetched, skipped, bytes (delivered this run),
    seconds, delivered ({nodeId: bytes}; only bytes that actually arrived
    this run, so skipped blocks credit nobody), speeds ({nodeId: MB/s})."""
    os.makedirs(os.path.join(out, "bin"), exist_ok=True)
    ranker = PeerRanker(peers, seed=seed)
    total = len(mine) * num_layers
    stats = {"fetched": 0, "skipped": 0, "bytes": 0, "delivered": {}}
    t0 = time.time()
    probed = False

    def credit(peer, n):
        stats["delivered"][peer["nodeId"]] = stats["delivered"].get(peer["nodeId"], 0) + n
        stats["bytes"] += n

    for layer in range(num_layers):
        path = os.path.join(out, "bin", f"layer_{layer:02d}.bin")
        header = {"layer_idx": layer, "num_experts": num_experts,
                  "layout": {"expert_block_size": block_size,
                             "data_start": PAGE_SIZE, "tensors": tensors}}
        hj = json.dumps(header).encode()
        size = PAGE_SIZE + num_experts * block_size
        # r+b keeps a previous run's blocks; "wb" would truncate them away.
        exists = os.path.exists(path) and os.path.getsize(path) == size
        with open(path, "r+b" if exists else "wb") as f:
            fd = f.fileno()
            os.pwrite(fd, hj + b"\x00" * (PAGE_SIZE - len(hj)), 0)
            if not exists:
                # sparse: sizing the file to the full layout leaves holes
                # where other machines' experts live, so the engine's fixed
                # offsets still land while the disk only holds our slice.
                f.truncate(size)

            todo = []
            for eid in mine:
                want = blocks.get(f"{layer}:{eid}")
                if _have_block(fd, PAGE_SIZE + eid * block_size, block_size, want):
                    stats["skipped"] += 1
                else:
                    todo.append(eid)

            if todo and not probed:
                # the first block we actually need doubles as the probe:
                # every peer serves it once, timed, and the fastest copy is
                # the one that lands on disk.
                probed = True
                eid = todo[0]
                res = ranker.probe(layer, eid, blocks.get(f"{layer}:{eid}"))
                log("peer speeds (probe, one block each):")
                log(ranker.table())
                got = None
                for peer in peers:
                    data, _ = res.get(peer["nodeId"], (None, 0))
                    if data is not None:
                        credit(peer, len(data))
                        got = data
                if got is not None:
                    os.pwrite(fd, got, PAGE_SIZE + eid * block_size)
                    stats["fetched"] += 1
                    todo = todo[1:]

            def one(eid):
                data, peer = fetch_block(ranker, layer, eid, blocks.get(f"{layer}:{eid}"))
                return eid, data, peer

            with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
                for eid, data, peer in pool.map(one, todo):
                    os.pwrite(fd, data, PAGE_SIZE + eid * block_size)
                    credit(peer, len(data))
                    stats["fetched"] += 1
            done = stats["fetched"] + stats["skipped"]
            rate = stats["bytes"] / 1e6 / max(0.1, time.time() - t0)
            log(f"  layer {layer + 1}/{num_layers}  {done}/{total} blocks  "
                f"({stats['skipped']} already had)  {rate:.0f} MB/s", end="\r", flush=True)

    stats["seconds"] = time.time() - t0
    stats["speeds"] = dict(ranker.speed)
    return stats


def fetch_from_hf(args, out):
    """Take a partition directly from the source checkpoint on HuggingFace.

    Same sparse layout on disk, same header, same block hashes — the only
    difference is where the bytes come from. If the coordinator already has
    a manifest for the model, every block is verified against it; if not
    (you are the first machine), the hashes of what you fetched are written
    to manifest.partial.json so the publisher can merge them."""
    from .hf_source import HFCheckpoint
    hf = HFCheckpoint(args.from_hf, jobs=max(2, args.jobs))
    num_layers, num_experts = hf.num_layers, hf.num_experts
    lay0 = hf.public_layout(0)
    block_size = lay0["expert_block_size"]
    if args.roster == "auto" or (args.roster and args.me):
        # Same rule the node applies at start: rendezvous hashing over the
        # live roster (plus me), so a machine knows its share before it
        # downloads a byte. Re-run later to top up if the roster grew.
        from .hrw import partition as hrw_partition
        me = args.me or os.uname().nodename.lower()[:20]
        model_key = args.model or f"{num_layers}x{num_experts}"
        if args.roster == "auto":
            try:
                info = _get(f"{args.coordinator}/api/yield/roster?model={model_key}&me={me}")
                roster = info.get("roster") or [me]
            except Exception as e:  # noqa: BLE001
                print(f"roster lookup failed ({type(e).__name__}); taking the whole model")
                roster = [me]
        else:
            roster = args.roster.split(",")
        if me not in roster:
            roster.append(me)
        mine = hrw_partition(roster, me, num_experts)
        print(f"roster: {len(roster)} machine(s) on {model_key} — my share as {me!r}: "
              f"{len(mine)}/{num_experts} experts")
        args.partition = ",".join(str(e) for e in mine)
    elif args.partition:
        mine = parse_partition(args.partition, num_experts)
    else:
        sys.exit("give --partition, or --roster auto --me <node-id>")
    total = len(mine) * num_layers
    print(f"{args.from_hf}: {num_layers} layers x {num_experts} experts, "
          f"{block_size / 1e6:.2f} MB blocks")
    print(f"partition: {len(mine)} experts/layer = {total} blocks, "
          f"{total * block_size / 1e9:.1f} GB (full model would be "
          f"{num_experts * num_layers * block_size / 1e9:.0f} GB)\n")

    want = {}
    if args.model:
        try:
            info = _get(f"{args.coordinator}/api/yield/peers?model={args.model}&manifest=1",
                        key=args.key)
            want = (info.get("manifest") or {}).get("blocks") or {}
            print(f"verifying against the published manifest ({len(want)} blocks)")
        except Exception as e:  # noqa: BLE001
            print(f"no manifest to verify against ({type(e).__name__}); trusting the source")

    hf.write_config(out)
    got_hashes = {}
    stats = {"fetched": 0, "skipped": 0, "bytes": 0}
    t0 = time.time()
    for layer in range(num_layers):
        lay = hf.expert_layout(layer)
        tensors = {k: v for k, v in lay["tensors"].items()}
        path = os.path.join(out, "bin", f"layer_{layer:02d}.bin")
        header = {"layer_idx": layer, "num_experts": num_experts,
                  "layout": {"expert_block_size": block_size,
                             "data_start": PAGE_SIZE, "tensors": tensors}}
        hj = json.dumps(header).encode()
        size = PAGE_SIZE + num_experts * block_size
        exists = os.path.exists(path) and os.path.getsize(path) == size
        with open(path, "r+b" if exists else "wb") as f:
            fd = f.fileno()
            os.pwrite(fd, hj + b"\x00" * (PAGE_SIZE - len(hj)), 0)
            if not exists:
                f.truncate(size)
            todo = []
            for eid in mine:
                w = want.get(f"{layer}:{eid}")
                if w and _have_block(fd, PAGE_SIZE + eid * block_size, block_size, w):
                    stats["skipped"] += 1
                    got_hashes[f"{layer}:{eid}"] = w
                else:
                    todo.append(eid)

            def one(eid, layer=layer, lay=lay):
                data = hf.fetch_expert_block(layer, eid, lay)
                h = hashlib.sha256(data).hexdigest()
                w = want.get(f"{layer}:{eid}")
                if w and w != h:
                    raise RuntimeError(f"block {layer}:{eid} from {args.from_hf} does not "
                                       f"match the published manifest — refusing to write it")
                return eid, data, h
            with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
                for eid, data, h in pool.map(one, todo):
                    os.pwrite(fd, data, PAGE_SIZE + eid * block_size)
                    got_hashes[f"{layer}:{eid}"] = h
                    stats["fetched"] += 1
                    stats["bytes"] += len(data)
        done = stats["fetched"] + stats["skipped"]
        rate = stats["bytes"] / 1e6 / max(0.1, time.time() - t0)
        print(f"  layer {layer + 1}/{num_layers}  {done}/{total} blocks  "
              f"({stats['skipped']} already had)  {rate:.0f} MB/s", end="\r", flush=True)
    secs = time.time() - t0
    print(f"\n{len(got_hashes)} blocks on disk in {secs:.0f}s: {stats['fetched']} fetched, "
          f"{stats['skipped']} already had; {stats['bytes'] / 1e9:.1f} GB at "
          f"{stats['bytes'] / 1e6 / max(0.1, secs):.0f} MB/s")
    with open(os.path.join(out, "manifest.partial.json"), "w") as f:
        json.dump({"model_type": hf.tc.get("model_type"), "source": args.from_hf,
                   "num_layers": num_layers, "num_experts": num_experts,
                   "expert_block_size": block_size, "tensors": lay0["tensors"],
                   "blocks": got_hashes}, f)

    if args.pinned:
        print("\npinned trunk:")
        hf.download_pinned(out)
    if args.ngram:
        print("\nn-gram tables:")
        hf.download_ngram(out)

    real = sum(os.stat(os.path.join(dp, fn)).st_blocks * 512
               for dp, _, fns in os.walk(os.path.join(out, "bin")) for fn in fns)
    print(f"on disk: {real / 1e9:.1f} GB (sparse)")
    part = f"--roster auto --me {args.me}" if args.roster else f"--partition {args.partition}"
    nxt = "" if args.pinned else f"\n  expert-fetch --from-hf {args.from_hf} {part} -o {out} --pinned   # trunk, if this machine also drives"
    print(f"\nNext:{nxt}\n  expert-node --model-dir {out} --roster auto --me <id> --join <key>")


def main():
    p = argparse.ArgumentParser(description="Fetch this machine's expert partition")
    p.add_argument("--model", default=None,
                   help="manifest key, e.g. 48x128 or a model name (optional with --from-hf)")
    p.add_argument("--partition", default=None, help="0-31 · 0,3,9 · half:0")
    p.add_argument("--roster", default=None,
                   help="'auto' (ask the coordinator) or a comma list of node ids; "
                        "with --me, computes the partition by rendezvous hashing")
    p.add_argument("--me", default=None, help="this machine's node id (with --roster)")
    p.add_argument("-o", "--out", required=True, help="streaming dir to create")
    p.add_argument("--key", default=os.environ.get("HERO_RUN_KEY"), help="hr_live_ key")
    p.add_argument("--coordinator", default="https://herorunai.com")
    p.add_argument("--peers", default=None, help="comma-separated peer URLs (skip discovery)")
    p.add_argument("--jobs", type=int, default=6, help="parallel block fetches")
    p.add_argument("--node-id", default=None,
                   help="this machine's node id, for crediting the peers that seeded it")
    p.add_argument("--from-hf", default=None, metavar="REPO",
                   help="pull your partition straight from a HuggingFace MLX checkpoint "
                        "by byte range (e.g. Vontra/Qwen3.8-Flash-Next-MLX-4bit); "
                        "no peer needs to hold the model first")
    p.add_argument("--pinned", action="store_true",
                   help="with --from-hf: also fetch pinned.safetensors (the resident trunk)")
    p.add_argument("--ngram", action="store_true",
                   help="with --from-hf: also fetch the n-gram tables (drivers only)")
    p.add_argument("--no-attest", action="store_true",
                   help="do not report who delivered the blocks (they go unpaid)")
    args = p.parse_args()

    out = os.path.expanduser(args.out)
    os.makedirs(os.path.join(out, "bin"), exist_ok=True)

    if args.from_hf:
        return fetch_from_hf(args, out)
    if not args.model:
        p.error("--model is required unless --from-hf is given")

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

    if not args.partition:
        p.error("--partition is required (or use --from-hf with --roster auto)")
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
    stats = fetch_partition(out, blocks, mine, num_layers, num_experts, block_size,
                            peers, jobs=args.jobs, tensors=manifest.get("tensors"))
    delivered = stats["delivered"]
    done = stats["fetched"] + stats["skipped"]

    real = sum(os.stat(os.path.join(out, "bin", f)).st_blocks * 512
               for f in os.listdir(os.path.join(out, "bin")))
    # --- credit the machines that seeded us --------------------------------
    if delivered and not args.no_attest and args.key:
        me = args.node_id or f"{os.uname().nodename.lower()[:16]}-fetch"
        payload = json.dumps({
            "receiverId": me,
            "deliveries": [{"nodeId": n, "bytes": b} for n, b in delivered.items()],
        }).encode()
        req = urllib.request.Request(
            f"{args.coordinator}/api/yield/attest", data=payload,
            headers={"Content-Type": "application/json", "x-api-key": args.key,
                     "User-Agent": "expert-fetch/0.3"})
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                res = json.loads(r.read())
            n = len(res.get("credited") or [])
            print(f"\n\ncredited {n} seeder(s) for delivering your partition")
        except Exception as e:
            print(f"\n\ncould not credit seeders ({type(e).__name__}) — "
                  f"they served you anyway")

    secs = stats["seconds"]
    print(f"\n{done} blocks verified in {secs:.0f}s: {stats['fetched']} fetched, "
          f"{stats['skipped']} already had")
    print(f"downloaded {stats['bytes'] / 1e6:.0f} MB at "
          f"{stats['bytes'] / 1e6 / max(0.1, secs):.1f} MB/s")
    for n, b in sorted(delivered.items(), key=lambda kv: -kv[1]):
        print(f"  {n[:24]:<24} {b / 1e6:8.0f} MB")
    print(f"on disk: {real / 1e9:.1f} GB (sparse)")
    print(f"\nNext: copy pinned.safetensors + config.json from a peer or the "
          f"source repo, then:\n  expert-node --model-dir {out} "
          f"--experts {','.join(str(e) for e in mine[:4])}… --join <key>")


if __name__ == "__main__":
    main()
