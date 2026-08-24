#!/usr/bin/env python3
"""Machine Yield sidecar — join ANY inference box to the network.

The MLX node serves expert partitions; this sidecar serves PROOF. It sits
beside whatever engine the machine actually runs — FreeToken on a DGX
Spark or a gaming PC, vLLM, llama.cpp, Ollama — and gives Machine Yield
the three things it needs from hardware it can't run MLX on:

  1. proof of weights — the model files are content-addressed into fixed
     4 MB chunks; the coordinator fetches random chunks and re-hashes
     them against the published manifest while timing the transfer
  2. proof of liveness — heartbeats, plus a health poll of the local
     engine's OpenAI-compatible endpoint
  3. registration — node id, engine, models, committed bytes

DRAFT-ONLY nodes: omit --model-path and the sidecar becomes a pure Fast
Token draft node — it proxies /v1/* to its engine (a local model server
or a hosted provider like Cerebras via --engine-key) and earns standing
through uptime and counted draft service instead of weight custody.

Stdlib only: no mlx, no fastapi — it runs wherever Python runs.

  expert-sidecar --model-path ~/models/GLM-4.6-FTW \\
      --engine-url http://127.0.0.1:8000 \\
      --join <hero-api-key> --advertise-url https://spark1.example.com

Generate + publish the manifest once (operator side):
  expert-sidecar --model-path ~/models/GLM-4.6-FTW --write-manifest
"""
import argparse
import hashlib
import json
import os
import socket
import threading
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

from .yield_client import YieldClient

CHUNK = 4 * 1024 * 1024


def weight_files(model_path):
    """The files worth proving: everything above 1 MB, sorted for a
    stable chunk order."""
    if os.path.isfile(model_path):
        return [model_path]
    out = []
    for root, _, names in os.walk(model_path):
        for n in sorted(names):
            p = os.path.join(root, n)
            if not os.path.islink(p) and os.path.getsize(p) > 1024 * 1024:
                out.append(p)
    return sorted(out)


def build_index(model_path):
    """Global chunk index: [(file, offset, size)], stable across machines
    holding the same files."""
    index = []
    for f in weight_files(model_path):
        size = os.path.getsize(f)
        for off in range(0, size, CHUNK):
            index.append((f, off, min(CHUNK, size - off)))
    return index


def read_chunk(entry):
    f, off, size = entry
    with open(f, "rb") as fh:
        fh.seek(off)
        return fh.read(size)


def build_manifest(model_path, model_key):
    """Chunk manifest in the coordinator's block format: keys "0:<idx>"
    so the existing /block/{a}/{b} challenge path needs no changes."""
    index = build_index(model_path)
    blocks = {}
    for i, entry in enumerate(index):
        blocks[f"0:{i}"] = hashlib.sha256(read_chunk(entry)).hexdigest()
        if i % 200 == 0:
            print(f"  hashed {i + 1}/{len(index)} chunks", end="\r", flush=True)
    total = sum(s for _, _, s in index)
    print(f"\n{len(index)} chunks, {total / 1e9:.1f} GB")
    return {"model_type": model_key, "num_layers": 1, "num_experts": len(index),
            "expert_block_size": CHUNK, "blocks": blocks}


def engine_health(engine_url):
    try:
        with urllib.request.urlopen(f"{engine_url}/v1/models", timeout=5) as r:
            data = json.loads(r.read())
        return {"up": True,
                "models": [m.get("id") for m in data.get("data", [])][:8]}
    except Exception as e:
        return {"up": False, "error": str(e)[:120]}


def main():
    p = argparse.ArgumentParser(description="Machine Yield sidecar")
    p.add_argument("--model-path", default=None,
                   help="Model weights dir/file the engine serves "
                        "(omit for a draft-only node)")
    p.add_argument("--model-key", default=None,
                   help="Manifest key (default: basename of model-path)")
    p.add_argument("--engine-url", default="http://127.0.0.1:8000",
                   help="Local engine's OpenAI-compatible base URL")
    p.add_argument("--engine-name", default="freetoken",
                   help="Engine label reported to the coordinator")
    p.add_argument("--engine-key", default=None,
                   help="API key for a hosted engine (e.g. Cerebras); "
                        "injected as a Bearer header on proxied /v1 calls. "
                        "Falls back to ENGINE_API_KEY env.")
    p.add_argument("--port", type=int, default=8311)
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (challenge endpoint only serves "
                        "already-published weights, but keep it deliberate)")
    p.add_argument("--join", metavar="API_KEY", default=None)
    p.add_argument("--coordinator", default="https://herorunai.com")
    p.add_argument("--node-id", default=None)
    p.add_argument("--advertise-url", default=None)
    p.add_argument("--write-manifest", action="store_true",
                   help="Hash the model into <model-path>.manifest.json "
                        "and exit (publish it via the coordinator's "
                        "manifest route)")
    args = p.parse_args()

    engine_key = args.engine_key or os.environ.get("ENGINE_API_KEY")
    model_path = os.path.expanduser(args.model_path) if args.model_path else None
    model_key = args.model_key or (
        os.path.basename(model_path.rstrip("/")) if model_path else "draft-only")

    if args.write_manifest:
        m = build_manifest(model_path, model_key)
        out = model_path.rstrip("/") + ".manifest.json"
        with open(out, "w") as f:
            json.dump(m, f)
        print(f"manifest -> {out}\npublish: POST {args.coordinator}"
              f"/api/yield/manifest with model={model_key!r}")
        return

    index = build_index(model_path) if model_path else []
    total_gb = sum(s for _, _, s in index) / 1e9
    role = "weights+draft" if index else "draft-only"
    print(f"sidecar ({role}): {len(index)} chunks / {total_gb:.1f} GB; "
          f"engine {args.engine_name} at {args.engine_url}")

    stats = {"draft_requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            # Draft service: forward /v1/* to the local engine so one
            # public URL serves both custody proofs and Fast Token drafts.
            if not self.path.startswith("/v1/"):
                self.send_response(404); self.end_headers(); return
            try:
                n = int(self.headers.get("Content-Length", 0))
                hdrs = {"Content-Type": "application/json",
                        "User-Agent": "expert-sidecar/0.3"}
                if engine_key:
                    hdrs["Authorization"] = f"Bearer {engine_key}"
                req = urllib.request.Request(
                    args.engine_url.rstrip("/") + self.path,
                    data=self.rfile.read(n), headers=hdrs)
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read()
                stats["draft_requests"] += 1
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            except Exception as e:
                self.send_response(502)
                self.end_headers()
                self.wfile.write(str(e).encode()[:200])

        def do_GET(self):
            parts = self.path.strip("/").split("/")
            if len(parts) == 3 and parts[0] == "block":
                try:
                    idx = int(parts[2])
                    data = read_chunk(index[idx])
                except Exception:
                    self.send_response(404); self.end_headers(); return
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.end_headers()
                self.wfile.write(data)
            elif self.path == "/health":
                body = json.dumps({
                    "status": "ok", "engine": args.engine_name,
                    "engine_health": engine_health(args.engine_url),
                    "chunks": len(index), "gb": round(total_gb, 1),
                    "draft_requests": stats["draft_requests"],
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_response(404); self.end_headers()

        def log_message(self, *a):
            pass

    if args.join:
        node_id = args.node_id or f"{socket.gethostname()}-{args.port}"
        advertise = args.advertise_url or f"http://{args.host}:{args.port}"
        yc = YieldClient(args.coordinator, args.join, node_id, advertise, {
            "model_config": {"num_layers": 1, "num_experts": len(index),
                             "expert_block_size": CHUNK,
                             "model_key": model_key},
            "experts_per_layer": len(index),
            "expert_ids": [],
            "est_gb": round(total_gb, 2),
            "engine": args.engine_name,
            "role": role,
        })
        yc.join()
        yc.start_heartbeat(lambda: {
            "engine": engine_health(args.engine_url),
            "draft_requests": stats["draft_requests"],
        })

    HTTPServer.allow_reuse_address = True
    print(f"serving challenges on http://{args.host}:{args.port}")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
