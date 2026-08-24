# Expert Sniper — the Expert Network

**Connect ordinary computers into a network that runs MoE models none of them
could run alone.**

MoE (Mixture-of-Experts) models activate only 8 of 128-256 experts per token,
so ~97% of the weights are unused per computation. That sparsity has a
consequence most inference stacks ignore: the experts don't have to live on
the machine doing the thinking. They can live on *any* machine — on its SSD,
served over the network — because what moves between boxes is activations
(~4 KB per layer), while what moves within a box is expert weights
(~13.5 MB per layer). Interconnect is never the bottleneck; pooled SSD
bandwidth is the resource, and it scales linearly with every machine that
joins:

```
feasible when   N x SSD_bandwidth x cache_hit_rate  >  bytes_per_token x target_tok/s
                └ number of machines in the network
```

Expert Sniper builds this in two tiers:

1. **One machine** (`src/mlx_expert_sniper/`): pin the small always-needed
   weights (attention, norms, router — ~0.9-1.4 GB) in RAM and stream only
   the active experts from SSD via `F_NOCACHE` + `pread`, with a right-sized
   LRU expert cache and threaded prefetch. A 17-21 GB model runs in a few GB
   of RAM.
2. **A network of machines** (`src/expert_network/`): each node owns a
   partition of the experts, loaded from its own SSD into its own RAM; a
   driver runs attention locally and dispatches expert compute to whichever
   nodes own the active experts. Every machine added is more expert RAM and
   more SSD bandwidth. Partition assignment is rendezvous hashing over a
   shared node roster — every node derives the same assignment with zero
   coordination, and adding a node moves only ~1/N of the experts.

Measured, two nodes + driver (localhost, OLMoE-1B-7B, 64 experts split
32/32 by roster): **19.3 tok/s, TTFT 0.6 s, 2.9 ms/layer round trip,
~0.24 MB of network traffic per token** — faster than the same model
streaming from SSD on one machine (14.9 tok/s), because resident partitions
take the SSD out of the token loop entirely.

```bash
pip install -e ".[network]"
mlx-sniper download olmoe-1b-7b -o ~/models/olmoe-stream   # 3.6 GB demo model

# each machine (or terminal) derives its partition from the shared roster:
expert-node --model-dir ~/models/olmoe-stream --roster mini-a,mini-b --me mini-a --port 8301
expert-node --model-dir ~/models/olmoe-stream --roster mini-a,mini-b --me mini-b --port 8302

expert-net ~/models/olmoe-stream --nodes http://127.0.0.1:8301,http://127.0.0.1:8302 --chat
```

Nodes bind `127.0.0.1` by default and have no authentication — pass
`--host 0.0.0.0` only on a trusted network. The decode loop is a LAN/metro
design (per-layer round trips); the WAN's role is distributing the
content-addressed expert partitions, never the token loop.

## Quick start (Apple Silicon, 16 GB+)

```bash
pip install -e .

mlx-sniper download qwen3-30b          # download + preprocess + calibrate
mlx-sniper run  ~/models/qwen3-30b-stream -p "Hello" -v
mlx-sniper chat ~/models/qwen3-30b-stream
mlx-sniper serve ~/models/qwen3-30b-stream        # Ollama-compatible API
mlx-sniper eval ~/models/qwen3-30b-stream          # perplexity on held-out text
```

`mlx-sniper download list` shows supported models (Qwen3-30B/Coder-30B,
Qwen3.5-35B/122B, Qwen3-Next-80B, Qwen3-235B; Gemma 4-26B experimental).
Already have an MLX 4-bit checkpoint? `mlx-sniper preprocess <src> <out>`.

## How it works

1. **Preprocess** splits the checkpoint into `pinned.safetensors` (RAM) and
   per-layer `bin/layer_XX.bin` files of 16 KB-aligned expert blocks (SSD).
2. **Calibrate** (one-time) right-sizes the expert LRU cache for your RAM,
   records a cross-layer co-activation matrix, and sweeps the routing bias
   with a perplexity gate: a bias only ships if held-out perplexity stays
   within 5% of the unbiased baseline, measured on the same forward pass
   that serves.
3. **Serve** runs the model with threaded `pread` prefetch of the next
   layer's predicted + selected experts, fused active-expert FFN via
   `gather_qmm`, and the router nudged toward already-cached experts.

## Performance (v0.2, measured)

Qwen3-30B-A3B 4-bit (17 GB on disk, 128 experts), M4 Mac Mini 16 GB with
other apps running, 886-expert (2.3 GB) cache, 100-token generations,
decode-mode perplexity on held-out text:

| Routing bias | Speed | Decode ppl | Cache hit rate |
|---|---|---|---|
| 0.0 (default) | 1.15 tok/s | 4.18 | 51% |
| 0.5 (opt-in: `--ppl-tolerance 1.10`) | 1.70 tok/s (+48%) | 4.42 (+5.7%) | 62% |
| 1.0 | — | 6.38 (+53%) | quality cliff |
| 1.5 | — | 14.11 (+238%) | quality cliff |

TTFT ~11 s; sustained SSD streaming 1.7–2.3 GB/s at 1.2–1.9 ms/expert.

OLMoE-1B-7B (3.6 GB, 64 experts — the demo model), same machine:
**14.9 tok/s single-machine streaming** (85% hit rate, TTFT 1.3 s) and
**19.3 tok/s as a two-node Expert Network**. Prefetch is early-router
prediction (layer *i+1*'s pinned router run on layer *i*'s hidden state —
97% recall@16 measured), with speculative reads parked in a victim buffer
and promoted to the main cache only when a demand uses them.

Three honest findings from re-measuring on the fixed code (details in
RESEARCH.md):

1. **The routing bias trades quality for speed** — it is not free. Earlier
   releases shipped bias 1.0–1.5 as a "sweet spot" validated by a two-prompt
   substring check; decode-mode perplexity shows 1.0+ degrades the model
   badly. v0.2 defaults to bias 0 and gates any bias on measured perplexity.
2. **Prefill perplexity cannot see bias damage** (prefill activates ~80% of
   experts per layer, so the cache-aware bias barely engages) — the
   calibration gate must measure token-by-token decode.
3. **Co-activation prefetch is bandwidth-neutral here**: fixing the bug that
   discarded its reads doubled consumed prefetches (1,810 → 3,407 per 100
   tokens) at unchanged tok/s. The old "70% prediction accuracy → speedup"
   framing was not realizable.

Practical note: on a 16 GB machine, cache sizing must respect *available*
RAM, not total — an oversized expert cache pushes the OS into swap on the
same SSD the experts stream from, and throughput collapses ~50× (measured
0.21 tok/s vs 1.7). `calibrate` handles this automatically.

Earlier published figures (5.37 tok/s 35B, 3.34 tok/s 30B, "92% cache hit")
were measured on code with corrupted hit-rate accounting and a bias level
that decode-ppl shows was damaging quality; treat them as superseded.

An `--expert-cache-size` madvise patch for llama.cpp (see `llama-cpp/`)
produced 0.57 tok/s for a 30B MoE on an 8 GB M2 Air where stock llama.cpp
produced no output.

## Repository layout

- `src/mlx_expert_sniper/` — the pip-installable package (MLX, Apple Silicon)
- `llama-cpp/` — expert-cache patch for llama.cpp (cross-platform, GGUF) —
  original sources + an `apply.sh` that patches your own llama.cpp checkout
- `sniper-router/` — thin client for driving a remote sniper/llama-server
- `bench/` — research benchmark scripts behind the RESEARCH.md numbers
- `tests/` — unit tests (`python -m pytest tests/`)
- `RESEARCH.md` — full technical writeup

- `src/expert_network/` — the Expert Network: nodes, binary protocol,
  distributed driver, HRW roster placement

## Non-Mac hardware: DGX Spark, gaming PCs (sidecar)

The MLX engine is Apple Silicon, but Machine Yield is not. Any box running
its own MoE engine — FreeToken on a DGX Spark or an RTX rig, vLLM,
llama.cpp — joins through the sidecar, which proves what the yield system
actually needs proven: that the machine holds the weights (content-
addressed 4 MB chunk challenges), that it is alive (heartbeats + a health
poll of the engine's OpenAI-compatible endpoint), and what it serves.

```bash
# on the Spark: FreeToken serving GLM/DeepSeek on :8000, then
expert-sidecar --model-path ~/models/GLM-4.6 --engine-url http://127.0.0.1:8000     --join <api-key> --advertise-url https://your-tunnel.example.com

# once per model, publish its chunk manifest:
expert-sidecar --model-path ~/models/GLM-4.6 --write-manifest
```

Stdlib-only — no mlx, no extra deps. Expert-partition serving on CUDA
(the full network tier) is future work; see the issues.

## Security notes

- `mlx-sniper serve` and the agent CLIs bind `127.0.0.1` by default. There is
  no authentication layer — do not expose them to untrusted networks.
- The agent tools (`llama-cpp/sniper.py`, `sniper-router/router.py`) can
  execute model-proposed shell commands (`/shell`). Use with prompts and
  models you trust.

## License

MIT (see LICENSE). llama.cpp integration notes in NOTICE.
