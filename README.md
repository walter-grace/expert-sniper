# Expert Sniper — Run MoE Models Larger Than Your RAM

MoE (Mixture-of-Experts) models activate only 8 of 128-256 experts per token,
so ~97% of the weights are unused per computation. Expert Sniper exploits that
sparsity on consumer hardware: pin the small always-needed weights
(attention, norms, router — ~1.4 GB) in RAM, and stream only the active
experts from SSD with `F_NOCACHE` + `pread`, a right-sized LRU expert cache,
cross-layer co-activation prefetch, and cache-aware routing bias.

```
Model on disk: 17-21 GB      RAM pinned: ~0.9-1.4 GB
The rest streams from SSD on demand.
```

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

Multi-Mac distributed expert sharding (pipeline-parallel over 3 Mac Minis)
exists in the research tree and will be published separately once it's
hardened for public use.

## Security notes

- `mlx-sniper serve` and the agent CLIs bind `127.0.0.1` by default. There is
  no authentication layer — do not expose them to untrusted networks.
- The agent tools (`llama-cpp/sniper.py`, `sniper-router/router.py`) can
  execute model-proposed shell commands (`/shell`). Use with prompts and
  models you trust.

## License

MIT (see LICENSE). llama.cpp integration notes in NOTICE.
