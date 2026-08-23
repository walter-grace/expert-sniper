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

## Performance

Measured on an M4 Mac Mini, 16 GB RAM (see RESEARCH.md for methodology):

| Model | Size on disk | Speed | Notes |
|---|---|---|---|
| Qwen3.5-35B-A3B 4-bit | 19.5 GB | 5.37 tok/s | 256 experts, TTFT 2.9 s |
| Qwen3-30B-A3B 4-bit | 17.2 GB | 3.34 tok/s | 128 experts |

> These numbers predate the v0.2 correctness fixes (honest cache-hit
> accounting, working co-activation prefetch, perplexity-gated bias) and are
> being re-measured; v0.2 numbers will replace them here.

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
