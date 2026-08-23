# bench/ — research benchmarks

These are the measurement scripts behind the numbers in RESEARCH.md. They are
kept as research artifacts, not maintained tooling. Install the package first
(`pip install -e ..`), then point them at a model via env vars:

```bash
SNIPER_EXPERT_DIR=~/models/qwen3-30b-stream/bin python bench_quick.py
```

| Script | What it measures | Format notes |
|---|---|---|
| `test_ternary_sensitivity.py` | Cosine similarity of ternary-quantized down_proj vs 4-bit (source of the ~0.89 ternary / 0.81 1-bit figures) | Reads layer `.bin` files directly |
| `test_fsbr_smoothing.py` | FSBR group-smoothing effect on ternary quality | Reads layer `.bin` files directly |
| `bench_quick.py` | SSD pread vs mmap-fallback latency (5 tokens × 5 layers × 8 experts) | **Legacy format**: expects the older `mlp.switch_mlp.*` tensor keys and an `experts/` layout — predates the current `mlx-sniper download` output |
| `bench_mixed.py` | Full mixed-precision fallback A/B (cache off vs cache+ternary fallback) | **Legacy format**, and needs a down_proj fallback buffer (`SNIPER_FALLBACK_PATH`) — the buffer-builder is not part of this repo |

The two legacy scripts document methodology for numbers already published;
running them against current-format models requires renaming the tensor keys
(drop the `mlp.` prefix) and generating a fallback buffer.
