# Contributing

The interesting problems here are measured, not speculative — every claim in
RESEARCH.md comes with the number that earned it, and contributions should
keep that bar.

## Setup

Apple Silicon Mac (mlx requires it), Python 3.10+:

```bash
pip install -e ".[network]" pytest
python -m pytest tests/ -q          # unit tests, no model needed
```

For anything touching the engines or the network, validate on the demo
model (3.6 GB, minutes to build):

```bash
mlx-sniper download olmoe-1b-7b -o ~/models/olmoe-stream
mlx-sniper run ~/models/olmoe-stream -p "hello" -v
```

Two nodes + driver on one machine exercises the full Expert Network:

```bash
expert-node --model-dir ~/models/olmoe-stream --roster a,b --me a --port 8301
expert-node --model-dir ~/models/olmoe-stream --roster a,b --me b --port 8302
expert-net ~/models/olmoe-stream --nodes http://127.0.0.1:8301,http://127.0.0.1:8302 --chat
```

## Ground rules

- **Measure before and after.** A performance PR without numbers on real
  hardware (chip, RAM, model) is a conversation, not a change. `-v` prints
  the reader stats; `mlx-sniper eval --decode` measures quality.
- **Quality gates run in decode mode.** Prefill perplexity is blind to
  cache-aware routing effects (see RESEARCH.md v0.2 findings).
- **Honest docs.** If a number is a projection, label it. If an experiment
  failed, the failure is worth writing down.
- Unit tests (`tests/`) run on synthetic fixtures and must pass without any
  model on disk — CI has no 17 GB checkpoint.

## Where help is wanted

The open issues are the roadmap; the deepest ones are marked. Highlights:
pinned-only driver loading, persistent draft-model KV for speculation,
LAN/Thunderbolt multi-machine measurements, and adaptive prefetch gated on
link utilization.
