# Working on this repo (for coding agents)

You are probably here because someone pointed an agent at this project. This
file is the short version of what matters.

## What this is

Two tiers that run Mixture-of-Experts models on hardware that "cannot" hold
them:

- `src/mlx_expert_sniper/` — one machine: pin the always-needed weights
  (~1 GB), stream active experts from SSD with `F_NOCACHE` + `pread`, cache
  the hot ones, prefetch the next layer's experts by running its router one
  layer early.
- `src/expert_network/` — many machines: each holds a partition of the
  experts resident in RAM; a driver runs attention locally and dispatches
  expert compute to whoever owns the active ones. Partitions come from
  rendezvous hashing over a shared roster.

Apple Silicon only for the engine (MLX has no x86 build). The sidecar and
fetch tools are stdlib and run anywhere.

## The one rule: measure, then claim

Every performance number in README.md and RESEARCH.md is measured on stated
hardware. Four claims in this project's history were killed by measurement,
including two of the author's own. If you add a number, say what machine
produced it. If you cannot measure it, label it a projection.

**Benchmark on a quiet machine.** Three separate results in this repo were
poisoned by contention (another model loaded, swap pressure) and had to be
retracted. Close the other servers first.

**Quality gates run in decode mode.** Prefill perplexity is blind to
cache-aware routing damage — 4.33 prefill vs 13.1 decode for the same
setting. Use `mlx-sniper eval --decode`.

## Setup

```bash
pip install -e ".[network]" pytest
python -m pytest tests/ -q          # 21 tests, no model needed
mlx-sniper download olmoe-1b-7b -o ~/models/olmoe-stream   # 3.6 GB demo
```

Two nodes plus a driver on one machine exercises the network tier:

```bash
expert-node --model-dir ~/models/olmoe-stream --partition 0-31  --port 8301
expert-node --model-dir ~/models/olmoe-stream --partition 32-63 --port 8303
expert-net ~/models/olmoe-stream --nodes http://127.0.0.1:8301,http://127.0.0.1:8303 --chat
```

## Where the interesting problems are

Read RESEARCH.md before proposing an optimization — several obvious ones are
already measured and dead:

- Co-activation prefetch: doubled prefetch hits, zero throughput change.
- Speculative decoding on the SSD tier: loses three ways, because the verify
  batch's wider expert union evicts the expert cache.
- Speculative decoding on the mesh: loses at low acceptance, because the
  batch's payload doubles per-hop cost. The break-even is
  `tok/forward > payload ratio` and it has never been tested with a good
  drafter on real hardware. **That is the most valuable open measurement.**

The open issues are the roadmap. Highest value first: real LAN numbers across
two physical machines (#3), pinned-only driver loading (#1), persistent draft
KV (#2).

## Things that will bite you

- `mlx_lm.load_model` wants a `Path`, and calls `get_model_classes(config=…)`
  by keyword.
- Size any cache against *available* RAM, never total. Sizing against total
  put a streaming node into swap on the same SSD it was reading from and cost
  a factor of eight.
- A node's stdout must stay line-buffered; a machine killed by the OOM killer
  otherwise takes its explanation with it.
- macOS `rsync` is 2.6.9: no `--info=progress2`.

## Joining the live network (optional)

The network side has a coordinator that pays machine operators. The engine
does not depend on it — everything above runs with no account and no key. If
you want a machine on the live network:

```bash
curl -fsSL https://herorunai.com/join.sh | sh -s -- <hr_live_key>
```

That installs the client, builds the model, opens a Cloudflare quick tunnel
so a machine behind NAT is reachable, takes the partition the roster assigns
it, and registers.
