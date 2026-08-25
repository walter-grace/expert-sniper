# What the Expert Network can and cannot do

Measured on a 16 GB M4 Mac mini, a 6-vCPU VPS, and the internet between
them. The arithmetic here decides the shape of the whole system, so it is
worth being blunt about it.

## The governing constraint: layers are serial

Within a layer, expert dispatch is parallel — every node that owns an
active expert is asked at once. Across layers it is strictly serial:
layer *i+1* cannot start until layer *i* returns. So per-token latency is

```
    t_token  ≈  num_layers × round_trip_time
```

That single line is the whole story. Round trips do not amortise across
depth, and depth is a property of the model, not of the network.

Measured round trips: **1.1 ms** localhost, **81 ms** Mac→VPS over the
public internet (ICMP floor; HTTPS measured 117–216 ms).

| model | layers | localhost | LAN (~1 ms) | WAN (81 ms) |
|---|---|---|---|---|
| OLMoE | 16 | 57 tok/s | 63 tok/s | **0.8 tok/s** |
| Qwen3-Coder-30B | 48 | 19 tok/s | 21 tok/s | **0.3 tok/s** |
| GLM-5.2 | 78 | 12 tok/s | 13 tok/s | **0.2 tok/s** |

(Ceilings from latency alone; measured two-node localhost OLMoE came in at
19.3 tok/s, below the 57 ceiling, because compute and framing take the
rest.)

**Conclusion: the expert mesh is a LAN technology.** Pooling machines in
one building works. Pooling strangers' machines across the internet, for a
single request, does not — not by a small margin, but by two orders of
magnitude. Any pitch that implies otherwise is wrong.

## What is possible

**Pooling arithmetic.** Usable RAM per machine ≈ (RAM − 4 GB OS/driver).
A 16 GB Mac contributes ~10 GB; a 64 GB Studio ~58 GB.

| model (4-bit) | size | 16 GB Macs needed | 64 GB Studios |
|---|---|---|---|
| Qwen3-Coder-30B | 16 GB | 2 | 1 |
| Qwen3.5-122B | 65 GB | 7 | 2 |
| GLM-4.6 355B | 180 GB | 18 | 4 |
| GLM-5.2 | 418 GB | 42 | 8 |
| DeepSeek-V3 671B | 350 GB | 35 | 7 |

Eight Mac Studios on a switch could hold GLM-5.2 resident with a latency
ceiling around 13 tok/s. That is a real machine that does not otherwise
exist at that price, and it is the strongest version of this idea.

**What the internet IS good for here**, all latency-tolerant and all
measured working:
- **Distribution.** Pulling your partition from peers: 0.3 GB fetched and
  hash-verified over the public internet in 40 s. Distribution is
  throughput-bound, not latency-bound.
- **Proof and coordination.** Challenges, rosters, heartbeats, payment.
- **Drafting.** A remote draft node cost nothing measurable: 5.0
  tok/forward at 56% acceptance from a VPS in another country, because a
  draft is one round trip per *batch*, not per layer.
- **Throughput, not latency.** Many independent requests in flight tolerate
  WAN fine. One request does not.

So the honest architecture is two-tier: **WAN for distribution, proof and
money; LAN for the mesh that actually runs the model.**

## Limitations, in order of how much they hurt

1. **WAN inference is off the table** (above). LAN or Thunderbolt only.
2. **Every multi-machine number is localhost.** The design is sound and the
   protocol is real, but no two physical machines have run this yet. This
   is the single most valuable open measurement.
3. **No failover.** If a node vanishes mid-generation the request dies.
   Rendezvous hashing makes reassignment cheap (~1/N of experts move), but
   nothing re-plans yet.
4. **One driver, one request.** The driver is a single point of failure and
   serves serially; concurrency is untested and batching is unimplemented.
5. **Preparation needs one big disk.** Splitting a 418 GB checkpoint into
   expert blocks requires the whole checkpoint somewhere first. Fetching is
   distributed; preparing is not.
6. **Speculation does not rescue latency.** Measured losing on both tiers,
   for two different reasons — expert-read amplification on SSD, payload
   amplification on the mesh. See RESEARCH.md.
7. **Trust is sampled, not continuous.** Challenges verify random blocks
   periodically. A node that serves correct bytes when challenged and
   garbage otherwise would not be caught quickly. Per-request verification
   would cost more than it saves today.
8. **Apple Silicon only** for the engine; MLX has no x86 build.
9. **Cold start is slow.** A node loads its partition into RAM at startup
   (measured 6.5 GB in 3.9 s locally, but hours if it must first fetch that
   partition over a home connection).

## The question worth testing next

Two Apple Silicon machines on one LAN, one model split between them, with
a good drafter. Every term in `t_token ≈ layers × RTT` is then measurable
rather than projected, and the speculation break-even
(`tok/forward > payload ratio`) can finally be settled on the tier where
the arithmetic says it should win.
