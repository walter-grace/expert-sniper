# Expert Sniper Router — Scaling Architecture

## Level 1: Multi-Server Load Balancing

**Scales:** requests/sec (not single-request tok/s)
**Difficulty:** Easy (modify sniper-router)
**Cost:** Linear with workers

### Changes to router.py

```python
class ServerPool:
    """Round-robin load balancer across multiple inference backends."""

    def __init__(self, hosts: list[str]):
        self.servers = [RemoteServer(h) for h in hosts]
        self.idx = 0

    def next(self) -> RemoteServer:
        """Get next healthy server (round-robin)."""
        for _ in range(len(self.servers)):
            server = self.servers[self.idx % len(self.servers)]
            self.idx += 1
            if server.health():
                return server
        raise RuntimeError("No healthy servers available")

    def stream(self, messages, **kwargs):
        server = self.next()
        return server.stream(messages, **kwargs)
```

### Usage

```bash
python router.py \
  --server 192.168.1.244:8899 \   # Mac Mini (local, free)
  --server runpod-worker-1:8201 \  # RunPod GPU 1
  --server runpod-worker-2:8201    # RunPod GPU 2
```

---

## Level 2: Speculative Decode Pipeline

**Scales:** single-request tok/s (3-5x improvement)
**Difficulty:** Medium (new orchestration logic)
**Cost:** 2 servers (1 cheap draft + 1 verify)

### Architecture

```
User → Router → Draft Server (small model, fast) → K draft tokens
                                                          │
       Router → Verify Server (Expert Sniper, 35B) ← ────┘
                         │
                   batched_verify (K tokens at once)
                         │
                   accept/reject → emit tokens to user
```

### Changes to router.py

```python
class SpecDecodeRouter:
    """Speculative decode orchestrator with draft + verify servers."""

    def __init__(self, draft_host: str, verify_host: str, draft_len: int = 8):
        self.draft = RemoteServer(draft_host)   # Small model (1-3B)
        self.verify = RemoteServer(verify_host)  # Full model (35B)
        self.draft_len = draft_len

    def stream(self, messages, **kwargs):
        """
        1. Send prompt to both servers for prefill
        2. Loop:
           a. Draft server generates K tokens (fast)
           b. Verify server checks K tokens in batch (Expert Sniper)
           c. Accept valid prefix + 1 bonus token
           d. Yield accepted tokens to user
        """
        # Prefill on verify server
        # ... (standard chat completion to get first token)

        while not done:
            # Draft: generate K tokens from small model
            draft_response = self.draft.complete(
                messages + [{"role": "assistant", "content": generated_so_far}],
                max_tokens=self.draft_len,
                temperature=0.0,
            )
            draft_tokens = draft_response["tokens"]

            # Verify: check all K tokens in one batched pass
            verify_response = self.verify.verify_batch(
                messages,
                draft_tokens=draft_tokens,
            )

            # Accept/reject
            accepted = 0
            for i, (draft_tok, verify_tok) in enumerate(
                zip(draft_tokens, verify_response["tokens"])
            ):
                if draft_tok == verify_tok:
                    accepted += 1
                    yield draft_tok
                else:
                    yield verify_tok  # bonus token
                    break
            else:
                yield verify_response["bonus_token"]

            # Update context for next round
            generated_so_far += accepted_text
```

### Server requirements

| Server | Model | GPU | Cost | tok/s |
|--------|-------|-----|------|-------|
| Draft | Qwen3-0.6B | 16 GB ($0.58/hr) | Cheap | ~200 tok/s |
| Verify | Qwen3.5-35B (Expert Sniper) | 16-24 GB ($0.58-0.68/hr) | Cheap | ~5 tok/s per batch of K tokens |

### Expected performance

- Draft generates 8 tokens in ~40ms (200 tok/s × 8)
- Verify checks 8 tokens in ~200ms (Expert Sniper batched)
- Accept ~5-6 tokens per step
- Total: 5-6 tokens per 240ms = **~22 tok/s effective**
- vs baseline: 5 tok/s serial = **4.4x improvement**

### Why Expert Sniper matters here

Without Expert Sniper, verification of 8 tokens through a 35B MoE model reads
4.22 GB from SSD per step. With Expert Sniper (58% union dedup + 92% cache):
only 0.14 GB per step. That's the difference between 200ms verify and 2000ms verify.

---

## Level 3: Expert Parallelism

**Scales:** single-request tok/s via parallel expert compute
**Difficulty:** Hard (requires custom inference engine)
**Cost:** N cheap GPUs (cheaper per GPU, more GPUs)

### Architecture

```
sniper-router (expert dispatcher)
    │
    ├── GPU 0: Attention + Router + Shared Expert (1.4 GB)
    │          Runs attention layers, determines which experts are needed,
    │          dispatches hidden states to expert GPUs
    │
    ├── GPU 1: Experts 0-127 (6.5 GB at Q4)
    │          Receives hidden states, computes assigned experts,
    │          returns expert output
    │
    └── GPU 2: Experts 128-255 (6.5 GB at Q4)
               Same as GPU 1 for its expert shard
```

### How it works per token

```
1. GPU 0: input → embedding → attention layer 0 → layernorm
2. GPU 0: router gate → softmax → top-8 experts: [12, 45, 78, 134, 201, 210, 230, 255]
3. GPU 0: split by shard:
          GPU 1 gets: experts [12, 45, 78] with hidden state
          GPU 2 gets: experts [134, 201, 210, 230, 255] with hidden state
4. GPU 1 + GPU 2: compute expert FFN in PARALLEL
5. GPU 0: receive results, weighted sum, add shared expert, residual
6. Repeat for all 40 layers
7. GPU 0: final norm → lm_head → next token
```

### Communication overhead

Per layer: 2 × hidden_size × sizeof(float16) = 2 × 2048 × 2 = 8 KB
Per token (40 layers): 40 × 8 KB = 320 KB round-trip between GPUs

At 10 Gbps network: 320 KB / 1.25 GB/s = 0.26 ms per token overhead
At 1 Gbps LAN: 320 KB / 125 MB/s = 2.6 ms per token overhead

**With 10 Gbps, network overhead is negligible. With 1 Gbps, it adds ~2.6ms per token.**

### Changes needed

This is NOT just a router change — it requires a custom inference server:

```python
class ExpertShardServer:
    """Serves a subset of experts for one MoE model."""

    def __init__(self, model_dir, expert_range=(0, 128)):
        self.start, self.end = expert_range
        # Load only experts in [start, end) from the sniper binary files
        # Plus the LRU cache, routing bias, co-activation prefetch

    async def compute_experts(self, layer_idx, hidden_state, expert_ids, weights):
        """
        Compute expert FFN for assigned experts only.
        Returns weighted expert output.
        """
        my_experts = [e for e in expert_ids if self.start <= e < self.end]
        if not my_experts:
            return zeros

        expert_data = self.reader.get_experts(layer_idx, my_experts)
        return run_expert_ffn(hidden_state, expert_data, my_experts, weights)


class ExpertParallelRouter:
    """Orchestrates expert-parallel inference across GPU shards."""

    def __init__(self, attention_host, expert_hosts):
        self.attention = RemoteServer(attention_host)
        self.expert_shards = [RemoteServer(h) for h in expert_hosts]

    async def forward_one_token(self, token_id, kv_cache):
        # 1. Attention on main GPU
        hidden, expert_ids, expert_weights = await self.attention.run_attention(
            token_id, kv_cache
        )

        # 2. Dispatch to expert shards IN PARALLEL
        tasks = [
            shard.compute_experts(hidden, expert_ids, expert_weights)
            for shard in self.expert_shards
        ]
        results = await asyncio.gather(*tasks)

        # 3. Aggregate on main GPU
        return await self.attention.aggregate_and_continue(results)
```

### Expected performance

| Config | GPUs | Total cost/hr | Expert compute | tok/s |
|--------|------|---------------|----------------|-------|
| Single 48GB | 1 | $1.22 | Serial (8 experts) | 75 |
| 3× 16GB parallel | 3 | $1.74 | Parallel (3+5 split) | ~120-150 |
| 4× 16GB parallel | 4 | $2.32 | Parallel (2+2+2+2) | ~200+ |

The tok/s improvement comes from expert computation happening in parallel
across GPUs instead of serially on one GPU.

---

## Recommended Path

1. **Now:** Level 1 (multi-server load balancing) — 1 day to implement
2. **Next:** Level 2 (spec decode pipeline) — 1 week, biggest tok/s gain
3. **Future:** Level 3 (expert parallelism) — 1 month, requires custom engine

Level 2 is the sweet spot: 4x tok/s improvement with 2 cheap servers.
Expert Sniper's union batching + LRU cache make the verify step fast enough
to be practical. Without Expert Sniper, verification is too slow and spec
decode provides no benefit for MoE models.
