"""
MoE Expert Sniper — Read only active experts from SSD via F_NOCACHE + pread.

For a 256-expert model with 8 active per token:
  - Each expert: ~1.69 MB (4-bit quantized, moe_intermediate_size=512, hidden_size=2048)
  - Per layer: 8 × 1.69 MB = 13.5 MB
  - Per token (40 layers): ~540 MB
  - At 3-5 GB/s NVMe: ~108-180ms = 5.6-9.3 tok/s theoretical

Uses multi-threaded pread (8 workers) to saturate NVMe queue depth.
1-bit fallback buffer serves cache misses from mmap'd RAM while SSD
backfills the 4-bit version for the next token.
"""

import os
import json
import fcntl
import logging
import mmap
import threading
import time
import numpy as np
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor

F_NOCACHE = 48
PAGE_SIZE = 16384

log = logging.getLogger("mlx_expert_sniper.expert_io")


class DownProjFallback:
    """
    Ternary fallback buffer for down_proj only (mixed-precision strategy).

    On cache miss:
      gate_proj → pread from SSD (4-bit, full quality)
      up_proj   → pread from SSD (4-bit, full quality)
      down_proj → instant dequant from mmap'd ternary buffer
                  (~0.89 cosine estimated for ternary; 0.81 measured for
                  the 1-bit legacy format — see bench/test_ternary_sensitivity.py)

    Ternary: {-scale, 0, +scale} per group. Captures sparsity.
    Packing: 2 bits per value (00=zero, 01=+scale, 10=-scale), 4 per byte.

    Supports both formats:
      - expert_fallback_down_ternary_v1 (ternary, 1.5 GB)
      - expert_fallback_down_1bit_v1 (1-bit, 792 MB, legacy)
    """

    DOWN_SHAPE = (2048, 512)
    VALUES = 1048576  # 2048 * 512

    def __init__(self, path, group_size=128):
        self.group_size = group_size
        self.enabled = False
        self.fallback_hits = 0
        self.dequant_time = 0.0

        if not os.path.exists(path):
            print(f"  [fallback] file not found: {path}")
            return

        with open(path, "rb") as f:
            raw = f.read(PAGE_SIZE)
        depth = 0
        for i, b in enumerate(raw):
            if b == ord("{"):
                depth += 1
            elif b == ord("}"):
                depth -= 1
                if depth == 0:
                    self.header = json.loads(raw[: i + 1])
                    break

        self.fmt = self.header["format"]
        self.is_ternary = "ternary" in self.fmt
        self.num_layers = self.header["num_layers"]
        self.num_experts = self.header["num_experts"]
        self.data_start = self.header["data_start"]

        padded = self.VALUES + (-self.VALUES % group_size)
        n_groups = padded // group_size
        self.n_groups = n_groups
        self.scales_bytes = n_groups * 2  # fp16

        if self.is_ternary:
            # 2 bits per value, 4 per byte
            total_codes = n_groups * group_size
            self.packed_bytes = (total_codes + 3) // 4
            self.expert_size = self.header["expert_ternary_size"]
        else:
            # 1 bit per value, 8 per byte
            self.packed_bytes = n_groups * (group_size // 8)
            self.expert_size = self.header["expert_1bit_size"]

        self._fd = os.open(path, os.O_RDONLY)
        self._mm = mmap.mmap(self._fd, 0, access=mmap.ACCESS_READ)
        self.enabled = True

        import mlx.core as mx
        if self.is_ternary:
            # Masks for unpacking 2-bit codes: 4 values per byte
            self._shift_vals = mx.array([0, 2, 4, 6], dtype=mx.uint32)
        else:
            # Masks for unpacking 1-bit signs: 8 values per byte
            self._bit_masks = mx.array([128, 64, 32, 16, 8, 4, 2, 1], dtype=mx.uint8)

        file_mb = os.path.getsize(path) / 1024 / 1024
        mode = "ternary" if self.is_ternary else "1-bit"
        print(f"  [fallback] mmap'd {file_mb:.0f} MB down_proj {mode} buffer "
              f"({self.num_layers} layers × {self.num_experts} experts)")

    def get_down_proj_f16(self, layer_idx, expert_id):
        """
        Dequantize down_proj to float16 mx.array [2048, 512].
        MLX GPU ops — no numpy loops.
        """
        import mlx.core as mx

        t0 = time.time()
        offset = self.data_start + (layer_idx * self.num_experts + expert_id) * self.expert_size

        s_end = offset + self.scales_bytes
        p_end = s_end + self.packed_bytes

        scales_np = np.frombuffer(self._mm[offset:s_end], dtype=np.float16).copy()
        packed_np = np.frombuffer(self._mm[s_end:p_end], dtype=np.uint8).copy()

        scales = mx.array(scales_np).reshape(self.n_groups, 1)
        packed = mx.array(packed_np)

        if self.is_ternary:
            # Unpack 2-bit codes on GPU: each byte → 4 values
            # packed shape: (total_bytes,)
            # Expand: (total_bytes, 1) >> [0,2,4,6] & 0x3 → (total_bytes, 4)
            codes = (mx.expand_dims(packed.astype(mx.uint32), -1) >> self._shift_vals) & 0x3
            codes = codes.reshape(-1)[:self.n_groups * self.group_size]
            codes = codes.reshape(self.n_groups, self.group_size).astype(mx.float16)

            # Map: 0→0.0, 1→+1.0, 2→-1.0
            signs = mx.where(codes == 1, 1.0, mx.where(codes == 2, -1.0, 0.0))
            weights = signs * scales.astype(mx.float16)
        else:
            # 1-bit: unpack sign bits
            packed = packed.reshape(self.n_groups, self.group_size // 8)
            bits = (mx.expand_dims(packed, -1) & self._bit_masks) > 0
            bits = bits.reshape(self.n_groups, self.group_size).astype(mx.float16)
            weights = (2.0 * bits - 1.0) * scales.astype(mx.float16)

        weights = weights.reshape(-1)[:self.VALUES].reshape(self.DOWN_SHAPE)
        mx.eval(weights)

        self.fallback_hits += 1
        self.dequant_time += time.time() - t0
        return weights

    def close(self):
        if hasattr(self, "_mm") and self._mm:
            self._mm.close()
        if hasattr(self, "_fd"):
            os.close(self._fd)

    def stats(self):
        avg_ms = (self.dequant_time / self.fallback_hits * 1000) if self.fallback_hits > 0 else 0
        mode = "ternary" if self.is_ternary else "1-bit"
        return (f"fallback_hits={self.fallback_hits}, mode={mode}, "
                f"avg_dequant={avg_ms:.1f}ms, "
                f"total_dequant={self.dequant_time:.2f}s")


class LRUExpertCache:
    """LRU cache for parsed expert data. Skips SSD reads on cache hits.

    Thread-safe: `put` may be called from backfill threads while the main
    thread reads. `get` is the only accessor that counts hits/misses and
    refreshes recency — use `contains`/`peek`/`cached_ids` for probes so
    stats stay honest and eviction order stays true LRU.
    """

    def __init__(self, max_experts=100):
        self.max_experts = max_experts
        self.cache = OrderedDict()  # (layer_idx, expert_id) → parsed expert dict
        self.hits = 0
        self.misses = 0
        self._lock = threading.RLock()
        self._by_layer = defaultdict(set)  # layer_idx → {expert_id}

    def get(self, layer_idx, expert_id):
        key = (layer_idx, expert_id)
        with self._lock:
            if key in self.cache:
                self.hits += 1
                self.cache.move_to_end(key)
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, layer_idx, expert_id, data):
        key = (layer_idx, expert_id)
        with self._lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                self.cache[key] = data
            else:
                if len(self.cache) >= self.max_experts:
                    (l, e), _ = self.cache.popitem(last=False)
                    self._by_layer[l].discard(e)
                self.cache[key] = data
                self._by_layer[layer_idx].add(expert_id)

    def contains(self, layer_idx, expert_id):
        """Existence probe: no stats, no recency update."""
        with self._lock:
            return (layer_idx, expert_id) in self.cache

    def peek(self, layer_idx, expert_id):
        """Value or None: no stats, no recency update."""
        with self._lock:
            return self.cache.get((layer_idx, expert_id))

    def cached_ids(self, layer_idx):
        """Snapshot of expert ids cached for a layer (for the routing-bias mask)."""
        with self._lock:
            return list(self._by_layer.get(layer_idx, ()))

    def hit_rate(self):
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

    def stats(self):
        with self._lock:
            total = self.hits + self.misses
            rate = self.hits / total if total > 0 else 0.0
            return (f"cache: {len(self.cache)}/{self.max_experts} entries, "
                    f"hit_rate={rate:.1%} ({self.hits}/{total})")


class MoEExpertReader:
    """
    Reads specific experts from concatenated layer files via F_NOCACHE + pread.
    Expert offset = data_start + expert_id × expert_block_size

    With fallback_path set, cache misses are served instantly from a 1-bit
    mmap buffer while the 4-bit version loads from SSD for the next token.
    """

    def __init__(self, expert_dir, num_layers, num_workers=8, cache_size=0,
                 fallback_path=None):
        self.expert_dir = expert_dir
        self.num_layers = num_layers
        self.executor = ThreadPoolExecutor(
            max_workers=num_workers, thread_name_prefix="sniper-read")
        # Backfill parses on its own single worker: submitting it to the read
        # pool would let backfill jobs occupy every worker while blocking on
        # read futures queued behind them (pool self-deadlock).
        self.backfill_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="sniper-backfill")

        # LRU cache (0 = disabled)
        self.lru = LRUExpertCache(max_experts=cache_size) if cache_size > 0 else None

        # 1-bit down_proj fallback buffer
        self.fallback = None
        if fallback_path:
            self.fallback = DownProjFallback(fallback_path)
            if not self.fallback.enabled:
                self.fallback = None

        # Parse all layer headers
        self.headers = {}
        self.fds = {}
        for i in range(num_layers):
            path = f"{expert_dir}/layer_{i:02d}.bin"
            with open(path, "rb") as f:
                raw = f.read(PAGE_SIZE)
            self.headers[i] = json.loads(raw.rstrip(b"\x00"))

        # Precompute layout info
        h0 = self.headers[0]["layout"]
        self.expert_block_size = h0["expert_block_size"]
        self.data_start = h0["data_start"]
        self.tensor_layout = h0["tensors"]

        # Fail fast if a fallback buffer is configured but the streaming
        # format doesn't carry the down_proj tensor it needs.
        if self.fallback and "switch_mlp.down_proj.weight" not in self.tensor_layout:
            raise ValueError(
                "fallback_path is set but this model's tensor layout has no "
                "'switch_mlp.down_proj.weight' — the mixed-precision fallback "
                f"can't map into it (layout keys: {list(self.tensor_layout)})")

        # Stats. Counters are mutated by the main thread and the backfill
        # thread — guard with _stats_lock.
        self._stats_lock = threading.Lock()
        self.read_time = 0.0
        self.reads = 0
        self.bytes_read = 0
        self.cache_hits = 0
        self.prefetch_hits = 0
        self.backfill_errors = 0

        # Prefetch state. Only touched from the caller thread
        # (prefetch_experts / get_experts) — no lock needed.
        self.prefetch_futures = {}

        # Victim buffer for speculative reads. Promoting every completed
        # prefetch straight into the main LRU pollutes it — each speculative
        # insert evicts a demand-proven expert (measured on the OLMoE lab:
        # naive promotion cut 17.4 tok/s to 10.8). Speculation lands here,
        # is promoted only when a demand actually uses it, and a wrong guess
        # ages out without costing a main-cache slot. Caller-thread only.
        self.victim = OrderedDict()  # (layer, eid) → parsed expert
        self.victim_cap = max(16, cache_size // 16) if cache_size else 16
        self.victim_hits = 0

        self._fd_lock = threading.Lock()

    def _get_fd(self, layer_idx):
        with self._fd_lock:
            if layer_idx not in self.fds:
                path = f"{self.expert_dir}/layer_{layer_idx:02d}.bin"
                fd = os.open(path, os.O_RDONLY)
                fcntl.fcntl(fd, F_NOCACHE, 1)
                self.fds[layer_idx] = fd
            return self.fds[layer_idx]

    def _read_expert(self, layer_idx, expert_id):
        """Read one expert's data via pread. Thread-safe."""
        fd = self._get_fd(layer_idx)
        offset = self.data_start + expert_id * self.expert_block_size

        # Read the full expert block
        data = os.pread(fd, self.expert_block_size, offset)
        return data

    def _parse_expert_data(self, raw_data, expert_id):
        """Parse raw bytes into MLX arrays for one expert."""
        import mlx.core as mx

        # Map dtype strings to MLX dtypes
        MLX_DTYPES = {
            "uint32": mx.uint32, "float16": mx.float16, "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }

        result = {}
        for name, info in self.tensor_layout.items():
            inner_offset = info["inner_offset"]
            nbytes = info["nbytes"]
            shape = info["shape_per_expert"]
            dtype_str = info["dtype"].replace("mlx.core.", "")
            mlx_dtype = MLX_DTYPES.get(dtype_str, mx.float16)

            arr_bytes = raw_data[inner_offset:inner_offset + nbytes]
            # Create MLX array directly from bytes (handles bfloat16 correctly)
            flat = mx.array(np.frombuffer(arr_bytes, dtype=np.uint8))
            arr = flat.view(mlx_dtype).reshape(shape)
            result[name] = arr

        return result

    def prefetch_experts(self, layer_idx, expert_ids):
        """Launch parallel preads for experts not cached and not already in
        flight. Non-blocking. Repeat calls for the same layer MERGE — the
        forward pass prefetches both predicted and active experts for the
        next layer, and the second call must not discard the first's reads.
        """
        pending = self.prefetch_futures.setdefault(layer_idx, {})
        for eid in expert_ids:
            if eid in pending:
                continue  # already in flight
            if self.lru and self.lru.contains(layer_idx, eid):
                continue
            pending[eid] = self.executor.submit(self._read_expert, layer_idx, eid)

    def reset_prefetch(self):
        """Cancel and drop all outstanding prefetch futures (new request)."""
        for futures in self.prefetch_futures.values():
            for fut in futures.values():
                fut.cancel()
        self.prefetch_futures.clear()

    def _read_expert_partial(self, layer_idx, expert_id):
        """Read gate_proj + up_proj from SSD (skip down_proj). Thread-safe."""
        fd = self._get_fd(layer_idx)
        offset = self.data_start + expert_id * self.expert_block_size
        # Read only gate + up + their scales/biases (first 1,179,648 bytes)
        # down_proj starts at inner_offset 1,179,648
        down_offset = self.tensor_layout["switch_mlp.down_proj.weight"]["inner_offset"]
        data = os.pread(fd, down_offset, offset)
        return data

    def _parse_expert_partial(self, raw_data, expert_id):
        """Parse gate_proj + up_proj from partial SSD read."""
        import mlx.core as mx
        MLX_DTYPES = {
            "uint32": mx.uint32, "float16": mx.float16, "float32": mx.float32,
            "bfloat16": mx.bfloat16,
        }
        result = {}
        for name, info in self.tensor_layout.items():
            # Skip down_proj tensors — they come from fallback
            if "down_proj" in name:
                continue
            inner_offset = info["inner_offset"]
            nbytes = info["nbytes"]
            shape = info["shape_per_expert"]
            dtype_str = info["dtype"].replace("mlx.core.", "")
            mlx_dtype = MLX_DTYPES.get(dtype_str, mx.float16)
            if inner_offset + nbytes > len(raw_data):
                continue
            arr_bytes = raw_data[inner_offset:inner_offset + nbytes]
            flat = mx.array(np.frombuffer(arr_bytes, dtype=np.uint8))
            arr = flat.view(mlx_dtype).reshape(shape)
            result[name] = arr
        return result

    def get_experts(self, layer_idx, expert_ids):
        """
        Get parsed expert data for active experts.

        Mixed-precision fallback strategy:
          Cache HIT  → all 3 projections from 4-bit cache (gather_qmm)
          Cache MISS → gate+up from SSD pread (4-bit, 2/3 I/O)
                     → down from mmap fallback buffer (instant;
                       0.81 cosine measured 1-bit, ~0.89 estimated ternary)
                     → async backfill: full expert from SSD into cache

        Returns: dict[expert_id] → dict[tensor_name → mx.array]
          For cache hits: all tensors are 4-bit quantized (uint32 + bf16 scales/biases)
          For fallback:   gate+up are 4-bit, down_proj.weight is float16 mx.array

        The caller (run_expert_ffn) checks dtype to decide gather_qmm vs matmul per projection.
        """
        t0 = time.time()

        experts = {}
        futures = self.prefetch_futures.pop(layer_idx, {})
        backfill_futures = {}
        bytes_read = 0
        cache_hits = 0
        prefetch_hits = 0

        for eid in expert_ids:
            # 1. Check LRU cache
            if self.lru:
                cached = self.lru.get(layer_idx, eid)
                if cached is not None:
                    experts[eid] = cached
                    cache_hits += 1
                    continue

            # 1.5 Victim buffer — a correct speculation earns promotion
            vkey = (layer_idx, eid)
            if vkey in self.victim:
                parsed = self.victim.pop(vkey)
                experts[eid] = parsed
                self.victim_hits += 1
                if self.lru:
                    self.lru.put(layer_idx, eid, parsed)
                continue

            # 2. Check prefetched data (already read from SSD)
            if eid in futures:
                raw = futures[eid].result()
                parsed = self._parse_expert_data(raw, eid)
                experts[eid] = parsed
                bytes_read += len(raw)
                prefetch_hits += 1
                if self.lru:
                    self.lru.put(layer_idx, eid, parsed)
                continue

            # 3. Cache miss
            if self.fallback:
                # Mixed precision: gate+up from SSD, down from fallback buffer
                raw_partial = self._read_expert_partial(layer_idx, eid)
                parsed = self._parse_expert_partial(raw_partial, eid)
                bytes_read += len(raw_partial)

                # down_proj from fallback buffer (instant)
                down_f16 = self.fallback.get_down_proj_f16(layer_idx, eid)
                parsed["switch_mlp.down_proj.weight"] = down_f16

                experts[eid] = parsed

                # Async backfill: read FULL expert from SSD for next token's cache
                backfill_futures[eid] = self.executor.submit(
                    self._read_expert, layer_idx, eid
                )
            else:
                # No fallback — full synchronous SSD read
                raw = self._read_expert(layer_idx, eid)
                parsed = self._parse_expert_data(raw, eid)
                experts[eid] = parsed
                bytes_read += len(raw)
                if self.lru:
                    self.lru.put(layer_idx, eid, parsed)

        # Leftover prefetched futures (prefetched but not activated this
        # token): a completed read is already paid for — park it in the
        # victim buffer (NOT the main LRU, which it would pollute); a
        # queued-not-started read frees its worker via cancel.
        for eid, fut in futures.items():
            if eid in experts:
                continue
            if fut.done():
                try:
                    raw = fut.result()
                    self.victim[(layer_idx, eid)] = self._parse_expert_data(raw, eid)
                    self.victim.move_to_end((layer_idx, eid))
                    while len(self.victim) > self.victim_cap:
                        self.victim.popitem(last=False)
                    bytes_read += len(raw)
                except Exception as e:
                    log.warning("prefetch read failed L%d E%d: %s",
                                layer_idx, eid, e)
            else:
                fut.cancel()

        if backfill_futures:
            self._schedule_backfill(layer_idx, backfill_futures)

        with self._stats_lock:
            self.read_time += time.time() - t0
            self.reads += len(expert_ids)
            self.bytes_read += bytes_read
            self.cache_hits += cache_hits
            self.prefetch_hits += prefetch_hits
        return experts

    def _schedule_backfill(self, layer_idx, futures):
        """Parse and cache full 4-bit expert from SSD in background."""
        def _do_backfill():
            for eid, future in futures.items():
                try:
                    raw = future.result()
                    parsed = self._parse_expert_data(raw, eid)
                    with self._stats_lock:
                        self.bytes_read += len(raw)
                    if self.lru:
                        self.lru.put(layer_idx, eid, parsed)
                except Exception as e:
                    with self._stats_lock:
                        self.backfill_errors += 1
                    log.warning("backfill failed L%d E%d: %s", layer_idx, eid, e)
        self.backfill_executor.submit(_do_backfill)

    def stats(self):
        if self.reads == 0:
            return "No reads yet"
        ssd_reads = self.reads - self.cache_hits
        avg_ms = self.read_time / self.reads * 1000
        throughput = self.bytes_read / self.read_time / 1e9 if self.read_time > 0 else 0
        s = (f"reads={self.reads}, ssd_reads={ssd_reads}, cache_hits={self.cache_hits}, "
             f"prefetch_hits={self.prefetch_hits}, victim_hits={self.victim_hits}, "
             f"avg={avg_ms:.1f}ms/expert, "
             f"throughput={throughput:.1f} GB/s, "
             f"total_bytes={self.bytes_read/1e9:.2f} GB, "
             f"total_time={self.read_time:.1f}s")
        if self.backfill_errors:
            s += f"\n  backfill_errors={self.backfill_errors}"
        if self.lru:
            s += f"\n  {self.lru.stats()}"
        if self.fallback:
            s += f"\n  {self.fallback.stats()}"
        return s

    def close(self):
        self.reset_prefetch()
        # Wait for in-flight workers: a prefetch task can lazily open a new
        # layer fd while we are closing the others, which is a "dictionary
        # changed size during iteration" on the fd table (seen in CI).
        self.executor.shutdown(wait=True, cancel_futures=True)
        self.backfill_executor.shutdown(wait=True, cancel_futures=True)
        for fd in list(self.fds.values()):
            os.close(fd)
        self.fds.clear()
        if self.fallback:
            self.fallback.close()
