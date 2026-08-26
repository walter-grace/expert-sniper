"""Streamed n-gram tables for qwen4_exp PLE layers.

The hashed n-gram table of Qwen3.8-Flash-Next is ~320M rows x 160 dims
(4-bit: 100 bytes/row, ~32 GB) — far too big to pin. Every token only
touches ``ngram_heads`` (16) rows though, so the table is read by row:

  DiskNGramSource   bin/ngram/layout.json + shard_I.{weight,scales,biases}
                    raw row files (HFCheckpoint.download_ngram /
                    download._write_ngram_tensor), one os.pread per row
  HFNGramSource     HTTP Range reads straight from the checkpoint through
                    HFCheckpoint.locate() + read_range(); one lookup's rows
                    are coalesced into ranges and fetched on a thread pool

Both implement models.qwen4_exp.NGramSource (``rows(shard_idx, ids)``) and
share the dequantisation + LRU in ``RowNGramSource``. ``make_ngram_source``
picks one from ``streaming.ngram_source`` in config.json.
"""
import json
import os
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .models.qwen4_exp import NGramSource

ST_NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32,
         "I32": np.int32, "U8": np.uint8, "I8": np.int8}
ST_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "U8": 1, "I8": 1}
KINDS = ("weight", "scales", "biases")


class RowLRU:
    """(shard, row) -> dict of raw bytes per kind. Small: rows are ~100 B."""
    def __init__(self, max_rows=200_000):
        self.max_rows = max_rows
        self._d = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key):
        with self._lock:
            v = self._d.get(key)
            if v is None:
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return v

    def put(self, key, v):
        with self._lock:
            self._d[key] = v
            self._d.move_to_end(key)
            if len(self._d) > self.max_rows:
                self._d.popitem(last=False)

    def __len__(self):
        return len(self._d)


class RowNGramSource(NGramSource):
    """Row-addressed table with optional affine quantisation. Subclasses
    implement ``_read_rows(kind, shard_idx, ids) -> list[bytes]`` for the
    rows that missed the LRU."""

    def __init__(self, tensor_meta, group_size=32, bits=4, lru_rows=200_000):
        """tensor_meta: {"shard_7.weight": {"shape": [rows, w], "dtype": "U32",
        "row_bytes": n}, ...} — the layout.json / safetensors geometry."""
        self.meta = tensor_meta
        self.group_size, self.bits = group_size, bits
        self.quantized = any(k.endswith(".scales") for k in tensor_meta)
        self.kinds = KINDS if self.quantized else ("weight",)
        w0 = next(v for k, v in tensor_meta.items() if k.endswith(".weight"))
        self.rows_per_shard = int(w0["shape"][0])
        self.num_shards = sum(1 for k in tensor_meta if k.endswith(".weight"))
        self.lru = RowLRU(lru_rows)
        self.bytes_read = 0
        self.reads = 0

    # ---- subclass hook ------------------------------------------------------
    def _read_rows(self, kind, shard_idx, ids):
        raise NotImplementedError

    # ---- NGramSource ----------------------------------------------------------
    def rows(self, shard_idx, ids):
        import mlx.core as mx
        ids_np = np.array(ids).astype(np.int64).reshape(-1)
        uniq, inverse = np.unique(ids_np, return_inverse=True)
        raw = [None] * len(uniq)
        missing = []
        for i, r in enumerate(uniq):
            v = self.lru.get((shard_idx, int(r)))
            if v is None:
                missing.append(i)
            else:
                raw[i] = v
        if missing:
            miss_ids = [int(uniq[i]) for i in missing]
            per_kind = {k: self._read_rows(k, shard_idx, miss_ids) for k in self.kinds}
            for j, i in enumerate(missing):
                v = {k: per_kind[k][j] for k in self.kinds}
                raw[i] = v
                self.lru.put((shard_idx, miss_ids[j]), v)
            self.reads += len(miss_ids)
        arrays = {}
        for k in self.kinds:
            m = self.meta[f"shard_{shard_idx}.{k}"]
            buf = b"".join(v[k] for v in raw)
            a = np.frombuffer(buf, dtype=ST_NP[m["dtype"]]).reshape(len(uniq), *m["shape"][1:])
            a = mx.array(a)
            if m["dtype"] == "BF16":
                a = a.view(mx.bfloat16)
            arrays[k] = a
        if self.quantized:
            out = mx.dequantize(arrays["weight"], arrays["scales"], arrays["biases"],
                                group_size=self.group_size, bits=self.bits)
        else:
            out = arrays["weight"]
        return out[mx.array(inverse.astype(np.int32))]

    def stats(self):
        return (f"ngram rows={self.reads}, bytes={self.bytes_read / 1e6:.1f} MB, "
                f"lru={len(self.lru)} hit_rate={self.lru.hits / max(1, self.lru.hits + self.lru.misses):.1%}")


class DiskNGramSource(RowNGramSource):
    """bin/ngram/<shard_i.kind> raw files: row r at r * row_bytes."""

    def __init__(self, ngram_dir, group_size=32, bits=4, lru_rows=200_000):
        with open(os.path.join(ngram_dir, "layout.json")) as f:
            layout = json.load(f)
        super().__init__(layout["tensors"], group_size, bits, lru_rows)
        self.dir = ngram_dir
        self._fds = {}
        self._lock = threading.Lock()

    def _fd(self, name):
        with self._lock:
            fd = self._fds.get(name)
            if fd is None:
                fd = os.open(os.path.join(self.dir, self.meta[name]["file"]), os.O_RDONLY)
                self._fds[name] = fd
            return fd

    def _read_rows(self, kind, shard_idx, ids):
        name = f"shard_{shard_idx}.{kind}"
        rb = self.meta[name]["row_bytes"]
        fd = self._fd(name)
        out = []
        for r in ids:
            b = os.pread(fd, rb, r * rb)
            if len(b) != rb:
                raise IOError(f"short n-gram row read {name}[{r}]")
            out.append(b)
        self.bytes_read += rb * len(ids)
        return out

    def close(self):
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()


class HFNGramSource(RowNGramSource):
    """Rows by HTTP Range straight from the safetensors shard on HF.

    A CDN request costs ~0.4 s regardless of size, so the rows of one lookup
    are sorted, neighbours closer than ``coalesce_bytes`` are merged into one
    range, and the ranges go out concurrently on a thread pool."""

    def __init__(self, hf, layer_idx, group_size=32, bits=4, lru_rows=200_000,
                 coalesce_bytes=256 * 1024, workers=16):
        self.hf = hf
        self.layer_idx = layer_idx
        prefix = f"{hf.prefix}layers.{layer_idx}.ple.ple_embedding.ngram_embedding."
        meta, self._loc = {}, {}
        for name in hf.ngram_names():
            if not name.startswith(prefix):
                continue
            local = name[len(prefix):]
            shard, dtype, shape, s, e = hf.locate(name)
            meta[local] = {"shape": [int(x) for x in shape], "dtype": dtype,
                           "row_bytes": int(np.prod(shape[1:])) * ST_SIZE[dtype], "source": name}
            self._loc[local] = (shard, s)
        if not meta:
            raise ValueError(f"no n-gram tensors for layer {layer_idx} in {hf.repo}")
        super().__init__(meta, group_size, bits, lru_rows)
        self.coalesce = coalesce_bytes
        self.pool = ThreadPoolExecutor(max_workers=workers)
        self.requests = 0

    def _read_rows(self, kind, shard_idx, ids):
        name = f"shard_{shard_idx}.{kind}"
        rb = self.meta[name]["row_bytes"]
        shard, base = self._loc[name]
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        # coalesce sorted rows into byte ranges
        ranges = []  # (start_row, end_row_inclusive)
        for i in order:
            r = ids[i]
            if ranges and (r - ranges[-1][1] - 1) * rb <= self.coalesce:
                ranges[-1][1] = max(ranges[-1][1], r)
            else:
                ranges.append([r, r])

        def fetch(rg):
            a, b = rg
            return rg, self.hf.read_range(shard, base + a * rb, base + (b + 1) * rb - 1)
        blobs = {}
        for (a, b), data in self.pool.map(fetch, ranges):
            blobs[a] = (b, data)
            self.bytes_read += len(data)
        self.requests += len(ranges)
        starts = sorted(blobs)
        out = [None] * len(ids)
        for i, r in enumerate(ids):
            # find the range containing r
            j = max(k for k in starts if k <= r)
            b, data = blobs[j]
            off = (r - j) * rb
            out[i] = data[off:off + rb]
        return out

    def stats(self):
        return super().stats() + f", http_requests={self.requests}"


def make_ngram_source(kind, config, model_dir, layer_idx, hf=None):
    """``kind``: "disk" | "hf" (memory is handled inside the model)."""
    q = config.get("quantization") or {}
    gs, bits = int(q.get("group_size", 32)), int(q.get("bits", 4))
    streaming = config.get("streaming", {})
    if kind == "disk":
        d = os.path.join(model_dir, streaming.get("ngram_dir") or "bin/ngram")
        return DiskNGramSource(d, gs, bits)
    if kind == "hf":
        if hf is None:
            from expert_network.hf_source import HFCheckpoint
            repo = streaming.get("source_repo")
            if not repo:
                raise ValueError("streaming.source_repo is required for ngram_source=hf")
            hf = HFCheckpoint(repo, jobs=4)
        return HFNGramSource(hf, layer_idx, gs, bits)
    raise ValueError(f"unknown ngram_source {kind!r} (memory|disk|hf)")
