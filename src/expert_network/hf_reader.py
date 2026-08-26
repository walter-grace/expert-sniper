"""HFExpertReader: the engine's reader interface served from local sparse
bin/ files first and the HuggingFace checkpoint for the holes.

Satisfies what the engines call on ``engine.reader``:
  get_experts(layer, ids) -> {eid: {tensor: mx.array}}
  prefetch_experts(layer, ids) / reset_prefetch()
  lru.contains(layer, eid) / lru.cached_ids(layer)
  stats() / close()

Blocks are parsed with the layer header layout exactly like
expert_network.node.parse_expert_block. Which local blocks are real (not
sparse holes) comes from manifest.partial.json when present, else from a
non-zero probe of the block's first bytes.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from mlx_expert_sniper.expert_io import LRUExpertCache
from .hf_source import HFCheckpoint, PAGE_SIZE
from .node import parse_expert_block


class HFExpertReader:
    def __init__(self, model_dir, repo=None, num_layers=None, cache_size=600,
                 jobs=12, hf=None, log=print):
        self.model_dir = model_dir
        self.expert_dir = os.path.join(model_dir, "bin")
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        repo = repo or cfg.get("streaming", {}).get("source_repo")
        self.hf = hf or HFCheckpoint(repo, jobs=jobs)
        self.num_layers = num_layers or self.hf.num_layers
        self.lru = LRUExpertCache(max_experts=cache_size)
        self.pool = ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="hf-expert")
        self.prefetch_futures = {}          # layer -> {eid: future}
        self._fds, self._headers = {}, {}
        self._layouts = {}
        self._lock = threading.Lock()
        self.log = log
        # local blocks known to be real
        self.local = set()
        mp = os.path.join(model_dir, "manifest.partial.json")
        if os.path.exists(mp):
            self.local = {tuple(int(x) for x in k.split(":"))
                          for k in json.load(open(mp)).get("blocks", {})}
        self.reads = self.local_hits = self.hf_reads = self.bytes_read = 0
        self.read_time = 0.0

    # ---- local sparse files ---------------------------------------------------
    def _layer(self, layer):
        with self._lock:
            if layer in self._headers:
                return self._fds[layer], self._headers[layer]
            path = os.path.join(self.expert_dir, f"layer_{layer:02d}.bin")
            if not os.path.exists(path):
                self._fds[layer], self._headers[layer] = None, None
                return None, None
            fd = os.open(path, os.O_RDONLY)
            hdr = json.loads(os.pread(fd, PAGE_SIZE, 0).rstrip(b"\x00"))
            self._fds[layer], self._headers[layer] = fd, hdr
            return fd, hdr

    def layout(self, layer):
        _, hdr = self._layer(layer)
        if hdr is not None:
            return hdr["layout"]
        with self._lock:
            if layer not in self._layouts:
                self._layouts[layer] = self.hf.expert_layout(layer)
            return self._layouts[layer]

    def _read_local(self, layer, eid):
        fd, hdr = self._layer(layer)
        if fd is None:
            return None
        lay = hdr["layout"]
        off = lay["data_start"] + eid * lay["expert_block_size"]
        if self.local and (layer, eid) not in self.local:
            return None
        if not self.local:
            probe = os.pread(fd, 4096, off)
            if not any(probe):
                return None
        return os.pread(fd, lay["expert_block_size"], off)

    # ---- fetch one block ----------------------------------------------------
    def _fetch(self, layer, eid):
        raw = self._read_local(layer, eid)
        src = "local"
        if raw is None:
            src = "hf"
            hl = self._layouts.get(layer)
            if hl is None:
                with self._lock:
                    hl = self._layouts.get(layer)
                    if hl is None:
                        hl = self._layouts[layer] = self.hf.expert_layout(layer)
            raw = self.hf.fetch_expert_block(layer, eid, hl)
        return src, raw

    # ---- reader interface ---------------------------------------------------
    def prefetch_experts(self, layer, expert_ids):
        futs = self.prefetch_futures.setdefault(layer, {})
        for eid in expert_ids:
            if eid in futs or self.lru.contains(layer, eid):
                continue
            futs[eid] = self.pool.submit(self._fetch, layer, eid)

    def reset_prefetch(self):
        self.prefetch_futures.clear()

    def get_experts(self, layer, expert_ids):
        t0 = time.time()
        futs = self.prefetch_futures.pop(layer, {})
        out, pending = {}, {}
        for eid in expert_ids:
            c = self.lru.get(layer, eid)
            if c is not None:
                out[eid] = c
                continue
            pending[eid] = futs.get(eid) or self.pool.submit(self._fetch, layer, eid)
        tl = self.layout(layer)["tensors"]
        for eid, f in pending.items():
            src, raw = f.result()
            parsed = parse_expert_block(raw, tl)
            self.lru.put(layer, eid, parsed)
            out[eid] = parsed
            self.bytes_read += len(raw)
            if src == "local":
                self.local_hits += 1
            else:
                self.hf_reads += 1
        # keep prefetches that were not consumed
        for eid, f in futs.items():
            if eid not in pending and not self.lru.contains(layer, eid):
                try:
                    src, raw = f.result()
                    self.lru.put(layer, eid, parse_expert_block(raw, tl))
                except Exception:  # noqa: BLE001
                    pass
        self.reads += len(expert_ids)
        self.read_time += time.time() - t0
        return out

    def stats(self):
        return (f"reads={self.reads}, local={self.local_hits}, hf={self.hf_reads}, "
                f"bytes={self.bytes_read / 1e9:.2f} GB, wait={self.read_time:.1f}s; "
                f"{self.lru.stats()}")

    def close(self):
        self.reset_prefetch()
        self.pool.shutdown(wait=False)
        for fd in self._fds.values():
            if fd is not None:
                os.close(fd)
        self._fds.clear()
