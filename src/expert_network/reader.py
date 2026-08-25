"""Distributed expert reader — the driver-side client.

Implements the reader interface `generate.make_forward` expects, but with
`compute_distributed`: the expert FFN is evaluated remotely on whichever
nodes own the active experts, and the float partials are summed here.
Broadcast model: every node receives the request and masks to its own
partition (nodes not owning any active expert return zeros).

Optional hot-expert cache (`hot_cache_gb`, default 0 = off): the driver keeps
a bounded LRU of raw expert blocks pulled from nodes via /block/{layer}/{eid}
and computes those experts locally. A node's partial is float16(sum over the
experts it owns), so splitting one node's expert set between the driver and
the node would change the rounding. The cache therefore works per node,
all-or-nothing: a node's round trip is skipped only when EVERY active expert
it owns is cached, and its partial is then produced here by the same
`compute_partial` the node runs, over the same expert set, summed in the same
node order — bit-identical to the uncached path. Misses warm the cache in the
background (fetch-and-store, one in-flight fetch per block).
"""
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from mlx_expert_sniper.expert_io import LRUExpertCache

from .protocol import pack_request, unpack_response

log = logging.getLogger("expert_network.reader")


class DistributedExpertReader:
    def __init__(self, node_urls, timeout=60, hot_cache_gb=0.0, layout=None,
                 sessions=None):
        """
        node_urls:    expert node base URLs (broadcast targets).
        hot_cache_gb: byte budget for the driver-side hot-expert cache
                      (0 = off, nothing changes for existing users).
        layout:       the streaming format's `layout` dict (header of
                      bin/layer_00.bin): needs `expert_block_size` and
                      `tensors`. Required when hot_cache_gb > 0.
        sessions:     optional {url: session-like} transport override
                      (tests inject in-process fake nodes here).
        """
        self.node_urls = [u.rstrip("/") for u in node_urls]
        self.timeout = timeout
        self.sessions = {}
        for url in self.node_urls:
            if sessions and url in sessions:
                self.sessions[url] = sessions[url]
                continue
            s = requests.Session()
            s.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=1,
                                           max_retries=2))
            self.sessions[url] = s
        self.executor = ThreadPoolExecutor(max_workers=len(self.node_urls))
        self.requests_sent = 0
        self.network_time = 0.0
        self.bytes_sent = 0
        self.bytes_received = 0

        # --- hot-expert cache (driver-side) ---
        self.hot_cache_gb = hot_cache_gb
        self.layout = layout
        self.hot = None
        self.hot_hits = 0       # active experts served from the cache
        self.hot_misses = 0     # active experts not in the cache
        self.hot_skipped = 0    # node round trips skipped
        self.hot_fetch_errors = 0
        self._owner = {}        # expert_id -> node url (None = nobody serves it)
        self._partition = {}    # node url -> set(expert_ids) once known
        self._inflight = {}     # (layer, eid) -> Future  (fetch dedupe)
        self._hot_lock = threading.Lock()
        self._fetch_executor = None
        if hot_cache_gb > 0:
            if not layout or "expert_block_size" not in layout \
                    or "tensors" not in layout:
                raise ValueError("hot_cache_gb > 0 needs the streaming "
                                 "layout (expert_block_size + tensors)")
            self.block_size = int(layout["expert_block_size"])
            budget = int(hot_cache_gb * 1e9)
            max_experts = budget // self.block_size
            if max_experts < 1:
                log.warning("hot cache budget %.3f GB < one expert block "
                            "(%d bytes); cache disabled", hot_cache_gb,
                            self.block_size)
            else:
                self.hot = LRUExpertCache(max_experts=max_experts)
                self._fetch_executor = ThreadPoolExecutor(
                    max_workers=max(1, len(self.node_urls)),
                    thread_name_prefix="hot-fetch")
                self._learn_partitions()
        # make_forward's routing-bias mask reads reader.lru.cached_ids().
        self.lru = self.hot

    # --- reader interface (no-ops: nodes hold everything resident) ---
    def prefetch_experts(self, layer_idx, expert_ids):
        pass

    def reset_prefetch(self):
        pass

    def get_experts(self, layer_idx, expert_ids):  # pragma: no cover
        raise RuntimeError("DistributedExpertReader computes remotely — "
                           "make_forward should call compute_distributed")

    # --- hot cache internals ---
    def _learn_partitions(self):
        """Ask each node for its full partition (best effort; older nodes
        without /partition are learned one expert at a time from /block)."""
        for url in self.node_urls:
            try:
                r = self.sessions[url].get(f"{url}/partition", timeout=5)
                if r.status_code != 200:
                    continue
                ids = set(int(e) for e in r.json().get("expert_ids", []))
            except Exception:
                continue
            with self._hot_lock:
                self._partition[url] = ids
                for e in ids:
                    self._owner[e] = url

    def _fetch_block(self, layer_idx, eid):
        """GET one raw expert block from its owner (probing nodes when the
        owner is unknown), parse, store. Runs on the fetch executor."""
        from .node import parse_expert_block
        with self._hot_lock:
            owner = self._owner.get(eid, "unknown")
        candidates = [owner] if owner not in ("unknown", None) else \
            ([] if owner is None else list(self.node_urls))
        raw = None
        for url in candidates:
            try:
                r = self.sessions[url].get(f"{url}/block/{layer_idx}/{eid}",
                                          timeout=self.timeout)
            except Exception as e:
                log.warning("block fetch L%d E%d from %s failed: %s",
                            layer_idx, eid, url, e)
                continue
            if r.status_code == 200 and len(r.content) == self.block_size:
                raw = r.content
                with self._hot_lock:
                    self._owner[eid] = url
                    self._partition.setdefault(url, set()).add(eid)
                self.bytes_received += len(raw)
                break
        if raw is None:
            if owner == "unknown" and candidates:
                with self._hot_lock:
                    self._owner[eid] = None   # nobody serves it; stop asking
            self.hot_fetch_errors += 1
            return False
        parsed = parse_expert_block(raw, self.layout["tensors"])
        # Materialize here: the arrays are built on this fetch thread and
        # consumed on the compute thread, and MLX will not evaluate a lazy
        # graph that carries another thread's stream.
        import mlx.core as mx
        mx.eval(*parsed.values())
        self.hot.put(layer_idx, eid, parsed)
        return True

    def _warm(self, layer_idx, eid):
        """Schedule a fetch unless cached or already in flight."""
        key = (layer_idx, eid)
        with self._hot_lock:
            if key in self._inflight or self.hot.contains(layer_idx, eid):
                return
            if self._owner.get(eid, "unknown") is None:
                return
            fut = self._fetch_executor.submit(self._fetch_block, layer_idx, eid)
            self._inflight[key] = fut

        def _done(f, key=key):
            with self._hot_lock:
                self._inflight.pop(key, None)
        fut.add_done_callback(_done)

    def wait_hot(self):
        """Block until every in-flight block fetch has landed (tests/tooling)."""
        while True:
            with self._hot_lock:
                futs = list(self._inflight.values())
            if not futs:
                return
            for f in futs:
                try:
                    f.result()
                except Exception:
                    pass

    def _plan(self, layer_idx, active_ids):
        """Decide, per node, whether its partial can be produced locally.
        Returns (local: {url: [eids]}, remote: [url]). A node is local only
        when the owner of every active expert is known and every expert this
        node owns is cached — anything less would split one node's float16
        partial and change the rounding."""
        local, remote = {}, []
        with self._hot_lock:
            owners = {e: self._owner.get(e, "unknown") for e in active_ids}
        all_known = all(o != "unknown" for o in owners.values())
        for url in self.node_urls:
            mine = [e for e in active_ids if owners[e] == url]
            if all_known and mine and all(self.hot.contains(layer_idx, e)
                                          for e in mine):
                local[url] = mine
            elif all_known and not mine and url in self._partition:
                # Node owns none of the active experts: its partial is zeros.
                local[url] = []
            else:
                remote.append(url)
        return local, remote

    def _local_partial(self, url, layer_idx, mine, active_ids, h_np, inds_np,
                       wt_np):
        from .node import compute_partial
        if not mine:
            return np.zeros(h_np.shape, dtype=np.float16)
        experts = {}
        for e in mine:
            data = self.hot.get(layer_idx, e)   # counts the hit, refreshes LRU
            if data is None:                    # evicted between plan and use
                return None
            experts[(layer_idx, e)] = data
        partition = self._partition[url]
        num_experts = max(int(inds_np.max()) if inds_np.size else 0,
                          max(active_ids), max(partition)) + 1
        _, out = compute_partial(experts, partition, num_experts,
                                 layer_idx, active_ids, h_np, inds_np, wt_np)
        return out

    # --- the actual data path ---
    def compute_distributed(self, layer_idx, x, active_ids, inds, weights):
        import mlx.core as mx

        h_np = np.array(x.astype(mx.float16))
        inds_np = np.array(inds).astype(np.int32)
        wt_np = np.array(weights.astype(mx.float32))
        # Same bytes the node unpacks: pack/unpack round-trips through the
        # wire format so local and remote compute see identical inputs.
        payload = pack_request(layer_idx, active_ids, h_np, inds_np, wt_np)
        active_ids = sorted(active_ids)

        t0 = time.time()

        local, remote = ({}, list(self.node_urls))   # url -> partial ndarray
        if self.hot is not None:
            plan, remote = self._plan(layer_idx, active_ids)
            cached = set()
            for url, mine in plan.items():
                out = self._local_partial(url, layer_idx, mine, active_ids,
                                          h_np, inds_np, wt_np)
                if out is None:
                    remote.append(url)
                else:
                    local[url] = out
                    cached.update(mine)
            self.hot_hits += len(cached)
            missing = [e for e in active_ids if e not in cached]
            self.hot_misses += len(missing)
            self.hot_skipped += len(local)
            for e in missing:
                self._warm(layer_idx, e)

        def post(url):
            r = self.sessions[url].post(f"{url}/compute_bin", data=payload,
                                        timeout=self.timeout)
            r.raise_for_status()
            return r.content

        futures = {u: self.executor.submit(post, u) for u in remote}
        total = None
        received = 0
        for url in self.node_urls:          # fixed order: summation is exact
            if url in local:
                partial = local[url]
            else:
                content = futures[url].result()
                received += len(content)
                _, partial = unpack_response(content)
            partial = partial.astype(np.float32)
            total = partial if total is None else total + partial

        self.network_time += time.time() - t0
        self.requests_sent += len(remote)
        self.bytes_sent += len(payload) * len(remote)
        self.bytes_received += received
        return mx.array(total.astype(np.float16))

    def health(self):
        out = {}
        for url in self.node_urls:
            try:
                out[url] = self.sessions[url].get(f"{url}/health", timeout=5).json()
            except Exception as e:
                out[url] = {"status": f"unreachable: {e}"}
        return out

    def stats(self):
        avg = (self.network_time / self.requests_sent * 1000
               * len(self.node_urls)) if self.requests_sent else 0
        s = (f"network: {self.requests_sent} requests to "
             f"{len(self.node_urls)} nodes, "
             f"avg={avg:.1f}ms/layer, "
             f"sent={self.bytes_sent/1e6:.1f} MB, "
             f"recv={self.bytes_received/1e6:.1f} MB, "
             f"total_time={self.network_time:.1f}s, "
             f"hot_hits={self.hot_hits}, hot_misses={self.hot_misses}")
        if self.hot is not None:
            s += (f"\n  hot cache: {len(self.hot.cache)}/{self.hot.max_experts} "
                  f"experts ({len(self.hot.cache) * self.block_size / 1e9:.2f}"
                  f"/{self.hot_cache_gb:.2f} GB), "
                  f"skipped_round_trips={self.hot_skipped}, "
                  f"fetch_errors={self.hot_fetch_errors}")
        return s

    def close(self):
        self.executor.shutdown(wait=False)
        if self._fetch_executor is not None:
            self._fetch_executor.shutdown(wait=True, cancel_futures=True)
        for s in self.sessions.values():
            s.close()
