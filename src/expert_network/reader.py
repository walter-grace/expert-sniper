"""Distributed expert reader — the driver-side client.

Implements the reader interface `generate.make_forward` expects, but with
`compute_distributed`: the expert FFN is evaluated remotely on whichever
nodes own the active experts, and the float partials are summed here.
Broadcast model: every node receives the request and masks to its own
partition (nodes not owning any active expert return zeros).
"""
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from requests.adapters import HTTPAdapter

from .protocol import pack_request, unpack_response


class DistributedExpertReader:
    def __init__(self, node_urls, timeout=60):
        self.node_urls = [u.rstrip("/") for u in node_urls]
        self.timeout = timeout
        self.lru = None  # no driver-side cache (yet) — bias masks stay off
        self.sessions = {}
        for url in self.node_urls:
            s = requests.Session()
            s.mount("http://", HTTPAdapter(pool_connections=1, pool_maxsize=1,
                                           max_retries=2))
            self.sessions[url] = s
        self.executor = ThreadPoolExecutor(max_workers=len(self.node_urls))
        self.requests_sent = 0
        self.network_time = 0.0
        self.bytes_sent = 0
        self.bytes_received = 0

    # --- reader interface (no-ops: nodes hold everything resident) ---
    def prefetch_experts(self, layer_idx, expert_ids):
        pass

    def reset_prefetch(self):
        pass

    def get_experts(self, layer_idx, expert_ids):  # pragma: no cover
        raise RuntimeError("DistributedExpertReader computes remotely — "
                           "make_forward should call compute_distributed")

    # --- the actual data path ---
    def compute_distributed(self, layer_idx, x, active_ids, inds, weights):
        import mlx.core as mx

        h_np = np.array(x.astype(mx.float16))
        inds_np = np.array(inds).astype(np.int32)
        wt_np = np.array(weights.astype(mx.float32))
        payload = pack_request(layer_idx, active_ids, h_np, inds_np, wt_np)

        t0 = time.time()

        def post(url):
            r = self.sessions[url].post(f"{url}/compute_bin", data=payload,
                                        timeout=self.timeout)
            r.raise_for_status()
            return r.content

        futures = [self.executor.submit(post, u) for u in self.node_urls]
        total = None
        received = 0
        for fut in futures:
            content = fut.result()
            received += len(content)
            _, partial = unpack_response(content)
            partial = partial.astype(np.float32)
            total = partial if total is None else total + partial

        self.network_time += time.time() - t0
        self.requests_sent += len(self.node_urls)
        self.bytes_sent += len(payload) * len(self.node_urls)
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
        return (f"network: {self.requests_sent} requests to "
                f"{len(self.node_urls)} nodes, "
                f"avg={avg:.1f}ms/layer, "
                f"sent={self.bytes_sent/1e6:.1f} MB, "
                f"recv={self.bytes_received/1e6:.1f} MB, "
                f"total_time={self.network_time:.1f}s")

    def close(self):
        self.executor.shutdown(wait=False)
        for s in self.sessions.values():
            s.close()
