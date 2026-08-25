"""DistributedExpertReader hot-expert cache (issue #5).

Two fake nodes live in-process behind a mocked transport (the reader's
`sessions` dict), serving a tiny synthetic 4-bit model through the real wire
format and the real `compute_partial`. No model download, no sockets.

(a) cache off: transport traffic and output are exactly as before;
(b) cache on: after warming, the repeat request skips the transport for the
    nodes whose active set is cached, and the summed output is bit-identical
    to the uncached output.
"""
import threading

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from expert_network.node import compute_partial, parse_expert_block
from expert_network.protocol import pack_response, unpack_request
from expert_network.reader import DistributedExpertReader

NUM_LAYERS = 2
NUM_EXPERTS = 8
D = 64          # hidden
I = 64          # intermediate
GROUP = 64
BITS = 4
TOP_K = 2
NODES = ["http://n0", "http://n1"]
PARTITIONS = {"http://n0": set(range(0, 4)), "http://n1": set(range(4, 8))}


def _make_model(seed=0):
    """Blocks + layout in sniper streaming format, tensors quantized like the
    real preprocess (uint32 weights, float16 scales/biases)."""
    rng = np.random.default_rng(seed)
    names = []
    for proj, shape in (("gate_proj", (I, D)), ("up_proj", (I, D)),
                        ("down_proj", (D, I))):
        w = mx.array(rng.standard_normal(shape).astype(np.float16))
        q, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
        names.append((proj, q, s, b))
    # layout from expert 0's tensors (shapes/dtypes are the same for all)
    layout_tensors, off = {}, 0
    for proj, q, s, b in names:
        for kind, arr, dt in (("weight", q, "uint32"), ("scales", s, "float16"),
                              ("biases", b, "float16")):
            nb = arr.size * arr.itemsize
            layout_tensors[f"switch_mlp.{proj}.{kind}"] = {
                "inner_offset": off, "nbytes": nb,
                "shape_per_expert": list(arr.shape), "dtype": dt}
            off += nb
    layout = {"expert_block_size": off, "data_start": 16384,
              "tensors": layout_tensors}

    blocks = {}
    for layer in range(NUM_LAYERS):
        for eid in range(NUM_EXPERTS):
            parts = []
            for proj, shape in (("gate_proj", (I, D)), ("up_proj", (I, D)),
                                ("down_proj", (D, I))):
                w = mx.array(rng.standard_normal(shape).astype(np.float16))
                q, s, b = mx.quantize(w, group_size=GROUP, bits=BITS)
                for arr in (q, s, b):
                    parts.append(np.array(arr).tobytes())
            raw = b"".join(parts)
            assert len(raw) == off
            blocks[(layer, eid)] = raw
    return layout, blocks


class _Resp:
    def __init__(self, status, content=b"", json_obj=None):
        self.status_code = status
        self.content = content
        self._json = json_obj

    def json(self):
        return self._json

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeNode:
    """In-process expert node: same compute_partial, same wire format."""

    def __init__(self, url, partition, layout, blocks, serve_partition=True):
        self.url = url
        self.partition = set(partition)
        self.layout = layout
        self.serve_partition = serve_partition
        self.experts = {(l, e): parse_expert_block(raw, layout["tensors"])
                        for (l, e), raw in blocks.items() if e in self.partition}
        # Materialized at load, like load_partition: the node computes on the
        # reader's worker thread here, and MLX will not evaluate a lazy graph
        # built under another thread's stream.
        mx.eval(*[a for d in self.experts.values() for a in d.values()])
        self.blocks = blocks
        self.posts = 0
        self.block_gets = []
        self.lock = threading.Lock()

    def post(self, url, data, timeout=None):
        assert url == f"{self.url}/compute_bin"
        with self.lock:
            self.posts += 1
        layer, ids, h, inds, w = unpack_request(data)
        n, out = compute_partial(self.experts, self.partition, NUM_EXPERTS,
                                 layer, ids, h, inds, w)
        return _Resp(200, pack_response(n, out))

    def get(self, url, timeout=None):
        path = url[len(self.url):]
        if path == "/partition":
            if not self.serve_partition:
                return _Resp(404)
            return _Resp(200, json_obj={"expert_ids": sorted(self.partition)})
        if path == "/health":
            return _Resp(200, json_obj={"status": "ok"})
        if path.startswith("/block/"):
            layer, eid = (int(x) for x in path.split("/")[2:4])
            with self.lock:
                self.block_gets.append((layer, eid))
            if eid not in self.partition:
                return _Resp(404)
            return _Resp(200, self.blocks[(layer, eid)])
        return _Resp(404)

    def close(self):
        pass


@pytest.fixture(scope="module")
def model():
    return _make_model()


def _reader(model, hot_cache_gb=0.0, serve_partition=True):
    layout, blocks = model
    nodes = {u: FakeNode(u, PARTITIONS[u], layout, blocks, serve_partition)
             for u in NODES}
    r = DistributedExpertReader(NODES, hot_cache_gb=hot_cache_gb,
                                layout=layout, sessions=nodes)
    return r, nodes


def _inputs(seed, T=3):
    rng = np.random.default_rng(seed)
    x = mx.array(rng.standard_normal((1, T, D)).astype(np.float16))
    inds = np.stack([rng.choice(NUM_EXPERTS, TOP_K, replace=False)
                     for _ in range(T)])[None]            # [1, T, k]
    w = rng.random((1, T, TOP_K)).astype(np.float32)
    w /= w.sum(-1, keepdims=True)
    active = sorted(set(int(e) for e in inds.flatten()))
    return x, active, mx.array(inds.astype(np.int32)), mx.array(w)


def _run(r, layer, seed):
    x, active, inds, w = _inputs(seed)
    return np.array(r.compute_distributed(layer, x, active, inds, w)), active


def test_cache_off_is_unchanged(model):
    r, nodes = _reader(model)                       # default: off
    assert r.hot is None and r.lru is None
    out, active = _run(r, 0, 1)
    assert out.dtype == np.float16 and out.shape == (1, 3, D)
    assert all(n.posts == 1 for n in nodes.values())
    assert all(n.block_gets == [] for n in nodes.values())
    assert r.requests_sent == len(NODES)
    assert "hot_hits=0, hot_misses=0" in r.stats()
    # Reference: sum of the nodes' partials in node order (the old code path)
    x, active, inds, w = _inputs(1)
    total = None
    for u in NODES:
        _, p = compute_partial(nodes[u].experts, nodes[u].partition, NUM_EXPERTS,
                               0, active, np.array(x), np.array(inds),
                               np.array(w))
        p = p.astype(np.float32)
        total = p if total is None else total + p
    assert np.array_equal(out, total.astype(np.float16))
    r.close()


def test_cache_on_second_request_skips_transport_and_matches(model):
    layout, _ = model
    ref, _ = _reader(model)
    cached, nodes = _reader(model, hot_cache_gb=layout["expert_block_size"]
                            * NUM_EXPERTS * NUM_LAYERS / 1e9)
    assert cached.hot.max_experts == NUM_EXPERTS * NUM_LAYERS
    for seed in (7, 8):
        for layer in range(NUM_LAYERS):
            # 1st call: everything misses -> broadcast as before, warm async
            out1, active = _run(cached, layer, seed)
            want, _ = _run(ref, layer, seed)
            assert np.array_equal(out1, want)
            cached.wait_hot()
            posts_before = {u: n.posts for u, n in nodes.items()}
            gets_before = {u: list(n.block_gets) for u, n in nodes.items()}
            # 2nd call, same experts: no transport, identical bytes out
            out2, _ = _run(cached, layer, seed)
            assert np.array_equal(out2, want)
            assert all(n.posts == posts_before[u] for u, n in nodes.items())
            assert all(n.block_gets == gets_before[u] for u, n in nodes.items())
    assert cached.hot_hits > 0 and cached.hot_misses > 0
    assert cached.hot_skipped > 0
    s = cached.stats()
    assert f"hot_hits={cached.hot_hits}" in s and f"hot_misses={cached.hot_misses}" in s
    ref.close(); cached.close()


def test_partial_coverage_keeps_node_remote_and_exact(model):
    """A node whose active set is only partly cached must still be asked
    remotely (splitting its float16 partial would change rounding)."""
    layout, _ = model
    ref, _ = _reader(model)
    r, nodes = _reader(model, hot_cache_gb=layout["expert_block_size"] / 1e9)
    assert r.hot.max_experts == 1                  # byte budget honored
    x = mx.array(np.random.default_rng(3).standard_normal((1, 1, D)).astype(np.float16))
    inds = mx.array(np.array([[[0, 1]]], dtype=np.int32))   # both on node n0
    w = mx.array(np.array([[[0.6, 0.4]]], dtype=np.float32))
    want = np.array(ref.compute_distributed(0, x, [0, 1], inds, w))
    r.compute_distributed(0, x, [0, 1], inds, w)
    r.wait_hot()
    assert len(r.hot.cache) == 1
    got = np.array(r.compute_distributed(0, x, [0, 1], inds, w))
    assert np.array_equal(got, want)
    assert nodes["http://n0"].posts == 2           # n0 still asked
    # n1's partition is known from /partition at construction, so the
    # reader knows it owns none of {0, 1} and never asks it, even cold.
    assert nodes["http://n1"].posts == 0
    assert r.hot_hits == 0
    ref.close(); r.close()


def test_inflight_dedupe_and_ownership_probe(model):
    """Without /partition the owner is learned by probing /block; a burst of
    misses for the same expert issues one fetch."""
    layout, _ = model
    r, nodes = _reader(model, hot_cache_gb=1.0, serve_partition=False)
    assert r._partition == {}
    x, active, inds, w = _inputs(5)
    for _ in range(4):
        r.compute_distributed(1, x, active, inds, w)   # misses each time
    r.wait_hot()
    gets = [g for n in nodes.values() for g in n.block_gets]
    for e in active:
        # exactly one successful fetch per expert (plus at most one 404 probe)
        owner = r._owner[e]
        assert owner in NODES and e in PARTITIONS[owner]
        assert nodes[owner].block_gets.count((1, e)) == 1
    assert len(gets) <= 2 * len(active)
    assert all(r.hot.contains(1, e) for e in active)
    r.close()


def test_budget_below_one_block_disables_cache(model):
    layout, _ = model
    r, _ = _reader(model, hot_cache_gb=(layout["expert_block_size"] - 1) / 1e9)
    assert r.hot is None
    r.close()


def test_hot_cache_requires_layout():
    with pytest.raises(ValueError):
        DistributedExpertReader(NODES, hot_cache_gb=1.0)
