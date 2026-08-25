"""expert-fetch: nearest-peer dispatch and resume/dedup (issue #14).

Peers are in-process http.server instances serving synthetic blocks with
known hashes; nothing leaves the loopback interface. Stdlib only.
"""

import hashlib
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

# Test the copy of the package next to this file, not whatever is installed.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from expert_network.fetch_partition import PAGE_SIZE, fetch_partition  # noqa: E402

NUM_LAYERS = 2
NUM_EXPERTS = 8
BLOCK = 4096
MINE = [1, 3, 5, 6]


def block_bytes(layer, eid):
    return hashlib.sha256(f"{layer}:{eid}".encode()).digest() * (BLOCK // 32)


MANIFEST = {f"{l}:{e}": hashlib.sha256(block_bytes(l, e)).hexdigest()
            for l in range(NUM_LAYERS) for e in range(NUM_EXPERTS)}


class Peer:
    """One seeder. `delay` seconds per block simulates a slow link."""

    def __init__(self, node_id, delay=0.0):
        self.node_id = node_id
        self.delay = delay
        self.requests = 0
        self._lock = threading.Lock()
        peer = self

        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                parts = self.path.strip("/").split("/")
                if len(parts) != 3 or parts[0] != "block":
                    self.send_response(404); self.end_headers(); return
                with peer._lock:
                    peer.requests += 1
                if peer.delay:
                    time.sleep(peer.delay)
                data = block_bytes(int(parts[1]), int(parts[2]))
                self.send_response(200)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    @property
    def info(self):
        return {"nodeId": self.node_id, "url": f"http://127.0.0.1:{self.srv.server_address[1]}"}

    def close(self):
        self.srv.shutdown()
        self.srv.server_close()


@pytest.fixture
def peers():
    fast = Peer("fast")
    slow = Peer("slow", delay=0.03)
    yield fast, slow
    fast.close()
    slow.close()


def pull(out, peers, **kw):
    return fetch_partition(out, MANIFEST, MINE, NUM_LAYERS, NUM_EXPERTS, BLOCK,
                           [p.info for p in peers], jobs=4, log=lambda *a, **k: None, **kw)


def read_block(out, layer, eid):
    with open(os.path.join(out, "bin", f"layer_{layer:02d}.bin"), "rb") as f:
        f.seek(PAGE_SIZE + eid * BLOCK)
        return f.read(BLOCK)


def test_fast_peer_gets_majority_of_requests(tmp_path, peers):
    fast, slow = peers
    stats = pull(str(tmp_path), peers, seed=1)
    total = len(MINE) * NUM_LAYERS
    assert stats["fetched"] == total and stats["skipped"] == 0
    assert stats["speeds"]["fast"] > stats["speeds"]["slow"] > 0
    # each peer serves the probe block once; after that the fast one leads
    assert fast.requests > slow.requests
    assert fast.requests > total * 0.6
    assert stats["delivered"]["fast"] > stats["delivered"]["slow"]
    assert stats["bytes"] == sum(stats["delivered"].values())
    for l in range(NUM_LAYERS):
        for e in MINE:
            assert read_block(str(tmp_path), l, e) == block_bytes(l, e)


def test_second_run_fetches_nothing(tmp_path, peers):
    fast, slow = peers
    pull(str(tmp_path), peers, seed=1)
    before = fast.requests + slow.requests
    stats = pull(str(tmp_path), peers, seed=1)
    assert stats["fetched"] == 0
    assert stats["skipped"] == len(MINE) * NUM_LAYERS
    assert stats["bytes"] == 0 and stats["delivered"] == {}
    assert fast.requests + slow.requests == before
    for l in range(NUM_LAYERS):
        for e in MINE:
            assert read_block(str(tmp_path), l, e) == block_bytes(l, e)


def test_corrupted_block_is_refetched(tmp_path, peers):
    fast, slow = peers
    pull(str(tmp_path), peers, seed=1)
    path = tmp_path / "bin" / "layer_01.bin"
    with open(path, "r+b") as f:
        f.seek(PAGE_SIZE + 5 * BLOCK + 100)
        f.write(b"\xff" * 16)
    assert read_block(str(tmp_path), 1, 5) != block_bytes(1, 5)
    before = fast.requests + slow.requests
    stats = pull(str(tmp_path), peers, seed=1)
    assert stats["fetched"] == 1
    assert stats["skipped"] == len(MINE) * NUM_LAYERS - 1
    assert stats["bytes"] == sum(stats["delivered"].values()) > 0
    # the single refetch doubles as the probe: one request per peer
    assert fast.requests + slow.requests == before + 2
    assert read_block(str(tmp_path), 1, 5) == block_bytes(1, 5)
    # untouched neighbours are still intact and other machines' holes stay holes
    assert read_block(str(tmp_path), 1, 3) == block_bytes(1, 3)
    assert read_block(str(tmp_path), 1, 0) == b"\x00" * BLOCK
