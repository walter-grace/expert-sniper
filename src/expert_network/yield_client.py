"""Machine Yield client — joins this node to a coordinator.

With --join, the node registers itself using the operator's API key and
heartbeats while serving. The coordinator scores the machine by what it can
PROVE: it fetches known content-addressed expert blocks from /block/... and
verifies the sha256 while timing the transfer, so bandwidth, latency, and
uptime are measured, never self-reported. Yield share is computed
coordinator-side from stake and that proven machine score.
"""
import json
import threading
import time
import urllib.request

HEARTBEAT_SECONDS = 60


class YieldClient:
    def __init__(self, coordinator, api_key, node_id, advertise_url, info):
        self.coordinator = coordinator.rstrip("/")
        self.api_key = api_key
        self.node_id = node_id
        self.advertise_url = advertise_url
        self.info = info
        self._stop = threading.Event()
        self._thread = None

    def _post(self, path, payload):
        req = urllib.request.Request(
            f"{self.coordinator}{path}",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"},
            method="POST")
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())

    def join(self):
        """Register with the coordinator. Never fatal: a coordinator outage (a 500, a timeout) must
        not take the node down — it keeps serving and the heartbeat loop keeps retrying the join."""
        try:
            resp = self._post("/api/yield/join", {
                "node_id": self.node_id,
                "url": self.advertise_url,
                **self.info,
            })
            self.joined = not resp.get("error")
            print(f"[yield] joined as {self.node_id!r}: "
                  f"{resp.get('status', resp)}")
            return resp
        except Exception as e:
            self.joined = False
            print(f"[yield] join failed (will retry every {HEARTBEAT_SECONDS}s, still serving): {e}")
            return {"error": str(e)}

    def start_heartbeat(self, get_stats):
        def loop():
            while not self._stop.wait(HEARTBEAT_SECONDS):
                if not getattr(self, "joined", False):
                    self.join()
                    continue
                try:
                    resp = self._post("/api/yield/heartbeat", {
                        "node_id": self.node_id,
                        "stats": get_stats(),
                    })
                    # The coordinator forgot us (record expired, store reset): re-join next tick.
                    if isinstance(resp, dict) and resp.get("error"):
                        print(f"[yield] heartbeat rejected: {resp.get('error')} — re-joining")
                        self.joined = False
                except Exception as e:
                    print(f"[yield] heartbeat failed: {e}")
                    if "404" in str(e):
                        self.joined = False
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
