"""
Ollama-compatible HTTP server for Expert Sniper.

Implements /api/tags, /api/chat, /api/generate, /api/version.
Compatible with Open WebUI, Continue.dev, and any Ollama client.

Serving is intentionally single-threaded: the engine drives one Metal
stream, so concurrent forward passes would corrupt each other's KV cache.
Requests queue at the socket; one long generation blocks the next client.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, os, time

_engine = None
_bias = 0.0
_model_dir = None
_model_name = "expert-sniper"
_model_size = 0
_draft_cache = {}


def _get_engine():
    global _engine, _bias, _model_name, _model_size
    if _engine is not None:
        return _engine

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    print("Loading model...", flush=True)

    from .generate import load_engine
    _engine, _bias, model_type = load_engine(_model_dir)

    try:
        with open(os.path.join(_model_dir, "config.json")) as f:
            config = json.load(f)
        _model_name = config.get("model_type", model_type) or model_type
    except Exception:
        _model_name = model_type
    _model_size = 0
    for root, _, names in os.walk(_model_dir):
        for name in names:
            p = os.path.join(root, name)
            if not os.path.islink(p):  # layer_XX.bin symlinks alias real files
                _model_size += os.lstat(p).st_size
    print(f"  Model loaded ({_model_name}, bias={_bias}).", flush=True)
    return _engine


class OllamaHandler(BaseHTTPRequestHandler):
    def _cors(self):
        # Browser chat UIs (e.g. a page on another origin talking to this
        # local server) need CORS; the server still binds 127.0.0.1.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/tags":
            self._json_response({
                "models": [{
                    "name": _model_name,
                    "model": _model_name,
                    "size": _model_size,
                    "details": {"family": _model_name,
                                "quantization_level": "Q4_0"},
                }]
            })
        elif self.path in ("/race", "/"):
            from .race_page import RACE_HTML
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self._cors()
            self.end_headers()
            self.wfile.write(RACE_HTML.encode())
        elif self.path == "/api/version":
            self._json_response({"version": "0.2.0-sniper"})
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/generate"):
            self.send_response(404)
            self.end_headers()
            return

        # Parse the request and load the engine BEFORE committing to a 200,
        # so failures surface as HTTP errors instead of truncated bodies.
        try:
            content_len = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(content_len))
            if "messages" in body:
                messages = body["messages"]
            else:
                messages = [{"role": "user",
                             "content": body.get("prompt", "hello")}]
            stream = body.get("stream", True)
            opts = body.get("options", {})
            max_tokens = opts.get("num_predict", 200)
            engine = _get_engine()
        except Exception as e:
            self.send_error(500, str(e))
            return

        # A client that stops reading mid-stream (closed tab) must not
        # wedge the single-threaded server on a blocking write forever.
        self.connection.settimeout(60)

        print(f"[req] {'fast-token' if opts.get('spec') else 'standard'} "
              f"start: {messages[-1].get('content', '')[:40]!r}", flush=True)

        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self._cors()
        self.end_headers()

        spec_stats = {}
        if opts.get("spec"):
            from .speculative import (spec_generate_stream, RemoteDraft,
                                      ModelDraft)
            global _draft_cache
            if opts.get("draft_url"):
                draft = RemoteDraft(opts["draft_url"], opts.get("draft_model"),
                                    engine.tokenizer)
            elif opts.get("draft_model"):
                # Local in-process drafting — the fastest, simplest option.
                # Loaded once and cached across requests.
                path = os.path.expanduser(opts["draft_model"])
                if _draft_cache.get("path") != path:
                    _draft_cache = {"path": path,
                                    "draft": ModelDraft(path, engine.tokenizer)}
                draft = _draft_cache["draft"]
            else:
                draft = None
            gen = spec_generate_stream(engine, messages, bias=_bias,
                                       max_tokens=max_tokens,
                                       k=int(opts.get("spec_k", 8)),
                                       draft=draft, stats=spec_stats)
        else:
            from .generate import generate_stream
            gen = generate_stream(engine, messages, bias=_bias,
                                  max_tokens=max_tokens)
        t0 = time.time()
        total_tokens = 0
        full_response = ""
        preview = messages[-1].get("content", "")[:40]

        try:
            for token_text in gen:
                total_tokens += 1
                full_response += token_text
                if stream:
                    self._ndjson({"model": _model_name,
                                  "message": {"role": "assistant",
                                              "content": token_text},
                                  "done": False})
        except (BrokenPipeError, ConnectionResetError, TimeoutError, OSError):
            print(f"  [client gone after {total_tokens} tok] {preview}")
            return

        elapsed = time.time() - t0
        done = {
            "model": _model_name,
            "message": {"role": "assistant",
                        "content": "" if stream else full_response},
            "done": True,
            "total_duration": int(elapsed * 1e9),
            "eval_count": total_tokens,
            "eval_duration": int(elapsed * 1e9),
        }
        if spec_stats.get("forwards"):
            done["spec"] = {
                "forwards": spec_stats["forwards"],
                "drafted": spec_stats["drafted"],
                "accepted": spec_stats["accepted"],
            }
        try:
            self._ndjson(done)
        except (BrokenPipeError, ConnectionResetError):
            pass
        tps = total_tokens / elapsed if elapsed > 0 else 0
        print(f"  [{total_tokens} tok, {tps:.1f} tok/s, {elapsed:.1f}s] {preview}")

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self._cors()
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _ndjson(self, data):
        self.wfile.write((json.dumps(data) + "\n").encode())
        self.wfile.flush()

    def log_message(self, format, *args):
        pass


def run_server(model_dir, host="127.0.0.1", port=11434):
    global _model_dir
    _model_dir = model_dir

    print(f"mlx-sniper serve")
    print(f"  Model:  {model_dir}")
    print(f"  Listen: http://{host}:{port}")
    print(f"  API:    Ollama-compatible (/api/tags, /api/chat, /api/generate)")
    print()

    _get_engine()  # Pre-load

    print(f"\nReady. Listening on http://{host}:{port}")
    print(f"  Test: curl http://localhost:{port}/api/tags")
    print(f"  Chat: curl http://localhost:{port}/api/chat -d "
          f"'{{\"model\":\"{_model_name}\",\"messages\":"
          f"[{{\"role\":\"user\",\"content\":\"hello\"}}]}}'")
    print()

    import socket
    HTTPServer.allow_reuse_address = True
    server = HTTPServer((host, port), OllamaHandler)
    server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()
