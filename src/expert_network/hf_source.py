"""Read a model straight out of a HuggingFace checkpoint, one byte range at a
time — so a machine can take its slice of a 112 GB model without ever
holding the model.

safetensors puts every tensor at a known offset inside its shard, and the
MLX conversions store each layer's experts stacked as [num_experts, ...].
So expert `e` of layer `L` is a fixed byte range of nine tensors, and HF's
CDN answers HTTP Range requests. A node that owns 1/8 of the experts pulls
1/8 of the bytes, writes them into the same sparse layer files the engine
reads, and is serving twenty minutes later.

Three jobs live here:

  HFCheckpoint(repo).fetch_expert_block(L, e)   the bytes of one block
  HFCheckpoint(repo).download_pinned(out)        the small resident trunk
  HFCheckpoint(repo).download_ngram(out)         hashed n-gram tables, as
                                                  row-addressable files the
                                                  driver reads on demand
  HFCheckpoint(repo).hash_all_blocks()           a manifest, streamed, for
                                                  `mlx-sniper publish --from-hf`
"""
import hashlib
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

PAGE_SIZE = 16384
UA = "expert-fetch/0.4 (hf-source)"

# The nine expert tensors, in the order the streaming format lays them out.
TENSOR_ORDER = [
    "switch_mlp.gate_proj.weight", "switch_mlp.gate_proj.scales", "switch_mlp.gate_proj.biases",
    "switch_mlp.up_proj.weight", "switch_mlp.up_proj.scales", "switch_mlp.up_proj.biases",
    "switch_mlp.down_proj.weight", "switch_mlp.down_proj.scales", "switch_mlp.down_proj.biases",
]

DTYPE_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "U32": 4, "I32": 4, "I64": 8, "U8": 1, "I8": 1, "BOOL": 1}
# safetensors dtype → the string the engines' readers expect in layer headers
DTYPE_MLX = {"BF16": "mlx.core.bfloat16", "F16": "mlx.core.float16", "F32": "mlx.core.float32",
             "U32": "mlx.core.uint32", "I32": "mlx.core.int32", "I64": "mlx.core.int64",
             "U8": "mlx.core.uint8", "I8": "mlx.core.int8"}


def _nbytes(shape, dtype):
    n = 1
    for d in shape:
        n *= int(d)
    return n * DTYPE_SIZE[dtype]


class HFCheckpoint:
    def __init__(self, repo, revision="main", token=None, jobs=12):
        self.repo = repo
        self.revision = revision
        self.token = token or os.environ.get("HF_TOKEN")
        self.jobs = jobs
        self.base = f"https://huggingface.co/{repo}/resolve/{revision}/"
        self._headers = {}          # shard → parsed safetensors header
        self._data_start = {}       # shard → byte offset of tensor data
        self._cdn = {}              # shard → (signed CDN url, expiry ts)
        self._lock = threading.Lock()
        # One CDN range request costs ~0.4 s whatever its size, and a single
        # connection tops out well under the line rate, so the nine slices of
        # a block are read concurrently on a shared pool. Measured: 7.5 MB/s
        # sequential → ~30 MB/s (line-limited) with 48 slices in flight.
        self._slices = ThreadPoolExecutor(max_workers=max(9, jobs * 4))
        self.config = self._json("config.json")
        try:
            index = self._json("model.safetensors.index.json")
            self.weight_map = index["weight_map"]
        except urllib.error.HTTPError:
            names = self._probe_single_shard()
            self.weight_map = {k: "model.safetensors" for k in names}
        self.shards = sorted(set(self.weight_map.values()))
        self.tc = self.config.get("text_config", self.config)
        self.num_layers = int(self.tc["num_hidden_layers"])
        self.num_experts = int(self.tc.get("num_experts") or self.tc.get("n_routed_experts"))
        # tensor-name prefix in front of "layers." ("language_model.model." for
        # multimodal checkpoints, "model." for text-only ones)
        probe = next(k for k in self.weight_map if ".layers.0.mlp.switch_mlp.gate_proj.weight" in k
                     or k.endswith("layers.0.mlp.switch_mlp.gate_proj.weight"))
        self.prefix = probe.split("layers.0.")[0]

    # ---- http ---------------------------------------------------------------

    def _req(self, url, rng=None):
        h = {"User-Agent": UA}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        if rng is not None:
            h["Range"] = f"bytes={rng[0]}-{rng[1]}"
        return urllib.request.Request(url, headers=h)

    def _json(self, name):
        with urllib.request.urlopen(self._req(self.base + name), timeout=60) as r:
            return json.loads(r.read())

    def _probe_single_shard(self):
        hdr = self._header("model.safetensors")
        return [k for k in hdr if k != "__metadata__"]

    def _shard_url(self, shard):
        """The CDN URL a shard redirects to, cached until it expires. Saves a
        huggingface.co round trip per range request."""
        with self._lock:
            url, exp = self._cdn.get(shard, (None, 0))
        if url and time.time() < exp - 60:
            return url
        req = self._req(self.base + shard, rng=(0, 0))
        with urllib.request.urlopen(req, timeout=60) as r:
            final = r.geturl()
        exp = time.time() + 600
        if "Expires=" in final:
            try:
                exp = int(final.split("Expires=")[1].split("&")[0])
            except ValueError:
                pass
        with self._lock:
            self._cdn[shard] = (final, exp)
        return final

    def read_range(self, shard, start, end, tries=5):
        """Inclusive byte range of a shard file, with retries and CDN re-resolve."""
        last = None
        for attempt in range(tries):
            try:
                url = self._shard_url(shard)
                with urllib.request.urlopen(self._req(url, rng=(start, end)), timeout=120) as r:
                    if r.status not in (200, 206):
                        raise IOError(f"HTTP {r.status}")
                    data = r.read()
                if len(data) != end - start + 1:
                    raise IOError(f"short read {len(data)} != {end - start + 1}")
                return data
            except Exception as e:  # noqa: BLE001 — network, retry
                last = e
                with self._lock:
                    self._cdn.pop(shard, None)
                time.sleep(min(30, 1.5 ** attempt))
        raise IOError(f"range read failed for {shard} [{start}-{end}]: {last}")

    # ---- safetensors layout --------------------------------------------------

    def _header(self, shard):
        with self._lock:
            if shard in self._headers:
                return self._headers[shard]
        n = struct.unpack("<Q", self.read_range(shard, 0, 7))[0]
        hdr = json.loads(self.read_range(shard, 8, 8 + n - 1))
        with self._lock:
            self._headers[shard] = hdr
            self._data_start[shard] = 8 + n
        return hdr

    def locate(self, name):
        """→ (shard, dtype, shape, abs_start, abs_end_exclusive)"""
        shard = self.weight_map[name]
        hdr = self._header(shard)
        info = hdr[name]
        a, b = info["data_offsets"]
        ds = self._data_start[shard]
        return shard, info["dtype"], info["shape"], ds + a, ds + b

    def download_tensor(self, name):
        shard, dtype, shape, s, e = self.locate(name)
        return dtype, shape, self.read_range(shard, s, e - 1)

    # ---- experts ------------------------------------------------------------

    def expert_layout(self, layer):
        """Per-layer layout identical to what download._write_layer produces,
        plus the source locations needed to fetch single experts."""
        tensors, sources, offset = {}, {}, 0
        for t in TENSOR_ORDER:
            name = f"{self.prefix}layers.{layer}.mlp.{t}"
            shard, dtype, shape, s, e = self.locate(name)
            assert int(shape[0]) == self.num_experts, (name, shape)
            per_shape = [int(x) for x in shape[1:]]
            per_bytes = _nbytes(per_shape, dtype)
            tensors[t] = {"inner_offset": offset, "nbytes": per_bytes,
                          "shape_per_expert": per_shape, "dtype": DTYPE_MLX[dtype]}
            sources[t] = (shard, s, per_bytes)
            offset += per_bytes
        block = ((offset + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
        return {"expert_block_size": block, "data_start": PAGE_SIZE,
                "tensors": tensors, "_sources": sources, "_payload": offset}

    def fetch_expert_block(self, layer, eid, layout=None):
        """The exact bytes of one block: nine tensor slices + zero pad."""
        layout = layout or self.expert_layout(layer)

        def slice_(t):
            shard, s, per = layout["_sources"][t]
            a = s + eid * per
            return self.read_range(shard, a, a + per - 1)
        out = bytearray()
        for piece in self._slices.map(slice_, TENSOR_ORDER):
            out.extend(piece)
        out.extend(b"\x00" * (layout["expert_block_size"] - len(out)))
        return bytes(out)

    def public_layout(self, layer=0):
        lay = self.expert_layout(layer)
        return {k: v for k, v in lay.items() if not k.startswith("_")}

    def hash_all_blocks(self, layers=None, log=print):
        """Stream every expert block through sha256 without storing any.
        This is the publisher's job: ~75 GB for Flash-Next, no disk needed."""
        layers = list(range(self.num_layers)) if layers is None else layers
        blocks = {}
        t0, done, nbytes = time.time(), 0, 0
        for L in layers:
            lay = self.expert_layout(L)

            def one(eid, L=L, lay=lay):
                return eid, hashlib.sha256(self.fetch_expert_block(L, eid, lay)).hexdigest()
            with ThreadPoolExecutor(max_workers=self.jobs) as pool:
                for eid, h in pool.map(one, range(self.num_experts)):
                    blocks[f"{L}:{eid}"] = h
                    done += 1
                    nbytes += lay["expert_block_size"]
            rate = nbytes / 1e6 / max(0.1, time.time() - t0)
            log(f"  hashed layer {L + 1}/{self.num_layers}  {done} blocks  {rate:.0f} MB/s")
        return blocks

    # ---- the resident trunk --------------------------------------------------

    def is_expert(self, name):
        return ".mlp.switch_mlp." in name

    def is_ngram(self, name):
        return "ngram_embedding.shard_" in name

    def is_vision(self, name):
        return name.startswith(("vision_tower", "visual", "model.visual"))

    def pinned_names(self, include_vision=False):
        return [k for k in self.weight_map
                if not self.is_expert(k) and not self.is_ngram(k)
                and (include_vision or not self.is_vision(k))]

    def download_pinned(self, out, include_vision=False, log=print):
        """pinned.safetensors: everything that is neither an expert nor an
        n-gram row. ~4 GB for Flash-Next. Resumable per tensor via a parts dir."""
        import numpy as np
        import mlx.core as mx
        NP = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "U32": np.uint32,
              "I32": np.int32, "I64": np.int64, "U8": np.uint8, "I8": np.int8}
        MXV = {"BF16": mx.bfloat16}
        dst = os.path.join(out, "pinned.safetensors")
        if os.path.exists(dst):
            log(f"  pinned.safetensors already present")
            return dst
        names = self.pinned_names(include_vision)
        log(f"  pinned: {len(names)} tensors")
        arrays = {}
        t0, nbytes = time.time(), 0

        def one(name):
            dtype, shape, data = self.download_tensor(name)
            return name, dtype, shape, data
        with ThreadPoolExecutor(max_workers=self.jobs) as pool:
            for i, (name, dtype, shape, data) in enumerate(pool.map(one, names)):
                arr = np.frombuffer(data, dtype=NP[dtype]).reshape([int(x) for x in shape])
                a = mx.array(arr)
                if dtype in MXV:
                    a = a.view(MXV[dtype])
                arrays[name] = a
                nbytes += len(data)
                if i % 200 == 0:
                    log(f"    {i}/{len(names)}  {nbytes / 1e9:.2f} GB  "
                        f"{nbytes / 1e6 / max(0.1, time.time() - t0):.0f} MB/s")
        mx.save_safetensors(dst, arrays)
        log(f"  saved pinned.safetensors: {nbytes / 1e9:.2f} GB ({len(arrays)} keys)")
        return dst

    # ---- n-gram tables -------------------------------------------------------

    def ngram_names(self):
        return sorted(k for k in self.weight_map if self.is_ngram(k))

    def download_ngram(self, out, log=print):
        """Row-addressable raw files: bin/ngram/<shard>.<weight|scales|biases>
        plus layout.json. Each is the tensor's bytes verbatim, so row r of a
        [rows, w] tensor is at r * w * itemsize — one pread per lookup.
        Resumable: a file whose size already matches is skipped."""
        names = self.ngram_names()
        if not names:
            return None
        d = os.path.join(out, "bin", "ngram")
        os.makedirs(d, exist_ok=True)
        layout = {"tensors": {}, "prefix": self.prefix}
        jobs = []
        for name in names:
            shard, dtype, shape, s, e = self.locate(name)
            local = name.split("ngram_embedding.")[1]  # shard_7.weight
            layout["tensors"][local] = {"shape": [int(x) for x in shape], "dtype": dtype,
                                        "row_bytes": _nbytes(shape[1:], dtype),
                                        "file": local, "source": name}
            path = os.path.join(d, local)
            if os.path.exists(path) and os.path.getsize(path) == e - s:
                continue
            jobs.append((name, path, shard, s, e))
        with open(os.path.join(d, "layout.json"), "w") as f:
            json.dump(layout, f)
        total = sum(e - s for _, _, _, s, e in jobs)
        log(f"  n-gram tables: {len(names)} tensors, {len(jobs)} to fetch "
            f"({total / 1e9:.1f} GB)")
        t0, nbytes = time.time(), 0
        CHUNK = 64 * 1024 * 1024

        def one(job):
            name, path, shard, s, e = job
            tmp = path + ".part"
            with open(tmp, "wb") as f:
                for a in range(s, e, CHUNK):
                    f.write(self.read_range(shard, a, min(e, a + CHUNK) - 1))
            os.replace(tmp, path)
            return e - s
        with ThreadPoolExecutor(max_workers=max(1, self.jobs // 2)) as pool:
            for i, n in enumerate(pool.map(one, jobs)):
                nbytes += n
                if i % 8 == 0 or i == len(jobs) - 1:
                    log(f"    {i + 1}/{len(jobs)}  {nbytes / 1e9:.1f} GB  "
                        f"{nbytes / 1e6 / max(0.1, time.time() - t0):.0f} MB/s")
        return d

    # ---- config ---------------------------------------------------------------

    def write_config(self, out):
        """config.json for a streaming dir: the full text config, plus the
        quantization recipe and where the streamed parts live."""
        cfg = dict(self.tc)
        cfg["model_type"] = self.config.get("model_type", cfg.get("model_type"))
        cfg["text_model_type"] = self.tc.get("model_type")
        cfg["num_experts"] = self.num_experts
        cfg["tie_word_embeddings"] = self.config.get("tie_word_embeddings",
                                                     self.tc.get("tie_word_embeddings", False))
        cfg["quantization"] = self.config.get("quantization", {"bits": 4, "group_size": 64})
        cfg["streaming"] = {"pinned_file": "pinned.safetensors", "expert_dir": "bin",
                            "ngram_dir": "bin/ngram" if self.ngram_names() else None,
                            "source_repo": self.repo}
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "config.json"), "w") as f:
            json.dump(cfg, f, indent=1)
        return cfg
