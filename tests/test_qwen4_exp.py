"""Numerical parity of models/qwen4_exp.py against transformers' Qwen4ExpForCausalLM,
plus an engine-level check that the streaming forward equals the model.

A tiny random config with every feature enabled (hyper-connections, PLE +
hashed n-grams, QSA indexer with a budget small enough that the sparse path
runs, GatedDeltaNet with sigmoid output gate, top-2 MoE + shared expert) is
instantiated in float32 on both sides, the HF state dict is converted through
``Model.sanitize`` and logits are compared for a prefill and for a cached
two-step decode.

The HF parity tests skip when torch / transformers (main, with qwen4_exp) are
not importable, so the default CI environment stays green; the engine test
only needs mlx.
"""
import json
import os

import numpy as np
import pytest

pytest.importorskip("mlx.core")
import mlx.core as mx
import mlx.nn as nn

from mlx_expert_sniper.models.qwen4_exp import Model, ModelArgs, QSAKVCache

ATOL = 1e-3
EOS = 1

TINY = dict(
    vocab_size=64,
    hidden_size=64,
    num_hidden_layers=2,
    layer_types=["linear_attention", "full_attention"],
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    hidden_act="silu",
    output_gate_type="sigmoid",
    linear_conv_kernel_dim=4,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    num_experts=8,
    num_experts_per_tok=2,
    moe_intermediate_size=32,
    shared_expert_intermediate_size=32,
    norm_topk_prob=True,
    hc_count=4,
    hc_lowrank=8,
    ple_layer_ids=[1],
    ple_embed_dim=128,  # 4 n-gram heads x 32 (quantizable head dim)
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=97,
    make_ngram_vocab_size_divisible_by=8,
    seed=1234,
    split_ngram_parts=4,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=16,
    indexer_budget=4,
    indexer_compress_ratio=2,
    max_position_embeddings=512,
    tie_word_embeddings=False,
    eos_token_id=EOS,
    bos_token_id=0,
    pad_token_id=None,
    rope_parameters={"rope_theta": 10000.0, "partial_rotary_factor": 0.5,
                     "rope_type": "default", "mrope_section": [3, 3, 2]},
    initializer_range=0.2,
)


def _hf_model():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    try:
        from transformers import Qwen4ExpForCausalLM, Qwen4ExpTextConfig
    except ImportError:
        pytest.skip("transformers has no qwen4_exp")
    torch.manual_seed(0)
    cfg = Qwen4ExpTextConfig(**TINY)
    cfg._attn_implementation = "eager"
    model = Qwen4ExpForCausalLM(cfg).float().eval()
    # Give the zero-initialised parameters real values so every path is exercised.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.ndim >= 1 and torch.count_nonzero(p) == 0:
                p.normal_(0.0, 0.3)
    return model


def _mlx_model(hf):
    args = ModelArgs(**{k: v for k, v in TINY.items() if k in ModelArgs.__dataclass_fields__})
    model = Model(args)
    sd = {k: mx.array(v.detach().cpu().numpy()) for k, v in hf.state_dict().items()}
    weights = model.sanitize(sd, hf_norms=True)
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model


@pytest.fixture(scope="module")
def models():
    hf = _hf_model()
    return hf, _mlx_model(hf)


@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def _tokens():
    rng = np.random.RandomState(1)
    ids = rng.randint(2, TINY["vocab_size"], size=(1, 12))
    ids[0, 5] = EOS  # exercise the n-gram segment reset
    return ids


def test_prefill_logits_match(models, torch):
    hf, mlx_model = models
    ids = _tokens()
    with torch.no_grad():
        ref = hf(torch.tensor(ids), use_cache=False).logits.numpy()
    out = np.array(mlx_model(mx.array(ids)))
    diff = np.abs(out - ref).max()
    print(f"prefill max|diff|={diff:.3e}  ref|max|={np.abs(ref).max():.3f}")
    assert diff < ATOL
    # 12 tokens / ratio 2 = 6 complete blocks > block_topk 2: the sparse path ran.
    assert 12 // TINY["indexer_compress_ratio"] > TINY["indexer_budget"] // TINY["indexer_compress_ratio"]


def test_indexer_mask_is_sparse_and_matters(models, torch):
    """The QSA mask must drop tokens for late queries, and dense attention
    must NOT reproduce HF (otherwise parity would say nothing about QSA)."""
    hf, mlx_model = models
    ids = _tokens()
    attn = mlx_model.model.layers[1].self_attn
    tm = mlx_model.model
    h = tm.embed(mx.array(ids))
    h = tm.layers[0](h, input_ids=mx.array(ids), mask=None)
    x, _, _ = tm.layers[1].attn_hyper_connection(h)
    m = np.array(attn.indexer(x, 0, None))[0, 0]  # [L, T]
    L = ids.shape[1]
    causal = np.tril(np.ones((L, L), dtype=bool))
    assert not m[~causal].any()                    # never attends to the future
    assert m[causal].sum() < causal.sum()          # some visible tokens dropped
    # every query keeps <= budget tokens from complete blocks + its tail
    r, budget = TINY["indexer_compress_ratio"], TINY["indexer_budget"]
    for q in range(L):
        nb = (q + 1) // r
        kept_blocks = m[q, : nb * r].sum()
        assert kept_blocks <= budget
        assert m[q, nb * r: q + 1].all()
    # dense attention diverges from HF
    with torch.no_grad():
        ref = hf(torch.tensor(ids), use_cache=False).logits.numpy()
    attn.indexer, saved = None, attn.indexer
    try:
        dense = np.array(mlx_model(mx.array(ids)))
    finally:
        attn.indexer = saved
    assert np.abs(dense - ref).max() > ATOL


def test_decode_with_cache_matches(models, torch):
    hf, mlx_model = models
    ids = _tokens()
    prompt, steps = ids[:, :10], ids[:, 10:12]
    from transformers import DynamicCache
    with torch.no_grad():
        pkv = DynamicCache(config=hf.config)
        ref_logits = [hf(torch.tensor(prompt), past_key_values=pkv, use_cache=True).logits[:, -1].numpy()]
        for t in range(steps.shape[1]):
            ref_logits.append(hf(torch.tensor(steps[:, t:t + 1]), past_key_values=pkv,
                                 use_cache=True).logits[:, -1].numpy())
        full = hf(torch.tensor(ids), use_cache=False).logits.numpy()

    cache = mlx_model.make_cache()
    assert isinstance(cache[1], QSAKVCache)
    got = [np.array(mlx_model(mx.array(prompt), cache=cache)[:, -1])]
    for t in range(steps.shape[1]):
        got.append(np.array(mlx_model(mx.array(steps[:, t:t + 1]), cache=cache)[:, -1]))

    for i, (g, r) in enumerate(zip(got, ref_logits)):
        d = np.abs(g - r).max()
        print(f"decode step {i} max|diff|={d:.3e}")
        assert d < ATOL
    # cached decode == uncached full-sequence logits at the same positions
    d = np.abs(np.stack(got, 1) - full[:, 9:12]).max()
    print(f"decode vs full prefill max|diff|={d:.3e}")
    assert d < ATOL


def test_ngram_ids_match_hf(models, torch):
    hf, mlx_model = models
    ids = _tokens()
    ple = hf.model.layers[0].ple.ple_embedding
    with torch.no_grad():
        emb_ref = ple(torch.tensor(ids), None).numpy()
    emb = np.array(mlx_model.model.layers[0].ple.ple_embedding(mx.array(ids), None))
    assert np.abs(emb - emb_ref).max() < 1e-6
    assert list(np.array(mlx_model.model.layers[0].ple.ple_embedding.layer_multipliers)) == \
        ple.layer_multipliers.tolist()


def test_loads_checkpoint_with_vision_and_mtp_keys(models):
    hf, mlx_model = models
    sd = {k: mx.array(v.detach().cpu().numpy()) for k, v in hf.state_dict().items()}
    sd = {"language_model." + k: v for k, v in sd.items()}
    sd["vision_tower.blocks.0.attn.qkv.weight"] = mx.zeros((4, 4))
    sd["language_model.mtp.layers.0.self_attn.q_proj.weight"] = mx.zeros((4, 4))
    weights = mlx_model.sanitize(sd)  # hf_norms auto-detected from the conv1d layout
    assert not any(k.startswith(("vision_tower", "mtp")) for k in weights)
    ref = mlx_model.sanitize(
        {"language_model." + k: mx.array(v.detach().cpu().numpy()) for k, v in hf.state_dict().items()},
        hf_norms=True)
    for k in weights:
        assert np.array_equal(np.array(weights[k]), np.array(ref[k])), k
    mlx_model.load_weights(list(weights.items()), strict=True)


# --------------------------------------------------------------------------- #
# Engine: streaming forward == model forward on a tiny sniper-format model dir
# --------------------------------------------------------------------------- #

PAGE_SIZE = 16384


def _write_streaming_dir(model, args, out_dir):
    """pinned.safetensors (language_model. prefix, vision key) + bin/layer_XX.bin."""
    from mlx.utils import tree_flatten
    params = dict(tree_flatten(model.parameters()))
    pinned = {"language_model." + k: v for k, v in params.items() if "switch_mlp" not in k}
    pinned["vision_tower.patch_embed.proj.weight"] = mx.zeros((2, 2))
    mx.save_safetensors(os.path.join(out_dir, "pinned.safetensors"), pinned)

    bin_dir = os.path.join(out_dir, "bin")
    os.makedirs(bin_dir, exist_ok=True)
    names = [f"switch_mlp.{p}.{t}" for p in ("gate_proj", "up_proj", "down_proj")
             for t in ("weight", "scales", "biases")]
    for li, layer in enumerate(model.layers):
        tensors, off = {}, 0
        per_expert = []
        for e in range(args.num_experts):
            blob = b""
            for n in names:
                arr = params[f"model.layers.{li}.mlp.{n}"][e]
                b = np.array(arr).tobytes()
                if e == 0:
                    tensors[n] = {"inner_offset": off, "nbytes": len(b),
                                  "shape_per_expert": list(arr.shape), "dtype": str(arr.dtype).replace("mlx.core.", "")}
                    off += len(b)
                blob += b
            per_expert.append(blob)
        header = json.dumps({"layout": {"expert_block_size": off, "data_start": PAGE_SIZE,
                                        "tensors": tensors}}).encode()
        assert len(header) < PAGE_SIZE
        with open(os.path.join(bin_dir, f"layer_{li:02d}.bin"), "wb") as f:
            f.write(header.ljust(PAGE_SIZE, b"\x00"))
            for blob in per_expert:
                f.write(blob)

    config = {k: v for k, v in TINY.items() if k in ModelArgs.__dataclass_fields__}
    config.update({"model_type": "qwen4_exp",
                   "quantization": {"bits": 4, "group_size": 32, "mode": "affine"},
                   "streaming": {"pinned_file": "pinned.safetensors", "expert_dir": "bin"},
                   "ngram_in_memory": True, "norm_weights_hf": False})
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f)


def test_engine_forward_matches_model(tmp_path, monkeypatch):
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_expert_sniper import engine_qwen4exp
    from mlx_expert_sniper.calibrate import _build_engine

    mx.random.seed(0)
    args = ModelArgs(**{k: v for k, v in TINY.items() if k in ModelArgs.__dataclass_fields__})
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4, class_predicate=lambda p, m: isinstance(m, SwitchLinear))
    mx.eval(model.parameters())
    model.eval()
    _write_streaming_dir(model, args, str(tmp_path))

    ids = _tokens()
    cache = model.make_cache()
    ref = [np.array(model(mx.array(ids[:, :10]), cache=cache))]
    ref.append(np.array(model(mx.array(ids[:, 10:11]), cache=cache)))

    class _Tok:  # engine.load wants a tokenizer; none is needed here
        eos_token_id = EOS
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _Tok()))
    engine = _build_engine(str(tmp_path), cache_size=64)
    assert isinstance(engine, engine_qwen4exp.MoESniperEngineQwen4Exp)
    assert engine.own_forward and engine.reader is not None
    engine.reset_cache()
    routed = []
    got = [np.array(engine.forward(mx.array(ids[:, :10]), on_route=lambda i, a, g: routed.append((i, len(a)))))]
    got.append(np.array(engine.forward(mx.array(ids[:, 10:11]))))
    assert len(routed) == args.num_hidden_layers and all(n == 10 * args.num_experts_per_tok for _, n in routed)
    for g, r in zip(got, ref):
        d = np.abs(g - r).max()
        print(f"engine vs model max|diff|={d:.3e}")
        assert d < 1e-4


# --------------------------------------------------------------------------- #
# Streamed n-gram sources: Disk / HF rows == in-memory shards
# --------------------------------------------------------------------------- #

def _quantized_shards(num_shards=4, rows=24, dim=64, gs=32, bits=4):
    """Random in-memory quantized shard module + raw row files geometry."""
    from mlx_expert_sniper.models.qwen4_exp import NGramShards, InMemoryNGramSource
    shards = NGramShards(num_shards, rows, dim)
    nn.quantize(shards, group_size=gs, bits=bits)
    mx.eval(shards.parameters())
    return shards, InMemoryNGramSource(shards, rows)


def _write_ngram_dir(shards, d, num_shards, rows, dim, gs, bits):
    os.makedirs(d, exist_ok=True)
    layout = {"tensors": {}}
    for i in range(num_shards):
        m = getattr(shards, f"shard_{i}")
        for kind, dt in (("weight", "U32"), ("scales", "BF16"), ("biases", "BF16")):
            t = m[kind]
            b = np.array(t.view(mx.uint16) if t.dtype == mx.bfloat16 else t).tobytes()
            local = f"shard_{i}.{kind}"
            with open(os.path.join(d, local), "wb") as f:
                f.write(b)
            layout["tensors"][local] = {"shape": list(t.shape), "dtype": dt,
                                        "row_bytes": len(b) // t.shape[0], "file": local,
                                        "source": f"language_model.model.layers.0.ple.ple_embedding.ngram_embedding.{local}"}
    with open(os.path.join(d, "layout.json"), "w") as f:
        json.dump(layout, f)
    return layout


def test_disk_ngram_source_matches_memory(tmp_path):
    from mlx_expert_sniper.ngram_source import DiskNGramSource
    mx.random.seed(1)
    ns, rows, dim, gs, bits = 4, 24, 64, 32, 4
    shards, mem = _quantized_shards(ns, rows, dim, gs, bits)
    # scales/biases are fp32 in a float model; cast to bf16 like the checkpoint
    for i in range(ns):
        m = getattr(shards, f"shard_{i}")
        m.scales, m.biases = m.scales.astype(mx.bfloat16), m.biases.astype(mx.bfloat16)
    _write_ngram_dir(shards, str(tmp_path / "ngram"), ns, rows, dim, gs, bits)
    disk = DiskNGramSource(str(tmp_path / "ngram"), gs, bits)
    assert disk.rows_per_shard == rows and disk.num_shards == ns
    ids = mx.array(np.random.RandomState(0).randint(0, ns * rows, size=(2, 5, 3)))
    a, b = np.array(mem.lookup(ids).astype(mx.float32)), np.array(disk.lookup(ids).astype(mx.float32))
    assert a.shape == (2, 5, 3, dim) and np.abs(a - b).max() < 1e-6
    # second lookup is served from the LRU
    n = disk.reads
    mx.eval(disk.lookup(ids))
    assert disk.reads == n and disk.lru.hits > 0
    disk.close()


def test_hf_ngram_source_matches_memory(tmp_path, monkeypatch):
    from mlx_expert_sniper.ngram_source import HFNGramSource
    mx.random.seed(2)
    ns, rows, dim, gs, bits = 4, 24, 64, 32, 4
    shards, mem = _quantized_shards(ns, rows, dim, gs, bits)
    for i in range(ns):
        m = getattr(shards, f"shard_{i}")
        m.scales, m.biases = m.scales.astype(mx.bfloat16), m.biases.astype(mx.bfloat16)
    layout = _write_ngram_dir(shards, str(tmp_path / "ngram"), ns, rows, dim, gs, bits)
    # a fake checkpoint: every tensor lives at a distinct offset of one "shard file"
    blob, loc = bytearray(), {}
    for local, m in layout["tensors"].items():
        data = open(tmp_path / "ngram" / local, "rb").read()
        loc[m["source"]] = ("model-00001.safetensors", m["dtype"], m["shape"], len(blob), len(blob) + len(data))
        blob += data

    class FakeHF:
        prefix = "language_model.model."
        repo = "fake/repo"
        def __init__(self):
            self.requests = []
        def ngram_names(self):
            return sorted(loc)
        def locate(self, name):
            return loc[name]
        def read_range(self, shard, s, e):
            self.requests.append((s, e))
            return bytes(blob[s:e + 1])
    hf = FakeHF()
    src = HFNGramSource(hf, layer_idx=0, group_size=gs, bits=bits, coalesce_bytes=64)
    ids = mx.array(np.random.RandomState(1).randint(0, ns * rows, size=(1, 7, 3)))
    a, b = np.array(mem.lookup(ids).astype(mx.float32)), np.array(src.lookup(ids).astype(mx.float32))
    assert np.abs(a - b).max() < 1e-6
    # adjacent rows were coalesced: fewer requests than rows*kinds
    n_rows = len(np.unique(np.array(ids)))
    assert 0 < len(hf.requests) < 3 * n_rows or n_rows <= 2
    # consecutive ids -> one request per kind
    hf.requests.clear()
    src2 = HFNGramSource(hf, layer_idx=0, group_size=gs, bits=bits, coalesce_bytes=64)
    out = np.array(src2.lookup(mx.array([[[3, 4, 5, 6]]])).astype(mx.float32))
    assert len(hf.requests) == 3
    assert np.abs(out - np.array(mem.lookup(mx.array([[[3, 4, 5, 6]]])).astype(mx.float32))).max() < 1e-6


def test_engine_with_disk_ngram_source(tmp_path, monkeypatch):
    """Same streaming dir as the engine test, but the n-gram shards are
    removed from pinned and served from bin/ngram row files."""
    from mlx_lm.models.switch_layers import SwitchLinear
    from mlx_expert_sniper.calibrate import _build_engine
    mx.random.seed(3)
    args = ModelArgs(**{k: v for k, v in TINY.items() if k in ModelArgs.__dataclass_fields__})
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4,
                class_predicate=lambda p, m: isinstance(m, SwitchLinear) or ".ngram_embedding.shard_" in p)
    mx.eval(model.parameters())
    model.eval()
    _write_streaming_dir(model, args, str(tmp_path))
    # strip shards from pinned, write them as row files
    pinned = mx.load(str(tmp_path / "pinned.safetensors"))
    pinned = {k: v for k, v in pinned.items() if "ngram_embedding.shard_" not in k}
    mx.save_safetensors(str(tmp_path / "pinned2.safetensors"), pinned)
    shards = model.model.layers[0].ple.ple_embedding.ngram_embedding
    geo = model.model.layers[0].ple.ple_embedding
    for i in range(geo.num_shards):
        m = getattr(shards, f"shard_{i}")
        m.scales, m.biases = m.scales.astype(mx.bfloat16), m.biases.astype(mx.bfloat16)
    _write_ngram_dir(shards, str(tmp_path / "bin" / "ngram"), geo.num_shards, geo.rows_per_shard, geo.head_dim, 32, 4)
    cfg = json.load(open(tmp_path / "config.json"))
    cfg["streaming"]["ngram_source"] = "disk"
    cfg["streaming"]["ngram_dir"] = "bin/ngram"
    cfg["streaming"]["pinned_file"] = "pinned2.safetensors"
    json.dump(cfg, open(tmp_path / "config.json", "w"))

    ids = _tokens()
    ref = np.array(model(mx.array(ids[:, :6])))

    class _Tok:
        eos_token_id = EOS
    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(lambda *a, **k: _Tok()))
    engine = _build_engine(str(tmp_path), cache_size=64)
    assert engine.ngram_source_kind == "disk" and 0 in engine.ngram_sources
    assert not hasattr(engine.model.model.layers[0].ple.ple_embedding, "ngram_embedding")
    engine.reset_cache()
    got = np.array(engine.forward(mx.array(ids[:, :6])))
    d = np.abs(got - ref).max()
    print(f"engine(disk ngram) vs model max|diff|={d:.3e}")
    assert d < 1e-3


# --------------------------------------------------------------------------- #
# HFExpertReader: local sparse bin first, HF for holes
# --------------------------------------------------------------------------- #

def test_hf_expert_reader_local_then_hf(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")  # expert_network.node imports it
    from mlx_lm.models.switch_layers import SwitchLinear
    from expert_network.hf_reader import HFExpertReader
    mx.random.seed(4)
    args = ModelArgs(**{k: v for k, v in TINY.items() if k in ModelArgs.__dataclass_fields__})
    model = Model(args)
    nn.quantize(model, group_size=32, bits=4, class_predicate=lambda p, m: isinstance(m, SwitchLinear))
    mx.eval(model.parameters())
    _write_streaming_dir(model, args, str(tmp_path))
    cfg = json.load(open(tmp_path / "config.json"))
    cfg["streaming"]["source_repo"] = "fake/repo"
    json.dump(cfg, open(tmp_path / "config.json", "w"))
    # punch holes: zero out experts >= 4 in every layer and record which are real
    full_blocks = {}
    hdr = json.loads(open(tmp_path / "bin" / "layer_00.bin", "rb").read(PAGE_SIZE).rstrip(b"\x00"))["layout"]
    bs, ds = hdr["expert_block_size"], hdr["data_start"]
    for li in range(args.num_hidden_layers):
        p = tmp_path / "bin" / f"layer_{li:02d}.bin"
        with open(p, "r+b") as f:
            for e in range(args.num_experts):
                f.seek(ds + e * bs)
                full_blocks[(li, e)] = f.read(bs)
                if e >= 4:
                    f.seek(ds + e * bs)
                    f.write(b"\x00" * bs)
    json.dump({"blocks": {f"{l}:{e}": "x" for (l, e) in full_blocks if e < 4}},
              open(tmp_path / "manifest.partial.json", "w"))

    class FakeHF:
        num_layers = args.num_hidden_layers
        def __init__(self):
            self.fetched = []
        def expert_layout(self, layer):
            return dict(hdr)
        def fetch_expert_block(self, layer, eid, layout=None):
            self.fetched.append((layer, eid))
            return full_blocks[(layer, eid)]
    hf = FakeHF()
    reader = HFExpertReader(str(tmp_path), num_layers=args.num_hidden_layers, cache_size=16, hf=hf)
    reader.prefetch_experts(1, [2, 6])
    got = reader.get_experts(0, [1, 5, 7])
    assert sorted(got) == [1, 5, 7]
    assert sorted(hf.fetched) == [(0, 5), (0, 7), (1, 6)]
    ref = model.model.layers[0].mlp.switch_mlp
    for e in (1, 5, 7):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            for t in ("weight", "scales", "biases"):
                assert np.array_equal(np.array(got[e][f"switch_mlp.{proj}.{t}"]),
                                      np.array(getattr(ref, proj)[t][e]))
    got1 = reader.get_experts(1, [2, 6])
    assert sorted(hf.fetched) == [(0, 5), (0, 7), (1, 6)]  # served from prefetch, no refetch
    assert reader.lru.contains(0, 5) and reader.lru.contains(1, 2)
    assert reader.local_hits == 2 and reader.hf_reads == 3
    assert "reads=" in reader.stats()
    reader.close()
