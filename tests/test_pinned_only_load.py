"""load_engine(pinned_only=True): an Expert Network driver must load from a
dir holding only pinned.safetensors + config.json — no bin/ — and get an
engine whose reader slot is free for DistributedExpertReader (issue #1).

Uses a tiny synthetic OLMoE (2 layers, 4 experts, 64-dim) so the real
engine load path runs: config parse, model build, quantize predicate,
load_weights. The tokenizer is stubbed; no HF download, no model on disk."""

import json
import os

import mlx.core as mx
import pytest
from mlx.utils import tree_flatten

from mlx_expert_sniper.generate import load_engine

CONFIG = {
    "model_type": "olmoe",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-5,
    "vocab_size": 32,
    "num_experts": 4,
    "num_experts_per_tok": 2,
    "tie_word_embeddings": False,
}


@pytest.fixture
def pinned_only_dir(tmp_path, monkeypatch):
    """config.json + pinned.safetensors and nothing else — the driver's
    footprint. The pinned tensors are the model's own non-expert params so
    load_weights sees the exact names/shapes preprocess would write."""
    from mlx_lm.models.olmoe import Model, ModelArgs

    with open(tmp_path / "config.json", "w") as f:
        json.dump(CONFIG, f)
    model = Model(ModelArgs(**CONFIG))
    pinned = {k: v for k, v in tree_flatten(model.parameters())
              if "switch_mlp" not in k}
    mx.save_safetensors(str(tmp_path / "pinned.safetensors"), pinned)

    # The engine loads a tokenizer from the model dir; there is none here.
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained",
                        lambda *a, **k: "stub-tokenizer")
    return str(tmp_path)


def test_pinned_only_load_needs_no_bin_dir(pinned_only_dir):
    assert not os.path.exists(os.path.join(pinned_only_dir, "bin"))

    engine, bias, model_type = load_engine(pinned_only_dir, pinned_only=True)

    assert model_type == "olmoe"
    assert bias == 0.0
    assert engine.reader is None            # slot is free for the network
    assert engine.num_layers == CONFIG["num_hidden_layers"]
    assert engine.num_experts == CONFIG["num_experts"]
    assert engine.tokenizer == "stub-tokenizer"
    # Attention weights really landed (not just a constructed model)
    loaded = dict(tree_flatten(engine.model.parameters()))
    assert "model.embed_tokens.weight" in loaded
    assert not os.path.exists(os.path.join(pinned_only_dir, "bin"))


def test_pinned_only_engine_accepts_distributed_reader(pinned_only_dir):
    """The driver's sequence: load pinned, install the network reader, reset
    the KV cache. No node is contacted — the reader only opens sessions."""
    from expert_network.reader import DistributedExpertReader

    engine, _, _ = load_engine(pinned_only_dir, pinned_only=True)
    engine.reader = DistributedExpertReader(["http://127.0.0.1:1"])
    try:
        engine.reset_cache()                # calls reader.reset_prefetch()
        assert engine.reader.lru is None    # bias masks stay off
    finally:
        engine.reader.close()


def test_default_load_still_requires_bin_dir(pinned_only_dir):
    """Negative control: the single-machine path is unchanged and refuses a
    dir with no bin/ (cache sizing reads a layer header)."""
    with pytest.raises(FileNotFoundError):
        load_engine(pinned_only_dir)
