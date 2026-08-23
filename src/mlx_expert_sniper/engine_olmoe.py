#!/usr/bin/env python3
"""MoE Sniper — OLMoE-1B-7B via SSD streaming. 64 experts x 16 layers;
small enough (3.6 GB) to demo the full pipeline and the Expert Network on
one machine."""
import json, os, gc
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .expert_io import MoEExpertReader
from .coactivation import CoActivationTracker

MODEL_DIR = None  # set by generate.load_engine / calibrate._build_engine
BITS = 4
GROUP_SIZE = 64


class MoESniperEngineOlmoe:
    def __init__(self, cache_size=512, enable_prediction=True):
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self.num_layers = 16
        self.coact = None
        self._cache_size = cache_size
        self._enable_prediction = enable_prediction

    def load(self):
        if MODEL_DIR is None:
            raise RuntimeError(
                "engine_olmoe.MODEL_DIR is not set — load via "
                "generate.load_engine(model_dir), which sets it")
        with open(f"{MODEL_DIR}/config.json") as f:
            config = json.load(f)
        self.num_layers = config["num_hidden_layers"]
        self.num_experts = config["num_experts"]

        from mlx_lm.models.olmoe import Model, ModelArgs
        args = ModelArgs(
            model_type=config.get("model_type", "olmoe"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=self.num_layers,
            intermediate_size=config.get("intermediate_size", 1024),
            num_attention_heads=config["num_attention_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            num_experts=self.num_experts,
            num_experts_per_tok=config["num_experts_per_tok"],
            norm_topk_prob=config.get("norm_topk_prob", False),
            head_dim=config.get("head_dim"),
            max_position_embeddings=config.get("max_position_embeddings", 4096),
            num_key_value_heads=config["num_key_value_heads"],
            attention_bias=config.get("attention_bias", False),
            mlp_bias=config.get("mlp_bias", False),
            rope_theta=config.get("rope_theta", 10000.0),
            rope_scaling=config.get("rope_scaling"),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
        )

        self.model = Model(args)
        from mlx_lm.models.switch_layers import SwitchLinear

        mx.set_memory_limit(6 * 1024**3)
        mx.set_cache_limit(256 * 1024**2)

        pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")

        # Quantize per the CHECKPOINT's recipe (see engine_30b for why).
        qcfg = dict(config.get("quantization") or {})
        q_group = qcfg.pop("group_size", GROUP_SIZE)
        q_bits = qcfg.pop("bits", BITS)
        pinned_keys = set(pinned.keys())
        def should_quantize(path, module):
            if path in qcfg:
                return qcfg[path]
            if isinstance(module, SwitchLinear):
                return True
            if isinstance(module, (nn.Linear, nn.Embedding)):
                return f"{path}.scales" in pinned_keys
            return False
        nn.quantize(self.model, group_size=q_group, bits=q_bits,
                    class_predicate=should_quantize)

        self.model.load_weights(list(pinned.items()), strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters())
                  if "switch_mlp" not in name]
        mx.eval(*params)
        del pinned; gc.collect(); mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9
        streaming = config.get("streaming", {"expert_dir": "bin"})
        expert_dir = os.path.join(MODEL_DIR, streaming["expert_dir"])
        self.reader = MoEExpertReader(expert_dir, self.num_layers,
                                      num_workers=8, cache_size=self._cache_size)
        self.coact = CoActivationTracker(self.num_layers, warmup_tokens=3)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        return pinned_gb

    def reset_cache(self):
        from mlx_lm.models.cache import KVCache
        self.cache = [KVCache() for _ in range(self.num_layers)]
        if self.reader:
            self.reader.reset_prefetch()
