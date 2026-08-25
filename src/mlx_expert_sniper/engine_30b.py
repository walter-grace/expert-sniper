#!/usr/bin/env python3
"""MoE Sniper — Qwen3-30B-A3B via SSD streaming on M4 Mac Mini."""
import json, sys, os, time, gc
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .expert_io import MoEExpertReader
from .coactivation import CoActivationTracker

MODEL_DIR = None  # set by generate.load_engine / calibrate._build_engine
BITS = 4
GROUP_SIZE = 64

def run_expert_ffn(x, expert_data, top_k_indices, top_k_weights):
    active_ids = sorted(expert_data.keys())
    id_to_local = {eid: i for i, eid in enumerate(active_ids)}
    inds_np = np.array(top_k_indices)
    local_np = np.vectorize(lambda v: id_to_local.get(int(v), 0))(inds_np)
    local_indices = mx.array(local_np)

    def stack_proj(proj):
        w = mx.stack([expert_data[eid][f"switch_mlp.{proj}.weight"] for eid in active_ids])
        s = mx.stack([expert_data[eid][f"switch_mlp.{proj}.scales"] for eid in active_ids])
        b = mx.stack([expert_data[eid][f"switch_mlp.{proj}.biases"] for eid in active_ids])
        return w, s, b

    gate_w, gate_s, gate_b = stack_proj("gate_proj")
    up_w, up_s, up_b = stack_proj("up_proj")
    down_w, down_s, down_b = stack_proj("down_proj")

    x_exp = mx.expand_dims(x, (-2, -3))
    gate_out = mx.gather_qmm(x_exp, gate_w, scales=gate_s, biases=gate_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=BITS)
    up_out = mx.gather_qmm(x_exp, up_w, scales=up_s, biases=up_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=BITS)
    hidden = nn.silu(gate_out) * up_out
    down_out = mx.gather_qmm(hidden, down_w, scales=down_s, biases=down_b,
        rhs_indices=local_indices, transpose=True, group_size=GROUP_SIZE, bits=BITS)
    out = down_out.squeeze(-2)
    out = (out * top_k_weights[..., None]).sum(axis=-2)
    return out


class MoESniperEngine30B:
    def __init__(self, cache_size=3000, enable_prediction=True,
                 pinned_only=False):
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self.num_layers = 48
        self.coact = None
        self._cache_size = cache_size
        self._enable_prediction = enable_prediction
        # pinned_only: load attention/routing weights but open no
        # bin/ layer files — the caller installs its own reader
        # (Expert Network driver). See generate.load_engine.
        self._pinned_only = pinned_only

    def load(self):
        if MODEL_DIR is None:
            raise RuntimeError(
                "engine_30b.MODEL_DIR is not set — load via "
                "generate.load_engine(model_dir), which sets it")
        with open(f"{MODEL_DIR}/config.json") as f:
            config = json.load(f)
        self.num_layers = config["num_hidden_layers"]
        self.num_experts = config["num_experts"]
        streaming = config["streaming"]

        from mlx_lm.models.qwen3_moe import Model, ModelArgs
        args = ModelArgs(
            model_type=config.get("model_type"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=self.num_layers,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            max_position_embeddings=config.get("max_position_embeddings", 40960),
            head_dim=config.get("head_dim"),
            tie_word_embeddings=config.get("tie_word_embeddings", True),
            num_experts=config["num_experts"],
            num_experts_per_tok=config["num_experts_per_tok"],
            moe_intermediate_size=config["moe_intermediate_size"],
            norm_topk_prob=config.get("norm_topk_prob", True),
            intermediate_size=config.get("intermediate_size", 6144),
            decoder_sparse_step=config.get("decoder_sparse_step", 1),
            mlp_only_layers=config.get("mlp_only_layers", []),
            rope_theta=config.get("rope_theta", 1000000.0),
        )

        self.model = Model(args)
        from mlx_lm.models.switch_layers import SwitchLinear

        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)

        pinned = mx.load(f"{MODEL_DIR}/pinned.safetensors")

        # Quantize per the CHECKPOINT's recipe, not the current mlx_lm
        # conversion recipe: honor per-module overrides from the config's
        # quantization dict, and quantize a pinned module only if the
        # checkpoint actually shipped scales for it. (The conversion-time
        # quant_predicate says e.g. "gate at 8-bit" for checkpoints newer
        # than this one and crashes on older uniform-4-bit checkpoints.)
        qcfg = dict(config.get("quantization") or {})
        q_group = qcfg.pop("group_size", GROUP_SIZE)
        q_bits = qcfg.pop("bits", BITS)
        pinned_keys = set(pinned.keys())
        def should_quantize(path, module):
            if path in qcfg:
                return qcfg[path]  # per-module override from the checkpoint
            if isinstance(module, SwitchLinear):
                return True  # expert weights are streamed, always quantized
            if isinstance(module, (nn.Linear, nn.Embedding)):
                return f"{path}.scales" in pinned_keys
            return False
        nn.quantize(self.model, group_size=q_group, bits=q_bits,
                    class_predicate=should_quantize)

        self.model.load_weights(list(pinned.items()), strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters()) if "switch_mlp" not in name]
        mx.eval(*params)
        del pinned; gc.collect(); mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9
        expert_dir = os.path.join(MODEL_DIR, streaming["expert_dir"])
        if self._pinned_only:
            self.reader = None  # caller installs one; bin/ need not exist
        else:
            self.reader = MoEExpertReader(expert_dir, self.num_layers, num_workers=8, cache_size=self._cache_size)
        self.coact = CoActivationTracker(self.num_layers, warmup_tokens=3)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        return pinned_gb

    def reset_cache(self):
        from mlx_lm.models.cache import KVCache
        self.cache = [KVCache() for _ in range(self.num_layers)]
        if self.reader:
            self.reader.reset_prefetch()

    def forward(self, input_ids):
        from mlx_lm.models.base import create_attention_mask
        h = self.model.model.embed_tokens(input_ids)
        mask = create_attention_mask(h, self.cache[0])

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            normed = layer.input_layernorm(h)
            attn_out = layer.self_attn(normed, mask=mask, cache=self.cache[i])
            h = h + attn_out
            mx.eval(h)

            normed = layer.post_attention_layernorm(h)
            gates = layer.mlp.gate(normed)
            gates = mx.softmax(gates, axis=-1, precise=True)
            k = layer.mlp.top_k
            inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
            scores = mx.take_along_axis(gates, inds, axis=-1)
            if layer.mlp.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            mx.eval(inds, scores)

            active_ids = list(set(int(e) for e in np.array(inds).flatten()))

            # Record co-activation for learning
            self.coact.record_layer(i, active_ids)

            # Predictive prefetch: use co-activation to prefetch next layer
            if self.coact.ready and i + 1 < self.num_layers:
                predicted = self.coact.predict_next_layer(i, active_ids, top_k=6)
                # Score prediction against what actually fires (for stats)
                # (actual scoring happens next iteration when we see layer i+1)
                # Prefetch predicted experts not already cached
                if predicted:
                    to_fetch = []
                    for eid in predicted:
                        if self.reader.lru and not self.reader.lru.contains(i + 1, eid):
                            to_fetch.append(eid)
                    if to_fetch:
                        self.reader.prefetch_experts(i + 1, to_fetch)

            # Also do the standard 1-layer-ahead prefetch with router-selected IDs
            if i + 1 < self.num_layers:
                self.reader.prefetch_experts(i + 1, active_ids)

            expert_data = self.reader.get_experts(i, active_ids)
            expert_out = run_expert_ffn(normed, expert_data, inds, scores)

            h = h + expert_out
            mx.eval(h)
            del expert_data, expert_out, normed, attn_out
            mx.clear_cache()

        self.coact.end_token()

        h = self.model.model.norm(h)
        return self.model.lm_head(h)
