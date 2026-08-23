#!/usr/bin/env python3
"""MoE Sniper — Qwen3.5-35B-A3B via SSD streaming on M4 Mac Mini."""
import json, sys, os, time, gc
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .expert_io import MoEExpertReader
from .coactivation import CoActivationTracker

MODEL_DIR = "/Users/bigneek/models/qwen35-35b-stream"
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


class MoESniperEngineNext:
    def __init__(self, cache_size=3000, enable_prediction=True):
        self.model = None
        self.reader = None
        self.tokenizer = None
        self.cache = None
        self.num_layers = 40
        self.coact = None
        self._cache_size = cache_size
        self._enable_prediction = enable_prediction

    def load(self):
        with open(os.path.join(MODEL_DIR, "config.json")) as f:
            config = json.load(f)
        self.num_layers = config["num_hidden_layers"]
        streaming = config["streaming"]

        from mlx_lm.models.qwen3_next import Model, ModelArgs
        args = ModelArgs(
            model_type=config.get("model_type"),
            hidden_size=config["hidden_size"],
            num_hidden_layers=self.num_layers,
            num_attention_heads=config["num_attention_heads"],
            num_key_value_heads=config["num_key_value_heads"],
            rms_norm_eps=config["rms_norm_eps"],
            vocab_size=config["vocab_size"],
            max_position_embeddings=config.get("max_position_embeddings", 262144),
            head_dim=config.get("head_dim"),
            tie_word_embeddings=config.get("tie_word_embeddings", False),
            num_experts=config["num_experts"],
            num_experts_per_tok=config["num_experts_per_tok"],
            shared_expert_intermediate_size=config.get("shared_expert_intermediate_size"),
            moe_intermediate_size=config["moe_intermediate_size"],
            linear_num_value_heads=config.get("linear_num_value_heads"),
            linear_num_key_heads=config.get("linear_num_key_heads"),
            linear_key_head_dim=config.get("linear_key_head_dim"),
            linear_value_head_dim=config.get("linear_value_head_dim"),
            linear_conv_kernel_dim=config.get("linear_conv_kernel_dim"),
            full_attention_interval=config.get("full_attention_interval"),
            partial_rotary_factor=config.get("partial_rotary_factor", 0.25),
            intermediate_size=config.get("intermediate_size", 5120),
            decoder_sparse_step=config.get("decoder_sparse_step", 1),
            mlp_only_layers=config.get("mlp_only_layers", []),
            rope_theta=config.get("rope_theta", 10000000),
        )

        self.model = Model(args)
        from mlx_lm.models.switch_layers import SwitchLinear
        model_pred = self.model.quant_predicate
        def should_quantize(path, module):
            if isinstance(module, nn.Embedding): return True
            if isinstance(module, SwitchLinear): return True
            if not isinstance(module, nn.Linear): return False
            return model_pred(path, module)
        nn.quantize(self.model, group_size=GROUP_SIZE, bits=BITS,
                     class_predicate=should_quantize)

        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)

        pinned = mx.load(os.path.join(MODEL_DIR, "pinned.safetensors"))
        stripped = [(k.replace("language_model.", "", 1), v) for k, v in pinned.items()]
        self.model.load_weights(stripped, strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters()) if "switch_mlp" not in name]
        mx.eval(*params)
        del pinned; gc.collect(); mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9
        expert_dir = os.path.join(MODEL_DIR, streaming["expert_dir"])
        self.reader = MoEExpertReader(expert_dir, self.num_layers, num_workers=8, cache_size=self._cache_size)
        self.coact = CoActivationTracker(self.num_layers, warmup_tokens=3)

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-35B-A3B", trust_remote_code=True)
        return pinned_gb

    def reset_cache(self):
        self.cache = self.model.make_cache()

    def forward(self, input_ids):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        h = self.model.model.embed_tokens(input_ids)
        fa_mask = create_attention_mask(h, self.cache[self.model.model.fa_idx])
        ssm_mask = create_ssm_mask(h, self.cache[self.model.model.ssm_idx])

        for i in range(self.num_layers):
            layer = self.model.model.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            normed = layer.input_layernorm(h)
            if layer.is_linear:
                attn_out = layer.linear_attn(normed, mask=mask, cache=self.cache[i])
            else:
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

            self.coact.record_layer(i, active_ids)

            # Predictive prefetch
            if self._enable_prediction and self.coact.ready and i + 1 < self.num_layers:
                predicted = self.coact.predict_next_layer(i, active_ids, top_k=6)
                if predicted:
                    to_fetch = [eid for eid in predicted
                                if self.reader.lru and self.reader.lru.get(i + 1, eid) is None]
                    if to_fetch:
                        self.reader.prefetch_experts(i + 1, to_fetch)

            # Standard prefetch
            if i + 1 < self.num_layers:
                self.reader.prefetch_experts(i + 1, active_ids)

            expert_data = self.reader.get_experts(i, active_ids)
            expert_out = run_expert_ffn(normed, expert_data, inds, scores)

            shared_out = layer.mlp.shared_expert(normed)
            shared_gate = mx.sigmoid(layer.mlp.shared_expert_gate(normed))
            if shared_gate.ndim < shared_out.ndim:
                shared_gate = shared_gate[..., None]
            expert_out = expert_out + shared_gate * shared_out

            h = h + expert_out
            mx.eval(h)
            del expert_data, expert_out, normed, attn_out
            mx.clear_cache()

        self.coact.end_token()
        h = self.model.model.norm(h)
        return self.model.lm_head(h)
