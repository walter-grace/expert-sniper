#!/usr/bin/env python3
"""MoE Sniper — Qwen3.8-Flash-Next (HF ``qwen4_exp``) via SSD streaming.

Pinned weights (embeddings, hyper-connections, attention, GatedDeltaNet,
routers, shared experts, PLE projections and, for now, the n-gram tables)
stay in RAM; routed experts stream from ``streaming.expert_dir`` exactly as
in engine_next. The model class lives in ``models/qwen4_exp.py``; this file
only owns loading and the per-layer streaming forward.

config.json keys read here (the full HF ``text_config`` dict plus):
  model_type                "qwen4_exp"
  quantization              {"bits": 4, "group_size": 32, "mode": "affine"}
  streaming.pinned_file     pinned safetensors (default "pinned.safetensors")
  streaming.expert_dir      directory of layer_XX.bin expert files
  streaming.ngram_source    "memory" | "disk" | "hf" (default: memory unless
                            ngram_in_memory is false, then disk)
  streaming.ngram_dir       row files for "disk" (default bin/ngram)
  streaming.source_repo     HF repo for "hf" (and the tokenizer fallback)
  ngram_in_memory           legacy bool; ngram_source wins when present
  norm_weights_hf           optional bool: force whether the pinned norm
                            weights are raw HF (1 + w) values; auto-detected
                            from the conv1d layout when absent
  tokenizer_repo            optional HF id used when MODEL_DIR has no tokenizer
"""
import json, sys, os, time, gc
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from .expert_io import MoEExpertReader
from .coactivation import CoActivationTracker

MODEL_DIR = None  # set by generate.load_engine / calibrate._build_engine
BITS = 4
GROUP_SIZE = 32


def run_expert_ffn(x, expert_data, top_k_indices, top_k_weights):
    """Routed SwiGLU experts from streamed ``switch_mlp.*`` blocks (same
    layout as engine_next; group size follows the checkpoint)."""
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


class MoESniperEngineQwen4Exp:
    # generate/evaluate must call engine.forward (the hyper-connection
    # residual is not the plain pre-norm stream make_forward assumes).
    own_forward = True

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
        global BITS, GROUP_SIZE
        if MODEL_DIR is None:
            raise RuntimeError(
                "engine_qwen4exp.MODEL_DIR is not set — load via "
                "generate.load_engine(model_dir), which sets it")
        with open(os.path.join(MODEL_DIR, "config.json")) as f:
            config = json.load(f)
        # Accept either the flattened text_config or the nested HF layout.
        if "text_config" in config and "num_hidden_layers" not in config:
            merged = dict(config["text_config"])
            merged.update({k: v for k, v in config.items() if k != "text_config"})
            config = merged
        self.num_layers = config["num_hidden_layers"]
        self.num_experts = config["num_experts"]
        streaming = config.get("streaming", {})

        from .models.qwen4_exp import Model, ModelArgs
        # n-gram table residency: "memory" (shards pinned), "disk"
        # (bin/ngram row files), "hf" (HTTP Range from streaming.source_repo)
        ngram_source = streaming.get("ngram_source")
        if ngram_source is None:
            ngram_source = "memory" if config.get("ngram_in_memory", True) else "disk"
        self.ngram_source_kind = ngram_source
        args = ModelArgs.from_dict(dict(
            config,
            model_type=config.get("model_type", "qwen4_exp"),
            ngram_in_memory=(ngram_source == "memory"),
        ))
        self.model = Model(args)
        self.ngram_sources = {}
        if ngram_source != "memory":
            from .ngram_source import make_ngram_source
            hf = None
            for layer in self.model.layers:
                if layer.ple is None:
                    continue
                src = make_ngram_source(ngram_source, config, MODEL_DIR, layer.layer_idx, hf=hf)
                hf = getattr(src, "hf", hf)
                layer.ple.ple_embedding.set_source(src)
                self.ngram_sources[layer.layer_idx] = src
        from mlx_lm.models.switch_layers import SwitchLinear

        mx.set_memory_limit(14 * 1024**3)
        mx.set_cache_limit(512 * 1024**2)

        pinned_file = streaming.get("pinned_file", "pinned.safetensors")
        pinned = mx.load(os.path.join(MODEL_DIR, pinned_file))
        weights = self.model.sanitize(pinned, hf_norms=config.get("norm_weights_hf"))

        # Quantize per the CHECKPOINT's recipe (see engine_30b for why).
        qcfg = dict(config.get("quantization") or {})
        q_group = qcfg.pop("group_size", GROUP_SIZE)
        q_bits = qcfg.pop("bits", BITS)
        qcfg.pop("mode", None)
        GROUP_SIZE, BITS = q_group, q_bits
        pinned_keys = set(weights)
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

        self.model.load_weights(list(weights.items()), strict=False)
        params = [p for name, p in tree_flatten(self.model.parameters()) if "switch_mlp" not in name]
        mx.eval(*params)
        del pinned, weights; gc.collect(); mx.clear_cache()

        pinned_gb = sum(p.nbytes for p in params) / 1e9
        expert_dir = os.path.join(MODEL_DIR, streaming.get("expert_dir", "bin"))
        if self._pinned_only:
            self.reader = None  # caller installs one; bin/ need not exist
        else:
            self.reader = MoEExpertReader(expert_dir, self.num_layers, num_workers=8, cache_size=self._cache_size)
        self.coact = CoActivationTracker(self.num_layers, warmup_tokens=3)

        from transformers import AutoTokenizer
        candidates = [MODEL_DIR, config.get("tokenizer_repo"), streaming.get("source_repo"),
                      "Qwen/Qwen3.8-Flash-Next"]
        err = None
        for c in [c for c in candidates if c]:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(c, trust_remote_code=True)
                break
            except Exception as e:  # noqa: BLE001
                err = e
        else:
            raise RuntimeError(f"no tokenizer found (tried {candidates}): {err}")
        return pinned_gb

    def reset_cache(self):
        self.cache = self.model.make_cache()
        if self.reader:
            self.reader.reset_prefetch()

    def forward(self, input_ids, on_route=None):
        """Streaming forward. ``on_route(layer_idx, inds, scores)`` (flat
        python lists) lets calibration record routing without a second
        copy of this loop."""
        tm = self.model.model
        h = tm.embed(input_ids)
        fa_mask, ssm_mask = tm.masks(h, self.cache)

        for i in range(self.num_layers):
            layer = tm.layers[i]
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer.attention_block(h, input_ids, mask, self.cache[i], conv_mask=ssm_mask)
            mx.eval(h)

            x, hyper, inj = layer.mlp_hyper_connection(h)
            inds, scores = layer.mlp.route(x)
            mx.eval(inds, scores)

            inds_flat = [int(e) for e in np.array(inds).flatten()]
            active_ids = list(set(inds_flat))
            if on_route is not None:
                on_route(i, inds_flat,
                         [float(s) for s in np.array(scores.astype(mx.float32)).flatten()])
            self.coact.record_layer(i, active_ids)

            # Predictive prefetch
            if self._enable_prediction and self.coact.ready and i + 1 < self.num_layers:
                predicted = self.coact.predict_next_layer(i, active_ids, top_k=6)
                if predicted:
                    to_fetch = [eid for eid in predicted
                                if self.reader.lru and not self.reader.lru.contains(i + 1, eid)]
                    if to_fetch:
                        self.reader.prefetch_experts(i + 1, to_fetch)

            # Standard prefetch
            if i + 1 < self.num_layers:
                self.reader.prefetch_experts(i + 1, active_ids)

            expert_data = self.reader.get_experts(i, active_ids)
            expert_out = run_expert_ffn(x, expert_data, inds, scores)
            expert_out = expert_out + layer.mlp.shared(x)

            from .models.qwen4_exp import inject
            h = inject(hyper, inj, expert_out)
            mx.eval(h)
            del expert_data, expert_out, x, hyper, inj
            mx.clear_cache()

        self.coact.end_token()
        h = tm.hyper_connection_mixer(h)
        return self.model.lm_head(h)
