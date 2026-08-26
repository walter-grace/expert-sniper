"""
MLX model implementation for Qwen3.8-Flash-Next (HF model_type ``qwen4_exp``,
text part ``qwen4_exp_text``).

Text-only decoder. What is new relative to Qwen3-Next / Qwen3.5-MoE:

- Hyper-connections: the residual stream is ``hc_count`` copies of the hidden
  state (``hc_count * hidden_size`` wide). Every block reads a learned,
  low-rank-gated mix of the streams (``GatedResidual``) and writes its output
  back into each stream with a per-stream injection weight. A final mixer
  (``hyper_connection_mixer``) collapses the streams before ``lm_head``.
  There is no final RMSNorm.
- Per-Layer Embedding (PLE, ``ple_layer_ids``, one-indexed): hashed 2-/3-gram
  embeddings (``NGramEmbedding``) gate a value that is added to every stream,
  plus a dilated depthwise conv over the gated values. The n-gram table is
  huge (~45B rows on the real model) and is looked up through the
  ``NGramSource`` interface so it can be streamed later; the in-memory
  implementation reads the ``ngram_embedding.shard_{i}`` tables that are held
  as regular (quantizable) ``nn.Embedding`` modules.
- Qwen Sparse Attention (QSA): each full-attention layer has an indexer that
  pools keys into blocks of ``indexer_compress_ratio`` tokens, scores them with
  ReLU(q.k) summed over indexer heads, and keeps the top
  ``indexer_budget / indexer_compress_ratio`` blocks plus the incomplete tail.
  When the context holds at most ``indexer_budget / compress_ratio`` complete
  blocks every block is selected, so attention is exactly dense; we take the
  dense path then and only build the sparse mask beyond that point.
- GatedDeltaNet output gate uses ``output_gate_type`` (sigmoid on the real
  checkpoint) instead of SiLU.
- Routed MoE (softmax top-k, renormalised) + shared expert with a sigmoid
  shared gate. Experts are ``mlp.switch_mlp.{gate,up,down}_proj`` SwitchGLU
  modules so the streaming reader can supply them; the engine bypasses
  ``self.mlp.switch_mlp`` and feeds ``run_expert_ffn`` with streamed experts.

Skipped (TODO): the vision tower and the MTP head. ``sanitize`` drops their
keys so a full checkpoint loads with ``strict=False``.

RMSNorm weights are stored in MLX convention (the weight is the final
multiplier). HF stores ``w`` and applies ``1 + w`` for every
``Qwen4ExpTextRMSNorm``; ``sanitize`` shifts those weights when it detects a
raw HF checkpoint (conv1d weights still in ``[C, 1, K]`` layout, or MTP keys
present), the same heuristic mlx_lm uses for Qwen3.5.

Reference: transformers ``models/qwen4_exp/modular_qwen4_exp.py``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_lm.models.base import (
    BaseModelArgs,
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.gated_delta import gated_delta_update
from mlx_lm.models.qwen3_next import Qwen3NextMLP
from mlx_lm.models.switch_layers import SwitchGLU


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "qwen4_exp"
    hidden_size: int = 2560
    num_hidden_layers: int = 48
    num_attention_heads: int = 24
    num_key_value_heads: int = 2
    head_dim: int = 256
    vocab_size: int = 248320
    rms_norm_eps: float = 1e-6
    hidden_act: str = "silu"
    tie_word_embeddings: bool = False
    attention_bias: bool = False
    max_position_embeddings: int = 262144
    # layer layout
    layer_types: Optional[List[str]] = None
    full_attention_interval: int = 4
    # linear attention (GatedDeltaNet)
    linear_conv_kernel_dim: int = 4
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 48
    output_gate_type: Optional[str] = None
    # MoE
    num_experts: int = 512
    num_experts_per_tok: int = 10
    moe_intermediate_size: int = 640
    shared_expert_intermediate_size: int = 640
    norm_topk_prob: bool = True
    # hyper-connections
    hc_count: int = 4
    hc_lowrank: int = 320
    # PLE / n-gram
    ple_layer_ids: List[int] = field(default_factory=list)
    ple_embed_dim: Optional[int] = None
    ple_conv_kernel_size: int = 4
    ngram_size: int = 3
    heads_per_ngram: int = 8
    ngram_vocab_size_base: int = 20_000_000
    make_ngram_vocab_size_divisible_by: int = 128
    seed: int = 1234
    split_ngram_parts: int = 512
    eos_token_id: Any = None
    # QSA indexer (None -> plain dense attention)
    indexer_n_heads: Optional[int] = None
    indexer_kv_heads: Optional[int] = None
    indexer_head_dim: Optional[int] = None
    indexer_budget: Optional[int] = None
    indexer_compress_ratio: Optional[int] = None
    # RoPE
    rope_parameters: Optional[Dict[str, Any]] = None
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    # n-gram table residency (see NGramSource)
    ngram_in_memory: bool = True

    def __post_init__(self):
        if self.rope_parameters:
            self.rope_theta = float(self.rope_parameters.get("rope_theta", self.rope_theta))
            self.partial_rotary_factor = float(
                self.rope_parameters.get("partial_rotary_factor", self.partial_rotary_factor))
        if self.layer_types is None:
            self.layer_types = [
                "linear_attention" if (i + 1) % self.full_attention_interval else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        # HF renames "full_attention" -> "qwen_sparse_attention"; treat both as attention.
        self.layer_types = [
            "full_attention" if t in ("full_attention", "qwen_sparse_attention") else t
            for t in self.layer_types
        ]
        self.ple_layer_ids = sorted(set(self.ple_layer_ids or []))
        if self.ple_embed_dim is None:
            self.ple_embed_dim = self.hidden_size
        if self.output_gate_type is None:
            self.output_gate_type = self.hidden_act
        if self.hc_count <= 1:
            raise ValueError(f"qwen4_exp requires hc_count > 1, got {self.hc_count}")

    @property
    def use_indexer(self) -> bool:
        return self.indexer_n_heads is not None

    @property
    def eos_id(self) -> int:
        eos = self.eos_token_id
        if isinstance(eos, list):
            eos = eos[0]
        if eos is None:
            raise ValueError("eos_token_id must be set when PLE layers are enabled")
        return int(eos)


# --------------------------------------------------------------------------- #
# Norms
# --------------------------------------------------------------------------- #

class RMSNorm(nn.Module):
    """RMSNorm, optionally grouped along the last axis (``group_size``).

    MLX convention: ``weight`` is the final multiplier (HF stores ``w`` and
    applies ``1 + w``; ``Model.sanitize`` shifts raw HF weights).
    """
    def __init__(self, dims: int, eps: float = 1e-6, group_size: Optional[int] = None):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dims % group_size != 0:
            raise ValueError(f"dims ({dims}) must be divisible by group_size ({group_size})")

    def __call__(self, x: mx.array) -> mx.array:
        if self.group_size is None:
            return mx.fast.rms_norm(x, self.weight, self.eps)
        shape = x.shape
        xg = x.reshape(*shape[:-1], -1, self.group_size)
        xg = mx.fast.rms_norm(xg, None, self.eps).reshape(shape)
        return (xg * self.weight).astype(x.dtype)


class RMSNormGated(nn.Module):
    """``weight * rms_norm(x) * act(gate)``; ``act`` is sigmoid or silu."""
    def __init__(self, dims: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
        if activation not in ("silu", "sigmoid"):
            raise ValueError(f"unsupported output gate activation: {activation}")
        self.activation = activation

    def __call__(self, x: mx.array, gate: Optional[mx.array] = None) -> mx.array:
        y = mx.fast.rms_norm(x, self.weight, self.eps)
        if gate is None:
            return y.astype(x.dtype)
        g = gate.astype(mx.float32)
        g = mx.sigmoid(g) if self.activation == "sigmoid" else nn.silu(g)
        return (y.astype(mx.float32) * g).astype(x.dtype)


# --------------------------------------------------------------------------- #
# RoPE helpers (partial rotary, rotate-half layout, arbitrary positions)
# --------------------------------------------------------------------------- #

def _rope_cos_sin(positions: mx.array, rotary_dim: int, base: float) -> Tuple[mx.array, mx.array]:
    inv_freq = 1.0 / (base ** (mx.arange(0, rotary_dim, 2, dtype=mx.float32) / rotary_dim))
    freqs = positions.astype(mx.float32)[:, None] * inv_freq[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb), mx.sin(emb)


def _apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
    """``x[..., P, D]`` with ``cos/sin[P, rotary_dim]`` broadcast on the
    trailing two axes. Only the first ``rotary_dim`` features rotate."""
    rd = cos.shape[-1]
    xr, xp = x[..., :rd], x[..., rd:]
    half = rd // 2
    x1, x2 = xr[..., :half], xr[..., half:]
    rot = mx.concatenate([-x2, x1], axis=-1)
    xr = (xr * cos + rot * sin).astype(x.dtype)
    return mx.concatenate([xr, xp], axis=-1) if xp.shape[-1] else xr


# --------------------------------------------------------------------------- #
# Linear attention (GatedDeltaNet, Qwen3.5 layout + configurable output gate)
# --------------------------------------------------------------------------- #

class GatedDeltaNet(nn.Module):
    """Same math as mlx_lm.models.qwen3_5.GatedDeltaNet; the output-gate
    activation follows ``output_gate_type``. Cache slots: [0]=conv, [1]=state."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_v_heads = args.linear_num_value_heads
        self.num_k_heads = args.linear_num_key_heads
        self.head_k_dim = args.linear_key_head_dim
        self.head_v_dim = args.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        if self.num_v_heads % self.num_k_heads != 0:
            raise ValueError("linear_num_value_heads must be divisible by linear_num_key_heads")
        self.conv_kernel_size = args.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(self.conv_dim, self.conv_dim, kernel_size=self.conv_kernel_size,
                                groups=self.conv_dim, bias=False, padding=0)
        self.in_proj_qkv = nn.Linear(self.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
        self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
        self.dt_bias = mx.ones(self.num_v_heads)
        self.A_log = mx.log(mx.random.uniform(low=0.01, high=16, shape=(self.num_v_heads,)))
        self.norm = RMSNormGated(self.head_v_dim, eps=args.rms_norm_eps, activation=args.output_gate_type)
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None, cache: Optional[Any] = None) -> mx.array:
        B, S, _ = x.shape
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(x)
        a = self.in_proj_a(x)

        if cache is not None and cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            if cache.lengths is not None:
                ends = mx.clip(cache.lengths, 0, S)
                positions = (ends[:, None] + mx.arange(n_keep))[..., None]
                cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
            else:
                cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1] if cache else None
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        # mlx_lm's Metal kernel needs head_k_dim >= 32 (Dk / 32 lanes); the
        # reference ops path is bit-compatible and used for smaller heads.
        use_kernel = (not self.training) and self.head_k_dim >= 32
        out, state = gated_delta_update(q, k, v, a, b, self.A_log, self.dt_bias, state, mask,
                                        use_kernel=use_kernel)
        if cache is not None:
            cache[1] = state
            cache.advance(S)
        out = self.norm(out, z)
        return self.out_proj(out.reshape(B, S, -1))


# --------------------------------------------------------------------------- #
# QSA indexer + attention
# --------------------------------------------------------------------------- #

class QSAKVCache(KVCache):
    """KVCache plus the raw (pre-norm, pre-RoPE) indexer keys ``[B, T, dh]``."""
    def __init__(self):
        super().__init__()
        self.indexer_keys = None

    def update_indexer(self, raw_keys: mx.array) -> mx.array:
        if self.indexer_keys is None:
            self.indexer_keys = raw_keys
        else:
            self.indexer_keys = mx.concatenate([self.indexer_keys, raw_keys], axis=1)
        return self.indexer_keys


class QSAIndexer(nn.Module):
    """Qwen Sparse Attention token selector.

    Returns ``None`` when every query can see at most ``block_topk`` complete
    blocks (selection is then the identity and attention is dense), else a
    boolean mask ``[B, 1, L, T]`` (True = attend) that already encodes
    causality: selected blocks plus the incomplete tail of each query.
    """
    def __init__(self, args: ModelArgs):
        super().__init__()
        if args.indexer_kv_heads != 1:
            raise ValueError("qwen4_exp QSA requires indexer_kv_heads=1")
        self.n_heads = args.indexer_n_heads
        self.head_dim = args.indexer_head_dim
        self.budget = args.indexer_budget
        self.ratio = args.indexer_compress_ratio
        if self.budget % self.ratio != 0:
            raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
        self.block_topk = self.budget // self.ratio
        self.rotary_dim = int(args.head_dim * args.partial_rotary_factor)
        if self.rotary_dim > self.head_dim:
            raise ValueError("attention rotary dims must fit the indexer head")
        self.rope_theta = args.rope_theta
        self.index_qk_proj = nn.Linear(args.hidden_size, (self.n_heads + 1) * self.head_dim, bias=False)
        self.q_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)

    def __call__(self, x: mx.array, offset: int, cache: Optional[QSAKVCache]) -> Optional[mx.array]:
        B, L, _ = x.shape
        qk = self.index_qk_proj(x)
        q = qk[..., : self.n_heads * self.head_dim].reshape(B, L, self.n_heads, self.head_dim)
        raw_k = qk[..., self.n_heads * self.head_dim:]
        raw_keys = cache.update_indexer(raw_k) if cache is not None else raw_k
        T = raw_keys.shape[1]
        nb = T // self.ratio
        if nb <= self.block_topk:
            return None  # every complete block is selected -> dense attention

        r = self.ratio
        q = self.q_layernorm(q)
        qpos = offset + mx.arange(L)
        cos, sin = _rope_cos_sin(qpos, self.rotary_dim, self.rope_theta)
        q = _apply_rope(q.transpose(0, 2, 1, 3), cos, sin).transpose(0, 2, 1, 3)  # [B, L, H, dh]

        pooled = raw_keys[:, : nb * r].reshape(B, nb, r, self.head_dim)
        pooled = pooled.astype(mx.float32).mean(axis=2).astype(raw_keys.dtype)
        pooled = self.k_layernorm(pooled)
        kcos, ksin = _rope_cos_sin(mx.arange(nb) * r, self.rotary_dim, self.rope_theta)
        block_keys = _apply_rope(pooled, kcos, ksin)  # [B, nb, dh]

        scores = mx.einsum("blhd,bnd->blhn", q.astype(mx.float32), block_keys.astype(mx.float32))
        scores = mx.maximum(scores, 0).sum(axis=2) / math.sqrt(self.head_dim)  # [B, L, nb]

        n_complete = (qpos + 1) // r                                    # [L]
        block_ids = mx.arange(nb)
        valid = block_ids[None, :] < n_complete[:, None]                # [L, nb]
        scores = mx.where(valid[None], scores, -mx.inf)
        order = mx.argsort(-scores, axis=-1)[..., : self.block_topk]    # [B, L, k]
        sel_valid = mx.take_along_axis(mx.broadcast_to(valid[None], scores.shape), order, axis=-1)
        onehot = (order[..., None] == block_ids) & sel_valid[..., None]  # [B, L, k, nb]
        block_sel = onehot.any(axis=2)                                  # [B, L, nb]

        t = mx.arange(T)
        tok_block = mx.minimum(t // r, nb - 1)
        in_block = mx.take(block_sel, tok_block, axis=-1)               # [B, L, T]
        in_block = in_block & (t[None, None, :] < (n_complete * r)[None, :, None])
        tail = (t[None, :] >= (n_complete * r)[:, None]) & (t[None, :] <= qpos[:, None])  # [L, T]
        return (in_block | tail[None])[:, None]                          # [B, 1, L, T]


class Attention(nn.Module):
    """Qwen3.5-style gated attention (q_proj emits query + sigmoid gate),
    partial rotary, optional QSA indexer."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(args.hidden_size, self.n_heads * self.head_dim * 2, bias=args.attention_bias)
        self.k_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(args.hidden_size, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, args.hidden_size, bias=args.attention_bias)
        self.q_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = nn.RoPE(int(self.head_dim * args.partial_rotary_factor), traditional=False,
                            base=args.rope_theta)
        self.indexer = QSAIndexer(args) if args.use_indexer else None

    def __call__(self, x: mx.array, mask: Optional[Any] = None, cache: Optional[Any] = None) -> mx.array:
        B, L, _ = x.shape
        offset = cache.offset if cache is not None else 0
        qg = self.q_proj(x).reshape(B, L, self.n_heads, 2 * self.head_dim)
        queries, gate = mx.split(qg, 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(self.k_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim)).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)
        if self.indexer is not None:
            sparse_mask = self.indexer(x, offset, cache)
            if sparse_mask is not None:
                mask = sparse_mask
        out = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=self.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out * mx.sigmoid(gate))


# --------------------------------------------------------------------------- #
# MoE
# --------------------------------------------------------------------------- #

class SparseMoeBlock(nn.Module):
    """Softmax top-k router (renormalised) + SwitchGLU experts + shared expert
    with sigmoid gate. ``route`` / ``shared`` are exposed so the streaming
    engine can run the routed experts itself."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.num_experts = args.num_experts
        self.top_k = args.num_experts_per_tok
        self.norm_topk_prob = args.norm_topk_prob
        self.gate = nn.Linear(dim, args.num_experts, bias=False)
        self.switch_mlp = SwitchGLU(dim, args.moe_intermediate_size, args.num_experts)
        self.shared_expert = Qwen3NextMLP(dim, args.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(dim, 1, bias=False)

    def route(self, x: mx.array, logits: Optional[mx.array] = None) -> Tuple[mx.array, mx.array]:
        if logits is None:
            logits = self.gate(x)
        gates = mx.softmax(logits, axis=-1, precise=True)
        k = self.top_k
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores.astype(x.dtype)

    def shared(self, x: mx.array) -> mx.array:
        return mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)

    def __call__(self, x: mx.array) -> mx.array:
        inds, scores = self.route(x)
        y = self.switch_mlp(x, inds)
        y = (y * scores[..., None]).sum(axis=-2)
        return y + self.shared(x)


# --------------------------------------------------------------------------- #
# Hyper-connections
# --------------------------------------------------------------------------- #

class GatedResidual(nn.Module):
    """Reads a gated mix of the ``hc_count`` residual streams. With
    ``use_combine`` also returns the per-stream injection weights."""
    def __init__(self, args: ModelArgs, use_combine: bool = True):
        super().__init__()
        self.hc_count = args.hc_count
        self.hidden_size = args.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = RMSNorm(hc_hidden, eps=args.rms_norm_eps, group_size=self.hidden_size)
        self.input_mix_weight_down = nn.Linear(hc_hidden, args.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(args.hc_lowrank, hc_hidden, bias=False)
        self.block_inject_weight = nn.Linear(hc_hidden, self.hc_count, bias=False) if use_combine else None

    def __call__(self, hyper: mx.array):
        hn = self.hc_norm(hyper)
        w = nn.silu(self.input_mix_weight_down(hn) / self.hc_count)
        w = mx.sigmoid(self.input_mix_weight_up(w))
        shape = (*hyper.shape[:-1], self.hc_count, self.hidden_size)
        mixed = (w.reshape(shape) * hn.reshape(shape)).mean(axis=-2)
        if self.block_inject_weight is None:
            return mixed
        inj = 2 * mx.sigmoid(self.block_inject_weight(hn) / self.hc_count)
        return mixed, hyper, inj


def inject(hyper: mx.array, inj: mx.array, out: mx.array) -> mx.array:
    """``hyper + flatten(out[..., None, :] * inj[..., :, None])``."""
    add = out[..., None, :] * inj[..., :, None]
    return hyper + add.reshape(hyper.shape)


# --------------------------------------------------------------------------- #
# N-gram hashing (bit-exact port of the HF helpers)
# --------------------------------------------------------------------------- #

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def build_layer_multipliers(unigram_vocab_size: int, ngram_size: int, ple_layer_index: int, seed: int) -> List[int]:
    max_long = (1 << 63) - 1
    multiplier_max = max_long // max(unigram_vocab_size, 1)
    half_bound = max(1, multiplier_max // 2)
    base_seed = seed + _PRIME_1 * ple_layer_index
    out = []
    for index in range(ngram_size):
        value = (base_seed + _SPLITMIX_GAMMA * (index + 1)) & _MASK64
        out.append(2 * (_splitmix64(value) % half_bound) + 1)
    return out


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for d in range(3, math.isqrt(value) + 1, 2):
        if value % d == 0:
            return False
    return True


def _find_nth_prime_after(start: int, count: int) -> int:
    p = start
    for _ in range(count):
        p += 1
        while not _is_prime(p):
            p += 1
    return p


def ngram_table_geometry(args: ModelArgs, ple_layer_index: int) -> Dict[str, Any]:
    """Head vocab sizes / offsets / padded table size for one PLE layer."""
    ngram_heads = (args.ngram_size - 1) * args.heads_per_ngram
    sizes, offsets, total = [], [], 0
    for head_idx in range(ngram_heads):
        g = ple_layer_index * ngram_heads + head_idx
        size = _find_nth_prime_after(args.ngram_vocab_size_base - 1, g + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    div = args.make_ngram_vocab_size_divisible_by
    padded = math.ceil(total / div) * div
    return {
        "ngram_heads": ngram_heads,
        "head_vocab_sizes": sizes,
        "head_offsets": offsets,
        "total_vocab_size": total,
        "padded_vocab_size": padded,
        "head_dim": args.ple_embed_dim // ngram_heads,
        "num_shards": args.split_ngram_parts,
        "rows_per_shard": padded // args.split_ngram_parts,
    }


# --------------------------------------------------------------------------- #
# N-gram table source
# --------------------------------------------------------------------------- #

class NGramSource:
    """Row provider for one PLE layer's n-gram table, sharded as in the
    checkpoint (``ngram_embedding.shard_{i}``, equal row splits of the padded
    vocabulary). ``rows(shard_idx, ids)`` returns dequantised rows
    ``[len(ids), head_dim]`` for ids local to that shard. A streaming
    implementation (SSD / Expert Network) only has to implement this."""
    rows_per_shard: int

    def rows(self, shard_idx: int, ids: mx.array) -> mx.array:
        raise NotImplementedError

    def lookup(self, global_ids: mx.array) -> mx.array:
        """``global_ids[...]`` (int) -> ``[..., head_dim]``."""
        shape = global_ids.shape
        flat = global_ids.reshape(-1)
        shard = flat // self.rows_per_shard
        local = flat % self.rows_per_shard
        shard_np = np.array(shard)
        order = np.argsort(shard_np, kind="stable")
        inv = np.argsort(order)
        local_np = np.array(local)[order]
        sorted_shards = shard_np[order]
        chunks = []
        for s in np.unique(sorted_shards):
            sel = np.nonzero(sorted_shards == s)[0]
            chunks.append(self.rows(int(s), mx.array(local_np[sel])))
        rows = chunks[0] if len(chunks) == 1 else mx.concatenate(chunks, axis=0)
        rows = rows[mx.array(inv)]
        return rows.reshape(*shape, rows.shape[-1])


class InMemoryNGramSource(NGramSource):
    """Reads rows from the ``shard_{i}`` embedding modules held in the model
    (fp or ``nn.QuantizedEmbedding`` after ``nn.quantize``)."""
    def __init__(self, shards: "NGramShards", rows_per_shard: int):
        self.shards = shards
        self.rows_per_shard = rows_per_shard

    def rows(self, shard_idx: int, ids: mx.array) -> mx.array:
        return getattr(self.shards, f"shard_{shard_idx}")(ids)


class NGramShards(nn.Module):
    def __init__(self, num_shards: int, rows_per_shard: int, dim: int):
        super().__init__()
        for i in range(num_shards):
            setattr(self, f"shard_{i}", nn.Embedding(rows_per_shard, dim))


class NGramEmbedding(nn.Module):
    """Hashed n-gram embeddings. Cache slot [3] of the layer's ArraysCache
    holds the previous ``ngram_size - 1`` token ids."""
    CTX_SLOT = 3

    def __init__(self, args: ModelArgs, ple_layer_index: int):
        super().__init__()
        self.ngram_size = args.ngram_size
        self.context_len = args.ngram_size - 1
        self.heads_per_ngram = args.heads_per_ngram
        self.eos_token_id = args.eos_id
        geo = ngram_table_geometry(args, ple_layer_index)
        self.ngram_heads = geo["ngram_heads"]
        self.head_dim = geo["head_dim"]
        self.rows_per_shard = geo["rows_per_shard"]
        self.num_shards = geo["num_shards"]
        self.padded_vocab_size = geo["padded_vocab_size"]
        # Buffers (also present in the checkpoint; recomputed here so a
        # partial load is still correct).
        self.layer_multipliers = mx.array(
            build_layer_multipliers(args.vocab_size, args.ngram_size, ple_layer_index, args.seed), dtype=mx.int64)
        self.ngram_heads_vocab_sizes = mx.array(geo["head_vocab_sizes"], dtype=mx.int64)
        self.ngram_heads_offsets = mx.array(geo["head_offsets"], dtype=mx.int64)
        self.source: Optional[NGramSource] = None
        if args.ngram_in_memory:
            self.ngram_embedding = NGramShards(self.num_shards, self.rows_per_shard, self.head_dim)
            self.source = InMemoryNGramSource(self.ngram_embedding, self.rows_per_shard)

    def set_source(self, source: NGramSource):
        self.source = source

    def _shift_right_ignore_eos(self, ids: mx.array, shift: int) -> mx.array:
        if shift == 0:
            return ids
        B, T = ids.shape
        pos = mx.arange(T, dtype=mx.int64)
        eos_pos = mx.where(ids == self.eos_token_id, pos[None, :], -1)
        prev_eos_incl = mx.cummax(eos_pos, axis=1)
        prev_eos = mx.concatenate([mx.full((B, 1), -1, dtype=mx.int64), prev_eos_incl[:, :-1]], axis=1)
        pos_in_seg = pos[None, :] - (prev_eos + 1)
        src = pos - shift
        gather = mx.broadcast_to(mx.maximum(src, 0)[None, :], (B, T))
        shifted = mx.take_along_axis(ids, gather, axis=1)
        valid = (pos_in_seg >= shift) & (src[None, :] >= 0)
        return mx.where(valid, shifted, mx.array(self.eos_token_id, dtype=mx.int64))

    def ngram_ids(self, input_ids: mx.array, cache: Optional[Any]) -> mx.array:
        """``[B, L]`` token ids -> ``[B, L, ngram_heads]`` global table rows."""
        ids = input_ids.astype(mx.int64)
        B, L = ids.shape
        if cache is not None and cache[self.CTX_SLOT] is not None:
            prev = cache[self.CTX_SLOT]
        else:
            prev = mx.full((B, self.context_len), self.eos_token_id, dtype=mx.int64)
        history = mx.concatenate([prev, ids], axis=1)
        if cache is not None:
            cache[self.CTX_SLOT] = history[:, -self.context_len:]
        shifted = [self._shift_right_ignore_eos(history, s) for s in range(self.ngram_size)]
        blocks = []
        for n in range(2, self.ngram_size + 1):
            s0 = (n - 2) * self.heads_per_ngram
            s1 = s0 + self.heads_per_ngram
            mixed = shifted[0] * self.layer_multipliers[0]
            for p in range(1, n):
                mixed = mx.bitwise_xor(mixed, shifted[p] * self.layer_multipliers[p])
            sizes = self.ngram_heads_vocab_sizes[s0:s1]
            offs = self.ngram_heads_offsets[s0:s1]
            blocks.append(mx.remainder(mixed[..., None], sizes.reshape(1, 1, -1)) + offs.reshape(1, 1, -1))
        return mx.concatenate(blocks, axis=-1)[:, -L:]

    def __call__(self, input_ids: mx.array, cache: Optional[Any] = None) -> mx.array:
        if self.source is None:
            raise RuntimeError("NGramEmbedding has no source: set ngram_in_memory=True or call set_source()")
        ids = self.ngram_ids(input_ids, cache)
        rows = self.source.lookup(ids)                     # [B, L, heads, head_dim]
        return rows.reshape(*ids.shape[:-1], -1)


# --------------------------------------------------------------------------- #
# PLE layer
# --------------------------------------------------------------------------- #

class PLELayer(nn.Module):
    """Per-Layer Embedding: n-gram keyed value injected into every stream,
    plus a dilated depthwise conv. Cache slot [2] holds the conv state."""
    CONV_SLOT = 2

    def __init__(self, args: ModelArgs, ple_layer_index: int):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.hc_count = args.hc_count
        hc_hidden = self.hidden_size * self.hc_count
        self.ple_embedding = NGramEmbedding(args, ple_layer_index)
        kernel = args.ple_conv_kernel_size
        dilation = args.ngram_size
        self.short_conv_state_len = (kernel - 1) * dilation
        self.key_proj = nn.Linear(args.ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(args.ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = RMSNorm(hc_hidden, eps=args.rms_norm_eps, group_size=self.hidden_size)
        self.norm_query = RMSNorm(hc_hidden, eps=args.rms_norm_eps, group_size=self.hidden_size)
        self.norm_conv = RMSNorm(hc_hidden, eps=args.rms_norm_eps, group_size=self.hidden_size)
        self.conv1d = nn.Conv1d(hc_hidden, hc_hidden, kernel_size=kernel, groups=hc_hidden,
                                dilation=dilation, bias=False, padding=0)

    def _short_conv(self, x: mx.array, cache: Optional[Any]) -> mx.array:
        B, L, C = x.shape
        n = self.short_conv_state_len
        if cache is not None and cache[self.CONV_SLOT] is not None:
            state = cache[self.CONV_SLOT]
        else:
            state = mx.zeros((B, n, C), dtype=x.dtype)
        full = mx.concatenate([state, x], axis=1)
        if cache is not None:
            cache[self.CONV_SLOT] = mx.contiguous(full[:, -n:, :])
        return nn.silu(self.conv1d(full))

    def __call__(self, hidden: mx.array, input_ids: mx.array, cache: Optional[Any] = None,
                 conv_mask: Optional[mx.array] = None) -> mx.array:
        emb = self.ple_embedding(input_ids, cache)
        shape = (*hidden.shape[:-1], self.hc_count, self.hidden_size)
        key_n = self.norm_key(self.key_proj(emb)).reshape(shape)
        value = self.value_proj(emb)
        query_n = self.norm_query(hidden).reshape(shape)
        gate = (key_n * query_n).sum(axis=-1, keepdims=True) / math.sqrt(self.hidden_size)
        gate = mx.sqrt(mx.maximum(mx.abs(gate), 1e-6)) * mx.sign(gate)
        gated = mx.sigmoid(gate) * value[..., None, :]
        gated = gated.reshape(hidden.shape)
        gated_n = self.norm_conv(gated)
        if conv_mask is not None:
            gated = mx.where(conv_mask[..., None], gated, 0)
            gated_n = mx.where(conv_mask[..., None], gated_n, 0)
        return gated + self._short_conv(gated_n, cache)


# --------------------------------------------------------------------------- #
# Decoder layer
# --------------------------------------------------------------------------- #

class DecoderLayer(nn.Module):
    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_linear = args.layer_types[layer_idx] == "linear_attention"
        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)
        self.mlp = SparseMoeBlock(args)
        one_indexed = layer_idx + 1
        self.ple = None
        if one_indexed in args.ple_layer_ids:
            if not self.is_linear:
                raise ValueError("PLE is only supported on linear_attention layers")
            self.ple = PLELayer(args, args.ple_layer_ids.index(one_indexed))
        self.attn_hyper_connection = GatedResidual(args)
        self.mlp_hyper_connection = GatedResidual(args)

    def attention_block(self, h: mx.array, input_ids: Optional[mx.array], mask: Optional[Any],
                        cache: Optional[Any], conv_mask: Optional[mx.array] = None) -> mx.array:
        """PLE (if any) + attention/linear attention with hyper-connection
        read/write. Returns the updated ``hc_count * hidden`` stream."""
        if self.ple is not None:
            h = h + self.ple(h, input_ids, cache, conv_mask=conv_mask)
        x, hyper, inj = self.attn_hyper_connection(h)
        if self.is_linear:
            out = self.linear_attn(x, mask=mask, cache=cache)
        else:
            out = self.self_attn(x, mask=mask, cache=cache)
        return inject(hyper, inj, out)

    def __call__(self, h: mx.array, input_ids: Optional[mx.array] = None, mask: Optional[Any] = None,
                 cache: Optional[Any] = None, conv_mask: Optional[mx.array] = None) -> mx.array:
        h = self.attention_block(h, input_ids, mask, cache, conv_mask)
        x, hyper, inj = self.mlp_hyper_connection(h)
        return inject(hyper, inj, self.mlp(x))


# --------------------------------------------------------------------------- #
# Full model
# --------------------------------------------------------------------------- #

class Qwen4ExpTextModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DecoderLayer(args, i) for i in range(args.num_hidden_layers)]
        self.hyper_connection_mixer = GatedResidual(args, use_combine=False)
        self.ssm_idx = next((i for i, t in enumerate(args.layer_types) if t == "linear_attention"), None)
        self.fa_idx = next((i for i, t in enumerate(args.layer_types) if t == "full_attention"), None)

    def embed(self, inputs: mx.array) -> mx.array:
        h = self.embed_tokens(inputs)
        return mx.tile(h, (1, 1, self.args.hc_count))

    def masks(self, h: mx.array, cache):
        fa_mask = create_attention_mask(h, cache[self.fa_idx]) if self.fa_idx is not None else None
        ssm_mask = create_ssm_mask(h, cache[self.ssm_idx]) if self.ssm_idx is not None else None
        return fa_mask, ssm_mask

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        h = self.embed(inputs)
        if cache is None:
            cache = [None] * len(self.layers)
        fa_mask, ssm_mask = self.masks(h, cache)
        for layer, c in zip(self.layers, cache):
            mask = ssm_mask if layer.is_linear else fa_mask
            h = layer(h, input_ids=inputs, mask=mask, cache=c, conv_mask=ssm_mask)
        return self.hyper_connection_mixer(h)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen4ExpTextModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache: Optional[Any] = None) -> mx.array:
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(out)
        return self.lm_head(out)

    @property
    def layers(self):
        return self.model.layers

    def make_cache(self):
        # linear layers: [conv, ssm state, ple conv, ngram context]
        return [ArraysCache(size=4) if l.is_linear else QSAKVCache() for l in self.layers]

    # keys of (1 + w) norms in the HF checkpoint
    _SHIFTED_NORM_SUFFIXES = (
        ".q_norm.weight", ".k_norm.weight", ".q_layernorm.weight", ".k_layernorm.weight",
        ".hc_norm.weight", ".norm_key.weight", ".norm_query.weight", ".norm_conv.weight",
    )

    def sanitize(self, weights: Dict[str, mx.array], hf_norms: Optional[bool] = None) -> Dict[str, mx.array]:
        """Map HF / mlx-community checkpoint keys onto this module tree.

        - strips ``language_model.`` / ``model.language_model.`` prefixes
        - drops vision tower, MTP and (if tied) lm_head keys
        - HF experts ``mlp.experts.{gate_up_proj,down_proj}`` -> switch_mlp
        - a runtime ``ngram_embedding.weight`` -> ``split_ngram_parts`` shards
        - conv1d ``[C, 1, K]`` -> ``[C, K, 1]``
        - ``hf_norms`` (auto-detected when None): add 1 to the (1 + w) norms
        """
        out = {}
        for k, v in weights.items():
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model."):]
            elif k.startswith("language_model."):
                k = k[len("language_model."):]
            if k.startswith(("vision_tower.", "visual.", "model.visual.", "model.vision_tower.", "mtp.", "model.mtp.")):
                continue
            if self.args.tie_word_embeddings and k == "lm_head.weight":
                continue
            out[k] = v
        weights = out

        if hf_norms is None:
            hf_norms = any("mtp." in k for k in weights) or any(
                k.endswith("conv1d.weight") and v.shape[-1] != 1 for k, v in weights.items())

        result = {}
        for k, v in list(weights.items()):
            if k.endswith(".mlp.experts.gate_up_proj"):
                gate, up = mx.split(v, 2, axis=1)  # [E, 2I, D]
                base = k[: -len("experts.gate_up_proj")]
                result[base + "switch_mlp.gate_proj.weight"] = gate
                result[base + "switch_mlp.up_proj.weight"] = up
                continue
            if k.endswith(".mlp.experts.down_proj"):
                result[k[: -len("experts.down_proj")] + "switch_mlp.down_proj.weight"] = v
                continue
            if k.endswith("ple_embedding.ngram_embedding.weight"):
                n = self.args.split_ngram_parts
                base = k[: -len("weight")]
                for i, part in enumerate(mx.split(v, n, axis=0)):
                    result[f"{base}shard_{i}.weight"] = part
                continue
            if k.endswith("conv1d.weight") and v.ndim == 3 and v.shape[-1] != 1:
                v = v.moveaxis(2, 1)
            if hf_norms and k.endswith(self._SHIFTED_NORM_SUFFIXES) and v.ndim == 1:
                v = v + 1.0
            result[k] = v
        return result

    @property
    def quant_predicate(self):
        def predicate(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8}
            return True
        return predicate
