"""DFlash speculative decode runtime — vendored from z-lab/dflash @ main.

Source: https://github.com/z-lab/dflash/blob/main/dflash/model_mlx.py
License: MIT (Copyright (c) 2026 Z Lab)

Adapted for lumen-rs:
  - Removed top-level `stream_generate` (we wire DFlash into our own
    `MlxRunner.prefill` / `decode_step` API instead of running our own loop).
  - Kept model classes (`DFlashConfig`, `DFlashAttention`,
    `DFlashDecoderLayer`, `DFlashDraftModel`), `load_draft`, `_LayerHook`,
    `_get_layers`, `_patch_model`, and `_GDNStateCapture` verbatim — they are
    the correctness foundation for DFlash on MLX-served hybrid Qwen targets.
  - Added a small block-step helper (`run_block_step`) that the runner calls
    once per spec-decode block. Mirrors upstream `stream_generate`'s inner
    loop without the streaming/detokenization machinery.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx_lm.models.cache import (
    KVCache,
    RotatingKVCache,
    can_trim_prompt_cache,
    make_prompt_cache,
    trim_prompt_cache,
)
from mlx_lm.models.qwen3 import MLP
from mlx_lm.models.rope_utils import initialize_rope


try:
    import mlx_lm.models.gated_delta as _gd_mod
    _HAS_GDN = True
except ImportError:
    _HAS_GDN = False


_GDN_PATCH_LOCK = RLock()


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: Tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: Optional[Dict[str, Any]] = None
    sliding_window_size: Optional[int] = None


def _build_rope(head_dim, rope_theta, max_position_embeddings, rope_scaling):
    return initialize_rope(
        dims=head_dim,
        base=rope_theta,
        traditional=False,
        scaling_config=rope_scaling,
        max_position_embeddings=max_position_embeddings,
    )


class DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.scale = config.head_dim ** -0.5
        self.q_proj = nn.Linear(dim, self.n_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * config.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * config.head_dim, dim, bias=False)
        self.q_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        B, L, _ = x.shape
        S = x_ctx.shape[1]
        queries = self.q_proj(x)
        ctx_keys = self.k_proj(x_ctx)
        ctx_values = self.v_proj(x_ctx)
        prop_keys = self.k_proj(x)
        prop_values = self.v_proj(x)
        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        ctx_keys = self.k_norm(ctx_keys.reshape(B, S, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        prop_keys = self.k_norm(prop_keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
        keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
        keys = mx.concatenate([keys, prop_keys], axis=2)
        values = mx.concatenate([values, prop_values], axis=2)
        output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=self.scale)
        return self.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x, x_ctx, rope, cache):
        h = x + self.self_attn(self.input_layernorm(x), x_ctx, rope, cache)
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rope = _build_rope(
            config.head_dim,
            config.rope_theta,
            config.max_position_embeddings,
            config.rope_scaling,
        )
        self.embed_tokens = None
        self.lm_head = None

    def bind(self, target_model):
        if hasattr(target_model, "embed_tokens"):
            inner = target_model
        elif hasattr(target_model, "model") and hasattr(target_model.model, "embed_tokens"):
            inner = target_model.model
        elif (
            hasattr(target_model, "language_model")
            and hasattr(target_model.language_model, "model")
            and hasattr(target_model.language_model.model, "embed_tokens")
        ):
            inner = target_model.language_model.model
        else:
            raise AttributeError(
                f"Cannot find embed_tokens in {type(target_model).__name__}"
            )
        self.embed_tokens = inner.embed_tokens
        lm = getattr(target_model, "language_model", target_model)
        self.lm_head = (
            getattr(target_model, "lm_head", None)
            or getattr(lm, "lm_head", None)
            or self.embed_tokens.as_linear
        )
        return self

    def make_cache(self):
        if self.config.sliding_window_size is not None:
            return [
                RotatingKVCache(max_size=self.config.sliding_window_size, keep=0)
                for _ in self.layers
            ]
        return [KVCache() for _ in self.layers]

    def __call__(self, inputs, target_hidden, cache):
        h = self.embed_tokens(inputs)
        h_ctx = self.hidden_norm(self.fc(target_hidden))
        for layer, c in zip(self.layers, cache):
            h = layer(h, h_ctx, self.rope, c)
        return self.lm_head(self.norm(h))


def load_draft(draft_id: str, sliding_window_size: Optional[int] = None) -> DFlashDraftModel:
    """Load DFlash draft model from HF Hub (already-cached snapshot ok).

    `sliding_window_size`: optional rotating KV cache size for the draft.
    `None` = unbounded KV cache (matches upstream default).
    """
    if sliding_window_size is not None and sliding_window_size <= 0:
        raise ValueError(
            f"sliding_window_size must be positive or None, got {sliding_window_size}"
        )
    path = Path(snapshot_download(draft_id, allow_patterns=["*.safetensors", "*.json"]))
    cfg = json.loads((path / "config.json").read_text())
    config = DFlashConfig(
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["block_size"],
        target_layer_ids=tuple(cfg["dflash_config"]["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=cfg["dflash_config"]["mask_token_id"],
        rope_scaling=cfg.get("rope_scaling"),
        sliding_window_size=sliding_window_size,
    )
    weights = {
        k: v
        for f in path.glob("*.safetensors")
        for k, v in mx.load(str(f)).items()
    }
    model = DFlashDraftModel(config)
    model.load_weights(list(weights.items()))
    return model


# ────────────────────────────────────────────────────────────────────────────
# Target hidden state capture (proxy + list-indexed storage)
#
# Replaces target_layer_ids' DecoderLayer instances in-place with a simple
# proxy whose __call__ records the layer's output before delegating. Storage
# is a list on the target model (`model._hidden_states`), enumeration-indexed
# so `mx.concatenate(model._hidden_states, axis=-1)` is a one-liner that
# matches the draft's `fc` weight layout.
# ────────────────────────────────────────────────────────────────────────────


class _LayerHook:
    """Proxy wrapping a DecoderLayer; records output to storage[idx] on
    every call. Delegates attribute lookups to the wrapped layer so MLX's
    parameter machinery and other accessors keep working."""

    def __init__(self, layer, idx, storage):
        self._layer, self._idx, self._storage = layer, idx, storage

    def __call__(self, *args, **kwargs):
        self._storage[self._idx] = out = self._layer(*args, **kwargs)
        return out

    def __getattr__(self, name):
        return getattr(self._layer, name)


def _get_layers(model):
    """Locate the target's transformer-layer list across known MLX layouts."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def patch_model(model, layer_ids) -> int:
    """Install layer-output capture hooks on `model` for each layer index in
    `layer_ids`. Idempotent: skip if already patched. Returns number of
    layers actually wrapped."""
    if hasattr(model, "_hidden_states"):
        return 0
    model._hidden_states = [None] * len(layer_ids)
    layers = _get_layers(model)
    for i, lid in enumerate(layer_ids):
        layers[lid] = _LayerHook(layers[lid], i, model._hidden_states)
    return len(layer_ids)


def get_target_hidden(model):
    """After a target forward, return the captured hidden states concatenated
    along the channel axis, ready as `target_hidden` for the draft. Returns
    None when capture is uninstalled or no forward has populated all slots."""
    states = getattr(model, "_hidden_states", None)
    if states is None or any(s is None for s in states):
        return None
    return mx.concatenate(states, axis=-1)


# ────────────────────────────────────────────────────────────────────────────
# GatedDeltaNet rollback for hybrid Qwen3.5/3.6 targets
#
# Some Qwen3.5/3.6 layers use a linear-attention (gated delta) cache that
# isn't trim-able. To roll back a rejected DFlash block suffix on those
# targets, we capture every GDN forward's inputs, then on rollback we
# re-execute `gated_delta_update` over the accepted prefix only — that
# rebuilds the correct post-accept state.
# ────────────────────────────────────────────────────────────────────────────


class GDNStateCapture:
    """Class-method-monkey-patch on `mlx_lm.models.qwen3_5.GatedDeltaNet` that
    records each GDN layer's inputs during a forward, so a subsequent
    `rollback()` can replay the accepted-prefix subset and restore cache
    state without trim support. Use as a context manager or call `close()`
    explicitly."""

    def __init__(self):
        self.conv_data = []
        self._gdn_inputs = []
        self._gdn_cls = None
        self._orig_call = None
        self._patched_call = None
        self._closed = False
        _GDN_PATCH_LOCK.acquire()
        try:
            self._patch()
        except Exception:
            _GDN_PATCH_LOCK.release()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _patch(self):
        from mlx_lm.models.qwen3_5 import GatedDeltaNet

        self._gdn_cls = GatedDeltaNet
        self._orig_call = GatedDeltaNet.__call__
        capture = self

        def _capturing_gdn_call(self_layer, inputs, mask=None, cache=None):
            B, S, _ = inputs.shape
            if self_layer.sharding_group is not None:
                from mlx_lm.models.qwen3_5 import sum_gradients

                inputs = sum_gradients(self_layer.sharding_group)(inputs)
            qkv = self_layer.in_proj_qkv(inputs)
            z = self_layer.in_proj_z(inputs).reshape(
                B, S, self_layer.num_v_heads, self_layer.head_v_dim
            )
            b, a = self_layer.in_proj_b(inputs), self_layer.in_proj_a(inputs)
            conv_state = (
                cache[0]
                if (cache is not None and cache[0] is not None)
                else mx.zeros(
                    (B, self_layer.conv_kernel_size - 1, self_layer.conv_dim),
                    dtype=inputs.dtype,
                )
            )
            if mask is not None:
                qkv = mx.where(mask[..., None], qkv, 0)
            conv_input = mx.concatenate([conv_state, qkv], axis=1)
            capture.conv_data.append((conv_input, self_layer.conv_kernel_size))
            if cache is not None:
                cache[0] = conv_input[:, -(self_layer.conv_kernel_size - 1):]
            conv_out = nn.silu(self_layer.conv1d(conv_input))
            q, k, v = [
                t.reshape(B, S, h, d)
                for t, h, d in zip(
                    mx.split(conv_out, [self_layer.key_dim, 2 * self_layer.key_dim], -1),
                    [self_layer.num_k_heads, self_layer.num_k_heads, self_layer.num_v_heads],
                    [self_layer.head_k_dim, self_layer.head_k_dim, self_layer.head_v_dim],
                )
            ]
            state = cache[1] if cache else None
            inv_scale = k.shape[-1] ** -0.5
            q = (inv_scale ** 2) * mx.fast.rms_norm(q, None, 1e-6)
            k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
            capture._gdn_inputs.append(
                (q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask)
            )
            out, new_state = _gd_mod.gated_delta_update(
                q, k, v, a, b, self_layer.A_log, self_layer.dt_bias, state, mask, use_kernel=True
            )
            if cache is not None:
                cache[1] = new_state
            out = self_layer.norm(out, z)
            out = self_layer.out_proj(out.reshape(B, S, -1))
            if self_layer.sharding_group is not None:
                out = mx.distributed.all_sum(out, group=self_layer.sharding_group)
            return out

        self._patched_call = _capturing_gdn_call
        GatedDeltaNet.__call__ = _capturing_gdn_call

    def clear(self):
        self.conv_data.clear()
        self._gdn_inputs.clear()

    def close(self):
        if self._closed:
            return
        try:
            if self._gdn_cls is not None and self._gdn_cls.__call__ is self._patched_call:
                self._gdn_cls.__call__ = self._orig_call
        finally:
            self._closed = True
            self._gdn_cls = None
            self._orig_call = None
            self._patched_call = None
            _GDN_PATCH_LOCK.release()

    def rollback(self, cache, accepted, trim):
        n_non_trimmable = sum(1 for c in cache if not c.is_trimmable())
        assert n_non_trimmable == len(self._gdn_inputs), (
            f"non-trimmable cache count ({n_non_trimmable}) != "
            f"captured GDN inputs ({len(self._gdn_inputs)}); "
            "DFlash MLX rollback assumes every non-trimmable cache is a "
            "GatedDeltaNet layer"
        )
        j = 0
        for c in cache:
            if c.is_trimmable():
                c.trim(trim)
            else:
                q, k, v, a, b, A_log, dt_bias, init_state, mask = self._gdn_inputs[j]
                n = accepted + 1
                _, state = _gd_mod.gated_delta_update(
                    q[:, :n], k[:, :n], v[:, :n], a[:, :n], b[:, :n],
                    A_log, dt_bias, init_state,
                    None if mask is None else mask[:, :n],
                    use_kernel=True,
                )
                c.cache[1] = state
                conv_input, K = self.conv_data[j]
                c.cache[0] = conv_input[:, accepted + 1: accepted + K]
                j += 1
