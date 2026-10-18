#!/usr/bin/env python3
"""Tiny random-weights replica of a Qwen3.8-Flash-Next (qwen4_exp) MLX 4-bit checkpoint.

Builds the model with the vendored reference implementation (Slotstream's
Tools/reference/qwen4_exp.py, ported from the HF architecture), quantizes it the way the
mlx-community build is quantized (affine 4-bit group 32; router and shared-expert gate 8-bit
group 64), and writes:

  <out>/config.json            same schema as the real checkpoint, tiny sizes
  <out>/model.safetensors      real tensor names (language_model.model.layers.N...)
  <out>/reference/outputs.safetensors   ids + reference logits (full prefill, and prefill+decode) +
                               the residual stream after every layer, all float32

The weights are float32 (scales and biases too), so both the reference and the Rust port run
in float32 and agree to rounding error rather than bf16 noise.

usage: python scripts/golden/qwen4exp_tiny_fixture.py <reference_dir> crates/lumen-mlx/tests/fixtures/qwen4exp_tiny
       <reference_dir> is Slotstream's Tools/reference (github.com/carloslfu/slotstream, MIT);
       it needs mlx and mlx-lm (any recent version, e.g. the oMLX venv's).

The output is committed: the Rust tests (src/qwen4_exp/tests.rs) read it, so this only needs
re-running if the reference or the replica's shape changes. Seeds are fixed, so a rerun
reproduces the same bytes.
"""
import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

ref_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(ref_dir))
import qwen4_exp as q  # noqa: E402

SEED = 7
mx.random.seed(SEED)
rng = np.random.default_rng(SEED)

TEXT = dict(
    model_type="qwen4_exp_text",
    hidden_size=64,
    num_hidden_layers=8,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=32,
    vocab_size=128,
    rms_norm_eps=1e-6,
    full_attention_interval=4,
    num_experts=16,
    num_experts_per_tok=4,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_key_head_dim=32,
    linear_value_head_dim=32,
    linear_conv_kernel_dim=4,
    output_gate_type="sigmoid",
    hc_count=4,
    hc_lowrank=32,
    indexer_n_heads=2,
    indexer_kv_heads=1,
    indexer_head_dim=32,
    indexer_budget=16,
    indexer_compress_ratio=4,
    ngram_size=3,
    heads_per_ngram=2,
    ngram_vocab_size_base=1000,
    make_ngram_vocab_size_divisible_by=8,
    split_ngram_parts=4,
    ple_embed_dim=128,
    ple_layer_ids=[2],
    ple_conv_kernel_size=4,
    partial_rotary_factor=0.25,
    eos_token_id=7,
    bos_token_id=7,
    tie_word_embeddings=False,
    rope_parameters=dict(rope_type="default", rope_theta=10000.0, partial_rotary_factor=0.25),
)
TEXT["layer_types"] = [
    "full_attention" if (i + 1) % TEXT["full_attention_interval"] == 0 else "linear_attention"
    for i in range(TEXT["num_hidden_layers"])
]

args = q.ModelArgs(text_config=TEXT)
model = q.Model(args)

# Random but well-conditioned weights: the default inits leave A_log/dt_bias constant and the
# norms at one, which would hide wiring mistakes behind symmetric values.
def reinit(params, path=""):
    out = {}
    for k, v in params.items():
        p = f"{path}.{k}" if path else k
        if isinstance(v, dict):
            out[k] = reinit(v, p)
        elif isinstance(v, list):
            out[k] = [reinit(x, f"{p}.{i}") if isinstance(x, dict) else x for i, x in enumerate(v)]
        else:
            out[k] = v
    return out

flat = dict(tree_flatten(model.parameters()))
new = {}
INTEGER = ("layer_multipliers", "ngram_heads_vocab_sizes", "ngram_heads_offsets")
for name, w in flat.items():
    if name.endswith(INTEGER):
        new[name] = w
    elif name.endswith("A_log"):
        new[name] = mx.array(np.log(rng.uniform(1.0, 8.0, w.shape)).astype(np.float32))
    elif name.endswith("dt_bias"):
        new[name] = mx.array(rng.normal(0.0, 0.5, w.shape).astype(np.float32))
    elif name.endswith("norm.weight") or name.endswith("layernorm.weight") or name.endswith("_norm.weight") or "norm_" in name.split(".")[-2]:
        new[name] = mx.array(rng.uniform(0.7, 1.3, w.shape).astype(np.float32))
    elif name.endswith("conv1d.weight"):
        new[name] = mx.array(rng.normal(0.0, 0.4, w.shape).astype(np.float32))
    elif "ngram_embedding" in name:
        new[name] = mx.array(rng.normal(0.0, 1.0, w.shape).astype(np.float32))
    elif w.ndim >= 2:
        fan_in = w.shape[-1]
        new[name] = mx.array(rng.normal(0.0, 1.0 / np.sqrt(fan_in), w.shape).astype(np.float32))
    else:
        new[name] = w
model.load_weights(list(new.items()), strict=True)
model.eval()


def plan(path, module):
    if isinstance(module, (nn.Linear, nn.Embedding)) or hasattr(module, "to_quantized"):
        if path.endswith("mlp.gate") or path.endswith("mlp.shared_expert_gate"):
            return {"group_size": 64, "bits": 8, "mode": "affine"}
        return {"group_size": 32, "bits": 4, "mode": "affine"}
    return False


nn.quantize(model, group_size=32, bits=4, class_predicate=plan)
mx.eval(model.parameters())

# ---- reference outputs ------------------------------------------------------------------
N_PROMPT, N_DECODE = 40, 8
ids_np = rng.integers(0, TEXT["vocab_size"], size=(1, N_PROMPT + N_DECODE)).astype(np.int32)
# put an EOS inside the prompt so the n-gram segment-reset logic is exercised
ids_np[0, 11] = TEXT["eos_token_id"]
ids = mx.array(ids_np)

hidden_after = {}


def run(tokens, cache):
    return model(tokens, cache=cache)


# 1) one full prefill without a cache
full = model(ids, cache=None)
mx.eval(full)

# 2) prefill + one-token decode steps with caches
cache = model.make_cache()
logits_split = [model(ids[:, :N_PROMPT], cache=cache)]
for t in range(N_DECODE):
    logits_split.append(model(ids[:, N_PROMPT + t : N_PROMPT + t + 1], cache=cache))
split = mx.concatenate(logits_split, axis=1)
mx.eval(split)
err = float(mx.max(mx.abs(split - full)))
print(f"reference self-consistency, prefill+decode vs full forward: max |diff| = {err:.3e}")

# 3) residual stream after every layer (single full pass, hooked)
inner = model.model
captured_gids = []
_orig_call = q._ShardedEmbedding.__call__
def _capture(self, gid):
    captured_gids.append(gid)
    return _orig_call(self, gid)
q._ShardedEmbedding.__call__ = _capture
h = inner.embed_tokens(ids)
conv_mask = None
mask = q.create_attention_mask(h, None)
prev_ctx = mx.full((1, TEXT["ngram_size"] - 1), TEXT["eos_token_id"], ids.dtype)
h = mx.tile(h, (1, 1, inner.hc))
streams = {"embed_tiled": h}
for i, layer in enumerate(inner.layers):
    h = layer(h, inner.rope, mask, conv_mask, None, None, ids, prev_ctx)
    streams[f"after_layer_{i}"] = h
q._ShardedEmbedding.__call__ = _orig_call
final = inner.hyper_connection_mixer(h)
streams["final_mixed"] = final
mx.eval(list(streams.values()))
head = model.lm_head(final)
print("hooked pass vs full forward max |diff| =", float(mx.max(mx.abs(head - full))))

# The sparse path has to matter, or a port that ignored the indexer would pass: rerun with an
# unreachable budget (dense attention) and report how far the logits move.
saved = []
for layer in inner.layers:
    if layer.layer_type == "full_attention":
        saved.append((layer.self_attn.indexer, layer.self_attn.indexer.token_budget))
        layer.self_attn.indexer.token_budget = 10**9
dense = model(ids, cache=None)
mx.eval(dense)
for idx, budget in saved:
    idx.token_budget = budget
print("QSA sparse vs dense attention on this prompt: max |logit diff| =", float(mx.max(mx.abs(dense - full))))

out_dir.mkdir(parents=True, exist_ok=True)
import re

# The real checkpoint names the n-gram shards `shards.<i>`; the reference module calls them
# `shard_<i>` (setattr). The converter renames them, so do the same.
def checkpoint_name(k):
    return "language_model." + re.sub(r"ngram_embedding\.shard_(\d+)\.", r"ngram_embedding.shards.\1.", k)

weights = {checkpoint_name(k): v for k, v in tree_flatten(model.parameters())}
mx.save_safetensors(str(out_dir / "model.safetensors"), weights)

quant = {"group_size": 32, "bits": 4, "mode": "affine"}
for i in range(TEXT["num_hidden_layers"]):
    for leaf in ("mlp.gate", "mlp.shared_expert_gate"):
        quant[f"language_model.model.layers.{i}.{leaf}"] = {"group_size": 64, "bits": 8, "mode": "affine"}
config = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
    "language_model_only": True,
    "tie_word_embeddings": False,
    "eos_token_id": TEXT["eos_token_id"],
    "text_config": {**TEXT, "dtype": "float32"},
    "quantization": quant,
    "quantization_config": quant,
}
(out_dir / "config.json").write_text(json.dumps(config, indent=1))

fixture = {"ngram_gids": captured_gids[0].astype(mx.int64), "ids": mx.array(ids_np), "logits_full": full.astype(mx.float32), "logits_split": split.astype(mx.float32)}
for k, v in streams.items():
    fixture[k] = v.astype(mx.float32)
(out_dir / "reference").mkdir(exist_ok=True)
mx.save_safetensors(str(out_dir / "reference" / "outputs.safetensors"), fixture)
print("wrote", out_dir, "tensors:", len(weights), "prompt", N_PROMPT, "decode", N_DECODE)
print("top-1 of last position:", int(mx.argmax(full[0, -1])))
