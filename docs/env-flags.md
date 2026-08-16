# Env flags

GENERATED — do not edit. Regenerate with `cargo xtask flags --docs`;
`cargo xtask flags --check` fails when this file is stale.

Source of truth is the `lumen_flags::flag!` declaration next to the code
each flag gates; this file is its projection. Parse rule for every flag:
unset → default, `"0"` → off, any other value → on.

| Env | Default | Kind | Declared in |
|---|---|---|---|
| `LUMEN_GEMMA4_FUSE_DENSE_MLP` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_dense_mlp` |
| `LUMEN_GEMMA4_FUSE_EXPERTS` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_experts` |
| `LUMEN_GEMMA4_FUSE_LAYER_EPILOGUE` | off | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_layer_epilogue` |
| `LUMEN_GEMMA4_FUSE_NORM_ROUTING_EXPERTS` | off | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_norm_routing_experts` |
| `LUMEN_GEMMA4_FUSE_PRE_POST_NORM_DENSE_MLP` | off | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_pre_post_norm_dense_mlp` |
| `LUMEN_GEMMA4_FUSE_ROUTER` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_router` |
| `LUMEN_GEMMA4_FUSE_ROUTING_EXPERTS` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_routing_experts` |
| `LUMEN_GEMMA4_FUSE_SOFTCAP` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_softcap` |
| `LUMEN_MLX_KV_BF16` | on | Behavior | `lumen_mlx::qwen3_5_moe::imp::kv_bf16` |
| `LUMEN_MLX_NO_OVERLAP` | off | Optimization | `lumen_mlx::gemma4_backend::imp::no_overlap` |
| `LUMEN_NATIVE_ALLOC_REUSE` | on | Optimization | `lumen_mlx::qwen3_5_moe::imp::alloc_reuse` |
| `LUMEN_NATIVE_COMPILE` | on | Optimization | `lumen_mlx::native_ssm::imp::ssm_compile` |
| `LUMEN_NATIVE_COMPILE_ROUTING` | off | Optimization | `lumen_mlx::native_moe::imp::routing_compile` |
| `LUMEN_NATIVE_CONV_SLICE` | off | Optimization | `lumen_mlx::qwen3_5_moe::imp::conv_slice` |
| `LUMEN_NATIVE_FUSE_LINATTN_IN` | off | Optimization | `lumen_mlx::qwen3_5_moe::imp::fuse_linattn_in` |
| `LUMEN_NATIVE_FUSE_SIGMOID_MUL` | off | Optimization | `lumen_mlx::native_kernels::imp::sigmoid_mul_fuse` |
| `LUMEN_NATIVE_FUSE_SWIGLU` | on | Optimization | `lumen_mlx::native_moe::imp::swiglu_fuse` |
| `LUMEN_NATIVE_KV_STEP_PREALLOC` | on | Optimization | `lumen_mlx::native_cache::imp::kv_step_prealloc` |
| `LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE` | on | Optimization | `lumen_mlx::qwen3_5_moe::imp::linear_attn_scale_fuse` |
| `LUMEN_NATIVE_RMS_NORM_GATED_FUSED` | off | Optimization | `lumen_mlx::native_ssm::imp::rms_norm_gated_fused` |
| `LUMEN_NATIVE_TIMING` | off | Diagnostic | `lumen_mlx::native_runtime::imp::fine_timing` |

## Details

### `LUMEN_GEMMA4_FUSE_DENSE_MLP`

*Optimization, default on.*

Fuse the dense-MLP chain. Output-identical.

### `LUMEN_GEMMA4_FUSE_EXPERTS`

*Optimization, default on.*

Fuse the expert gather + GLU. Output-identical.

### `LUMEN_GEMMA4_FUSE_LAYER_EPILOGUE`

*Optimization, default off.*

Fuse the per-layer epilogue. Default OFF — opt-in; land path is enable,
 measure, promote if positive.

### `LUMEN_GEMMA4_FUSE_NORM_ROUTING_EXPERTS`

*Optimization, default off.*

Fold the pre/post norms into the fused routing+experts path. Default OFF —
 opt-in; land path is enable, measure, promote if positive.

### `LUMEN_GEMMA4_FUSE_PRE_POST_NORM_DENSE_MLP`

*Optimization, default off.*

Fold the pre/post RMSNorm into the dense-MLP fusion. Default OFF — opt-in;
 land path is enable, measure, promote if positive.

### `LUMEN_GEMMA4_FUSE_ROUTER`

*Optimization, default on.*

Fuse the MoE router into one dispatch. Output-identical.

### `LUMEN_GEMMA4_FUSE_ROUTING_EXPERTS`

*Optimization, default on.*

Fuse routing into the expert dispatch, replacing the two-slot
 routing_tail + experts path. LANDED 2026-05-17, bit-identical with the
 baseline (greedy tokens match exactly); 3-pair cool-state A/B at 8K
 showed +2.2% throughput (53.35 +/- 0.5 -> 54.5 +/- 0.7 tok/s).

### `LUMEN_GEMMA4_FUSE_SOFTCAP`

*Optimization, default on.*

Fuse the attention logit softcap. Output-identical.

### `LUMEN_MLX_KV_BF16`

*Behavior, default on.*

Store the full-attention KV cache in bf16 rather than f32.

 Task 007 measured the cache at ~67 KB per allocated slot on
 Qwen3.5-9B, which matches the f32 arithmetic exactly (8 full-attn
 layers × 4 KV heads × 256 head_dim × 2 for K+V × 4 bytes =
 64.0 KB). Halving it is the largest memory lever left after that
 task ruled out PagedAttention — worth ~1.5 GB at eight concurrent
 long-context sequences, against paging's 35 MB — and decode gets
 +1.6–4.1% faster because attention reads half the bytes.

 **`Behavior`, not `Optimization`: this is a numerics change.** The
 cast lands after k_norm and RoPE, so attention runs in bf16
 end-to-end and greedy output may differ from f32 — the equivalence
 matrix must never flip it expecting identical output. Default ON
 since the quality pass: 6,300 teacher-forced positions across two
 models, 99.73–99.83% top-1 agreement against an f32-vs-f32 control
 of exactly 100.000%, flat across context depth, every disagreement
 below the 1.5th percentile of the top1−top2 logit gap. `=0`
 restores f32. Harnesses: `examples/kv_bf16_ab.rs`,
 `examples/kv_bf16_quality.rs`.

### `LUMEN_MLX_NO_OVERLAP`

*Optimization, default off.*

Disable overlap scheduling on the sampled decode path, restoring the
 original fully-synchronous order. Default OFF, i.e. overlap is on.

 Output-identical by construction: everything that affects the token
 stream — sampling, `all_tokens.push`, thinking-budget force-close and
 channel-block, the EOS check, the runaway check, the hard break —
 stays synchronous and in the original order. Only the parser advance,
 detokenisation and SSE send of the *previous* token are deferred, and
 the parser is consumed solely by `emit_token_event` / `finalize`.

### `LUMEN_NATIVE_ALLOC_REUSE`

*Optimization, default on.*

Reuse per-layer scratch allocations across decode steps instead of
 reallocating. Default ON (LANDED 2026-05-11). Prerequisite for
 `LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE` (the fused-weight constants
 live in the reused `LinearAttnConstants`).

### `LUMEN_NATIVE_COMPILE`

*Optimization, default on.*

compute_g persistent compile cache. Default ON (LANDED Phase 3):
 wins σ-significantly across a 12-run 4-condition matrix
 (Δp50=−0.33 ms, Δtps=+1.96 vs legacy direct dispatch). `=0` falls
 back to the legacy direct-ops path.

### `LUMEN_NATIVE_COMPILE_ROUTING`

*Optimization, default off.*

Compile-wrap the MoE routing graph. Default OFF (A/B WASH). (Parse
 note: previously only `=1` enabled this; the uniform rule now
 accepts any non-`"0"`.)

### `LUMEN_NATIVE_CONV_SLICE`

*Optimization, default off.*

Conv-state advance via `slice` (view/copy) instead of `take_axis`
 (gather kernel). Bit-identical for the s==1 decode regime — same
 indices, cheaper Metal kernel. Default OFF (A/B WASH).

### `LUMEN_NATIVE_FUSE_LINATTN_IN`

*Optimization, default off.*

Fused input-side linear-attn kernel (conv+silu+q/k-norm). Built to
 completion, output bit-identical, measured 0 speedup — MLX async
 already overlaps the small launches. Kept as a reusable artifact;
 default OFF. (Parse note: previously only `=1` enabled this; the
 registry's uniform rule now accepts any non-`"0"` value.)

### `LUMEN_NATIVE_FUSE_SIGMOID_MUL`

*Optimization, default off.*

Compile-wrapped `sigmoid(gate) * other`, bit-identical to the
 two-op composition. Default OFF (anti-pattern #30 calibration).
 (Parse note: previously only `=1` enabled this; the uniform rule
 now accepts any non-`"0"`.)

### `LUMEN_NATIVE_FUSE_SWIGLU`

*Optimization, default on.*

Fuse `silu(gate) * up` into one compile dispatch. Default ON
 2026-05-11 — net WIN −0.671 ms (Welch t=−29σ) at n=10 STEPS=300 on
 Qwen3.6-35B-A3B-mxfp4; closes ~62% of the Native-vs-PyO3 decode gap.

### `LUMEN_NATIVE_KV_STEP_PREALLOC`

*Optimization, default on.*

Grow the full-attn KV buffer in `KV_CACHE_STEP` (256) token blocks
 and fill via `slice_update`, instead of a per-step
 `concatenate_axis`. The legacy concat path stays live as the `=0`
 emergency revert.

 Default ON since Phase G2: microbench long-prompt 35B-mxfp4 n=12
 interleaved A/B (PROMPT_LEN=2048, STEPS=1500) showed Δtps
 +4.67 tok/s (Welch t = +4.33σ), Δp50 −1.02 ms/step (t = −4.68σ),
 with a 32-step PyO3 golden bit-identical PASS for both paths.

 This flag is the ancestor of the whole `lumen_flags` design: it was
 the only one of the audited 370 env gates whose alternate path a
 test could reach, via a hand-rolled thread-local override that the
 macro now generates for every flag.

### `LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE`

*Optimization, default on.*

Absorb `scale_q/k` into the rms_norm weight (bit-identical by
 linearity of rms_norm weight scaling). Default ON 2026-05-11:
 thermal-clean A/B (n=10, STEPS=100) Δ=−0.315 ms, Welch t=−3.33σ.
 Only active when `LUMEN_NATIVE_ALLOC_REUSE` is also on.

### `LUMEN_NATIVE_RMS_NORM_GATED_FUSED`

*Optimization, default off.*

Fused rms_norm + silu(gate) + multiply Metal kernel. Default OFF —
 reserved for super-kernel composition. (Parse note: previously only
 `=1` enabled this; the uniform rule now accepts any non-`"0"`.)

### `LUMEN_NATIVE_TIMING`

*Diagnostic, default off.*

Per-step decode timing capture (`take_native_decode_timing_log`).
 Costs an eval barrier per timed stage; default OFF. (Parse note:
 previously a truthy list `1|true|TRUE|yes`; the uniform rule now
 accepts any non-`"0"`.)
