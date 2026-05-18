# Archived diagnostic benches

These bench examples chased hypotheses that were FALSIFIED, RESOLVED, or
SUPERSEDED during Phase 1.5 through Phase 1.9 work on the Gemma 4 26B-A4B
MLX-native path. They are preserved here for historical reference and
hypothesis replay; Cargo's example auto-discovery does NOT pick up subdirs,
so they do not build with the workspace.

| File | Phase | Hypothesis | Outcome |
|------|-------|-----------|---------|
| `bench_compile_overhead.rs` | 1, Task 003 | A5 compile slim-down would reduce per-call alloc cost | FALSIFIED — no net gain vs GPU launch regression (`notes/a5_compile_path_dead.md`) |
| `bench_full_attn_microbench.rs` | 1.8 post-M4.8 | full-attention head_dim=512 was 18× slower per KV-token | FALSIFIED — microbench showed only 1.4-1.6× FFI overhead (~1 ms/step). Gap closed at parity by Phase 1.8 RESOLVED. |
| `bench_ffi_alloc_cost.rs` | 1.5 | per-call `mlx_array_new` + drop FFI was dominant | SUBSUMED by G6-G op-count instrumentation in Phase 1.9 |
| `bench_drift_S_vs_1.rs` | 2.0 (track A) | lookahead K=3 single-tower batch-vs-step drift | OBSOLETE — single-tower lookahead path abandoned in favor of external draft |
| `bench_quantized_matmul_isolated.rs` | step 2 microbench | stream/sentinel-cache savings for quantized_matmul | FALSIFIED — predicted savings sub-noise (`notes/divergence_d4_conv_slice.md`) |
| `bench_mul_by_2_roundtrip.rs` | 1.8 M1.5 / M2.0 | Array↔MTL::Buffer bridge + bf16 proof | SUPERSEDED — M4.8 (`lumen_flash_attn_bf16` primitive) replaces the bridge-dispatch pattern entirely; the proof kernels are no longer reachable in production code |
| `bench_rms_norm_bf16_parity.rs` | 1.8 M2.2 / M2.4 | Standalone & fused rms_norm bf16 parity | SUPERSEDED — `LUMEN_GEMMA4_FUSED_QKNORM` gate is perf-DEFERRED + default OFF; parity was bit-identical at landing; if FUSED_QKNORM is ever revived, restore the transpose-half of this bench |

To replay: move the file back to `crates/lumen-mlx/examples/` and rebuild.
