// Phase 13 (`affine4_qmv_fast_gate_up_silu_mul_bf16in_bf16out`) was concluded
// NEGATIVE on the bf16 chain — qmv_fast + separate Candle silu_mul stayed at
// the BW/register pareto front. The kernel was removed from
// `affine4.metal` and its Rust wrappers in `affine4_gpu.rs` /
// `affine4_linear.rs` / `proj.rs`. This file used to host the parity tests
// for that kernel; it is intentionally left empty as a stub so the deletion
// is reflected in git history. Safe to delete on next commit.
