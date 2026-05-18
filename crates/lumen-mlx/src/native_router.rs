//! Native MLX MoE router (top-k expert selection).
//!
//! Reproduces the Qwen3.5-MoE routing pipeline:
//!
//! ```text
//! probs   = softmax(gate_logits, axis=-1, precise=true)
//! inds    = argpartition(probs, kth=-top_k, axis=-1)[..., -top_k:]
//! scores  = take_along_axis(probs, inds, axis=-1)
//! scores  = scores / scores.sum(axis=-1, keepdims=true)
//! ```
//!
//! `precise=true` matches mlx-lm's `mx.softmax(...precise=True)` setting; this
//! is significant for the routing path because the ranking can flip on tied
//! logits when softmax is not run in the precise variant.
//!
//! The full MoE forward (per-expert MXFP4 matmul + scatter-add combine)
//! reuses primitives validated elsewhere — `quantized_matmul_with_mode`
//! (3a.3 / 3b.7) for expert ffn weights, and `take_axis` (3b.1 quantized
//! embedding) for packed-weight row selection. This module focuses on the
//! routing primitive itself, which is the new composition.
//!
//! See `.ai/memory/active/mlx-rs-native-port/CONTEXT.md` Session 21+.

#[cfg(feature = "mlx-native")]
#[allow(dead_code)] // `route_top_k` is a primitive used by tests + reserved for future router refactor callsites
mod imp {
    use anyhow::{Context, Result};
    use mlx_rs::Array;
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};

    /// Run the four-op MoE routing pipeline. Returns
    /// `(top_k_indices: i32, top_k_scores: f32)` with shape
    /// `[..., top_k]` matching the leading dims of `gate_logits`.
    pub fn route_top_k(gate_logits: &Array, top_k: i32) -> Result<(Array, Array)> {
        let last_axis = (gate_logits.ndim() as i32) - 1;
        let num_experts = gate_logits.shape()[last_axis as usize];

        // 1. probs = softmax(logits, axis=-1, precise=true)
        let probs =
            mlx_rs::ops::softmax_axis(gate_logits, last_axis, /* precise */ Some(true))
                .context("softmax_axis on gate_logits failed")?;

        // 2. argpartition with kth = -top_k → top_k largest land in the last
        //    `top_k` positions (order within the top-k slice is deterministic
        //    for a fixed MLX build but unspecified by API contract).
        //    MLX accepts the negative kth directly.
        let partitioned = mlx_rs::ops::argpartition_axis(&probs, -top_k, last_axis)
            .context("argpartition_axis on probs failed")?;

        // 3. Slice the last `top_k` indices: probs[..., num_experts - top_k ..].
        let start = num_experts - top_k;
        let inds = partitioned.index((Ellipsis, start..));

        // 4. scores = take_along_axis(probs, inds, axis=-1)
        let scores = probs
            .take_along_axis(&inds, last_axis)
            .context("take_along_axis on probs failed")?;

        // 5. Renormalize so per-token weights sum to 1.
        let scores_sum = scores
            .sum_axis(last_axis, /* keep_dims */ true)
            .context("sum_axis for renormalize failed")?;
        let scores_normalized = scores
            .divide(&scores_sum)
            .context("scores divide for renormalize failed")?;

        Ok((inds, scores_normalized))
    }
}

#[cfg(feature = "mlx-native")]
#[allow(unused_imports)] // Consumed by Phase 3b model assembly in runner_native.rs.
pub(crate) use imp::route_top_k;

// router top-k bit-identical vs MLX reference.
#[cfg(all(test, feature = "mlx-native"))]
mod parity_tests {
    use super::imp::route_top_k;
    use mlx_rs::Array;
    use std::path::{Path, PathBuf};

    const MAGIC: &[u8; 4] = b"TQRO";
    const HEADER_LEN: usize = 4 /* magic */ + 4 * 4 /* 4 u32 */;

    fn fixture_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("lumen-mlx manifest dir has a parent (crates/)")
            .join("lumen-metal")
            .join("tests")
            .join("fixtures")
    }

    fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
    }
    fn slice_f32(bytes: &[u8]) -> Vec<f32> {
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
    fn slice_i32(bytes: &[u8]) -> Vec<i32> {
        bytes
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    struct RouterFixture {
        batch: usize,
        seq: usize,
        num_experts: usize,
        top_k: usize,
        logits: Vec<f32>,
        expected_inds: Vec<i32>,
        expected_scores: Vec<f32>,
    }

    fn load_fixture(name: &str) -> Option<RouterFixture> {
        let path = fixture_dir().join(format!("embed_router_{name}.bin"));
        let bytes = std::fs::read(&path).ok()?;
        if bytes.len() < HEADER_LEN || &bytes[..4] != MAGIC {
            panic!("router fixture {path:?} has bad header");
        }
        let batch = read_u32_le(&bytes, 4) as usize;
        let seq = read_u32_le(&bytes, 8) as usize;
        let num_experts = read_u32_le(&bytes, 12) as usize;
        let top_k = read_u32_le(&bytes, 16) as usize;

        let logits_count = batch * seq * num_experts;
        let topk_count = batch * seq * top_k;
        let mut cur = HEADER_LEN;
        let logits = slice_f32(&bytes[cur..cur + logits_count * 4]);
        cur += logits_count * 4;
        let expected_inds = slice_i32(&bytes[cur..cur + topk_count * 4]);
        cur += topk_count * 4;
        let expected_scores = slice_f32(&bytes[cur..cur + topk_count * 4]);

        Some(RouterFixture {
            batch,
            seq,
            num_experts,
            top_k,
            logits,
            expected_inds,
            expected_scores,
        })
    }

    fn run_fixture(name: &str) {
        let Some(fx) = load_fixture(name) else {
            eprintln!(
                "SKIP router/{name}: fixture missing under {}. Run `python scripts/generate_router_fixture.py` first.",
                fixture_dir().display()
            );
            return;
        };

        let logits = Array::from_slice(
            &fx.logits,
            &[fx.batch as i32, fx.seq as i32, fx.num_experts as i32],
        );

        let (inds, scores) =
            route_top_k(&logits, fx.top_k as i32).expect("route_top_k pipeline must succeed");

        assert_eq!(
            inds.shape(),
            &[fx.batch as i32, fx.seq as i32, fx.top_k as i32],
            "{name}: inds shape mismatch"
        );
        assert_eq!(
            scores.shape(),
            &[fx.batch as i32, fx.seq as i32, fx.top_k as i32],
            "{name}: scores shape mismatch"
        );

        // Coerce to canonical dtypes (i32 / f32) — fixture stores i32 / f32.
        let inds = inds
            .as_dtype(mlx_rs::Dtype::Int32)
            .expect("inds → i32 cast must succeed");
        let scores = scores
            .as_dtype(mlx_rs::Dtype::Float32)
            .expect("scores → f32 cast must succeed");
        inds.eval().expect("mlx eval (inds) must succeed");
        scores.eval().expect("mlx eval (scores) must succeed");

        let observed_inds: &[i32] = inds.as_slice();
        let observed_scores: &[f32] = scores.as_slice();

        assert_eq!(
            observed_inds.len(),
            fx.expected_inds.len(),
            "{name}: inds flat-size mismatch"
        );
        assert_eq!(
            observed_scores.len(),
            fx.expected_scores.len(),
            "{name}: scores flat-size mismatch"
        );

        let mut ind_mismatches = 0usize;
        for (i, (&got, &expected)) in observed_inds
            .iter()
            .zip(fx.expected_inds.iter())
            .enumerate()
        {
            if got != expected {
                if ind_mismatches < 5 {
                    eprintln!("router/{name}.inds[{i}]: {got} vs ref {expected}");
                }
                ind_mismatches += 1;
            }
        }
        assert_eq!(
            ind_mismatches, 0,
            "{name}: {ind_mismatches} index mismatches"
        );

        let mut score_mismatches = 0usize;
        for (i, (&got, &expected)) in observed_scores
            .iter()
            .zip(fx.expected_scores.iter())
            .enumerate()
        {
            if got.to_bits() != expected.to_bits() {
                if score_mismatches < 5 {
                    eprintln!(
                        "router/{name}.scores[{i}]: 0x{:08x} ({got}) vs ref 0x{:08x} ({expected})",
                        got.to_bits(),
                        expected.to_bits()
                    );
                }
                score_mismatches += 1;
            }
        }
        assert_eq!(
            score_mismatches, 0,
            "{name}: {score_mismatches} score mismatches"
        );
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn tiny_router_matches_mlx() {
        run_fixture("tiny");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn qwen_prod_router_matches_mlx() {
        run_fixture("qwen_prod");
    }

    #[test]
    #[ignore = "MLX FFI requires non-sandbox host with Metal device"]
    fn decode_batched_router_matches_mlx() {
        run_fixture("decode_batched");
    }
}
