//! Step-by-step parity check against MLX traces for the Qwen3-Next SparseMoeBlock.
//!
//! When [`moe_fixture`] disagrees with MLX at the block level, this test localizes the gap
//! stage-by-stage so we know exactly which step to blame — gate logits, softmax, top-k
//! selection, score normalization, routed sum, shared expert, scalar gate, or final add.
//!
//! Skipped when the weights fixture is missing (CI without the HF checkpoint).

use std::collections::HashMap;
use std::path::PathBuf;

use candle_core::{safetensors as cst, DType, Device, Tensor, D};
use candle_nn::{Linear, Module};
use lumen_model::qwen3_5_moe::moe::{
    MoeDims, SharedExpert, SparseMoeBlock, SparseMoeRuntime, SwitchMlp,
};

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures")
}

fn take_f32(map: &HashMap<String, Tensor>, name: &str) -> Tensor {
    let t = map.get(name).unwrap_or_else(|| panic!("missing `{name}`"));
    assert_eq!(t.dtype(), DType::F32, "`{name}` dtype");
    t.clone()
}

fn take_u32(map: &HashMap<String, Tensor>, name: &str) -> Tensor {
    let t = map.get(name).unwrap_or_else(|| panic!("missing `{name}`"));
    assert_eq!(t.dtype(), DType::U32, "`{name}` dtype");
    t.clone()
}

fn rel_l2(a: &Tensor, b: &Tensor) -> f32 {
    let d = (a - b).unwrap();
    let n = d.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt();
    let de = b.sqr().unwrap().sum_all().unwrap().to_scalar::<f32>().unwrap().sqrt().max(1e-12);
    n / de
}

fn max_abs(a: &Tensor, b: &Tensor) -> f32 {
    (a - b)
        .unwrap()
        .abs()
        .unwrap()
        .max_all()
        .unwrap()
        .to_scalar::<f32>()
        .unwrap()
}

#[test]
fn sparse_moe_trace_matches_mlx_layer0() {
    let dir = fixtures_dir();
    let weights_path = dir.join("layer0_moe_weights.safetensors");
    let trace_path = dir.join("layer0_moe_trace.safetensors");
    if !weights_path.exists() || !trace_path.exists() {
        eprintln!(
            "skipping: regenerate fixtures with\n  \
             python scripts/dump_qwen3_5_moe_layer_weights.py --layer 0 --block moe\n  \
             python scripts/dump_moe_trace.py"
        );
        return;
    }

    let device = Device::Cpu;
    let weights = cst::load(&weights_path, &device).unwrap();
    let trace = cst::load(&trace_path, &device).unwrap();

    let dims = MoeDims {
        hidden_size: 2048,
        num_experts: 256,
        moe_intermediate_size: 512,
        shared_expert_intermediate_size: 512,
    };
    let rt = SparseMoeRuntime {
        dims,
        top_k: 8,
        norm_topk_prob: true,
    };

    let gate = Linear::new(take_f32(&weights, "gate.weight"), None);
    let shared_expert_gate_l = Linear::new(take_f32(&weights, "shared_expert_gate.weight"), None);
    let se_gate = Linear::new(take_f32(&weights, "shared_expert.gate_proj.weight"), None);
    let se_up = Linear::new(take_f32(&weights, "shared_expert.up_proj.weight"), None);
    let se_down = Linear::new(take_f32(&weights, "shared_expert.down_proj.weight"), None);
    let se_gate_up = Linear::new(
        Tensor::cat(&[se_gate.weight(), se_up.weight()], 0).unwrap(),
        None,
    );
    let shared = SharedExpert::new(
        se_gate_up.into(),
        se_down.into(),
        dims.shared_expert_intermediate_size,
    );
    let switch = SwitchMlp::new(
        take_f32(&weights, "switch_mlp.gate_proj.weight"),
        take_f32(&weights, "switch_mlp.up_proj.weight"),
        take_f32(&weights, "switch_mlp.down_proj.weight"),
        dims,
    )
    .unwrap();

    let x_post_ln = take_f32(&trace, "x_post_ln"); // [L, hidden]
    let (seq, hidden) = (x_post_ln.dim(0).unwrap(), x_post_ln.dim(1).unwrap());
    assert_eq!(hidden, 2048);

    // 1. Gate logits.
    let our_logits = gate.forward(&x_post_ln).unwrap();
    let ref_logits = take_f32(&trace, "gates_logits");
    let gate_err = rel_l2(&our_logits, &ref_logits);
    let gate_max = max_abs(&our_logits, &ref_logits);
    println!("1. gate logits:     rel_L2 = {gate_err:.3e}, max_abs = {gate_max:.3e}");
    assert!(gate_err < 1e-2, "gate logits diverge");

    // 2. Softmax.
    let our_probs = candle_nn::ops::softmax_last_dim(&our_logits).unwrap();
    let ref_probs = take_f32(&trace, "gates_softmax");
    let sm_err = rel_l2(&our_probs, &ref_probs);
    let sm_max = max_abs(&our_probs, &ref_probs);
    println!("2. softmax probs:   rel_L2 = {sm_err:.3e}, max_abs = {sm_max:.3e}");
    assert!(sm_err < 1e-2, "softmax diverges");

    // 3. Top-k indices: check whether the *sets* agree (ordering may differ).
    let our_sorted = our_probs.arg_sort_last_dim(false).unwrap();
    let our_inds = our_sorted.narrow(D::Minus1, 0, rt.top_k).unwrap().contiguous().unwrap();
    let ref_inds = take_u32(&trace, "top_k_inds");
    let our_inds_v = our_inds.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    let ref_inds_v = ref_inds.flatten_all().unwrap().to_vec1::<u32>().unwrap();
    // When two experts' softmax probabilities are within bf16's ~4e-3 relative resolution,
    // MLX (bf16) and our f32 path can legitimately pick different top-k members. That's a
    // fundamental precision boundary, not a Rust bug. For each disagreement, print the
    // probability gap between the swapped experts to confirm it's genuinely a near-tie.
    let our_probs_v = our_probs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let ref_probs_v = ref_probs.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let n_experts = dims.num_experts;
    let mut mismatched_sets = 0;
    let mut near_tie_gaps: Vec<f32> = vec![];
    for t in 0..seq {
        let ours: std::collections::BTreeSet<u32> =
            our_inds_v[t * rt.top_k..(t + 1) * rt.top_k].iter().copied().collect();
        let refs: std::collections::BTreeSet<u32> =
            ref_inds_v[t * rt.top_k..(t + 1) * rt.top_k].iter().copied().collect();
        if ours != refs {
            mismatched_sets += 1;
            let only_ours: Vec<u32> = ours.difference(&refs).copied().collect();
            let only_refs: Vec<u32> = refs.difference(&ours).copied().collect();
            // Pair them up and report the f32 probability gap.
            for (&a, &b) in only_ours.iter().zip(only_refs.iter()) {
                let p_ours_a = our_probs_v[t * n_experts + a as usize];
                let p_ours_b = our_probs_v[t * n_experts + b as usize];
                let p_ref_a = ref_probs_v[t * n_experts + a as usize];
                let p_ref_b = ref_probs_v[t * n_experts + b as usize];
                let gap = (p_ours_a - p_ours_b).abs();
                near_tie_gaps.push(gap);
                println!(
                    "   token {t}: we chose expert {a} (p_f32={p_ours_a:.5e}, p_bf16={p_ref_a:.5e}), \
                     MLX chose {b} (p_f32={p_ours_b:.5e}, p_bf16={p_ref_b:.5e}); gap = {gap:.3e}"
                );
            }
        }
    }
    println!(
        "3. top-k set agreement: {} / {seq} tokens match (mismatches: {mismatched_sets})",
        seq - mismatched_sets
    );
    // Mismatches are permitted only when the probability gap is small enough to be within
    // bf16's ~4e-3 relative resolution. A large gap would indicate a real bug in our path.
    for g in &near_tie_gaps {
        assert!(
            *g < 5e-3,
            "top-k disagreement with prob gap {g:.3e} is larger than bf16 rounding (~4e-3); \
             this looks like a real routing bug, not a precision tie-break."
        );
    }

    // 4. Scores (gathered + renormalized). Because ordering differs inside the top-k slice
    //    and membership may differ on near-tie tokens, we compare only experts that appear
    //    in BOTH lists; a disagreement there would indicate a gather/normalization bug.
    let our_scores = our_probs.gather(&our_inds, D::Minus1).unwrap();
    let denom = our_scores.sum_keepdim(D::Minus1).unwrap();
    let our_scores_norm = our_scores.broadcast_div(&denom).unwrap();
    let ref_scores = take_f32(&trace, "top_k_scores");
    let our_scores_v = our_scores_norm.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let ref_scores_v = ref_scores.flatten_all().unwrap().to_vec1::<f32>().unwrap();
    let mut max_shared_score_diff = 0f32;
    for t in 0..seq {
        let ours: HashMap<u32, f32> = our_inds_v[t * rt.top_k..(t + 1) * rt.top_k]
            .iter()
            .copied()
            .zip(our_scores_v[t * rt.top_k..(t + 1) * rt.top_k].iter().copied())
            .collect();
        let refs: HashMap<u32, f32> = ref_inds_v[t * rt.top_k..(t + 1) * rt.top_k]
            .iter()
            .copied()
            .zip(ref_scores_v[t * rt.top_k..(t + 1) * rt.top_k].iter().copied())
            .collect();
        for (e, o) in &ours {
            if let Some(r) = refs.get(e) {
                let d = (o - r).abs();
                if d > max_shared_score_diff {
                    max_shared_score_diff = d;
                }
            }
        }
    }
    println!("4. renormalized scores (intersection): max_abs = {max_shared_score_diff:.3e}");
    // Loose because renorm divides by a sum that itself differs when the top-k set differs.
    assert!(
        max_shared_score_diff < 2e-2,
        "per-expert renormalized scores diverge on shared experts"
    );

    // 5. Shared expert out.
    let gate_up = shared.gate_up_proj.forward(&x_post_ln).unwrap();
    let our_shared_out = gate_up
        .narrow(1, 0, dims.shared_expert_intermediate_size)
        .unwrap()
        .contiguous()
        .unwrap();
    let our_shared_up = gate_up
        .narrow(
            1,
            dims.shared_expert_intermediate_size,
            dims.shared_expert_intermediate_size,
        )
        .unwrap()
        .contiguous()
        .unwrap();
    let our_shared_h = (candle_nn::ops::silu(&our_shared_out).unwrap() * our_shared_up).unwrap();
    let our_shared_down = shared.down_proj.forward(&our_shared_h).unwrap();
    let ref_shared_out = take_f32(&trace, "shared_out");
    let so_err = rel_l2(&our_shared_down, &ref_shared_out);
    println!("5. shared_expert out: rel_L2 = {so_err:.3e}");
    assert!(so_err < 1e-2, "shared expert diverges");

    // 6. Shared gate logit → sigmoid → scalar mixing.
    let our_shared_logit = shared_expert_gate_l.forward(&x_post_ln).unwrap();
    let ref_shared_logit = take_f32(&trace, "shared_gate_logit");
    let sl_err = rel_l2(&our_shared_logit, &ref_shared_logit);
    println!("6. shared gate logit: rel_L2 = {sl_err:.3e}");
    assert!(sl_err < 1e-2);

    let our_coef = candle_nn::ops::sigmoid(&our_shared_logit).unwrap();
    let ref_coef = take_f32(&trace, "shared_coef");
    let sc_err = rel_l2(&our_coef, &ref_coef);
    println!("6b. sigmoid coef:     rel_L2 = {sc_err:.3e}");
    assert!(sc_err < 1e-2);

    let our_shared_y = our_shared_down.broadcast_mul(&our_coef).unwrap();
    let ref_shared_y = take_f32(&trace, "shared_y");
    let sy_err = rel_l2(&our_shared_y, &ref_shared_y);
    println!("7. shared_y:           rel_L2 = {sy_err:.3e}");
    assert!(sy_err < 1e-2);

    // 8. End-to-end forward.
    let block = SparseMoeBlock::new(
        rt,
        gate.into(),
        shared_expert_gate_l.into(),
        shared,
        switch.into(),
    );
    let x_in = x_post_ln.unsqueeze(0).unwrap();
    let y_ours = block.forward(&x_in).unwrap().squeeze(0).unwrap();
    let y_ref = take_f32(&trace, "y_final");
    let y_err = rel_l2(&y_ours, &y_ref);
    let y_max = max_abs(&y_ours, &y_ref);
    println!("8. y_final:            rel_L2 = {y_err:.3e}, max_abs = {y_max:.3e}");

    // 9. Isolated routed branch: y - shared_y.
    let ours_routed = (&y_ours - &our_shared_y).unwrap();
    let ref_routed = take_f32(&trace, "y_routed");
    let routed_err = rel_l2(&ours_routed, &ref_routed);
    let routed_max = max_abs(&ours_routed, &ref_routed);
    println!("9. y_routed:           rel_L2 = {routed_err:.3e}, max_abs = {routed_max:.3e}");
}
