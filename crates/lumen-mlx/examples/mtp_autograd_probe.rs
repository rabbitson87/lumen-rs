//! PoC step (a): does autograd flow through the FULL dense MTP-block op set
//! into a LoRA placed on an EARLY internal linear (eh_proj)?
//!
//! This is the feasibility gate for "head-internal" MTP fine-tuning (the only
//! untested accept lever for dense Qwen — memory `mtplx-gdn-capture-commit-
//! analysis` + `mtplx-lever-catalog`). The dead C4 lm_head adapter trained on a
//! FROZEN post-block hidden (couldn't generalize). To change the hidden, a LoRA
//! must sit INSIDE the block and gradients must flow back through the block's
//! op chain: quantized_matmul (×8), fast::rope, fast::scaled_dot_product_
//! attention, fast::rms_norm, sigmoid/multiply/add. If those are autograd-
//! compatible, head-internal training needs NO Metal reimpl (MEDIUM effort);
//! if the chain breaks (a custom kernel with no vjp), it needs a bf16 reimpl
//! (multi-day).
//!
//! PASS = nonzero LoRA gradients (chain differentiable) AND CE decreases over a
//! few Adam steps (usable training signal). This uses the EXACT mlx_rs public
//! ops lumen's block wraps (`mlx_rs::fast::*`, `mlx_rs::ops::quantized_matmul`).
//!
//! Run: cargo run -p lumen-mlx --features mlx-native --example mtp_autograd_probe

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("mtp_autograd_probe requires --features mlx-native");
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use mlx_rs::Array;
    use mlx_rs::fast::{rms_norm, rope, scaled_dot_product_attention};
    use mlx_rs::ops::{quantize, quantized_matmul};
    use mlx_rs::transforms::value_and_grad_with_argnums;

    // ── mini dense-MTP-block dims (group_size=64 → all in-dims divisible) ──
    let (n, h, heads, hd, kv, ff, v, r) = (64i32, 64i32, 4i32, 16i32, 2i32, 128i32, 48i32, 4i32);
    let (gs, bits, eps, scale) = (64i32, 4i32, 1e-6f32, (hd as f32).powf(-0.5));
    mlx_rs::random::seed(11)?;

    // Frozen quantized weights for every linear in the block. q_*[out,in],
    // matmul uses transpose=true (x[.,in] @ Wᵀ → [.,out]).
    let mkq = |out: i32, in_: i32| -> anyhow::Result<(Array, Array, Array)> {
        let w = mlx_rs::random::normal::<f32>(&[out, in_], None, None, None)?
            .multiply(Array::from_f32(0.08))?;
        let (p, s, b) = quantize(&w, Some(gs), Some(bits))?;
        Ok((p, s, b))
    };
    let eh = mkq(h, 2 * h)?; // eh_proj: [h, 2h]  ← LoRA target
    let qw = mkq(heads * hd, h)?;
    let kw = mkq(kv * hd, h)?;
    let vw = mkq(kv * hd, h)?;
    let ow = mkq(h, heads * hd)?;
    let gw = mkq(ff, h)?;
    let uw = mkq(ff, h)?;
    let dw = mkq(h, ff)?;
    let lm = mkq(v, h)?;
    let _qmm = |x: &Array, q: &(Array, Array, Array)| -> anyhow::Result<Array> {
        quantized_matmul(x, &q.0, &q.1, Some(&q.2), Some(true), Some(gs), Some(bits))
            .map_err(Into::into)
    };

    // RMSNorm weights (≈1), constant.
    let ones = |d: i32| Array::ones::<f32>(&[d]);
    let (w_in, w_post, w_head, qn, kn) = (ones(h)?, ones(h)?, ones(h)?, ones(hd)?, ones(hd)?);

    // Constant block input: a [n,1,2h] "concat(enorm(embed), hnorm(hidden))".
    let concat = mlx_rs::random::normal::<f32>(&[n, 1, 2 * h], None, None, None)?;
    concat.eval()?;

    // Targets = argmax of an UNRELATED random logit set → base is "wrong",
    // so CE has a real gradient the LoRA can chase.
    let targets = {
        let rl = mlx_rs::random::normal::<f32>(&[n, v], None, None, None)?;
        mlx_rs::ops::indexing::argmax_axis(&rl, 1, false)?.as_dtype(mlx_rs::Dtype::Int32)?
    };
    targets.eval()?;

    // Full dense-block forward as a closure over the frozen weights, with a
    // trainable LoRA (A[r,2h], B[h,r]) added to eh_proj's output.
    let fwd = {
        let (concat, targets) = (concat.clone(), targets.clone());
        let (eh, qw, kw, vw, ow, gw, uw, dw, lm) = (
            eh.clone(),
            qw.clone(),
            kw.clone(),
            vw.clone(),
            ow.clone(),
            gw.clone(),
            uw.clone(),
            dw.clone(),
            lm.clone(),
        );
        let (w_in, w_post, w_head, qn, kn) = (
            w_in.clone(),
            w_post.clone(),
            w_head.clone(),
            qn.clone(),
            kn.clone(),
        );
        move |args: &[Array]| -> Vec<Array> {
            let a = &args[0]; // [r, 2h]
            let b = &args[1]; // [h, r]
            let g = |x: &Array, q: &(Array, Array, Array)| {
                quantized_matmul(x, &q.0, &q.1, Some(&q.2), Some(true), Some(gs), Some(bits))
                    .unwrap()
            };
            // eh_proj base + LoRA delta = (concat·Aᵀ)·Bᵀ
            let base = g(&concat, &eh); // [n,1,h]
            let delta = concat
                .matmul(a.transpose_axes(&[1, 0]).unwrap())
                .unwrap() // [n,1,r]
                .matmul(b.transpose_axes(&[1, 0]).unwrap())
                .unwrap(); // [n,1,h]
            let cur = base.add(&delta).unwrap();
            // attention sub-block
            let cur_n = rms_norm(&cur, &w_in, eps).unwrap();
            let q4 = g(&cur_n, &qw).reshape(&[n, 1, heads, hd]).unwrap();
            let qn4 = rms_norm(&q4, &qn, eps).unwrap();
            let qt = rope(
                qn4.transpose_axes(&[0, 2, 1, 3]).unwrap(),
                hd,
                false,
                Some(1.0e6f32),
                1.0,
                0,
                None,
            )
            .unwrap();
            let k4 = g(&cur_n, &kw).reshape(&[n, 1, kv, hd]).unwrap();
            let kn4 = rms_norm(&k4, &kn, eps).unwrap();
            let kt = rope(
                kn4.transpose_axes(&[0, 2, 1, 3]).unwrap(),
                hd,
                false,
                Some(1.0e6f32),
                1.0,
                0,
                None,
            )
            .unwrap();
            let vt = g(&cur_n, &vw)
                .reshape(&[n, 1, kv, hd])
                .unwrap()
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap();
            let attn = scaled_dot_product_attention(&qt, &kt, &vt, scale, None, None).unwrap();
            let attn_flat = attn
                .transpose_axes(&[0, 2, 1, 3])
                .unwrap()
                .reshape(&[n, 1, heads * hd])
                .unwrap();
            let o = g(&attn_flat, &ow);
            let cur = o.add(&cur).unwrap();
            // MLP sub-block (SwiGLU)
            let cur_n2 = rms_norm(&cur, &w_post, eps).unwrap();
            let gate = g(&cur_n2, &gw);
            let up = g(&cur_n2, &uw);
            let silu = gate.multiply(mlx_rs::ops::sigmoid(&gate).unwrap()).unwrap();
            let act = silu.multiply(&up).unwrap();
            let down = g(&act, &dw);
            let cur = down.add(&cur).unwrap();
            // head norm + lm_head → logits
            let norm_out = rms_norm(&cur, &w_head, eps).unwrap();
            let logits = g(&norm_out, &lm).reshape(&[n, v]).unwrap();
            // CE
            let lse = logits.logsumexp_axis(1, false).unwrap();
            let tl = logits
                .take_along_axis(targets.reshape(&[n, 1]).unwrap(), 1)
                .unwrap()
                .reshape(&[n])
                .unwrap();
            vec![lse.subtract(&tl).unwrap().mean(false).unwrap()]
        }
    };

    // LoRA params: A small random, B = 0 (delta starts at 0).
    let mut a = mlx_rs::random::normal::<f32>(&[r, 2 * h], None, None, None)?
        .multiply(Array::from_f32((1.0 / (2 * h) as f32).sqrt()))?;
    let mut b = Array::zeros::<f32>(&[h, r])?;
    a.eval()?;
    b.eval()?;

    let mut vg = value_and_grad_with_argnums(fwd, &[0i32, 1i32][..]);

    // Step 1: gradient-flow check (B starts 0 so dL/dA is 0; perturb B first).
    let (loss0, grads0) = vg(&[a.clone(), b.clone()])?;
    let gnorm = |g: &Array| -> f32 {
        let s = g.multiply(g).unwrap().sum(None).unwrap();
        s.eval().ok();
        s.item::<f32>().sqrt()
    };
    let (ga0, gb0) = (gnorm(&grads0[0]), gnorm(&grads0[1]));
    println!(
        "initial: loss={:.4}  |grad A|={ga0:.6}  |grad B|={gb0:.6}",
        loss0[0].item::<f32>()
    );
    // dL/dB must be nonzero (B feeds logits through the whole chain). dL/dA is
    // ~0 at init only because B=0 zeroes the delta's A-path — it becomes
    // nonzero once B moves, which the Adam loop below confirms.
    if !(gb0.is_finite() && gb0 > 1e-9) {
        return Err(anyhow::anyhow!(
            "GRADIENT DID NOT FLOW to LoRA B through the block (|grad B|={gb0}) — \
             a block op has no vjp; head-internal training would need a bf16 reimpl."
        ));
    }

    // Step 2: a few Adam steps — CE should drop (usable signal), and dL/dA
    // should turn nonzero (full chain trains).
    let (mut ma, mut va) = (
        Array::zeros::<f32>(&[r, 2 * h])?,
        Array::zeros::<f32>(&[r, 2 * h])?,
    );
    let (mut mb, mut vb) = (Array::zeros::<f32>(&[h, r])?, Array::zeros::<f32>(&[h, r])?);
    let (lr, b1, b2, ad_eps) = (5e-2f32, 0.9f32, 0.999f32, 1e-8f32);
    let adam =
        |p: &Array, gd: &Array, m: &mut Array, vv: &mut Array, t: i32| -> anyhow::Result<Array> {
            *m = m
                .multiply(Array::from_f32(b1))?
                .add(&gd.multiply(Array::from_f32(1.0 - b1))?)?;
            *vv = vv
                .multiply(Array::from_f32(b2))?
                .add(&gd.multiply(gd)?.multiply(Array::from_f32(1.0 - b2))?)?;
            let mhat = m.divide(Array::from_f32(1.0 - b1.powi(t)))?;
            let vhat = vv.divide(Array::from_f32(1.0 - b2.powi(t)))?;
            let upd = mhat.divide(&mlx_rs::ops::sqrt(&vhat)?.add(Array::from_f32(ad_eps))?)?;
            p.subtract(&upd.multiply(Array::from_f32(lr))?)
                .map_err(Into::into)
        };
    let mut last = loss0[0].item::<f32>();
    let mut max_ga = 0.0f32;
    for step in 1..=30 {
        let (loss, grads) = vg(&[a.clone(), b.clone()])?;
        max_ga = max_ga.max(gnorm(&grads[0]));
        a = adam(&a, &grads[0], &mut ma, &mut va, step)?;
        b = adam(&b, &grads[1], &mut mb, &mut vb, step)?;
        a.eval()?;
        b.eval()?;
        last = loss[0].item::<f32>();
        if step == 1 || step % 10 == 0 {
            println!("step {step:>2}: loss={last:.4}");
        }
    }
    println!(
        "FINAL: loss {:.4} → {last:.4}   max|grad A| over run = {max_ga:.6}",
        loss0[0].item::<f32>()
    );
    let dropped = last < loss0[0].item::<f32>() - 0.05;
    let a_trains = max_ga > 1e-9;
    if dropped && a_trains {
        println!(
            "✓ PASS: autograd flows through quantized_matmul + fast::sdpa + fast::rope + \
             fast::rms_norm into an in-block LoRA, and it trains. Head-internal MTP \
             fine-tuning is FEASIBLE without a Metal reimpl."
        );
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "chain differentiable but training weak (loss drop={}, A trains={}) — investigate",
            dropped,
            a_trains
        ))
    }
}
