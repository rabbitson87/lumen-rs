//! Isolated smoke test for the mlx-rs training primitives lumen needs for the
//! C4 trained MTP adapter (the only remaining accept lever after the geometric
//! correctors all died — see memory `mtp-c4-trained-adapter-pursuit`).
//!
//! Validates, in isolation (no model), that:
//!   1. `value_and_grad_with_argnums` flows gradient through `matmul` + a
//!      cross-entropy loss into two trainable LoRA matrices A, B,
//!   2. a hand-rolled Adam update on those raw `Array`s converges,
//!   3. a rank-r logit correction `logits = H·Wᵀ + (H·Aᵀ)·Bᵀ` can recover the
//!      right argmax when the frozen base `W` alone gets it wrong.
//!
//! If this passes, the same pattern wires into the MTP head's lm_head as the
//! minimal decisive C4 (train a low-rank logit correction on captured calib).
//!
//! Run:  cargo run -p lumen-mlx --features mlx-native --example smoke_lora_train

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("smoke_lora_train requires --features mlx-native");
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use mlx_rs::transforms::value_and_grad_with_argnums;
    use mlx_rs::{Array, Dtype};

    // ── synthetic problem ────────────────────────────────────────────────
    // n samples, hidden h, vocab v, rank r. Build a frozen base W and a
    // "true" low-rank delta D = Bt·At such that argmax((H·Wᵀ) + H·Dᵀ) = target
    // but argmax(H·Wᵀ) alone is wrong for most rows → only a learned low-rank
    // correction recovers accuracy.
    let (n, h, v, r) = (256usize, 64usize, 48usize, 4usize);
    let seed = 7u64;
    mlx_rs::random::seed(seed)?;

    let hmat = mlx_rs::random::normal::<f32>(&[n as i32, h as i32], None, None, None)?;
    let w = mlx_rs::random::normal::<f32>(&[v as i32, h as i32], None, None, None)?
        .multiply(Array::from_f32(0.1))?;
    // true low-rank correction (rank r): At [r,h], Bt [v,r]; D = Bt @ At.
    let at = mlx_rs::random::normal::<f32>(&[r as i32, h as i32], None, None, None)?;
    let bt = mlx_rs::random::normal::<f32>(&[v as i32, r as i32], None, None, None)?;
    // targets = argmax over the corrected logits (the thing we want to learn).
    let base_logits = hmat.matmul(&w.transpose_axes(&[1, 0])?)?;
    let true_delta = hmat
        .matmul(&at.transpose_axes(&[1, 0])?)? // H·Atᵀ  [n,r]
        .matmul(&bt.transpose_axes(&[1, 0])?)?; // ·Btᵀ   [n,v]
    let true_logits = base_logits.add(&true_delta.multiply(Array::from_f32(3.0))?)?;
    let targets =
        mlx_rs::ops::indexing::argmax_axis(&true_logits, 1, false)?.as_dtype(Dtype::Int32)?;
    targets.eval()?;

    let base_acc = argmax_match(&base_logits, &targets)?;
    println!("baseline (frozen W only) argmax-acc vs target = {base_acc:.3}");

    // ── trainable params ─────────────────────────────────────────────────
    // Small init: A ~ N(0, 1/h), B = 0  (standard LoRA init → delta starts 0).
    let mut a = mlx_rs::random::normal::<f32>(&[r as i32, h as i32], None, None, None)?
        .multiply(Array::from_f32((1.0 / h as f32).sqrt()))?;
    let mut b = Array::zeros::<f32>(&[v as i32, r as i32])?;
    a.eval()?;
    b.eval()?;

    // Adam moments.
    let (mut ma, mut va) = (
        Array::zeros::<f32>(&[r as i32, h as i32])?,
        Array::zeros::<f32>(&[r as i32, h as i32])?,
    );
    let (mut mb, mut vb) = (
        Array::zeros::<f32>(&[v as i32, r as i32])?,
        Array::zeros::<f32>(&[v as i32, r as i32])?,
    );
    let (lr, b1, b2, eps) = (5e-2f32, 0.9f32, 0.999f32, 1e-8f32);

    // Loss closure: args = [A, B], closes over H, W, targets as constants.
    // logits = H·Wᵀ + (H·Aᵀ)·Bᵀ ; loss = mean CE(logits, targets).
    let hmat_c = hmat.clone();
    let w_c = w.clone();
    let tgt_c = targets.clone();
    let loss_fn = move |args: &[Array]| -> Vec<Array> {
        let a = &args[0];
        let b = &args[1];
        let base = hmat_c
            .matmul(&w_c.transpose_axes(&[1, 0]).unwrap())
            .unwrap();
        let delta = hmat_c
            .matmul(&a.transpose_axes(&[1, 0]).unwrap())
            .unwrap()
            .matmul(&b.transpose_axes(&[1, 0]).unwrap())
            .unwrap();
        let logits = base.add(&delta).unwrap();
        // CE = mean( logsumexp(logits) - logits[target] )
        let lse = logits.logsumexp_axis(1, false).unwrap(); // [n]
        let tgt_logit = logits
            .take_along_axis(&tgt_c.reshape(&[n as i32, 1]).unwrap(), 1)
            .unwrap()
            .reshape(&[n as i32])
            .unwrap();
        let ce = lse.subtract(&tgt_logit).unwrap().mean(false).unwrap();
        vec![ce]
    };

    let mut vg = value_and_grad_with_argnums(loss_fn, &[0i32, 1i32][..]);

    let adam_step =
        |p: &Array, g: &Array, m: &mut Array, vv: &mut Array, t: i32| -> anyhow::Result<Array> {
            *m = m
                .multiply(Array::from_f32(b1))?
                .add(&g.multiply(Array::from_f32(1.0 - b1))?)?;
            *vv = vv
                .multiply(Array::from_f32(b2))?
                .add(&g.multiply(g)?.multiply(Array::from_f32(1.0 - b2))?)?;
            let bc1 = 1.0 - b1.powi(t);
            let bc2 = 1.0 - b2.powi(t);
            let mhat = m.divide(Array::from_f32(bc1))?;
            let vhat = vv.divide(Array::from_f32(bc2))?;
            let upd = mhat.divide(&mlx_rs::ops::sqrt(&vhat)?.add(Array::from_f32(eps))?)?;
            let np = p.subtract(&upd.multiply(Array::from_f32(lr))?)?;
            Ok(np)
        };

    let steps = 200;
    for step in 1..=steps {
        let (loss, grads) = vg(&[a.clone(), b.clone()])?;
        let ga = &grads[0];
        let gb = &grads[1];
        a = adam_step(&a, ga, &mut ma, &mut va, step)?;
        b = adam_step(&b, gb, &mut mb, &mut vb, step)?;
        a.eval()?;
        b.eval()?;
        if step == 1 || step % 40 == 0 || step == steps {
            let acc = {
                let base = hmat.matmul(&w.transpose_axes(&[1, 0])?)?;
                let delta = hmat
                    .matmul(&a.transpose_axes(&[1, 0])?)?
                    .matmul(&b.transpose_axes(&[1, 0])?)?;
                argmax_match(&base.add(&delta)?, &targets)?
            };
            println!(
                "step {step:>3}: loss={:.4}  argmax-acc={acc:.3}",
                loss[0].item::<f32>()
            );
        }
    }

    let final_acc = {
        let base = hmat.matmul(&w.transpose_axes(&[1, 0])?)?;
        let delta = hmat
            .matmul(&a.transpose_axes(&[1, 0])?)?
            .matmul(&b.transpose_axes(&[1, 0])?)?;
        argmax_match(&base.add(&delta)?, &targets)?
    };
    println!("FINAL argmax-acc = {final_acc:.3} (baseline {base_acc:.3})");
    if final_acc > base_acc + 0.2 {
        println!(
            "✓ PASS: trained LoRA recovered argmax accuracy → mlx-rs training primitives work."
        );
        Ok(())
    } else {
        Err(anyhow::anyhow!(
            "training did not improve accuracy enough ({base_acc:.3} → {final_acc:.3})"
        ))
    }
}

#[cfg(feature = "mlx-native")]
fn argmax_match(logits: &mlx_rs::Array, targets: &mlx_rs::Array) -> anyhow::Result<f32> {
    use mlx_rs::Dtype;
    let pred = mlx_rs::ops::indexing::argmax_axis(logits, 1, false)?.as_dtype(Dtype::Int32)?;
    let eq = pred.eq(targets)?.as_dtype(Dtype::Float32)?;
    let acc = eq.mean(false)?;
    acc.eval()?;
    Ok(acc.item::<f32>())
}
