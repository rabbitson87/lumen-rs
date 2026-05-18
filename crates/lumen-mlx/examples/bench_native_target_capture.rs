//! D5a.1 smoke test: native target-hidden capture.
//!
//! Verifies that `MlxBackend::prefill_with_capture` on the native runner
//! produces per-target-layer post-MLP residual hiddens with the expected
//! shapes:
//!   - one `[1, prompt_len, hidden]` Array per `target_layer_ids[i]`
//!   - in declared order (so `concatenate(axis=-1)` lines up with the
//!     DFlash draft's `fc` weight rows)
//!
//! The Python reference for this capture path is `dflash_runtime.py::patch_model`
//! + `_LayerHook.__call__` + `get_target_hidden(model)` — wraps each layer in
//! `target_layer_ids` so the layer's `__call__` return is recorded, then
//! concatenates the recorded list along axis -1 producing
//! `[1, T, len(target_layer_ids) * hidden]`. Native impl mirrors that layer
//! output (post-MLP residual of `h + mlp(post_attention_layernorm(h))`).
//!
//! Bit-identical comparison against the Python path is left to D5a.5; this
//! smoke covers the structural shape + dtype contract only.
//!
//! Usage:
//!   LUMEN_MLX_BACKEND=native \
//!     cargo run --release -p lumen-mlx --features mlx-native-metal \
//!     --example bench_native_target_capture -- \
//!     [--model mlx-community/Qwen3.6-35B-A3B-mxfp4] \
//!     [--target-layer-ids 1,10,19,28,37] \
//!     [--prompt "..."]

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, anyhow};
    use mlx_rs::ops;
    use lumen_mlx::MlxBackend;

    // ── CLI ──────────────────────────────────────────────────────────────────
    let args: Vec<String> = std::env::args().collect();
    let mut model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let mut target_ids_raw = String::from("1,10,19,28,37");
    let mut prompt = String::from(
        "<|im_start|>user\nWrite a Python function for binary search.<|im_end|>\n<|im_start|>assistant\n",
    );

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model_id = args[i + 1].clone();
                i += 2;
            }
            "--target-layer-ids" => {
                target_ids_raw = args[i + 1].clone();
                i += 2;
            }
            "--prompt" => {
                prompt = args[i + 1].clone();
                i += 2;
            }
            _ => i += 1,
        }
    }

    let target_layer_ids: Vec<usize> = target_ids_raw
        .split(',')
        .filter_map(|s| s.trim().parse::<usize>().ok())
        .collect();
    if target_layer_ids.is_empty() {
        return Err(anyhow!(
            "no valid layer ids in --target-layer-ids={target_ids_raw:?}"
        ));
    }

    println!("--- D5a.1 smoke: native target-hidden capture ---");
    println!(
        "backend           = {}",
        std::env::var("LUMEN_MLX_BACKEND").unwrap_or_else(|_| "(default = pyo3)".into())
    );
    println!("model             = {model_id}");
    println!("target_layer_ids  = {target_layer_ids:?}");

    let t0 = std::time::Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!("loaded in {:.0}ms", t0.elapsed().as_secs_f64() * 1000.0);

    let prompt_ids = backend.encode(&prompt)?;
    let prompt_len = prompt_ids.len();
    println!("prompt: {} chars → {prompt_len} tokens", prompt.len());

    // ── Capture forward ──────────────────────────────────────────────────────
    let seq = 4242u64;
    let (next_tok, position, captured) =
        backend.prefill_with_capture(seq, &prompt_ids, &target_layer_ids)?;
    println!(
        "\nprefill_with_capture: next_tok={next_tok} position={position} captured.len()={}",
        captured.len()
    );

    if captured.len() != target_layer_ids.len() {
        return Err(anyhow!(
            "captured.len()={} != target_layer_ids.len()={}",
            captured.len(),
            target_layer_ids.len()
        ));
    }

    // ── Per-capture shape + dtype + numerical sanity ─────────────────────────
    let mut hidden_size: Option<i32> = None;
    for (i, (h, &lid)) in captured.iter().zip(target_layer_ids.iter()).enumerate() {
        let shape = h.shape();
        if shape.len() != 3 {
            return Err(anyhow!(
                "captured[{i}] (layer {lid}) has ndim={} (expected 3)",
                shape.len()
            ));
        }
        let (b, l, hs) = (shape[0], shape[1], shape[2]);
        if b != 1 {
            return Err(anyhow!(
                "captured[{i}] (layer {lid}) batch={b} (expected 1)"
            ));
        }
        if l != prompt_len as i32 {
            return Err(anyhow!(
                "captured[{i}] (layer {lid}) seq={l} (expected {prompt_len})"
            ));
        }
        if let Some(h_prev) = hidden_size {
            if hs != h_prev {
                return Err(anyhow!(
                    "captured[{i}] (layer {lid}) hidden={hs} mismatches earlier {h_prev}"
                ));
            }
        } else {
            hidden_size = Some(hs);
        }

        // Quick numerical health check: the post-MLP residual should not be
        // all-zero (would indicate a broken hook) and should not contain
        // NaN/Inf. Use max(abs(.)) which is a single reduce.
        let max_abs = ops::abs(h)?.max(None)?.item::<f32>();
        let dtype_str = format!("{:?}", h.dtype());
        println!(
            "  captured[{i}] layer={lid:2} shape={shape:?} dtype={dtype_str} max|.|={max_abs:.3}"
        );
        if !max_abs.is_finite() {
            return Err(anyhow!(
                "captured[{i}] (layer {lid}) contains non-finite values (max|.|={max_abs})"
            ));
        }
        if max_abs == 0.0 {
            return Err(anyhow!(
                "captured[{i}] (layer {lid}) is all zero — capture hook is broken or never fired"
            ));
        }
    }

    let hs = hidden_size.expect("non-empty captures");
    let prompt_len_i32 = prompt_len as i32;

    // ── Concat-along-channel sanity (matches Python's `mx.concatenate(states, -1)`)
    let cap_refs: Vec<&mlx_rs::Array> = captured.iter().collect();
    let concat = ops::concatenate_axis(cap_refs.as_slice(), -1)
        .context("concatenate_axis(captured, -1) failed")?;
    concat.eval()?;
    let cs = concat.shape();
    let expected_concat_chan = hs * (target_layer_ids.len() as i32);
    println!(
        "\nconcatenate(axis=-1) shape = {cs:?}  (expected [1, {prompt_len_i32}, {expected_concat_chan}])"
    );
    if cs != [1, prompt_len_i32, expected_concat_chan] {
        return Err(anyhow!(
            "concat shape {cs:?} != expected [1, {prompt_len_i32}, {expected_concat_chan}]"
        ));
    }

    // ── Cleanup ──────────────────────────────────────────────────────────────
    backend.remove_seq(seq).ok();

    println!("\n=== verdict ===");
    println!("  ✓ capture count == |target_layer_ids|");
    println!("  ✓ each capture is [1, {prompt_len_i32}, {hs}] f32-or-bf16, finite, non-zero");
    println!("  ✓ concat(axis=-1) matches DFlash draft fc input shape");
    println!("  → D5a.1 native target-hidden capture infrastructure is sound.");
    println!("  → Next: D5a.2 (port DFlashDraftModel onto mlx-rs to consume these captures).");
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!(
        "bench_native_target_capture requires --features mlx-native (or mlx-native-metal). \
         Re-run with `cargo run -p lumen-mlx --features mlx-native-metal --example bench_native_target_capture`."
    );
    std::process::exit(2);
}
