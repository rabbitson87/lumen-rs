//! Lever H Step 2 / Flash Attention — A/B benchmark.
//!
//! Runs the full 4-run A/B protocol within ONE process invocation (single
//! model load) per `playbook_verification_protocol.md`:
//!   - 60s cool-down
//!   - Run 1 (forward):  KH=0 → KH=1
//!   - 60s cool-down
//!   - Run 2 (reverse):  KH=1 → KH=0
//! Pools n_per_variant × 2 samples each into the final Welch's t calculation.
//!
//! Bit-identical token check: each prompt's full token sequence is emitted
//! per run; a final cross-run comparison reports match counts.
//!
//! Defaults: 20 prompts × 20 tokens = 400 decode steps per sub-run, 800
//! pooled per variant (matches past Lever B/D/G A/B work).
//!
//! Usage:
//! ```sh
//! LUMEN_QWEN35_SHARDS="$HOME/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-mxfp4/snapshots/<sha>" \
//! cargo run --release -p lumen-model --example bench_lever_h_ab \
//!   --features turboquant-gpu
//! ```
//!
//! Environment:
//!   - `KH_BENCH_PROMPTS` (default 20): prompts per sub-run
//!   - `KH_BENCH_TOKENS`  (default 20): tokens per prompt
//!   - `KH_BENCH_COOLDOWN_S` (default 60): cool-down seconds between sub-runs
//!   - `KH_BENCH_QUICK=1`: shorten to n=4 prompts × 8 tokens × 5s cool-down
//!     (~2 min total) for a quick wiring sanity instead of full A/B
//!   - `LUMEN_QWEN35_SHARDS`: required, path to model snapshot dir
//!   - `MODEL_ID`: optional override of HF id

#![cfg(feature = "turboquant-gpu")]

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use lumen_metal::affine4_gpu::Affine4Context;
use lumen_metal::mxfp4_gpu::MxFp4Context;
use lumen_model::qwen3_5_moe::backend::Qwen35MoeBackend;

/// Static prompt set, varied to exercise different gate_up dispatch shapes.
/// Kept short so prefill cost doesn't dominate the per-step decode budget.
const PROMPTS: &[&str] = &[
    "<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nList 3 colors.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a fruit.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat day is it?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDescribe water.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine gravity.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is light?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a metal.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is air?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine love.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName an animal.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is fire?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine peace.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a planet.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is wind?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine joy.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a tree.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is rain?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine hope.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a bird.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is ice?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine truth.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a fish.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is sand?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine life.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a color.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nWhat is gold?<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nDefine time.<|im_end|>\n<|im_start|>assistant\n<think>\n",
    "<|im_start|>user\nName a flower.<|im_end|>\n<|im_start|>assistant\n<think>\n",
];

/// dflash (2026-05-02): builds a single long prompt (~12 K chars → ~2390
/// tokens) so flash_attn's N×N matrix-avoidance ROI can be measured in the
/// regime where the prior `flash_attn_metal_landed.md` quick A/B (Skv 14-23)
/// showed WASH. Mirrors `scripts/bench_b4_decode_ab.py::build_long_prompt`.
fn build_long_prompt(target_chars: usize) -> String {
    const BLOCK: &str = "Modern transformer architectures rely on a softmax \
attention mechanism whose memory and compute costs grow quadratically with the \
context length. The key-value cache stores intermediate keys and values across \
auto-regressive decoding but consumes substantial memory at long context lengths. \
State-space models such as Mamba-2 use a recurrent inner state to summarize the \
prefix into a constant-size hidden representation, allowing arbitrarily long \
contexts at constant per-token cost. Hybrid architectures interleave attention \
and SSM layers to retain global lookups while bounding the cache footprint. \
Mixture-of-experts routing dispatches each token to a small subset of expert \
feed-forward networks, multiplying parameter count without proportionally \
inflating per-token compute. Quantized weights packed in a four-bit floating \
format with per-block scales preserve accuracy while shrinking the model \
footprint to fit on consumer accelerators. ";
    let mut out = String::with_capacity(target_chars + 1024);
    out.push_str("<|im_start|>user\n");
    out.push_str(BLOCK);
    while out.len() < target_chars {
        out.push_str("\n\nAdditional context follows. ");
        out.push_str(BLOCK);
    }
    out.push_str(
        "\n\nNow, provide a thorough analysis of the above material.\
<|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
    out
}

/// Per-decode-step ms list for one sub-run: only the n_tokens-1 actual
/// decode steps are recorded (the first emitted token comes from prefill
/// argmax and is excluded). Returns (step_ms list, per-prompt token seqs).
fn run_subrun(
    backend: &mut Qwen35MoeBackend,
    label: &str,
    prompts: &[&str],
    n_tokens: usize,
) -> Result<(Vec<f64>, Vec<Vec<u32>>)> {
    let n_prompts = prompts.len();
    eprintln!("\n=== sub-run [{label}] {n_prompts} prompts × {n_tokens} tokens ===");

    let mut step_ms: Vec<f64> = Vec::with_capacity(n_prompts * (n_tokens - 1).max(1));
    let mut token_seqs: Vec<Vec<u32>> = Vec::with_capacity(n_prompts);

    for (p_idx, prompt) in prompts.iter().enumerate() {
        let ids = backend.encode(prompt)?;
        let t_total = Instant::now();
        let out = backend.generate_with_opts(&ids, n_tokens, 0.0, 1.0, 0, 1.0)?;
        let total_ms = t_total.elapsed().as_secs_f64() * 1000.0;
        let n_gen = out.len();
        if n_gen <= 1 {
            eprintln!("[bench {label}] prompt {p_idx}: only {n_gen} tokens, skipping");
            continue;
        }
        // Pool the decode-only per-step ms recorded by the backend (excludes
        // prefill cost). The first token comes from the prefill argmax so the
        // backend's decode loop only fires (n_gen - 1) times.
        let per_step = backend.last_decode_step_ms().to_vec();
        // Drop the first decode step — it's still cache-cold relative to the
        // steady-state. The remaining (n_gen - 2) steps are stable.
        for &m in per_step.iter().skip(1) {
            step_ms.push(m);
        }

        let pooled_mean = if per_step.len() > 1 {
            per_step.iter().skip(1).sum::<f64>() / (per_step.len() - 1) as f64
        } else {
            per_step.first().copied().unwrap_or(0.0)
        };
        println!(
            "BENCH run={label} p={p_idx} n_gen={n_gen} total_ms={total_ms:.2} decode_mean_ms={pooled_mean:.3} decode_tok_per_s={:.2} tokens={}",
            1000.0 / pooled_mean.max(1e-9),
            out.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
        );

        token_seqs.push(out);
    }

    let n = step_ms.len() as f64;
    let mean = step_ms.iter().sum::<f64>() / n.max(1.0);
    let var = step_ms.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n.max(1.0);
    let sd = var.sqrt();
    let tps = 1000.0 / mean;
    println!("SUBRUN run={label} n={} mean_ms={:.3} sd_ms={:.3} tok_per_s={:.2}", step_ms.len(), mean, sd, tps);

    Ok((step_ms, token_seqs))
}

fn welch_t(a: &[f64], b: &[f64]) -> (f64, f64, f64, f64) {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let mean_a = a.iter().sum::<f64>() / na;
    let mean_b = b.iter().sum::<f64>() / nb;
    let var_a = a.iter().map(|v| (v - mean_a).powi(2)).sum::<f64>() / (na - 1.0).max(1.0);
    let var_b = b.iter().map(|v| (v - mean_b).powi(2)).sum::<f64>() / (nb - 1.0).max(1.0);
    let se = (var_a / na + var_b / nb).sqrt().max(1e-12);
    let t = (mean_a - mean_b) / se;
    (mean_a, mean_b, t, se)
}

/// Toggle a lever via the env var + polarity selected by
/// `KH_BENCH_TARGET` (default: `post`):
/// - `post`         → `LUMEN_DISABLE_RMSNORM_FUSION`        (Lever H Step 2,    default ON  → DISABLE polarity)
/// - `input`        → `LUMEN_ENABLE_INPUT_RMSNORM_FUSION`   (Lever H Step 3,    default OFF → ENABLE polarity)
/// - `bf16_rmsnorm` → `LUMEN_BF16_RMSNORM`                  (Lever B L.3,       default OFF → ENABLE polarity)
/// - `conv1d_native`→ `LUMEN_CONV1D_NATIVE`                 (Lever C L.0.b,     default OFF → ENABLE polarity)
/// - `ssm_candle_queue`→ `LUMEN_DISABLE_SSM_CANDLE_QUEUE`   (Lever D L.2 LANDED,default ON  → DISABLE polarity)
/// - `rope_cache`   → `LUMEN_DISABLE_ROPE_CACHE`             (Lever E L.1.a,     default ON  → DISABLE polarity)
/// - `flash_attn`   → `LUMEN_DISABLE_FLASH_ATTN`            (Flash Attn,        default ON  → DISABLE polarity)
/// - `qknorm_rope`  → `LUMEN_DISABLE_QKNORM_ROPE`           (Lever E2 L.0,      default OFF → ENABLE polarity)
/// `value=1` selects "fused/enabled active"; `value=0` selects baseline of
/// THAT lever. The other lever retains its default state.
fn set_kh(value: u32) {
    let target = std::env::var("KH_BENCH_TARGET").unwrap_or_else(|_| "post".into());

    // Flash Attention uses a runtime-settable AtomicBool (not env-var OnceLock)
    // so we can toggle it within a single process.
    if target == "flash_attn" {
        lumen_metal::flash_attn::set_disabled(value == 0);
        return;
    }

    // SDPA vector port (MLX-style 32-way KV parallelism). Both KH=0 and KH=1
    // use the same `flash_attn_candle` entry; the dispatch routing inside
    // selects FA2 vs SDPA vector based on `set_sdpa_vector_enabled`.
    if target == "sdpa_vector" {
        lumen_metal::flash_attn::set_disabled(false);
        lumen_metal::flash_attn::set_sdpa_vector_enabled(value == 1);
        return;
    }

    // GDN (gated delta-net) SSM kernel port. KH=0 = Candle ops loop (8+
    // dispatches/timestep × 48 layers); KH=1 = single fused Metal kernel via
    // `set_enabled`. Toggles `LUMEN_USE_GDN_KERNEL` runtime gate.
    if target == "gdn_kernel" {
        lumen_metal::gated_delta::set_enabled(value == 1);
        return;
    }

    // to Affine4 (Lever B). KH=0 = legacy
    // Candle fallback with per-layer wait_until_completed + 5 Candle ops for
    // RMSNormGated + out_proj; KH=1 = same encoder, no wait, fused matmul.
    // 27B Dense decode only — MXFP4 (35B-A3B) already had this fast-path.
    if target == "affine4_post_conv_fusion" {
        unsafe {
            if value == 1 {
                std::env::remove_var("LUMEN_AFFINE4_POST_CONV_FUSION");
            } else {
                std::env::set_var("LUMEN_AFFINE4_POST_CONV_FUSION", "0");
            }
        }
        return;
    }

    // Lever F — bf16 activation chain. KH=0 = f32 chain (default production
    // path). KH=1 = `LUMEN_BF16_RMSNORM=1` + `LUMEN_BF16_OUT=1` together,
    // routing input_layernorm → qkv/in_proj_combined as bf16 → cast back to
    // f32 downstream. Targets BW-bound regime by halving activation read
    // bandwidth on unified memory.
    if target == "bf16_chain" {
        unsafe {
            if value == 1 {
                std::env::set_var("LUMEN_BF16_RMSNORM", "1");
                std::env::set_var("LUMEN_BF16_OUT", "1");
            } else {
                std::env::remove_var("LUMEN_BF16_RMSNORM");
                std::env::remove_var("LUMEN_BF16_OUT");
            }
        }
        return;
    }

    // Lever F isolated — `LUMEN_BF16_OUT` only. For decomposing a chain
    // result when bf16_chain shows WIN/NEGATIVE: which flag carries the signal?
    if target == "bf16_out" {
        unsafe {
            if value == 1 {
                std::env::set_var("LUMEN_BF16_OUT", "1");
            } else {
                std::env::remove_var("LUMEN_BF16_OUT");
            }
        }
        return;
    }

    // Workstream B Phase 9 — bf16 residual stream isolated A/B. Both arms
    // run with `LUMEN_BF16_RMSNORM=1` (B.5 prerequisite — input_layernorm
    // produces bf16). KH=1 additionally sets `LUMEN_BF16_RESIDUAL=1` so
    // the layer-level `h` carrier stays bf16 across one decoder layer
    // (boundary casts at o_proj / out_proj / native post-conv exit lifted;
    // single f32 cast at layer exit). Isolates B.9's contribution from the
    // pre-existing bf16 chain.
    if target == "bf16_residual" {
        unsafe {
            std::env::set_var("LUMEN_BF16_RMSNORM", "1");
            if value == 1 {
                std::env::set_var("LUMEN_BF16_RESIDUAL", "1");
            } else {
                std::env::remove_var("LUMEN_BF16_RESIDUAL");
            }
        }
        return;
    }

    // isolates ICB record-once-replay-many on
    // top of the bf16 residual chain. Both arms need `LUMEN_BF16_RMSNORM=1`
    // and `LUMEN_BF16_RESIDUAL=1` (the σ-NEGATIVE B.10 baseline); only
    // `LUMEN_ICB` toggles. KH=0 = standard dispatch (current σ ≈ -16),
    // KH=1 = ICB path. Bit-identical contract: tokens MUST match across arms
    // (same bf16 kernel output, just different command-buffer encoding).
    if target == "icb" {
        unsafe {
            std::env::set_var("LUMEN_BF16_RMSNORM", "1");
            std::env::set_var("LUMEN_BF16_RESIDUAL", "1");
            if value == 1 {
                std::env::set_var("LUMEN_ICB", "1");
            } else {
                std::env::remove_var("LUMEN_ICB");
            }
        }
        return;
    }

    // isolates the alloc-reuse
    // effect on top of Lever G's fused topk path. Both runs need
    // LUMEN_ENABLE_ROUTING_TOPK_FUSION=1; only LUMEN_ROUTER_ALLOC_REUSE
    // toggles between baseline (fresh Tensor::zeros each layer) and treatment
    // (cached buffer per layer). Force-set the underlying fusion gate ON for
    // both runs so the comparison stays apples-to-apples.
    if target == "routing_alloc_reuse" {
        unsafe {
            std::env::set_var("LUMEN_ENABLE_ROUTING_TOPK_FUSION", "1");
            if value == 1 {
                std::env::set_var("LUMEN_ROUTER_ALLOC_REUSE", "1");
            } else {
                std::env::remove_var("LUMEN_ROUTER_ALLOC_REUSE");
            }
        }
        return;
    }

    let (var, default_on) = match target.as_str() {
        "post" => ("LUMEN_DISABLE_RMSNORM_FUSION", true),
        "input" => ("LUMEN_ENABLE_INPUT_RMSNORM_FUSION", false),
        "bf16_rmsnorm" => ("LUMEN_BF16_RMSNORM", false),
        "conv1d_native" => ("LUMEN_CONV1D_NATIVE", false),
        "ssm_candle_queue" => ("LUMEN_DISABLE_SSM_CANDLE_QUEUE", true),
        "rope_cache" => ("LUMEN_DISABLE_ROPE_CACHE", true),
        "qknorm_rope" => ("LUMEN_DISABLE_QKNORM_ROPE", false),
        other => panic!(
            "KH_BENCH_TARGET must be 'post' | 'input' | 'bf16_rmsnorm' | 'conv1d_native' | 'ssm_candle_queue' | 'rope_cache' | 'flash_attn' | 'sdpa_vector' | 'gdn_kernel' | 'affine4_post_conv_fusion' | 'bf16_chain' | 'bf16_out' | 'bf16_residual' | 'qknorm_rope' | 'routing_alloc_reuse' | 'icb', got '{other}'"
        ),
    };
    unsafe {
        match (value == 1, default_on) {
            (true, true) | (false, false) => std::env::remove_var(var),
            (true, false) => std::env::set_var(var, "1"),
            (false, true) => std::env::set_var(var, "1"),
        }
    }
}

fn cooldown(secs: u64) {
    eprintln!("\n[cooldown] {secs}s ...");
    std::thread::sleep(std::time::Duration::from_secs(secs));
}

fn main() -> Result<()> {
    let shard_dir = std::env::var("LUMEN_QWEN35_SHARDS")
        .context("LUMEN_QWEN35_SHARDS required")?;
    let shard_dir = PathBuf::from(shard_dir);
    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());

    // Anti-pattern #36 mitigation: log perf-gate env state. Lever H A/B
    // toggles flash_attn directly; surface the surrounding state so the
    // measurement isn't confused with an unrelated fusion regression.
    for var in &[
        "LUMEN_DISABLE_FLASH_ATTN",
        "LUMEN_DISABLE_RESIDUAL_FUSION",
        "LUMEN_DISABLE_RMSNORM_FUSION",
        "LUMEN_DISABLE_MOE_GATE_UP_SILU_MUL_FUSION",
        "LUMEN_DISABLE_MOE_WSUM_FUSION",
        "LUMEN_DISABLE_QKNORM_ROPE",
        "LUMEN_USE_SDPA_VECTOR",
        "LUMEN_FA_GQA_INKERNEL",
    ] {
        match std::env::var(var) {
            Ok(v) => eprintln!("[lever-h env] {var}={v} (explicit)"),
            Err(_) => eprintln!("[lever-h env] {var}=(unset)"),
        }
    }

    let quick = std::env::var("KH_BENCH_QUICK").map(|v| v == "1").unwrap_or(false);
    // dflash (2026-05-02): when set, replaces PROMPTS with a single ~12 K-char
    // long prompt (~2390 tokens) so flash_attn / SDPA benches measure the
    // long-Skv regime where N×N matrix avoidance has theoretical ROI.
    let long_mode = std::env::var("KH_BENCH_LONG").map(|v| v == "1").unwrap_or(false);
    let n_prompts: usize = if long_mode {
        1
    } else {
        std::env::var("KH_BENCH_PROMPTS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(if quick { 4 } else { 20 })
    };
    let n_tokens: usize = std::env::var("KH_BENCH_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick { 8 } else { 20 });
    let cool_s: u64 = std::env::var("KH_BENCH_COOLDOWN_S")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(if quick { 5 } else { 60 });

    // Long-mode owns the prompt String; short-mode borrows from the static
    // PROMPTS slice. Build once and select the slice via owned `Vec<&str>`.
    let long_prompt: String;
    let active_prompts_owned: Vec<&str>;
    let long_chars: usize = std::env::var("KH_BENCH_LONG_CHARS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12000);
    let prompts_slice: Vec<&str> = if long_mode {
        long_prompt = build_long_prompt(long_chars);
        eprintln!("[long mode] prompt chars = {}", long_prompt.len());
        active_prompts_owned = vec![long_prompt.as_str()];
        active_prompts_owned
    } else {
        if n_prompts > PROMPTS.len() {
            anyhow::bail!("KH_BENCH_PROMPTS {n_prompts} > PROMPTS.len() {}", PROMPTS.len());
        }
        PROMPTS.iter().take(n_prompts).copied().collect()
    };

    eprintln!("=== Lever H A/B bench (single-process protocol) ===");
    eprintln!("model: {model_id}");
    eprintln!(
        "n_prompts={}, n_tokens={n_tokens}, cool-down={cool_s}s, quick={quick}, long_mode={long_mode}",
        prompts_slice.len()
    );

    // Affine4 ctx is required for 27B Dense (mode="affine" 4-bit). Unused for
    // 35B-A3B-mxfp4 (just a small idle GPU pipeline cache). Always supply it
    // so the bench harness loads any quantized variant correctly without
    // dequantizing weights to CPU f32 (which blows resident memory ~5×).
    let gpu_ctx = std::sync::Arc::new(MxFp4Context::new()?);
    let affine4_ctx = std::sync::Arc::new(Affine4Context::new()?);
    let mut backend =
        Qwen35MoeBackend::load_with_affine4(&model_id, &shard_dir, gpu_ctx, affine4_ctx)?;

    // KH_BENCH_PIN_STATE=0|1 — single-state profiling mode for Xcode
    // Instruments traces. Skips the A/B toggle entirely; runs `n_prompts ×
    // n_tokens` decode in the pinned state only. The Python automation
    // (`scripts/profile_b10_xcode.py`) uses this to capture two separate
    // .trace files (one per KH state) for clean dispatch-timeline diff.
    if let Ok(pin) = std::env::var("KH_BENCH_PIN_STATE") {
        let pin_value: u32 = pin.parse().context("KH_BENCH_PIN_STATE must be 0 or 1")?;
        if pin_value > 1 {
            anyhow::bail!("KH_BENCH_PIN_STATE must be 0 or 1, got {pin_value}");
        }
        eprintln!("\n[pin mode] running only KH={pin_value}, no A/B toggle");
        let warm_ids = backend.encode(prompts_slice[0])?;
        set_kh(pin_value);
        let _ = backend.generate_with_opts(&warm_ids, 4, 0.0, 1.0, 0, 1.0)?;
        cooldown(cool_s);
        set_kh(pin_value);
        let (subrun_ms, _) = run_subrun(
            &mut backend,
            &format!("PIN_KH{pin_value}"),
            &prompts_slice,
            n_tokens,
        )?;
        let mean_ms: f64 = subrun_ms.iter().sum::<f64>() / subrun_ms.len() as f64;
        eprintln!(
            "\n[pin mode done] KH={pin_value} mean={:.3} ms ({:.2} tok/s) n={}",
            mean_ms,
            1000.0 / mean_ms,
            subrun_ms.len()
        );
        return Ok(());
    }

    // Pre-warm BOTH paths so neither pays JIT/cache cost during measurement.
    eprintln!("\n[warmup] pre-warming KH=0 + KH=1 paths (1 prompt × 4 tokens each)...");
    let warm_ids = backend.encode(prompts_slice[0])?;
    set_kh(0);
    let _ = backend.generate_with_opts(&warm_ids, 4, 0.0, 1.0, 0, 1.0)?;
    set_kh(1);
    let _ = backend.generate_with_opts(&warm_ids, 4, 0.0, 1.0, 0, 1.0)?;

    cooldown(cool_s);

    // ── Run 1 (forward): KH=0 → KH=1 ──────────────────────────────────────
    set_kh(0);
    let (run1_kh0_ms, run1_kh0_tok) = run_subrun(&mut backend, "R1_KH0", &prompts_slice, n_tokens)?;
    set_kh(1);
    let (run1_kh1_ms, run1_kh1_tok) = run_subrun(&mut backend, "R1_KH1", &prompts_slice, n_tokens)?;

    cooldown(cool_s);

    // ── Run 2 (reverse): KH=1 → KH=0 ──────────────────────────────────────
    set_kh(1);
    let (run2_kh1_ms, run2_kh1_tok) = run_subrun(&mut backend, "R2_KH1", &prompts_slice, n_tokens)?;
    set_kh(0);
    let (run2_kh0_ms, run2_kh0_tok) = run_subrun(&mut backend, "R2_KH0", &prompts_slice, n_tokens)?;

    // ── Pooled stats (Welch's t between KH=0 and KH=1) ────────────────────
    let mut pool_kh0 = run1_kh0_ms;
    pool_kh0.extend_from_slice(&run2_kh0_ms);
    let mut pool_kh1 = run1_kh1_ms;
    pool_kh1.extend_from_slice(&run2_kh1_ms);

    let (mean0, mean1, t, se) = welch_t(&pool_kh0, &pool_kh1);
    let delta_ms = mean1 - mean0;
    let pct = 100.0 * delta_ms / mean0;

    println!();
    println!("======================================================================");
    let bench_target = std::env::var("KH_BENCH_TARGET").unwrap_or_else(|_| "post".into());
    println!("A/B [{bench_target}] Welch's t pooled n={} per variant", pool_kh0.len());
    println!("  KH=0 mean: {:.3} ms ({:.2} tok/s)", mean0, 1000.0 / mean0);
    println!("  KH=1 mean: {:.3} ms ({:.2} tok/s)", mean1, 1000.0 / mean1);
    println!("  Δ (KH=1 - KH=0): {delta_ms:+.3} ms ({pct:+.2}%) se={se:.3}");
    println!("  Welch's σ: {t:+.2}  ({})",
        if t.abs() >= 5.0 { "STRONG SIGNAL" }
        else if t.abs() >= 2.0 { "MILD SIGNAL" }
        else { "WASH (noise)" });
    println!("======================================================================");

    // ── Bit-identical token check across runs ─────────────────────────────
    let mut id_match_kh0 = 0usize;
    let mut id_match_kh1 = 0usize;
    let mut cross_match = 0usize;
    for p in 0..run1_kh0_tok.len().min(run2_kh0_tok.len()) {
        if run1_kh0_tok[p] == run2_kh0_tok[p] { id_match_kh0 += 1; }
    }
    for p in 0..run1_kh1_tok.len().min(run2_kh1_tok.len()) {
        if run1_kh1_tok[p] == run2_kh1_tok[p] { id_match_kh1 += 1; }
    }
    for p in 0..run1_kh0_tok.len().min(run1_kh1_tok.len()) {
        if run1_kh0_tok[p] == run1_kh1_tok[p] { cross_match += 1; }
    }
    let total = run1_kh0_tok.len();
    println!("Bit-identical:");
    println!("  KH=0 R1↔R2: {id_match_kh0}/{total}");
    println!("  KH=1 R1↔R2: {id_match_kh1}/{total}");
    println!("  KH=0 ↔ KH=1 (R1):  {cross_match}/{total}");
    if cross_match < total {
        // Print first divergence as diagnostic
        for p in 0..run1_kh0_tok.len().min(run1_kh1_tok.len()) {
            if run1_kh0_tok[p] != run1_kh1_tok[p] {
                let kh0 = &run1_kh0_tok[p];
                let kh1 = &run1_kh1_tok[p];
                let div_idx = kh0.iter().zip(kh1.iter()).position(|(a, b)| a != b).unwrap_or(0);
                println!(
                    "  first divergence: prompt {p} at token {div_idx}: KH=0={} KH=1={}",
                    kh0[div_idx], kh1[div_idx]
                );
                break;
            }
        }
    }

    Ok(())
}
