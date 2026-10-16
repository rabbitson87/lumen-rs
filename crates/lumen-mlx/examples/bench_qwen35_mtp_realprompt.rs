//! Real-prompt MTP accept-rate A/B (Phase B for the MTP head).
//!
//! Same setup as `bench_qwen35_mtp_ab` (load mxfp4 trunk + enable the
//! bf16 `mtp.*` head from an HF-original snapshot), but instead of a
//! SYNTHETIC token ramp it tokenizes a REAL natural-language prompt and
//! lets the model generate a coherent (greedy) continuation. The accept
//! rate measured here reflects in-distribution decoding — the number that
//! actually decides whether self-speculative MTP is worth pursuing.
//!
//! Why this matters: `bench_qwen35_mtp_ab` feeds gibberish tokens, so the
//! MTP head drafts off-distribution and accept_rate≈0 — useless for the
//! "does MTP win" question. Real coherent context lifts accept toward its
//! true value.
//!
//! Required env:
//!   LUMEN_MLX_BACKEND=native
//!   LUMEN_QWEN35_HF_ORIGINAL=<dir with mtp.* shards>
//! Optional env:
//!   MODEL_ID   trunk id (default mlx-community/Qwen3.6-35B-A3B-mxfp4)
//!   PROMPT     prompt text (default: a coding/explanation instruction)
//!   GEN        tokens to generate per arm (default 128)
//!   K          MTP draft count (default 3)
//!   WARMUP     warmup steps/cycles before timing (default 4)
//!
//! Usage:
//!   LUMEN_MLX_BACKEND=native \
//!   LUMEN_QWEN35_HF_ORIGINAL=~/models/Qwen3.6-35B-A3B-mtp-orig \
//!     cargo run --release -p lumen-mlx --features mlx-native \
//!       --example bench_qwen35_mtp_realprompt

use std::time::Instant;

use anyhow::{Result, anyhow};
use lumen_mlx::{
    MlxBackend, MtpLoadQuant, MtpMlpConfig, MtpMoeConfig, Qwen35MtpDims, load_block_from_hf,
};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() -> Result<()> {
    if std::env::var("LUMEN_MLX_BACKEND")
        .map(|v| v.trim().to_ascii_lowercase())
        .unwrap_or_default()
        != "native"
    {
        unsafe {
            std::env::set_var("LUMEN_MLX_BACKEND", "native");
        }
        eprintln!("[bench] forced LUMEN_MLX_BACKEND=native (MTP requires native runner)");
    }

    let model_id =
        std::env::var("MODEL_ID").unwrap_or_else(|_| "mlx-community/Qwen3.6-35B-A3B-mxfp4".into());
    let prompt_text = std::env::var("PROMPT").unwrap_or_else(|_| {
        // A coherent instruction-style prompt wrapped in the Qwen chat
        // template so the model decodes in-distribution like a real chat
        // turn. Override via PROMPT for other workloads.
        "<|im_start|>user\nExplain, in clear prose, how a hash map achieves \
         average O(1) lookup, what causes collisions, and how open addressing \
         differs from separate chaining. Give a concrete example.\
         <|im_end|>\n<|im_start|>assistant\n"
            .to_string()
    });
    let n_gen = env_usize("GEN", 128);
    let k = env_usize("K", 3);
    let warmup = env_usize("WARMUP", 4);

    let hf_dir = std::env::var("LUMEN_QWEN35_HF_ORIGINAL").map_err(|_| {
        anyhow!(
            "LUMEN_QWEN35_HF_ORIGINAL must point at an HF-original snapshot with `mtp.*` shards"
        )
    })?;
    let hf_path = std::path::PathBuf::from(&hf_dir);
    if !hf_path.is_dir() {
        return Err(anyhow!("{hf_dir} is not a directory"));
    }

    // Config-driven dims + MLP variant: parse the trunk's config.json so the
    // same harness covers 35B-A3B (MoE) and 9B/27B (Dense). When MODEL_ID is an
    // HF id (no local config.json), fall back to a built-in 35B-A3B MoE config.
    let cfg_path = std::path::Path::new(&model_id).join("config.json");
    let cfg: serde_json::Value = match std::fs::read_to_string(&cfg_path) {
        Ok(raw) => serde_json::from_str(&raw)?,
        Err(_) => {
            println!("(no local config.json — using built-in 35B-A3B MoE config)");
            serde_json::json!({
                "text_config": {
                    "hidden_size": 2048, "num_attention_heads": 16,
                    "num_key_value_heads": 2, "head_dim": 256,
                    "rope_theta": 10_000_000, "partial_rotary_factor": 0.25,
                    "rms_norm_eps": 1e-6, "num_experts": 256,
                    "num_experts_per_tok": 8, "moe_intermediate_size": 512,
                    "shared_expert_intermediate_size": 512
                }
            })
        }
    };
    let tc = cfg.get("text_config").unwrap_or(&cfg);
    let gi = |k: &str| tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    let gf = |k: &str| tc.get(k).and_then(|v| v.as_f64());
    // rope_theta / partial_rotary live top-level on 9B but nested under
    // `rope_parameters` on 27B (mrope schema). Check both.
    let rp = tc.get("rope_parameters");
    let grp = |k: &str| rp.and_then(|r| r.get(k)).and_then(|v| v.as_f64());
    let hidden = gi("hidden_size").ok_or_else(|| anyhow!("config missing hidden_size"))?;
    let n_heads = gi("num_attention_heads").unwrap_or(16);
    let n_kv = gi("num_key_value_heads").unwrap_or(2);
    let head_dim = gi("head_dim").unwrap_or(256);
    let rope_theta = gf("rope_theta")
        .or_else(|| grp("rope_theta"))
        .unwrap_or(10_000_000.0) as f32;
    let partial = gf("partial_rotary_factor")
        .or_else(|| grp("partial_rotary_factor"))
        .unwrap_or(0.25);
    let rope_dim = (head_dim as f64 * partial).round() as usize;
    let rms_eps = gf("rms_norm_eps").unwrap_or(1e-6) as f32;
    let n_experts = gi("num_experts").unwrap_or(0);
    let dims = Qwen35MtpDims {
        hidden_size: hidden,
        num_attention_heads: n_heads,
        num_key_value_heads: n_kv,
        head_dim,
        rope_theta,
        rope_dim,
        rms_norm_eps: rms_eps,
        attn_output_gate: true,
    };
    let mlp_cfg = if n_experts > 0 {
        MtpMlpConfig::Moe(MtpMoeConfig {
            num_experts: n_experts as i32,
            num_experts_per_tok: gi("num_experts_per_tok").unwrap_or(8) as i32,
            moe_intermediate_size: gi("moe_intermediate_size").unwrap_or(512),
            shared_expert_intermediate_size: gi("shared_expert_intermediate_size").unwrap_or(512),
            norm_topk_prob: true,
        })
    } else {
        MtpMlpConfig::Dense {
            intermediate_size: gi("intermediate_size")
                .ok_or_else(|| anyhow!("dense config missing intermediate_size"))?,
        }
    };
    println!(
        "config: hidden={hidden} heads={n_heads} kv={n_kv} head_dim={head_dim} rope_dim={rope_dim} experts={n_experts} mlp={}",
        if n_experts > 0 { "MoE" } else { "Dense" }
    );

    println!("=== Real-prompt MTP accept-rate A/B (config-driven) ===");
    println!("trunk: {model_id}");
    println!("mtp:   {hf_dir}");
    println!("n_gen={n_gen}  K={k}  warmup={warmup}");

    let t_load = Instant::now();
    let mut backend = MlxBackend::load(&model_id)?;
    println!(
        "trunk loaded in {:.0} ms",
        t_load.elapsed().as_secs_f64() * 1000.0
    );

    // Tokenize the real prompt BEFORE taking the mutable qwen borrow.
    let prompt = backend.encode(&prompt_text)?;
    println!("prompt tokens: {}", prompt.len());
    let qwen = backend
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("backend is not the Qwen3.5 family"))?;

    let quant = match std::env::var("MTP_QUANT").as_deref() {
        Ok("mxfp4") => MtpLoadQuant::Mxfp4,
        Ok("affine4_32") => MtpLoadQuant::Affine4 { group_size: 32 },
        _ => MtpLoadQuant::Affine4 { group_size: 64 },
    };
    println!(
        "head linear quant: {}",
        std::env::var("MTP_QUANT").unwrap_or_else(|_| "affine4_64".into())
    );
    let block = load_block_from_hf(&hf_path, dims, mlp_cfg, quant)?;
    qwen.enable_qwen35_mtp(block)?;
    if !qwen.qwen35_mtp_enabled() {
        return Err(anyhow!(
            "MTP enable claimed Ok but qwen35_mtp_enabled()=false"
        ));
    }
    println!("mtp head enabled.");

    // ───────── Pass A — baseline single-token decode ─────────
    let seq_a: u64 = 2001;
    let (mut last_a, mut pos_a) = qwen.prefill(seq_a, &prompt)?;
    for _ in 0..warmup {
        let (n, p) = qwen.decode_step(seq_a, last_a, pos_a)?;
        last_a = n;
        pos_a = p;
    }
    let mut step_ms: Vec<f64> = Vec::with_capacity(n_gen);
    let mut base_tokens: Vec<u32> = Vec::with_capacity(n_gen);
    for _ in 0..n_gen {
        let t0 = Instant::now();
        let (n, p) = qwen.decode_step(seq_a, last_a, pos_a)?;
        step_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        base_tokens.push(n);
        last_a = n;
        pos_a = p;
    }
    qwen.remove_seq(seq_a)?;
    let base_total: f64 = step_ms.iter().sum();
    let base_tps = n_gen as f64 / (base_total / 1000.0);
    let base_mean = base_total / n_gen as f64;

    // ───────── Pass B — MTP cycles (real context) ─────────
    // MTP_TEMP>0 → Leviathan rejection-sampling acceptance (else greedy).
    let mtp_temp: f32 = std::env::var("MTP_TEMP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let mtp_topp: f32 = std::env::var("MTP_TOPP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    println!("mtp acceptance: temp={mtp_temp} top_p={mtp_topp}");
    let seq_b: u64 = 2002;
    let (mut last_b, _pos_b) = qwen.prefill(seq_b, &prompt)?;
    for _ in 0..warmup {
        let w = qwen.qwen35_mtp_step(seq_b, last_b, k, &Default::default())?;
        last_b = *w.committed.last().expect("warmup commit");
    }
    let mut emitted = 0usize;
    let mut cycle_ms: Vec<f64> = Vec::new();
    let mut accepted_total = 0usize;
    let mut attempted_total = 0usize;
    let mut mtp_tokens: Vec<u32> = Vec::new();
    while emitted < n_gen {
        let t0 = Instant::now();
        let out = qwen.qwen35_mtp_step(seq_b, last_b, k, &Default::default())?;
        cycle_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        accepted_total += out.n_accepted;
        attempted_total += out.n_attempted;
        for t in &out.committed {
            mtp_tokens.push(*t);
            emitted += 1;
            if emitted >= n_gen {
                break;
            }
        }
        last_b = *out.committed.last().expect("non-empty commit");
    }
    qwen.remove_seq(seq_b)?;
    let cycles = cycle_ms.len();
    let mtp_total: f64 = cycle_ms.iter().sum();
    let mtp_tps = emitted as f64 / (mtp_total / 1000.0);
    let mtp_mean = mtp_total / cycles as f64;
    let accept_rate = if attempted_total > 0 {
        accepted_total as f64 / attempted_total as f64
    } else {
        0.0
    };
    let emit_per_cycle = emitted as f64 / cycles as f64;

    // ───────── Report ─────────
    let speedup = mtp_tps / base_tps;
    println!();
    println!("=== Results ===");
    println!("baseline:  {base_tps:>6.1} tok/s  (mean step {base_mean:.2} ms)");
    println!("mtp K={k}:   {mtp_tps:>6.1} tok/s  (mean cycle {mtp_mean:.2} ms, {cycles} cycles)");
    println!(
        "  accept_rate = {accept_rate:.3} ({accepted_total}/{attempted_total} drafts accepted)"
    );
    println!("  emit/cycle  = {emit_per_cycle:.2}  (max {})", k + 1);
    println!("  speedup     = {speedup:.3}x  (>= 1.0 = MTP wins)");

    // Lossless check: greedy MTP must reproduce the baseline greedy stream
    // exactly (commit only verified tokens). With WARMUP=0 both passes start
    // from the same prefill, so the streams must match token-for-token up to
    // the shorter length. First divergence localizes any correctness bug.
    let cmp_len = base_tokens.len().min(mtp_tokens.len());
    let first_div = (0..cmp_len).find(|&i| base_tokens[i] != mtp_tokens[i]);
    let matched = first_div.unwrap_or(cmp_len);
    println!();
    match first_div {
        None => println!(
            "lossless check: MATCH {matched}/{cmp_len} tokens (base==mtp over compared prefix)"
        ),
        Some(i) => println!(
            "lossless check: DIVERGE at idx {i} (base={} mtp={}) — matched {matched}/{cmp_len}",
            base_tokens[i], mtp_tokens[i]
        ),
    }

    // Decode mut borrow has ended; show generated text to confirm the
    // context stayed coherent (in-distribution). No `drop(qwen)` — it is a
    // borrow, so dropping it freed nothing; NLL ends the borrow at its last
    // use, which is what the call was really relying on.
    let base_text = backend.decode(&base_tokens).unwrap_or_default();
    let mtp_text = backend.decode(&mtp_tokens).unwrap_or_default();
    println!();
    println!("--- baseline continuation (first ~300 chars) ---");
    println!("{}", base_text.chars().take(300).collect::<String>());
    println!();
    println!("--- mtp continuation (first ~300 chars) ---");
    println!("{}", mtp_text.chars().take(300).collect::<String>());

    Ok(())
}
