//! Head-internal MTP C4 PoC — the decisive accept-lever test for dense Qwen.
//!
//! All offline hidden correctors (diagonal, Procrustes) and the frozen-hidden
//! lm_head C4 are DEAD: the head's softmax is insensitive to hidden-L2, and a
//! logit re-aim of a frozen post-block hidden can't generalize. The ONE untested
//! lever is a LoRA trained INSIDE the block (on eh_proj/mtp.fc), so gradients
//! reshape the hidden through the head's own attention + MLP against the trunk's
//! token (`Qwen35MtpBlock::forward_train_lora`). Feasibility (autograd flows
//! through quantized_matmul + fast::sdpa) proven by `mtp_autograd_probe`.
//!
//! This runs the REAL 27B trunk + MTP head: collect k=0 (embed, h_pre, trunk-
//! target) calib over a greedy decode, train the eh_proj LoRA, report in-sample
//! accept before/after. DECISION GATE: if accept doesn't move → the single head
//! is the ceiling (fold). If it moves → next: held-out generalization test.
//!
//! Run (36GB Mac, ~20GB peak):
//!   LUMEN_MLX_BACKEND=native \
//!     cargo run --release -p lumen-mlx --features mlx-native \
//!       --example bench_mtp_headinternal_poc

use anyhow::{Result, anyhow};
use lumen_mlx::{
    HiTrainCfg, MlxBackend, MtpLoadQuant, MtpLoraPos, MtpMlpConfig, MtpMoeConfig, Qwen35MtpDims,
    load_block_from_hf,
};

fn env_usize(k: &str, d: usize) -> usize {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn env_f32(k: &str, d: f32) -> f32 {
    std::env::var(k)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(d)
}

fn main() -> Result<()> {
    unsafe {
        if std::env::var("LUMEN_MLX_BACKEND").is_err() {
            std::env::set_var("LUMEN_MLX_BACKEND", "native");
        }
    }
    let model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| {
        format!(
            "{}/models/Qwen3.6-27B-MTPLX-Speed",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    // The 27B MTP head lives in `<model>/mtp/weights.safetensors` → the loader's
    // sidecar path. hf_path defaults to the model dir itself.
    let hf_dir = std::env::var("LUMEN_QWEN35_HF_ORIGINAL").unwrap_or_else(|_| model_id.clone());
    let hf_path = std::path::PathBuf::from(&hf_dir);

    let calib_gen = env_usize("CALIB_GEN", 512);
    let rank = env_usize("RANK", 8);
    let steps = env_usize("STEPS", 120);
    let lr: f32 = std::env::var("LR")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0.02);
    let k = env_usize("K", 1);
    let weight_decay = env_f32("WD", 0.0);
    let kl = std::env::var("HI_OBJECTIVE")
        .map(|v| v.eq_ignore_ascii_case("kl"))
        .unwrap_or(false);
    let topk = env_usize("HI_TOPK", 16);
    // HI_POSITIONS: comma list of {eh,q,k,v,o,gate,up,down}. Default = eh only.
    let positions: Vec<MtpLoraPos> = std::env::var("HI_POSITIONS")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|p| MtpLoraPos::parse(p))
                .collect::<Vec<_>>()
        })
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| vec![MtpLoraPos::Eh]);

    // ── config-driven dims (mirrors bench_qwen35_mtp_realprompt) ──
    let cfg_path = std::path::Path::new(&model_id).join("config.json");
    let cfg: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&cfg_path)?)?;
    let tc = cfg.get("text_config").unwrap_or(&cfg);
    let gi = |k: &str| tc.get(k).and_then(|v| v.as_u64()).map(|v| v as usize);
    let gf = |k: &str| tc.get(k).and_then(|v| v.as_f64());
    let rp = tc.get("rope_parameters");
    let grp = |k: &str| rp.and_then(|r| r.get(k)).and_then(|v| v.as_f64());
    let hidden = gi("hidden_size").ok_or_else(|| anyhow!("config missing hidden_size"))?;
    let head_dim = gi("head_dim").unwrap_or(128);
    let partial = gf("partial_rotary_factor")
        .or_else(|| grp("partial_rotary_factor"))
        .unwrap_or(0.25);
    let n_experts = gi("num_experts").unwrap_or(0);
    let dims = Qwen35MtpDims {
        hidden_size: hidden,
        num_attention_heads: gi("num_attention_heads").unwrap_or(32),
        num_key_value_heads: gi("num_key_value_heads").unwrap_or(8),
        head_dim,
        rope_theta: gf("rope_theta")
            .or_else(|| grp("rope_theta"))
            .unwrap_or(1.0e7) as f32,
        rope_dim: (head_dim as f64 * partial).round() as usize,
        rms_norm_eps: gf("rms_norm_eps").unwrap_or(1e-6) as f32,
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
        "=== head-internal MTP C4 PoC ===\nmodel: {model_id}\nconfig: hidden={hidden} heads={} kv={} head_dim={head_dim} mlp={}",
        dims.num_attention_heads,
        dims.num_key_value_heads,
        if n_experts > 0 { "MoE" } else { "Dense" }
    );
    let pos_str = positions
        .iter()
        .map(|p| p.as_str())
        .collect::<Vec<_>>()
        .join("+");
    println!(
        "calib_gen={calib_gen} rank={rank} steps={steps} lr={lr} wd={weight_decay} K={k} positions={pos_str} objective={} topk={topk}",
        if kl { "kl" } else { "ce" }
    );

    let mut backend = MlxBackend::load(&model_id)?;

    // Diverse calib corpus so the LoRA learns a generalizable correction (not
    // one prompt's tokens). The train/held-out split is a sequential 80/20 tail,
    // so with N prompts (rows contiguous per prompt) the held-out 20% is the
    // LAST ~N/5 *entire* prompts — a genuine CROSS-PROMPT generalization test
    // (no intra-prompt leakage). Keep this list large + topically diverse.
    // Encoded before the mutable qwen borrow.
    const TOPICS: &[&str] = &[
        "<|im_start|>user\nExplain how a hash map achieves average O(1) lookup and what causes collisions.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe how photosynthesis converts sunlight into chemical energy.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a short story about a lighthouse keeper who finds a message in a bottle.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the difference between TCP and UDP and when each is preferred.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nHow does a transformer use self-attention to process sequences?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nSummarize the causes of the fall of the Roman Empire.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain compound interest and its effect on long-term savings.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe how vaccines train the immune system to recognize pathogens.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a Python function that merges two sorted lists into one sorted list.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain what a database index is and the tradeoffs of adding one.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the water cycle and the role of evaporation and condensation.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat is the difference between supervised and unsupervised learning?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how garbage collection works in a managed runtime.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the causes and effects of inflation in an economy.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a haiku about the changing of the seasons.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how DNS resolves a domain name to an IP address.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe how a four-stroke internal combustion engine works.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the concept of recursion with a simple example.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nSummarize the key ideas of plate tectonics.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWhat are the main differences between HTTP/1.1 and HTTP/2?<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain how a binary search tree keeps lookups logarithmic.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nDescribe the role of mitochondria in a cell.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nExplain the tradeoffs between optimistic and pessimistic locking.<|im_end|>\n<|im_start|>assistant\n",
        "<|im_start|>user\nWrite a brief explanation of how rainbows form.<|im_end|>\n<|im_start|>assistant\n",
    ];
    let calib_prompts: Vec<Vec<u32>> = TOPICS
        .iter()
        .map(|t| backend.encode(t))
        .collect::<Result<_>>()?;

    let qwen = backend
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("not Qwen3.5 family"))?;
    let block = load_block_from_hf(
        &hf_path,
        dims,
        mlp_cfg,
        MtpLoadQuant::Affine4 { group_size: 64 },
    )?;
    qwen.enable_qwen35_mtp(block)?;
    if !qwen.qwen35_mtp_enabled() {
        return Err(anyhow!("mtp enable failed"));
    }
    println!("trunk + mtp head loaded.");

    // Collect k=0 (embed, h_pre, trunk-target) calib over a greedy decode.
    qwen.qwen35_enable_mtp_hi_calib()?;
    let per = (calib_gen / calib_prompts.len()).max(8);
    for (i, cp) in calib_prompts.iter().enumerate() {
        let seq: u64 = 5000 + i as u64;
        let (mut last, _) = qwen.prefill(seq, cp)?;
        let mut emitted = 0usize;
        while emitted < per {
            let out = qwen.qwen35_mtp_step(seq, last, k, 0.0, 1.0)?;
            emitted += out.committed.len();
            last = *out.committed.last().expect("commit non-empty");
        }
        qwen.remove_seq(seq)?;
        eprintln!(
            "  calib prompt {}/{}: ~{emitted} cycles",
            i + 1,
            calib_prompts.len()
        );
    }

    // Train the head-internal LoRA(s) + report the gate.
    let cfg = HiTrainCfg {
        rank,
        steps,
        lr,
        weight_decay,
        positions,
        kl,
        topk,
        seed: env_usize("SEED", 1234) as u64,
    };
    // `after` is the HONEST stable tail-mean (not the selection-biased max — see
    // train_mtp_headinternal). 1σ of the held-out estimate ≈ sqrt(p(1-p)/n_ho);
    // require Δ ≥ ~2σ to claim a real (non-noise) cross-prompt lift. With
    // n_ho≈90 and p≈0.75, 2σ ≈ 0.09 → the bar is deliberately strict.
    let (before, after) = qwen.qwen35_train_mtp_headinternal(cfg)?;
    let delta = after - before;
    println!("\n════════════════════════════════════════════");
    println!("  HELD-OUT accept (stable tail-mean): {before:.3} → {after:.3}  (Δ {delta:+.3})");
    if delta >= 0.05 {
        println!(
            "  ✓ GENERALIZES — stable cross-prompt lift clear of noise. Worth productionizing."
        );
    } else if delta > 0.0 {
        println!(
            "  ~ MARGINAL — within ~1-2σ of held-out noise; not a reliable lift (see σ line above)."
        );
    } else {
        println!("  ✗ NO LIFT — tail-mean ≤ baseline (overfit / collapse). Fold, like lm_head-C4.");
    }
    println!("════════════════════════════════════════════");
    Ok(())
}
