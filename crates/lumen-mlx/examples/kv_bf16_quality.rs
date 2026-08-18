//! How much does bf16 KV storage actually change the model's predictions?
//!
//! `kv_bf16_ab` settled the memory and throughput question (−33 KB per cache
//! slot, +1.6 to +4.1% decode) but left quality on thin evidence: four
//! operating points, 32 greedy tokens each, over prompts made of random filler
//! words. That is weak in two specific ways, and this harness fixes both.
//!
//! **Filler prompts are the wrong distribution.** A random word salad leaves
//! the next-token distribution flat, so argmax is close to a coin flip between
//! near-equal logits and a dtype perturbation flips it far more readily than it
//! would on text the model finds predictable. Here the context is real: a
//! handful of realistic seed prompts across domains and languages, each
//! extended by the model's own greedy continuation, which is in-distribution by
//! construction and can be grown to any length.
//!
//! **Free-running greedy comparison is brittle.** Once one argmax flips, every
//! later token is a *different* continuation rather than a worse one, so the
//! match rate collapses for reasons that say nothing about quality. This
//! harness teacher-forces instead: both conditions are fed the identical token
//! sequence and their predictions compared position by position, via
//! `forward_probe`, which returns a per-position argmax over `[1, K, vocab]`
//! logits. One 4,000-token sequence therefore yields ~4,000 independent
//! comparisons rather than one cascading trajectory.
//!
//! Agreement is also bucketed by context depth, because the question that
//! matters for a serving default is whether error accumulates as the cache
//! fills — a uniform 99.5% is a very different result from 100% early and 97%
//! late, even at the same average.
//!
//! `--control` runs both conditions in f32. Any disagreement it reports is MLX
//! scheduling nondeterminism, and it is the floor the real numbers must be read
//! against; the memory A/B needed exactly this control to avoid attributing a
//! measurement artifact to the dtype.
//!
//! ```text
//! MODEL_ID=~/models/Qwen3.5-9B-MTPLX-Speed \
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!   --example kv_bf16_quality -- --extend 600
//!
//! # determinism floor
//! ... --example kv_bf16_quality -- --control
//! ```

use anyhow::{Result, anyhow};
use lumen_mlx::metal_memory::clear_cache;
use lumen_mlx::{MlxBackend, MlxQwen35Backend, set_kv_store_bf16};

/// Realistic seeds across domains and scripts. Deliberately not filler: the
/// point is to measure on text whose next token the model has an opinion about.
const SEEDS: &[(&str, &str)] = &[
    (
        "code",
        "Explain, step by step, how a write-ahead log lets a database survive a crash \
         mid-transaction. Cover what is written before the data pages change, what recovery \
         reads on restart, and why fsync ordering matters. Then sketch the smallest correct \
         implementation in Rust, with the durability guarantee each function provides.",
    ),
    (
        "prose-en",
        "Write a careful, unhurried description of a coastal fishing town waking up in \
         winter: the light before sunrise, the sounds carrying over cold water, the routines \
         of people who have done the same work for decades, and the way the weather decides \
         what the day will be.",
    ),
    (
        "prose-ko",
        "겨울 새벽의 항구 도시가 깨어나는 풍경을 차분하게 묘사해 주세요. 해 뜨기 전의 빛, \
         찬 물 위로 퍼지는 소리, 수십 년째 같은 일을 해 온 사람들의 습관, 그리고 날씨가 \
         하루를 결정하는 방식에 대해 구체적으로 써 주세요.",
    ),
    (
        "reasoning",
        "A train leaves station A at 09:00 travelling at 80 km/h. A second train leaves \
         station B, 260 km away, at 09:30 travelling toward A at 100 km/h. Work out when and \
         where they meet, showing every step, then explain which assumptions in the problem \
         are unrealistic and how the answer changes if you drop each one.",
    ),
    (
        "technical-ko",
        "트랜스포머 추론에서 KV 캐시가 왜 필요한지, 그리고 캐시를 f32 대신 bf16으로 저장하면 \
         메모리와 정확도에 각각 어떤 영향이 있는지 설명해 주세요. 어텐션 계산 과정을 단계별로 \
         짚으면서 설명해 주시고, 실제로 문제가 되는 경우를 예로 들어 주세요.",
    ),
    (
        "structured",
        "Produce a JSON object describing a fictional library catalogue with at least eight \
         books. Each entry needs title, author, year, language, subject tags, and a one-line \
         summary. After the JSON, explain the schema choices you made and where they would \
         break down for a real catalogue.",
    ),
];

/// Teacher-forced probe window. `forward_probe` runs one *unchunked* forward
/// and materializes `[1, K, vocab]` logits, so K is bounded by memory rather
/// than by taste: at this vocab size, K=2000 is already ~2 GB of logits.
const PROBE_WINDOW: usize = 384;

struct Extended {
    label: &'static str,
    /// Prompt followed by the model's own greedy continuation.
    tokens: Vec<u32>,
    prompt_len: usize,
}

/// Build a realistic long sequence: prefill the seed, then greedily extend it
/// under f32. Using the model's own output keeps the probe context
/// in-distribution at any length without shipping a text fixture.
fn extend_seed(
    backend: &mut MlxQwen35Backend,
    label: &'static str,
    prompt: &str,
    extend: usize,
    seq_id: u64,
) -> Result<Extended> {
    set_kv_store_bf16(false);
    let msgs = vec![("user".to_string(), prompt.to_string())];
    let prompt_ids = backend.build_chat_input(&msgs, false)?;
    let prompt_len = prompt_ids.len();

    let (mut last, mut pos) = backend.prefill(seq_id, &prompt_ids)?;
    let mut tokens = prompt_ids;
    tokens.push(last);
    for _ in 1..extend {
        let (tok, p) = backend.decode_step(seq_id, last, pos)?;
        tokens.push(tok);
        last = tok;
        pos = p;
    }
    backend.remove_seq(seq_id)?;
    let _ = clear_cache();

    Ok(Extended {
        label,
        tokens,
        prompt_len,
    })
}

/// Teacher-force `tokens` through the model with the bf16 flag pinned, and
/// return the per-position argmax plus the per-position max |logit|.
///
/// The sequence is walked in `PROBE_WINDOW` slices; each `forward_probe` call
/// advances the cache, so window boundaries are invisible to the model — this
/// is one continuous forward as far as attention is concerned, just evaluated
/// in bounded pieces.
fn teacher_force(
    backend: &mut MlxQwen35Backend,
    tokens: &[u32],
    bf16: bool,
    seq_id: u64,
) -> Result<(Vec<u32>, Vec<f32>, Vec<f32>)> {
    set_kv_store_bf16(bf16);
    // `forward_probe` needs a live sequence, so seed the cache with token 0 and
    // probe everything after it.
    backend.prefill(seq_id, &tokens[..1])?;

    let mut argmaxes = Vec::with_capacity(tokens.len());
    let mut max_abs = Vec::with_capacity(tokens.len());
    let mut top2_gap = Vec::with_capacity(tokens.len());
    let mut start = 1usize;
    while start < tokens.len() {
        let end = (start + PROBE_WINDOW).min(tokens.len());
        let rows = backend.forward_probe(seq_id, &tokens[start..end])?;
        argmaxes.extend_from_slice(&rows.row_argmaxes);
        max_abs.extend_from_slice(&rows.row_max_abs);
        top2_gap.extend_from_slice(&rows.row_top2_gap);
        start = end;
    }
    backend.remove_seq(seq_id)?;
    let _ = clear_cache();
    Ok((argmaxes, max_abs, top2_gap))
}

#[derive(Default)]
struct Bucket {
    total: usize,
    agree: usize,
}

impl Bucket {
    fn pct(&self) -> f64 {
        if self.total == 0 {
            100.0
        } else {
            self.agree as f64 / self.total as f64 * 100.0
        }
    }
}

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let model_id = std::env::var("MODEL_ID")
        .map_err(|_| anyhow!("set MODEL_ID to a local model directory or an HF repo id"))?;
    let mut extend = 600usize;
    let mut control = false;

    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--extend" => {
                extend = argv
                    .get(i + 1)
                    .and_then(|v| v.parse().ok())
                    .ok_or_else(|| anyhow!("--extend needs a positive integer"))?;
                i += 2;
            }
            "--control" => {
                control = true;
                i += 1;
            }
            other => return Err(anyhow!("unknown argument {other:?}")),
        }
    }

    println!("--- bf16 KV quality, teacher-forced ---");
    println!("model   = {model_id}");
    println!("seeds   = {} realistic prompts", SEEDS.len());
    println!("extend  = {extend} model-generated tokens per seed");
    if control {
        println!("MODE    = CONTROL (both conditions f32 — this is the determinism floor)");
    }
    println!();

    let mut backend = MlxBackend::load(&model_id)?;
    let backend = backend
        .as_qwen35_mut()
        .ok_or_else(|| anyhow!("this harness drives the Qwen3.5-family probe API"))?;

    let mut next_id: u64 = 1;
    let mut overall = Bucket::default();
    // Context-depth buckets. Whether disagreement grows as the cache fills is
    // the question a serving default actually turns on.
    let mut depth = [Bucket::default(), Bucket::default(), Bucket::default()];
    let depth_names = ["ctx <25%", "ctx 25-75%", "ctx >75%"];
    let mut worst_logit_delta = 0.0f64;
    // Logit gap (top1 - top2) under f32 at every position, and at just the
    // positions where the argmax flipped. If the flips sit at the bottom of the
    // overall gap distribution they are broken ties, not changed predictions.
    let mut gap_at_flips: Vec<f32> = Vec::new();
    let mut gap_all: Vec<f32> = Vec::new();

    for (label, prompt) in SEEDS {
        let ext = extend_seed(backend, label, prompt, extend, next_id)?;
        next_id += 1;

        let (a_arg, a_mag, a_gap) = teacher_force(backend, &ext.tokens, false, next_id)?;
        next_id += 1;
        let (b_arg, b_mag, _b_gap) = teacher_force(backend, &ext.tokens, !control, next_id)?;
        next_id += 1;

        if a_arg.len() != b_arg.len() {
            return Err(anyhow!(
                "{}: probe lengths differ ({} vs {})",
                ext.label,
                a_arg.len(),
                b_arg.len()
            ));
        }

        let n = a_arg.len();
        let mut agree = 0usize;
        let mut first_disagreement = None;
        for (k, (x, y)) in a_arg.iter().zip(&b_arg).enumerate() {
            let bucket = if k * 4 < n {
                0
            } else if k * 4 < n * 3 {
                1
            } else {
                2
            };
            depth[bucket].total += 1;
            if x == y {
                agree += 1;
                depth[bucket].agree += 1;
            } else {
                if first_disagreement.is_none() {
                    first_disagreement = Some(k);
                }
                if let Some(g) = a_gap.get(k) {
                    gap_at_flips.push(*g);
                }
            }
        }
        gap_all.extend_from_slice(&a_gap);
        for (x, y) in a_mag.iter().zip(&b_mag) {
            let d = (*x as f64 - *y as f64).abs() / (x.abs().max(1e-6) as f64);
            worst_logit_delta = worst_logit_delta.max(d);
        }
        overall.total += n;
        overall.agree += agree;

        println!(
            "  {:<13} prompt={:<5} total={:<6} positions={:<5}  top-1 agree {:>6}/{:<6} = {:>6.2}%  \
             first disagreement at {}",
            ext.label,
            ext.prompt_len,
            ext.tokens.len(),
            n,
            agree,
            n,
            agree as f64 / n as f64 * 100.0,
            match first_disagreement {
                Some(k) => format!("position {k}"),
                None => "never".into(),
            },
        );
    }

    println!("\n--- aggregate ---");
    println!(
        "teacher-forced top-1 agreement: {}/{} = {:.3}%   ({} independent positions)",
        overall.agree,
        overall.total,
        overall.pct(),
        overall.total,
    );
    println!("worst |logit| relative delta:   {:.4}", worst_logit_delta);
    println!("\nby context depth (does error accumulate as the cache fills?)");
    for (name, b) in depth_names.iter().zip(&depth) {
        println!(
            "  {name:<12} {:>7}/{:<7} = {:>6.3}%",
            b.agree,
            b.total,
            b.pct()
        );
    }

    // Are the flips ties or real changes? Compare the f32 logit gap at flipped
    // positions against the overall distribution. A flip at the bottom of that
    // distribution is a coin toss landing the other way; a flip up where the
    // model was confident would be a genuine change of prediction.
    if !gap_at_flips.is_empty() {
        gap_all.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let pctile = |q: f64| -> f32 {
            let i = ((gap_all.len() as f64 - 1.0) * q).round() as usize;
            gap_all[i]
        };
        let worst_flip = gap_at_flips
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        // Where the largest flip sits in the overall gap distribution.
        let rank = gap_all.partition_point(|g| *g < worst_flip) as f64 / gap_all.len() as f64;
        println!("\nlogit gap (top1 - top2) at the positions that flipped");
        let mut sorted = gap_at_flips.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        println!("  flipped positions : {sorted:.4?}");
        println!(
            "  all positions     : p1={:.4} p10={:.4} p50={:.4} p90={:.4}",
            pctile(0.01),
            pctile(0.10),
            pctile(0.50),
            pctile(0.90),
        );
        println!(
            "  largest flip {:.4} sits at the {:.1}th percentile of all gaps",
            worst_flip,
            rank * 100.0,
        );
    }

    let drift = depth[2].pct() - depth[0].pct();
    println!("\n--- verdict ---");
    if control {
        println!(
            "Control: {:.3}% agreement is the nondeterminism floor. Any bf16 number at or \
             above this is indistinguishable from noise.",
            overall.pct()
        );
        return Ok(());
    }
    println!(
        "Top-1 agreement {:.3}% over {} positions.",
        overall.pct(),
        overall.total
    );
    if drift.abs() < 0.5 {
        println!(
            "Deep context is within 0.5 points of shallow ({drift:+.3}) — no sign that bf16 \
             rounding accumulates as the cache fills, which is the failure mode that would rule \
             out a default flip."
        );
    } else {
        println!(
            "Deep context is {drift:+.3} points versus shallow — error tracks cache depth. \
             Investigate before considering a default flip; a long-context regression is exactly \
             what an average hides."
        );
    }
    println!(
        "Run with --control to get the determinism floor, and compare: agreement is only \
         meaningful relative to it."
    );
    Ok(())
}
