//! In-process A/B for `LUMEN_TOKENIZE_MEMO` — task 016 Phase 2, Gate 2.
//!
//! Loads a Qwen checkpoint once and plays the warm turns of an agentic
//! conversation: a ~35K-token system+tools head and a file-sized tool result,
//! then one short question per turn. Each turn does what the server does
//! before its first token — the prompt-size count, then the chat path (render,
//! encode, prefix-cache fork + extend, one decode step) — with the memo on or
//! off, alternating in balanced order (ON/OFF, OFF/ON, ...). The flag is
//! flipped with `set()`: it is a `lumen_flags` flag, latched per process
//! otherwise.
//!
//! Gate 2 asks for a time-to-first-token win beyond two standard errors.
//!
//! ```text
//! MODEL_ID=~/models/Qwen3.5-9B-MTPLX-Speed AB_PAIRS=12 \
//!   cargo run --release -p lumen-mlx --features mlx-native --example tokenize_ttft_ab
//! ```

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("tokenize_ttft_ab requires --features mlx-native");
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use lumen_mlx::chat_io::{ResolvedToolChoice, ToolDef};
    use lumen_mlx::text_tokenizer::tokenize_memo;
    use lumen_mlx::{MlxBackend, SamplingOverrides};
    use std::time::Instant;

    let model_id = std::env::var("MODEL_ID").context("set MODEL_ID to a Qwen checkpoint")?;
    let pairs: usize = std::env::var("AB_PAIRS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(12);

    // The replay's shape (scratch `gate0_replay.py` in task 016): the repo's
    // docs as the system prompt and as 30 tool descriptions.
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    let mut paths: Vec<_> = std::fs::read_dir(format!("{root}/docs"))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "md"))
        .collect();
    paths.sort();
    let mut docs = String::new();
    for p in paths {
        docs.push_str(&std::fs::read_to_string(p)?);
    }
    let chars: Vec<char> = docs.chars().collect();
    let slice = |from: usize, len: usize| -> String { chars.iter().skip(from).take(len).collect() };
    let system = slice(0, 40_000);
    let descriptions: Vec<String> = (0..30).map(|i| slice(i * 2311, 1800)).collect();
    let names: Vec<String> = (0..30).map(|i| format!("tool_{i}")).collect();
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"path": {"type": "string"}, "limit": {"type": "integer"}},
        "required": ["path"],
    });
    let tools: Vec<ToolDef<'_>> = names
        .iter()
        .zip(&descriptions)
        .map(|(name, description)| ToolDef {
            name,
            description: Some(description),
            parameters: Some(&schema),
            response: None,
        })
        .collect();
    let choice = ResolvedToolChoice::Auto;
    let file: String =
        std::fs::read_to_string(format!("{root}/crates/lumen-mlx/src/prefix_cache.rs"))?
            .chars()
            .take(8000)
            .collect();
    let mut messages: Vec<(String, String)> = vec![
        ("system".into(), system),
        (
            "user".into(),
            "Read crates/lumen-mlx/src/prefix_cache.rs for me.".into(),
        ),
        ("assistant".into(), "Here it is.".into()),
        ("user".into(), file),
    ];

    eprintln!("[ab] loading {model_id}");
    let mut backend = MlxBackend::load(&model_id).context("load")?;
    let stats = backend.tokenize_stats().context("no tokenizer stats")?;
    let ov = SamplingOverrides::default();

    // One turn, timed from the count to the first token. `messages` already
    // holds the new question.
    let mut turn = |messages: &[(String, String)]| -> anyhow::Result<(f64, f64)> {
        let before = stats.snapshot();
        let t0 = Instant::now();
        backend.build_chat_input_prefilled(messages, false, &tools, &choice, false, None)?;
        backend.chat_streaming(
            messages,
            1,
            0.0,
            1.0,
            &ov,
            false,
            None,
            &tools,
            &choice,
            None,
            |_| Ok(()),
        )?;
        let wall = t0.elapsed().as_secs_f64() * 1e3;
        let encode = stats.snapshot().since(&before).encode_ns as f64 / 1e6;
        Ok((wall, encode))
    };

    let mut n = 0;
    let mut ask = |messages: &mut Vec<(String, String)>| {
        n += 1;
        messages.push(("assistant".into(), format!("Answer {n}.")));
        messages.push((
            "user".into(),
            format!("Question {n}: what does line {n} do?"),
        ));
    };

    // Unmeasured: the cold prefill, then a warm turn on each side so both the
    // prefix cache and the memo hold the conversation so far.
    for on in [false, true, false] {
        tokenize_memo::set(on);
        ask(&mut messages);
        turn(&messages)?;
    }

    let (mut on, mut off) = (Vec::new(), Vec::new());
    for p in 0..pairs {
        let order = if p % 2 == 0 {
            [true, false]
        } else {
            [false, true]
        };
        for memo in order {
            tokenize_memo::set(memo);
            ask(&mut messages);
            let (wall, encode) = turn(&messages)?;
            println!(
                "  pair {p:>2} memo={:<3} first token {wall:>7.1} ms   encode {encode:>6.1} ms",
                if memo { "on" } else { "off" }
            );
            let side = if memo { &mut on } else { &mut off };
            side.push((wall, encode));
        }
    }
    tokenize_memo::clear();

    fn mean_sd(v: &[f64]) -> (f64, f64) {
        let n = v.len() as f64;
        let mean = v.iter().sum::<f64>() / n;
        let var = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
        (mean, var.sqrt())
    }
    fn report(what: &str, on: &[f64], off: &[f64]) {
        let ((m_on, s_on), (m_off, s_off)) = (mean_sd(on), mean_sd(off));
        let se = (s_on * s_on / on.len() as f64 + s_off * s_off / off.len() as f64).sqrt();
        println!(
            "{what:<12} memo off {m_off:>7.1} ± {s_off:>5.1} ms   on {m_on:>7.1} ± {s_on:>5.1} ms   \
             saved {:>6.1} ms   Welch t = {:.1}",
            m_off - m_on,
            (m_off - m_on) / se
        );
    }
    let column = |v: &[(f64, f64)], i: usize| -> Vec<f64> {
        v.iter().map(|x| if i == 0 { x.0 } else { x.1 }).collect()
    };
    report("first token", &column(&on, 0), &column(&off, 0));
    report("encode", &column(&on, 1), &column(&off, 1));
    Ok(())
}
