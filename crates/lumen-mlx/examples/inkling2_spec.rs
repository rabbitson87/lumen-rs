//! Speculative decoding with the Inkling 2 draft head, measured against the plain loop on the real
//! checkpoint: for each prompt and each depth, the greedy reply, how long decode took, and how
//! many drafts the model agreed with.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_spec -- \
//!     <model_dir> [--depths 0,7,3] [--max N] [--think] [--repeat R] "<prompt 1>" "<prompt 2>" ...
//! ```
//!
//! Depth 0 is the plain loop. Every depth runs from cold caches (the retained conversation is
//! cleared between runs), the replies are compared with the plain loop's, and a depth is
//! reported once per prompt (the first run of a process also pays the draft head's load and the
//! shader compiles, so a warm-up run goes first and is not reported).

#![cfg(feature = "mlx-native")]

use std::path::PathBuf;

use anyhow::{Result, anyhow};
use lumen_mlx::SamplingOverrides;
use lumen_mlx::chat_io::{ChatTurn, ResolvedToolChoice};
use lumen_mlx::inkling2::Inkling2Backend;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut value = |name: &str| -> Option<String> {
        let at = args.iter().position(|a| a == name)?;
        let v = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        v
    };
    let depths: Vec<usize> = value("--depths")
        .unwrap_or_else(|| "0,7".into())
        .split(',')
        .filter_map(|d| d.trim().parse().ok())
        .collect();
    let max_new: usize = value("--max").and_then(|s| s.parse().ok()).unwrap_or(200);
    let repeat: usize = value("--repeat").and_then(|s| s.parse().ok()).unwrap_or(1);
    let think = if let Some(at) = args.iter().position(|a| a == "--think") {
        args.remove(at);
        true
    } else {
        false
    };
    let dir = PathBuf::from(
        args.first()
            .ok_or_else(|| anyhow!("usage: <model_dir> ..."))?,
    );
    let prompts: Vec<String> = args[1..].to_vec();
    anyhow::ensure!(!prompts.is_empty(), "give at least one prompt");

    let mut backend = Inkling2Backend::from_dir("inkling2", &dir)?;
    let ov = SamplingOverrides {
        seed: Some(7),
        ..Default::default()
    };
    let run = |backend: &mut Inkling2Backend, prompt: &str, depth: usize| -> Result<(String, _)> {
        backend.clear_retained();
        backend.set_draft_depth(Some(depth));
        let turns = [ChatTurn::User(prompt)];
        let reply = backend.chat(
            &turns,
            &[],
            &[],
            &ResolvedToolChoice::Auto,
            max_new,
            0.0,
            1.0,
            think,
            &ov,
            &mut |_| Ok(()),
        )?;
        Ok((
            format!("{}\u{1}{}", reply.reasoning, reply.visible),
            backend.last_run(),
        ))
    };

    // Warm-up: the draft head's load, the shader compiles.
    let started = std::time::Instant::now();
    run(
        &mut backend,
        "Say hello.",
        *depths.iter().max().unwrap_or(&0),
    )?;
    println!("warm-up {:.1}s", started.elapsed().as_secs_f64());

    for prompt in &prompts {
        println!("\n=== {prompt:?}");
        let mut plain: Option<String> = None;
        for _ in 0..repeat {
            for &depth in &depths {
                let (text, s) = run(&mut backend, prompt, depth)?;
                let tps = s.generated_tokens as f64 / (s.decode_ms / 1e3).max(1e-9);
                let same = match &plain {
                    None if depth == 0 => {
                        plain = Some(text.clone());
                        "plain".to_string()
                    }
                    Some(p) => {
                        if *p == text {
                            "same as plain".to_string()
                        } else {
                            let at = p
                                .chars()
                                .zip(text.chars())
                                .take_while(|(a, b)| a == b)
                                .count();
                            format!("differs from plain at char {at} of {}", p.chars().count())
                        }
                    }
                    None => "no plain run to compare".to_string(),
                };
                let sp = s.spec;
                if sp.cycles > 0 {
                    println!(
                        "depth {depth}: {:>3} tokens, decode {:>6.0} ms = {:>5.1} tok/s | {:.2} tokens per cycle over {} cycles, {}/{} drafts agreed; draft {:.1} ms/cycle, verify {:.1} ms/cycle | {same}",
                        s.generated_tokens,
                        s.decode_ms,
                        tps,
                        sp.tokens_per_cycle(),
                        sp.cycles,
                        sp.accepted,
                        sp.drafted,
                        sp.draft_ms / sp.cycles as f64,
                        sp.verify_ms / sp.cycles as f64,
                    );
                } else {
                    println!(
                        "depth {depth}: {:>3} tokens, decode {:>6.0} ms = {:>5.1} tok/s | {same}",
                        s.generated_tokens, s.decode_ms, tps
                    );
                }
            }
        }
    }
    Ok(())
}
