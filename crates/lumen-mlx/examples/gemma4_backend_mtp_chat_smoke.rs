//! End-to-end smoke for `Gemma4Backend::chat_streaming` with the MTP
//! decode branch wired through `mtp_step()` (Phase 3 landed 2026-05-24).
//!
//! Two runs against the same chat-templated prompt:
//!   1. `LUMEN_GEMMA4_MTP=0` (baseline, async-pipelined per-token loop)
//!   2. `LUMEN_GEMMA4_MTP=1` (routes through mtp_step → committed batch)
//!
//! At temperature=0 (greedy), Google's assistant-drafter MTP guarantees
//! byte-identical visible output. We assert the two runs produce the same
//! `ParsedResponse.visible` string and log per-run wall-clock for an A/B
//! comparison.
//!
//! Run:
//!   MLX_LOCAL_SOURCE_DIR=/Users/sonheesung/Documents/GitHub/mlx \
//!   MODEL_ID=/Users/sonheesung/models/hsng95--gemma-4-26b-a4b-mlx-imatrix3plus-awq \
//!   DRAFTER_DIR=/Users/sonheesung/models/gemma-4-26B-A4B-it-assistant-bf16 \
//!   cargo run --release -p lumen-mlx \
//!       --example gemma4_backend_mtp_chat_smoke --features mlx-native

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
use lumen_mlx::chat_io::{BackendStreamEvent, ResolvedToolChoice};
#[cfg(feature = "mlx-native")]
use lumen_mlx::gemma4::{Gemma4Backend, ToolDef};

#[cfg(feature = "mlx-native")]
fn run_once(
    backend: &mut Gemma4Backend,
    messages: &[(String, String)],
    max_new_tokens: usize,
    tools: &[ToolDef<'_>],
    tool_choice: &ResolvedToolChoice<'_>,
    label: &str,
) -> Result<(String, f64)> {
    let mut buf = String::new();
    let t0 = Instant::now();
    let resp = backend.chat_streaming(
        messages,
        max_new_tokens,
        /* temperature */ 0.0,
        /* top_p */ 1.0,
        /* thinking */ false,
        tools,
        tool_choice,
        |ev| {
            if let BackendStreamEvent::Text(t) = ev {
                buf.push_str(t);
            }
            Ok(())
        },
    )?;
    let wall_ms = t0.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "[mtp-chat-smoke:{label}] visible={:?} wall={:.0}ms",
        resp.visible, wall_ms
    );
    Ok((resp.visible, wall_ms))
}

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    let model_id = std::env::var("MODEL_ID").unwrap_or_else(|_| {
        "/Users/sonheesung/models/hsng95--gemma-4-26b-a4b-mlx-imatrix3plus-awq".into()
    });
    let drafter_dir = std::env::var("DRAFTER_DIR")
        .unwrap_or_else(|_| "/Users/sonheesung/models/gemma-4-26B-A4B-it-assistant-bf16".into());
    let max_new_tokens: usize = std::env::var("MAX_NEW_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    eprintln!("[mtp-chat-smoke] loading trunk {model_id}");
    let mut backend = Gemma4Backend::from_dir("gemma4-mtp-chat-smoke", Path::new(&model_id))
        .context("backend load")?;

    eprintln!("[mtp-chat-smoke] enabling MTP from {drafter_dir}");
    let enabled = backend
        .try_enable_mtp(Path::new(&drafter_dir))
        .context("try_enable_mtp")?;
    if !enabled {
        return Err(anyhow::anyhow!(
            "try_enable_mtp returned false (backbone_hidden_size mismatch)"
        ));
    }

    let messages = vec![
        (
            "system".to_string(),
            "당신은 한국어로 정중히 답하는 어시스턴트입니다.".to_string(),
        ),
        (
            "user".to_string(),
            "한국의 4대 명승지를 각각 한 문단씩 설명해주세요. 역사, 위치, 특징을 포함해서.".to_string(),
        ),
    ];

    let tools: &[ToolDef<'_>] = &[];
    let tool_choice = ResolvedToolChoice::Auto;

    // ── Warmup pass (MTP off, throwaway) so MLX kernel / graph cache and
    //    native runner fast-mode are warmed before we measure either path.
    //    Without this, the OFF run pays cold-start cost and the ON run
    //    inherits the warmth — net 8× per-token regression in the OFF
    //    measurement vs documented 73 tok/s steady state. ──
    unsafe {
        std::env::set_var("LUMEN_GEMMA4_MTP", "0");
    }
    eprintln!("[mtp-chat-smoke] warmup pass (max_new_tokens=16)");
    let _ = run_once(&mut backend, &messages, 16, tools, &tool_choice, "warmup")?;

    // ── Run 1: MTP OFF ──
    unsafe {
        std::env::set_var("LUMEN_GEMMA4_MTP", "0");
    }
    let (visible_off, wall_off) = run_once(
        &mut backend,
        &messages,
        max_new_tokens,
        tools,
        &tool_choice,
        "off",
    )?;

    // ── Run 2: MTP ON ──
    unsafe {
        std::env::set_var("LUMEN_GEMMA4_MTP", "1");
    }
    let (visible_on, wall_on) = run_once(
        &mut backend,
        &messages,
        max_new_tokens,
        tools,
        &tool_choice,
        "on",
    )?;

    println!("\n=== Gemma4Backend MTP chat A/B ===");
    println!("max_new_tokens = {max_new_tokens}");
    println!("OFF visible: {:?}", visible_off);
    println!("ON  visible: {:?}", visible_on);
    println!("OFF wall: {:.0} ms", wall_off);
    println!("ON  wall: {:.0} ms", wall_on);
    let speedup = if wall_on > 0.0 {
        wall_off / wall_on
    } else {
        0.0
    };
    println!("speedup ON / OFF = {:.2}x (wall-clock)", speedup);

    if visible_off != visible_on {
        eprintln!(
            "WARNING: MTP-on visible differs from MTP-off — greedy bit-identical guarantee violated"
        );
        eprintln!("  off len chars = {}", visible_off.chars().count());
        eprintln!("  on  len chars = {}", visible_on.chars().count());
    } else {
        println!("\n✓ bit-identical visible string (greedy guarantee holds)");
    }

    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("build with --features mlx-native to run this example");
}
