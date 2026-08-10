//! End-to-end smoke for `Gemma4Backend::chat_with_prefix_cache`.
//!
//! Simulates the Moltis sports matching batch workload: 3 requests share
//! the same system prompt (~600 chars Korean instructions + candidate
//! list) but differ in the user query (team name to match). The prefix
//! cache should hit on requests 2 and 3, saving most of the prefill.
//!
//! Per-request wall time is reported so the caller can verify:
//!   - Request 1 (MISS): cold prefill + decode
//!   - Request 2-3 (HIT): suffix-only prefill + decode (much faster)
//!
//! Run:
//!   MODEL_ID=/path/to/models/gemma-4-26b-a4b-mlx-3bit \
//!   cargo run --release --features mlx-native -p lumen-mlx \
//!       --example gemma4_backend_prefix_cache_smoke

use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result};

#[cfg(feature = "mlx-native")]
fn main() -> Result<()> {
    use lumen_mlx::chat_io::ResolvedToolChoice;
    use lumen_mlx::gemma4::{Gemma4Backend, ToolDef};

    let model_id = std::env::var("MODEL_ID")
        .unwrap_or_else(|_| "/path/to/models/gemma-4-26b-a4b-mlx-3bit".into());

    eprintln!("[backend-prefix-smoke] loading {model_id}");
    let mut backend = Gemma4Backend::from_dir("gemma-4-26b-a4b-mlx-3bit", Path::new(&model_id))
        .context("backend load")?;

    // Shared system prompt — Moltis-style sports matching instructions
    // + candidate list. Long enough (~600 chars / ~300+ tokens) to make
    // the prefill win obvious.
    let system_prompt = r#"당신은 한국 스포츠 데이터 매칭 전문가입니다.
미매칭 팀/리그명을 표준명 후보와 비교해 JSON 으로 반환합니다.

[규칙]
1. Sports/국가/연령대/시즌이 일치하지 않으면 매칭하지 말 것
2. 의심되면 suggested_match=null
3. 응답은 **JSON 만**. 다른 텍스트, 마크다운, 설명 금지

[후보 팀]
- Manchester United (Premier League, England) - 별명: 맨유, 맨체스터 유나이티드, Man Utd
- Manchester City (Premier League, England) - 별명: 맨시, 맨체스터 시티, Man City
- Liverpool (Premier League, England) - 별명: 리버풀, LFC
- Chelsea (Premier League, England) - 별명: 첼시, CFC
- Real Madrid (La Liga, Spain) - 별명: 레알 마드리드, 레알, RMA
- FC Barcelona (La Liga, Spain) - 별명: 바르샤, 바르셀로나, FCB

[응답 형식]
{
  "suggested_match": "표준명" | null,
  "match_type": "팀",
  "confidence": 0-100,
  "reasoning": "한국어 1-2문장"
}"#;

    // 3 queries — same system, different user team name.
    let queries = vec!["맨유 매칭해줘", "바르샤 매칭해줘", "리버풀 매칭해줘"];

    let max_new_tokens = 200;
    let prefix_key = "moltis-sports-batch-001";
    let tools: &[ToolDef<'_>] = &[];
    let tool_choice = ResolvedToolChoice::Auto;

    println!("\n=== Gemma4Backend prefix_cache end-to-end smoke ===\n");
    for (i, query) in queries.iter().enumerate() {
        let messages = vec![
            ("system".to_string(), system_prompt.to_string()),
            ("user".to_string(), query.to_string()),
        ];

        let t0 = Instant::now();
        let response = backend
            .chat_with_prefix_cache(
                &messages,
                max_new_tokens,
                /* temperature */ 0.0,
                /* top_p */ 1.0,
                &Default::default(),
                /* thinking */ false,
                prefix_key,
                tools,
                &tool_choice,
            )
            .with_context(|| format!("request {} failed", i + 1))?;
        let wall_ms = t0.elapsed().as_secs_f64() * 1e3;

        let visible = response.visible.as_str();
        // Cap preview at 200 *chars* (not bytes) to avoid splitting UTF-8.
        let preview: String = visible.chars().take(200).collect();
        let suffix_marker = if visible.chars().count() > 200 {
            "…"
        } else {
            ""
        };
        let preview = format!("{preview}{suffix_marker}");
        println!(
            "Request {}: query={:?}  wall={:.0} ms",
            i + 1,
            query,
            wall_ms
        );
        println!("  response: {preview}");
        println!("  prefix_cache_count = {}", backend.prefix_cache_count());
        println!();
    }
    println!("=== Expected ===");
    println!("  Request 1: MISS (cold prefill), wall ~3-5s on M3 Max");
    println!("  Request 2-3: HIT (suffix-only prefill), wall ~0.5-1.5s");
    println!("  → request 2/3 should be ~3-5x faster than request 1");

    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() -> Result<()> {
    eprintln!("This example requires --features mlx-native");
    Ok(())
}
