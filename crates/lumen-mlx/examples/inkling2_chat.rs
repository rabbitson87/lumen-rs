//! A multi-turn chat with the Inkling 2 text model on a real checkpoint, through
//! [`lumen_mlx::inkling2::Inkling2Backend`]: the rendered prompt, the streamed reply split into
//! reasoning and answer, and what each turn cost (how much of the prompt the retained caches
//! supplied, prefill and decode time).
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example inkling2_chat -- \
//!     <model_dir> [--think] [--effort E] [--system S] [--weather] [--image PATH] [--max N] \
//!     [--temp T] "<turn 1>" "<turn 2>" ...
//! ```
//!
//! Each argument after the flags is a user turn; the reply to each is kept in the history the next
//! one is rendered from (reasoning included, as a client that sends it back would). `--think`
//! turns thinking on; `--effort` names a level (`low`, `medium`, `high`, `0.5`, ...). `--weather` declares a
//! `get_weather` tool whose result is always "15C and sunny". `--image` attaches a picture to the
//! first user turn (and the history keeps it, so later turns send it again).

#![cfg(feature = "mlx-native")]

use std::io::Write;
use std::path::PathBuf;

use anyhow::{Result, anyhow};
use lumen_mlx::SamplingOverrides;
use lumen_mlx::chat_io::{
    AssistantToolCall, BackendStreamEvent, ChatTurn, ResolvedToolChoice, ToolDef,
    join_reasoning_envelope,
};
use lumen_mlx::inkling2::Inkling2Backend;

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut value = |name: &str| -> Option<String> {
        let at = args.iter().position(|a| a == name)?;
        let v = args.get(at + 1).cloned();
        args.drain(at..(at + 2).min(args.len()));
        v
    };
    let max_new: usize = value("--max").and_then(|s| s.parse().ok()).unwrap_or(256);
    let temperature: f32 = value("--temp").and_then(|s| s.parse().ok()).unwrap_or(0.0);
    let effort = value("--effort");
    let system = value("--system");
    let image = value("--image");
    let with_weather = if let Some(at) = args.iter().position(|a| a == "--weather") {
        args.remove(at);
        true
    } else {
        false
    };
    let think = if let Some(at) = args.iter().position(|a| a == "--think") {
        args.remove(at);
        true
    } else {
        effort.is_some()
    };
    let dir = PathBuf::from(
        args.first()
            .ok_or_else(|| anyhow!("usage: <model_dir> [flags] \"turn\"..."))?,
    );
    let user_turns: Vec<String> = args[1..].to_vec();
    anyhow::ensure!(!user_turns.is_empty(), "give at least one user turn");

    let t = std::time::Instant::now();
    let mut backend = Inkling2Backend::from_dir("inkling2", &dir)?;
    println!(
        "loaded in {:.1}s, context {} tokens",
        t.elapsed().as_secs_f64(),
        backend.max_context()
    );

    let ov = SamplingOverrides {
        reasoning_effort_raw: effort,
        seed: Some(7),
        ..Default::default()
    };
    enum Msg {
        System(String),
        User(String),
        Assistant {
            text: String,
            calls: Vec<(String, String, serde_json::Value)>,
        },
        Tool {
            id: String,
            content: String,
        },
    }
    let mut history: Vec<Msg> = Vec::new();
    let picture: Option<Vec<u8>> = image
        .as_deref()
        .map(|path| std::fs::read(path).map_err(|e| anyhow!("read {path}: {e}")))
        .transpose()?;
    // The history index of the user turn the picture is attached to (the first one).
    let mut picture_at: Option<usize> = None;
    if let Some(s) = system {
        history.push(Msg::System(s));
    }
    let weather = serde_json::json!({
        "type": "object",
        "properties": {"city": {"type": "string", "description": "City name"}},
        "required": ["city"]
    });
    let tools: Vec<ToolDef<'_>> = if with_weather {
        vec![ToolDef {
            name: "get_weather",
            description: Some("Get the current weather of a city"),
            parameters: Some(&weather),
            response: None,
        }]
    } else {
        Vec::new()
    };
    let mut pending: Vec<Option<String>> = user_turns.iter().cloned().map(Some).collect();
    pending.reverse();
    let mut next_id = 0;
    while let Some(slot) = pending.pop() {
        if let Some(user) = slot {
            println!("\n=== user: {user}");
            if picture_at.is_none() && picture.is_some() {
                picture_at = Some(history.len());
            }
            history.push(Msg::User(user));
        }
        // Borrow the history as turns.
        let call_refs: Vec<Vec<AssistantToolCall<'_>>> = history
            .iter()
            .map(|m| match m {
                Msg::Assistant { calls, .. } => calls
                    .iter()
                    .map(|(id, name, args)| AssistantToolCall {
                        id,
                        name,
                        arguments: args,
                    })
                    .collect(),
                _ => Vec::new(),
            })
            .collect();
        let turns: Vec<ChatTurn<'_>> = history
            .iter()
            .enumerate()
            .map(|(i, m)| match m {
                Msg::System(s) => ChatTurn::System(s),
                Msg::User(s) => ChatTurn::User(s),
                Msg::Assistant { text, .. } => ChatTurn::Assistant {
                    text,
                    tool_calls: &call_refs[i],
                },
                Msg::Tool { id, content } => ChatTurn::Tool {
                    tool_call_id: id,
                    name: None,
                    content,
                },
            })
            .collect();
        let mut on_event = |ev: BackendStreamEvent<'_>| -> Result<()> {
            match ev {
                BackendStreamEvent::Reasoning(t) => print!("\x1b[2m{t}\x1b[0m"),
                BackendStreamEvent::Text(t) => print!("{t}"),
                BackendStreamEvent::ToolCallStart { name } => print!("[call {name}]"),
                _ => {}
            }
            std::io::stdout().flush()?;
            Ok(())
        };
        lumen_mlx::metal_memory::reset_peak_memory();
        let images: Vec<Vec<Vec<u8>>> = (0..turns.len())
            .map(|i| match (&picture, picture_at) {
                (Some(bytes), Some(at)) if at == i => vec![bytes.clone()],
                _ => Vec::new(),
            })
            .collect();
        let reply = backend.chat(
            &turns,
            &images,
            &tools,
            &ResolvedToolChoice::Auto,
            max_new,
            temperature,
            1.0,
            think,
            &ov,
            &mut on_event,
        )?;
        let s = backend.last_run();
        println!(
            "\n--- prompt {} tokens ({} from the retained caches), {} image(s) encoded in {:.0} ms, prefill {:.0} ms; {} generated in {:.0} ms ({:.1} tok/s); stop {:?}",
            s.prompt_tokens,
            s.reused_tokens,
            s.images_encoded,
            s.image_ms,
            s.prefill_ms,
            s.generated_tokens,
            s.decode_ms,
            s.generated_tokens as f64 / (s.decode_ms / 1e3).max(1e-9),
            reply.stop
        );
        fn gib<E>(r: std::result::Result<usize, E>) -> f64 {
            r.map_or(f64::NAN, |b| b as f64 / 1_073_741_824.0)
        }
        println!(
            "    memory: peak {:.1} GiB, active {:.1} GiB, cached {:.1} GiB",
            gib(lumen_mlx::metal_memory::get_peak_memory()),
            gib(lumen_mlx::metal_memory::get_active_memory()),
            gib(lumen_mlx::metal_memory::get_cache_memory()),
        );
        let text = if reply.reasoning.is_empty() {
            reply.visible.clone()
        } else {
            join_reasoning_envelope(&reply.reasoning, &reply.visible)
        };
        let calls: Vec<(String, String, serde_json::Value)> = reply
            .tool_calls
            .iter()
            .map(|c| {
                next_id += 1;
                (
                    format!("call_{next_id}"),
                    c.name.clone(),
                    c.arguments.clone(),
                )
            })
            .collect();
        for (id, name, args) in &calls {
            println!("tool call {id}: {name}({args})");
        }
        let answered: Vec<String> = calls.iter().map(|(id, ..)| id.clone()).collect();
        history.push(Msg::Assistant { text, calls });
        if !answered.is_empty() {
            for id in answered {
                history.push(Msg::Tool {
                    id,
                    content: "15C and sunny".into(),
                });
            }
            pending.push(None); // let the model read the results
        }
    }
    Ok(())
}
