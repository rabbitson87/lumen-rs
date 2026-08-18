//! Tier-0 replay of the `request_parse` fuzz inputs — the `lumen-server` twin
//! of `lumen-mlx/tests/fuzz_corpus_replay.rs`, split because the request types
//! live here. Same contract: the body mirrors `fuzz/fuzz_targets/
//! request_parse.rs` exactly, and drift between the two un-pins whatever
//! crasher the corpus was holding.
//!
//! The invariant is bare survival. These five `Deserialize` impls are the
//! first code an unauthenticated POST body reaches, so any parse outcome is
//! acceptable and a panic is a remote crash.

use std::path::PathBuf;

use lumen_server::types::{
    AnthropicRequest, ChatCompletionRequest, CompletionRequest, EmbeddingRequest,
    ImageGenerationRequest,
};

#[test]
fn replay_request_parse() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fuzz");
    let mut replayed = 0usize;
    for dir in [
        root.join("seeds/request_parse"),
        root.join("artifacts/request_parse"),
    ] {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue; // artifacts/ legitimately absent until a crash is found
        };
        for e in entries.filter_map(Result::ok) {
            let path = e.path();
            if !path.is_file() {
                continue;
            }
            let bytes =
                std::fs::read(&path).unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            let _ = serde_json::from_slice::<ChatCompletionRequest>(&bytes);
            let _ = serde_json::from_slice::<CompletionRequest>(&bytes);
            let _ = serde_json::from_slice::<AnthropicRequest>(&bytes);
            let _ = serde_json::from_slice::<EmbeddingRequest>(&bytes);
            let _ = serde_json::from_slice::<ImageGenerationRequest>(&bytes);
            replayed += 1;
        }
    }
    assert!(
        replayed > 0,
        "no committed request_parse inputs replayed — fuzz/seeds/request_parse/ must not be empty"
    );
}

/// The well-formed seed must actually deserialize as an OpenAI request — a
/// pure no-panic replay would stay green if the tools field were renamed and
/// every seed quietly started failing to parse.
#[test]
fn openai_seed_deserializes_with_its_tools() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fuzz/seeds/request_parse/seed_openai_tools");
    let bytes = std::fs::read(&path).expect("seed_openai_tools must exist");
    let req: ChatCompletionRequest =
        serde_json::from_slice(&bytes).expect("the well-formed seed must deserialize");
    let tools = req.tools.expect("seed declares tools");
    assert_eq!(tools.len(), 1, "seed declares exactly one tool");
}
