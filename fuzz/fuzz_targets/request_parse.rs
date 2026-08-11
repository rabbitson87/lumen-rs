//! Raw-bytes fuzz of the HTTP request deserializers.
//!
//! These five `Deserialize` impls are the first code every client byte
//! reaches, and until the lib split they were unreachable from any test
//! target. Raw bytes rather than a generated structure: serde's own error
//! paths — deeply nested arrays, duplicate keys, numbers that overflow the
//! target type, truncated UTF-8 inside a string escape — are the code under
//! test, and a structure-aware generator would only ever hand serde things it
//! already accepts. The structured, valid-ish side is covered by the
//! `ChatRequest` generator in the deterministic driver.
//!
//! Invariant: never panics. Any parse outcome is fine; a panic is a remote
//! crash triggerable by an unauthenticated POST body.
#![no_main]

use libfuzzer_sys::fuzz_target;
use lumen_server::types::{
    AnthropicRequest, ChatCompletionRequest, CompletionRequest, EmbeddingRequest,
    ImageGenerationRequest,
};

fuzz_target!(|data: &[u8]| {
    let _ = serde_json::from_slice::<ChatCompletionRequest>(data);
    let _ = serde_json::from_slice::<CompletionRequest>(data);
    let _ = serde_json::from_slice::<AnthropicRequest>(data);
    let _ = serde_json::from_slice::<EmbeddingRequest>(data);
    let _ = serde_json::from_slice::<ImageGenerationRequest>(data);
});
