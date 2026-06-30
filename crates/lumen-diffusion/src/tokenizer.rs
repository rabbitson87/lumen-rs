//! FLUX.2 prompt tokenization (pure-CPU, no MLX).
//!
//! The FLUX.2 text encoder (Mistral-Small-3.2 / Tekken tokenizer) wraps the
//! user prompt in a fixed instruction template, tokenizes it, then LEFT-pads
//! the id sequence to a fixed length (512) so real tokens sit at the tail.
//!
//! This mirrors exactly what the reference Swift FLUX.2 encoder emits, so that
//! arbitrary prompts can flow through the live Mistral encoder → DiT → VAE
//! pipeline instead of relying on pre-dumped token-id dumps.
//!
//! Pure Rust (`tokenizers` crate only) — runs fine without Metal / MLX.

use anyhow::Result;
use std::path::Path;
use tokenizers::Tokenizer;

/// Target (padded) sequence length the encoder consumes.
pub const FLUX_SEQ_LEN: usize = 512;

/// Tekken pad id used for LEFT padding.
pub const PAD_ID: i32 = 11;

// Tekken special-token ids (control tokens).
const BOS_ID: i32 = 1; // <s>
const SYSTEM_PROMPT_ID: i32 = 17; // [SYSTEM_PROMPT]
const SYSTEM_PROMPT_END_ID: i32 = 18; // [/SYSTEM_PROMPT]
const INST_ID: i32 = 3; // [INST]
const INST_END_ID: i32 = 4; // [/INST]

/// Exact T2I system message the FLUX.2 encoder injects.
pub const FLUX_SYSTEM_PROMPT: &str = "You are an AI that reasons about image descriptions. You give structured responses focusing on object relationships, object attribution and actions without speculation.";

/// Default `tokenizer.json`: shared with the Mistral-Small-3.2 model, resolved
/// from the local HuggingFace cache (4-bit encoder repo). Falls back to a
/// relative path (load fails with a clear message) if not downloaded.
pub fn default_tokenizer_json() -> std::path::PathBuf {
    crate::hf_cache::snapshot_path_or_rel(crate::repos::ENCODER_4BIT, "tokenizer.json")
}

/// Tokenize a prompt into the FLUX.2 template and LEFT-pad to [`FLUX_SEQ_LEN`].
///
/// Loads the tokenizer from [`default_tokenizer_json`]. Use
/// [`tokenize_flux_prompt_with`] to supply a different tokenizer path.
pub fn tokenize_flux_prompt(prompt: &str) -> Result<Vec<i32>> {
    tokenize_flux_prompt_with(prompt, default_tokenizer_json())
}

/// Like [`tokenize_flux_prompt`] but with an explicit tokenizer.json path.
pub fn tokenize_flux_prompt_with(
    prompt: &str,
    tokenizer_json: impl AsRef<Path>,
) -> Result<Vec<i32>> {
    let tok = load_tokenizer(tokenizer_json)?;
    let ids = build_flux_ids(&tok, prompt)?;
    left_pad(ids, FLUX_SEQ_LEN, PAD_ID)
}

/// Index of the first real (non-pad) token, given a padded sequence — i.e. the
/// `first_real` the encoder's attention mask needs. Convenience for callers.
pub fn first_real_index(padded: &[i32]) -> usize {
    padded
        .iter()
        .position(|&id| id != PAD_ID)
        .unwrap_or(padded.len())
}

fn load_tokenizer(path: impl AsRef<Path>) -> Result<Tokenizer> {
    let path = path.as_ref();
    Tokenizer::from_file(path)
        .map_err(|e| anyhow::anyhow!("load tokenizer {}: {e}", path.display()))
}

/// Build the raw id sequence for the FLUX template:
/// `<s>[SYSTEM_PROMPT]{SYSTEM}[/SYSTEM_PROMPT][INST]{prompt}[/INST]`.
///
/// The bracketed control tokens are inserted by id (the tokenizer's
/// `encode(.., add_special_tokens=false)` only handles the natural-language
/// spans), guaranteeing the exact id layout the Swift encoder produces.
fn build_flux_ids(tok: &Tokenizer, prompt: &str) -> Result<Vec<i32>> {
    let system_ids = encode_text(tok, FLUX_SYSTEM_PROMPT)?;
    let prompt_ids = encode_text(tok, prompt)?;

    let mut ids = Vec::with_capacity(system_ids.len() + prompt_ids.len() + 5);
    ids.push(BOS_ID);
    ids.push(SYSTEM_PROMPT_ID);
    ids.extend_from_slice(&system_ids);
    ids.push(SYSTEM_PROMPT_END_ID);
    ids.push(INST_ID);
    ids.extend_from_slice(&prompt_ids);
    ids.push(INST_END_ID);
    Ok(ids)
}

/// Encode a plain text span WITHOUT special tokens, as `i32` ids.
fn encode_text(tok: &Tokenizer, text: &str) -> Result<Vec<i32>> {
    let enc = tok
        .encode(text, false)
        .map_err(|e| anyhow::anyhow!("encode text span: {e}"))?;
    Ok(enc.get_ids().iter().map(|&u| u as i32).collect())
}

/// LEFT-pad `ids` to `len` with `pad`; real tokens end up at the tail.
fn left_pad(ids: Vec<i32>, len: usize, pad: i32) -> Result<Vec<i32>> {
    if ids.len() > len {
        anyhow::bail!(
            "tokenized sequence ({}) exceeds target length {len}; prompt too long",
            ids.len()
        );
    }
    let mut out = vec![pad; len - ids.len()];
    out.extend_from_slice(&ids);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALIDATION_PROMPT: &str =
        "a photorealistic close-up of a hummingbird hovering near a red flower";

    fn swift_tokens() -> Vec<i32> {
        let raw =
            std::fs::read_to_string("/tmp/swift_tokens.json").expect("read /tmp/swift_tokens.json");
        let v: Vec<i32> = serde_json::from_str(&raw).expect("parse swift_tokens.json");
        assert_eq!(v.len(), FLUX_SEQ_LEN, "swift reference must be 512 ids");
        v
    }

    /// The gate: byte-exact match against the Swift FLUX.2 encoder output.
    #[test]
    fn flux_tokenize_matches_swift() {
        let expected = swift_tokens();
        let got = tokenize_flux_prompt(VALIDATION_PROMPT).expect("tokenize");
        assert_eq!(got.len(), FLUX_SEQ_LEN, "must be 512 ids");

        if got != expected {
            // Surface the exact differing positions for diagnosis.
            let diffs: Vec<(usize, i32, i32)> = got
                .iter()
                .zip(expected.iter())
                .enumerate()
                .filter(|(_, (g, e))| g != e)
                .map(|(i, (g, e))| (i, *g, *e))
                .collect();
            panic!(
                "tokenization mismatch: {} differing ids (pos, got, swift): {:?}",
                diffs.len(),
                diffs
            );
        }
    }

    #[test]
    fn first_real_index_is_465() {
        let padded = tokenize_flux_prompt(VALIDATION_PROMPT).expect("tokenize");
        // 47 real tokens, left-padded into 512 → first real at 465.
        assert_eq!(first_real_index(&padded), FLUX_SEQ_LEN - 47);
    }
}
