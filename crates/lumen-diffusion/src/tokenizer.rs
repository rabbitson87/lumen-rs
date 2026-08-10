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

    /// Where to find a real Mistral tokenizer for the checkpoint-gated tests.
    ///
    /// These previously read `/tmp/swift_tokens.json` and the default HF cache
    /// path unconditionally, so they failed on every machine that had neither —
    /// permanently, and as hard errors rather than skips.
    fn tokenizer_json() -> Option<std::path::PathBuf> {
        if let Ok(dir) = std::env::var("LUMEN_DIFFUSION_TOKENIZER_DIR") {
            let p = std::path::Path::new(&dir).join("tokenizer.json");
            return p.exists().then_some(p);
        }
        let p = default_tokenizer_json();
        p.exists().then_some(p)
    }

    // ── Pure logic: runs everywhere, no artifact ──────────────────────────
    //
    // Left-padding and the first-real-token scan are lumen's own code; the
    // tokenizer itself is HuggingFace's. Splitting them means the part we can
    // actually get wrong is covered unconditionally, instead of riding along
    // with a multi-GB download.

    #[test]
    fn first_real_index_finds_the_padding_boundary() {
        let mut padded = vec![PAD_ID; FLUX_SEQ_LEN];
        padded[FLUX_SEQ_LEN - 3] = 42;
        padded[FLUX_SEQ_LEN - 2] = 43;
        padded[FLUX_SEQ_LEN - 1] = 44;
        assert_eq!(first_real_index(&padded), FLUX_SEQ_LEN - 3);
    }

    #[test]
    fn first_real_index_handles_the_degenerate_ends() {
        assert_eq!(
            first_real_index(&vec![PAD_ID; 8]),
            8,
            "all padding → one past the end, never a panic"
        );
        assert_eq!(
            first_real_index(&[7, PAD_ID, 9]),
            0,
            "leading token is already real"
        );
        assert_eq!(first_real_index(&[]), 0, "empty input");
    }

    #[test]
    fn padding_is_on_the_left_and_preserves_order() {
        // The encoder reads the tail, so a right-pad would silently feed it
        // padding and drop the prompt.
        // Deliberately not PAD_ID (11) — a real token equal to the pad byte
        // would make `first_real_index` skip it, which is worth knowing but is
        // not what this test is about.
        let ids = vec![101i32, 22, 33];
        let padded = left_pad(ids, 6, PAD_ID).expect("pad");
        assert_eq!(padded, vec![PAD_ID, PAD_ID, PAD_ID, 101, 22, 33]);
        assert_eq!(first_real_index(&padded), 3);
    }

    #[test]
    fn padding_refuses_to_truncate() {
        // Silently dropping the head of a too-long prompt would be worse than
        // failing: the caller would get an image for a prompt it never sent.
        assert!(
            left_pad(vec![1, 2, 3], 2, PAD_ID).is_err(),
            "a prompt longer than the window must error, not truncate"
        );
    }

    // ── Checkpoint-gated ─────────────────────────────────────────────────

    #[test]
    #[ignore = "needs a Mistral tokenizer; set LUMEN_DIFFUSION_TOKENIZER_DIR"]
    fn flux_prompt_tokenizes_to_the_padded_window() {
        let Some(tok) = tokenizer_json() else {
            eprintln!("skipping: set LUMEN_DIFFUSION_TOKENIZER_DIR");
            return;
        };
        let got = tokenize_flux_prompt_with(VALIDATION_PROMPT, &tok).expect("tokenize");
        assert_eq!(got.len(), FLUX_SEQ_LEN, "must be {FLUX_SEQ_LEN} ids");
        let first = first_real_index(&got);
        assert!(
            first > 0 && first < FLUX_SEQ_LEN,
            "prompt must be left-padded"
        );
        assert!(
            got[..first].iter().all(|&id| id == PAD_ID),
            "everything before the first real token must be padding"
        );
        assert!(
            got[first..].iter().all(|&id| id != PAD_ID),
            "no padding may appear inside the prompt"
        );
    }

    /// Byte-exact match against the Swift FLUX.2 encoder, when a dump is
    /// available. `LUMEN_DIFFUSION_SWIFT_TOKENS` points at a JSON array of
    /// `FLUX_SEQ_LEN` ids; the old hardcoded `/tmp/swift_tokens.json` is still
    /// honoured so an existing dump keeps working.
    #[test]
    #[ignore = "needs a tokenizer and a Swift dump; set LUMEN_DIFFUSION_SWIFT_TOKENS"]
    fn flux_tokenize_matches_swift() {
        let Some(tok) = tokenizer_json() else {
            eprintln!("skipping: set LUMEN_DIFFUSION_TOKENIZER_DIR");
            return;
        };
        let dump = std::env::var("LUMEN_DIFFUSION_SWIFT_TOKENS")
            .unwrap_or_else(|_| "/tmp/swift_tokens.json".to_string());
        let Ok(raw) = std::fs::read_to_string(&dump) else {
            eprintln!("skipping: no Swift dump at {dump}");
            return;
        };
        let expected: Vec<i32> = serde_json::from_str(&raw).expect("parse swift tokens");
        assert_eq!(expected.len(), FLUX_SEQ_LEN, "reference must be 512 ids");
        let got = tokenize_flux_prompt_with(VALIDATION_PROMPT, &tok).expect("tokenize");
        let diffs: Vec<(usize, i32, i32)> = got
            .iter()
            .zip(expected.iter())
            .enumerate()
            .filter(|(_, (g, e))| g != e)
            .map(|(i, (g, e))| (i, *g, *e))
            .collect();
        assert!(
            diffs.is_empty(),
            "tokenization mismatch: {} differing ids (pos, got, swift): {diffs:?}",
            diffs.len()
        );
    }
}
