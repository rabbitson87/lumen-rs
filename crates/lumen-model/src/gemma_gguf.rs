use anyhow::{Context, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_transformers::models::quantized_gemma4::ModelWeights;

use crate::loader;
use crate::sampling::{sample_token, sample_token_cpu};

/// Gemma 4 GGUF quantized model wrapper.
///
/// Loads Q4/Q8 quantized GGUF files for reduced memory usage.
pub struct GemmaGgufModel {
    model: ModelWeights,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl GemmaGgufModel {
    /// Enable TurboQuant compressed KV cache with Metal GPU kernels.
    ///
    /// `bits`: quantization bits (2, 3, or 4). 3-bit recommended.
    /// `n_layers`, `n_kv_heads`, `head_dim`: model architecture params.
    #[cfg(feature = "turboquant-gpu")]
    pub fn enable_compressed_kv(
        &mut self,
        bits: u32,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        use lumen_core::config::TurboQuantConfig;
        use lumen_metal::GpuCompressor;

        let config = TurboQuantConfig {
            bits,
            qjl_m: head_dim / 2,
            seed: 42,
            lloyd_max_iter: 1000,
            head_dim,
        };
        let max_seq_len: usize = std::env::var("TQ_MAX_SEQ")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192);
        let compressor = GpuCompressor::new(config, n_layers, n_kv_heads, max_seq_len)?;
        self.model.set_compressed_kv(Box::new(compressor));
        eprintln!(
            "TurboQuant GPU KV cache enabled: {}-bit, {} layers, {} heads, dim={}",
            bits, n_layers, n_kv_heads, head_dim
        );
        Ok(())
    }

    /// Enable vLLM-style PagedAttention KV cache on Metal GPU.
    ///
    /// Single-sequence mode — multi-sequence continuous batching is Phase 5.
    /// Threshold-gated like TurboQuant: used for decode when context is long
    /// enough. Prefill always falls back to standard SDPA.
    #[cfg(feature = "paged-kv")]
    pub fn enable_paged_kv(
        &mut self,
        n_layers: u32,
        n_kv_heads: u32,
        head_dim_sliding: u32,
        head_dim_global: u32,
        global_every: u32,
    ) -> Result<()> {
        use crate::paged_kv::PagedKVBackend;

        let budget_mb: usize = std::env::var("PAGED_KV_MB")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048);
        let block_size: u32 = std::env::var("PAGED_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(16);

        let backend = PagedKVBackend::new(
            self.device.clone(),
            budget_mb,
            block_size,
            n_layers,
            n_kv_heads,
            head_dim_sliding,
            head_dim_global,
            global_every,
        )?;
        self.model.set_compressed_kv(Box::new(backend));
        eprintln!(
            "PagedAttention KV cache enabled: {}MB budget, block_size={}, {} layers, {} kv_heads, head_dim={}/{} (sliding/global)",
            budget_mb, block_size, n_layers, n_kv_heads, head_dim_sliding, head_dim_global
        );
        Ok(())
    }
}

impl GemmaGgufModel {
    /// Load a Gemma 4 model from a local GGUF file.
    pub fn load(gguf_path: &str, tokenizer_id: &str) -> Result<Self> {
        eprintln!("Loading Gemma4 GGUF model: {gguf_path}");

        let device = loader::get_device()?;

        // Load GGUF
        let mut file = std::fs::File::open(gguf_path)
            .with_context(|| format!("failed to open {gguf_path}"))?;
        let content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| anyhow::anyhow!("GGUF parse error: {e}"))?;

        // Print metadata
        if let Some(arch) = content.metadata.get("general.architecture") {
            eprintln!("  architecture: {arch:?}");
        }

        let model = ModelWeights::from_gguf(content, &mut file, &device)
            .map_err(|e| anyhow::anyhow!("model load error: {e}"))?;
        eprintln!("  model loaded on {device:?}");

        // Load tokenizer from HF
        let tokenizer = loader::load_tokenizer(tokenizer_id)?;
        eprintln!(
            "  tokenizer loaded (vocab={})",
            tokenizer.get_vocab_size(false)
        );

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    /// Direct access to the underlying ModelWeights (used for continuous
    /// batching primitives: set_current_seq_id, forward, forward_batched_decode_v2).
    pub fn model_mut(&mut self) -> &mut ModelWeights {
        &mut self.model
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))
    }

    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::with_capacity(max_new_tokens);

        // Clear KV cache from previous requests
        self.model.clear_kv_cache();

        // Prefill
        let t_prefill = std::time::Instant::now();
        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, 0)?;
        let logits = logits.unsqueeze(0)?;
        let mut next_token = sample_token(&logits, temperature, top_p)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "  prefill: {} tokens in {:.0}ms ({:.1} tok/s)",
            input_ids.len(),
            prefill_ms,
            input_ids.len() as f64 / (prefill_ms / 1000.0)
        );
        generated.push(next_token);

        // Decode with per-step wall-clock timing
        let t_decode = std::time::Instant::now();

        for i in 1..max_new_tokens {
            if is_eos(next_token) {
                break;
            }
            let t_step = std::time::Instant::now();
            let seqlen_offset = input_ids.len() + i - 1;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, seqlen_offset)?;
            let logits_cpu = logits
                .squeeze(0)?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?;
            static REPEAT_PENALTY: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
            let rp = *REPEAT_PENALTY.get_or_init(|| {
                std::env::var("REPEAT_PENALTY")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(1.0)
            });

            next_token = sample_token_cpu(&logits_cpu, temperature, top_p, rp, &generated)?;

            eprintln!(
                "  step {}: {:.0}ms (tok={})",
                seqlen_offset,
                t_step.elapsed().as_secs_f64() * 1000.0,
                next_token,
            );
            generated.push(next_token);
        }
        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_decoded = generated.len();
        if n_decoded > 0 {
            eprintln!(
                "  decode: {} tokens in {:.0}ms ({:.1} tok/s)",
                n_decoded,
                decode_ms,
                n_decoded as f64 / (decode_ms / 1000.0)
            );
        }

        Ok(generated)
    }

    /// Speculative decoding: use first `draft_layers` layers as draft model.
    /// Generates `draft_n` candidate tokens per cycle, verifies with full model.
    pub fn generate_speculative(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
        draft_layers: usize,
        draft_n: usize,
    ) -> Result<Vec<u32>> {
        let mut generated = Vec::with_capacity(max_new_tokens);
        self.model.clear_kv_cache();

        // Prefill (full model)
        let t_prefill = std::time::Instant::now();
        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, 0)?;
        let logits = logits.unsqueeze(0)?;
        let mut next_token = sample_token(&logits, temperature, top_p)?;
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "  prefill: {} tokens in {:.0}ms ({:.1} tok/s)",
            input_ids.len(),
            prefill_ms,
            input_ids.len() as f64 / (prefill_ms / 1000.0)
        );
        generated.push(next_token);

        let mut pos = input_ids.len();
        let t_decode = std::time::Instant::now();
        let mut total_accepted = 0usize;
        let mut total_drafted = 0usize;

        while generated.len() < max_new_tokens {
            if is_eos(next_token) {
                break;
            }

            // --- Draft phase: generate candidates with first K layers ---
            let kv_snapshot = self.model.snapshot_kv_cache();
            let mut draft_tokens = Vec::with_capacity(draft_n);
            let mut draft_token = next_token;

            for di in 0..draft_n {
                let input = Tensor::new(&[draft_token], &self.device)?.unsqueeze(0)?;
                let logits =
                    self.model
                        .forward_with_layers(&input, pos + di, Some(draft_layers), false)?;
                let logits_cpu = logits
                    .squeeze(0)?
                    .to_dtype(candle_core::DType::F32)?
                    .to_vec1::<f32>()?;
                // Greedy draft (temperature=0)
                draft_token = logits_cpu
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i as u32)
                    .unwrap_or(0);
                draft_tokens.push(draft_token);
                if is_eos(draft_token) {
                    break;
                }
            }

            // --- Rollback KV cache to pre-draft state ---
            self.model.rollback_kv_cache(&kv_snapshot);

            // --- Verify: full model on [next_token, draft_0, ..., draft_N-1] ---
            let mut verify_input: Vec<u32> = Vec::with_capacity(1 + draft_tokens.len());
            verify_input.push(next_token);
            verify_input.extend_from_slice(&draft_tokens);

            let verify_tensor = Tensor::new(verify_input.as_slice(), &self.device)?.unsqueeze(0)?;
            let verify_logits = self.model.forward_with_layers(
                &verify_tensor,
                pos,
                None,
                true, // full model, all-position logits
            )?;
            // verify_logits: [1, N+1, vocab]
            let verify_logits = verify_logits
                .squeeze(0)?
                .to_dtype(candle_core::DType::F32)?;

            // --- Accept/reject ---
            let mut accepted = 0;
            for i in 0..draft_tokens.len() {
                let logits_i = verify_logits.i(i)?.to_vec1::<f32>()?;
                let verified = sample_token_cpu(&logits_i, temperature, top_p, 1.0, &generated)?;
                if verified == draft_tokens[i] {
                    generated.push(draft_tokens[i]);
                    accepted += 1;
                } else {
                    // Mismatch: use verified token instead
                    generated.push(verified);
                    next_token = verified;
                    break;
                }
            }

            total_accepted += accepted;
            total_drafted += draft_tokens.len();

            if accepted == draft_tokens.len() {
                // All accepted — sample next from last verify logit
                let last_logits = verify_logits.i(draft_tokens.len())?.to_vec1::<f32>()?;
                next_token = sample_token_cpu(&last_logits, temperature, top_p, 1.0, &generated)?;
                generated.push(next_token);
            }

            // Trim KV cache to only keep verified tokens
            let keep = accepted + 1; // +1 for the original next_token
            let trim_to: Vec<usize> = kv_snapshot.iter().map(|&s| s + keep).collect();
            self.model.rollback_kv_cache(&trim_to);
            pos += keep;

            eprintln!(
                "  spec: pos={}, accepted={}/{}, total_tok={}",
                pos,
                accepted,
                draft_tokens.len(),
                generated.len()
            );
        }

        let decode_ms = t_decode.elapsed().as_secs_f64() * 1000.0;
        let n_decoded = generated.len();
        let accept_rate = if total_drafted > 0 {
            total_accepted as f64 / total_drafted as f64
        } else {
            0.0
        };
        eprintln!(
            "  decode: {} tokens in {:.0}ms ({:.1} tok/s, accept={:.0}%)",
            n_decoded,
            decode_ms,
            n_decoded as f64 / (decode_ms / 1000.0),
            accept_rate * 100.0
        );

        Ok(generated)
    }

    pub fn chat(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        self.chat_with_options(messages, max_new_tokens, temperature, false)
    }

    pub fn chat_with_options(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        thinking: bool,
    ) -> Result<String> {
        let input_ids = self.build_chat_input(messages, thinking)?;

        if thinking {
            let total_budget = max_new_tokens * 2;
            let output_ids = self.generate(&input_ids, total_budget, temperature, 0.9)?;
            let full_text = self.decode(&output_ids)?;
            if let Some(end_pos) = full_text.find("<channel|>") {
                let after = &full_text[end_pos + "<channel|>".len()..];
                Ok(after.trim_start().to_string())
            } else {
                Ok(full_text)
            }
        } else {
            let output_ids = self.generate(&input_ids, max_new_tokens, temperature, 0.9)?;
            self.decode(&output_ids)
        }
    }

    /// Streaming chat: calls `on_token` with each new text fragment as it's decoded.
    /// For thinking mode, falls back to non-streaming (thinking output is stripped).
    pub fn chat_streaming<F>(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
        thinking: bool,
        mut on_token: F,
    ) -> Result<String>
    where
        F: FnMut(&str),
    {
        if thinking {
            let text = self.chat_with_options(messages, max_new_tokens, temperature, true)?;
            on_token(&text);
            return Ok(text);
        }

        let input_ids = self.build_chat_input(messages, false)?;

        let mut generated = Vec::with_capacity(max_new_tokens);
        self.model.clear_kv_cache();

        // Prefill
        let input_tensor = Tensor::new(&input_ids[..], &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, 0)?;
        let logits = logits.unsqueeze(0)?;
        let mut next_token = sample_token(&logits, temperature, 0.9)?;
        generated.push(next_token);

        // Incremental decode state — buffer incomplete multi-byte characters
        let mut prev_text = String::new();
        if let Ok(text) = self.decode(&generated) {
            if !text.is_empty() && !text.contains('\u{FFFD}') {
                on_token(&text);
                prev_text = text;
            }
        }

        // Decode loop
        static REPEAT_PENALTY: std::sync::OnceLock<f32> = std::sync::OnceLock::new();
        let rp = *REPEAT_PENALTY.get_or_init(|| {
            std::env::var("REPEAT_PENALTY")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(1.0)
        });

        let debug_timing = std::env::var("BATCHED_TIMING").is_ok();
        for i in 1..max_new_tokens {
            if is_eos(next_token) {
                break;
            }
            let t_step = std::time::Instant::now();
            let seqlen_offset = input_ids.len() + i - 1;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let t_fwd = std::time::Instant::now();
            let logits = self.model.forward(&input, seqlen_offset)?;
            let t_cpu = std::time::Instant::now();
            let logits_cpu = logits
                .squeeze(0)?
                .to_dtype(candle_core::DType::F32)?
                .to_vec1::<f32>()?;
            let t_sample = std::time::Instant::now();
            next_token = sample_token_cpu(&logits_cpu, temperature, 0.9, rp, &generated)?;
            generated.push(next_token);
            let t_done = std::time::Instant::now();
            if debug_timing {
                eprintln!(
                    "  seq step: tok={:.2}ms fwd={:.2}ms cpu={:.2}ms sample={:.2}ms total={:.2}ms",
                    (t_fwd - t_step).as_secs_f64() * 1000.0,
                    (t_cpu - t_fwd).as_secs_f64() * 1000.0,
                    (t_sample - t_cpu).as_secs_f64() * 1000.0,
                    (t_done - t_sample).as_secs_f64() * 1000.0,
                    (t_done - t_step).as_secs_f64() * 1000.0,
                );
            }

            // Incremental decode — skip if incomplete multi-byte (U+FFFD present)
            if let Ok(text) = self.decode(&generated) {
                if text.len() > prev_text.len() && !text.contains('\u{FFFD}') {
                    on_token(&text[prev_text.len()..]);
                    prev_text = text;
                }
            }
        }

        // Flush any remaining buffered text
        if let Ok(text) = self.decode(&generated) {
            if text.len() > prev_text.len() {
                let delta = &text[prev_text.len()..];
                let clean: String = delta.replace('\u{FFFD}', "");
                if !clean.is_empty() {
                    on_token(&clean);
                }
            }
        }

        self.decode(&generated)
    }

    /// Build input token IDs from chat messages.
    /// Build chat-template-formatted token ids for `messages`. Public so a
    /// multi-seq scheduler can tokenize prompts before prefill.
    pub fn build_chat_input(
        &self,
        messages: &[(String, String)],
        thinking: bool,
    ) -> Result<Vec<u32>> {
        let mut input_ids = vec![BOS_TOKEN];

        for (role, content) in messages {
            let gemma_role = match role.as_str() {
                "assistant" => "model",
                other => other,
            };
            input_ids.push(TURN_START);
            input_ids.extend_from_slice(&self.encode(&format!("{gemma_role}\n"))?);
            input_ids.extend_from_slice(&self.encode(content)?);
            input_ids.push(TURN_END);
            input_ids.extend_from_slice(&self.encode("\n")?);
        }

        input_ids.push(TURN_START);
        input_ids.extend_from_slice(&self.encode("model\n")?);

        if !thinking {
            input_ids.extend_from_slice(&[
                CHANNEL_START,
                THOUGHT_TOKEN,
                NEWLINE_TOKEN,
                CHANNEL_END,
            ]);
        }

        Ok(input_ids)
    }
}

fn is_eos(token: u32) -> bool {
    matches!(token, 1 | 106)
}

/// Public EOS check for Gemma 4 tokens (1 = `<eos>`, 106 = `<end_of_turn>`).
pub fn is_eos_gemma(token: u32) -> bool {
    is_eos(token)
}

const BOS_TOKEN: u32 = 2;
const TURN_START: u32 = 105; // <|turn>
const TURN_END: u32 = 106; // <turn|>
const CHANNEL_START: u32 = 100; // <|channel>
const CHANNEL_END: u32 = 101; // <channel|>
const THOUGHT_TOKEN: u32 = 45518; // "thought"
const NEWLINE_TOKEN: u32 = 107; // "\n"
