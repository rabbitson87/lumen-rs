use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::qwen2::ModelForCausalLM;

use crate::config::Qwen2Config;
use crate::loader;
use crate::sampling::sample_token;

/// Qwen2 model wrapper with tokenizer and generation support.
pub struct QwenModel {
    model: ModelForCausalLM,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    config: Qwen2Config,
}

impl QwenModel {
    /// Load a Qwen2 model from HuggingFace Hub.
    pub fn load(model_id: &str) -> Result<Self> {
        eprintln!("Loading model: {model_id}");

        let config: Qwen2Config = loader::load_config(model_id)?;
        eprintln!(
            "  layers={}, heads={}, kv_heads={}, hidden={}",
            config.num_hidden_layers,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.hidden_size
        );

        let tokenizer = loader::load_tokenizer(model_id)?;
        eprintln!(
            "  tokenizer loaded (vocab={})",
            tokenizer.get_vocab_size(false)
        );

        let (vb, device) = loader::load_weights(model_id, DType::F16)?;
        let model = ModelForCausalLM::new(&config, vb)?;
        eprintln!("  model loaded on {device:?}");

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    /// Load model with a specific config (for testing with smaller models).
    pub fn load_with_config(model_id: &str, config: Qwen2Config) -> Result<Self> {
        let tokenizer = loader::load_tokenizer(model_id)?;
        let (vb, device) = loader::load_weights(model_id, DType::F16)?;
        let model = ModelForCausalLM::new(&config, vb)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenizer encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("tokenizer decode error: {e}"))
    }

    /// Generate tokens autoregressively.
    pub fn generate(
        &mut self,
        input_ids: &[u32],
        max_new_tokens: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Vec<u32>> {
        self.model.clear_kv_cache();
        let mut generated = Vec::with_capacity(max_new_tokens);

        // Prefill: process all input tokens at once
        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, 0)?;
        let mut next_token = sample_token(&logits, temperature, top_p)?;
        generated.push(next_token);

        // Decode: one token at a time
        for i in 1..max_new_tokens {
            if is_eos(next_token) {
                break;
            }
            let seqlen_offset = input_ids.len() + i - 1;
            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, seqlen_offset)?;
            next_token = sample_token(&logits, temperature, top_p)?;
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Apply chat template and tokenize, without running the model. Used by
    /// the server to report exact `prompt_tokens` in OpenAI usage payloads.
    pub fn build_chat_input(&self, messages: &[(String, String)]) -> Result<Vec<u32>> {
        let prompt = format_chat_prompt(messages);
        self.encode(&prompt)
    }

    /// Apply chat template and generate a response.
    pub fn chat(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        let input_ids = self.build_chat_input(messages)?;
        let output_ids = self.generate(&input_ids, max_new_tokens, temperature, 0.9)?;
        self.decode(&output_ids)
    }
}

/// Check if token is an end-of-sequence token.
fn is_eos(token: u32) -> bool {
    // Qwen2.5 EOS tokens
    matches!(token, 151643 | 151644 | 151645)
}

/// Format messages into Qwen2.5 chat template.
fn format_chat_prompt(messages: &[(String, String)]) -> String {
    let mut prompt = String::new();
    for (role, content) in messages {
        prompt.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    prompt.push_str("<|im_start|>assistant\n");
    prompt
}
