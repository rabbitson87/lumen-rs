use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::gemma4;

use crate::loader;
use crate::sampling::sample_token;

pub use gemma4::config_e4b::Gemma3nConfig;

/// Gemma 4 E4B model wrapper with tokenizer and generation support.
///
/// Uses the E4B TextModel (AltUp + PLE + Laurel architecture).
pub struct GemmaModel {
    model: gemma4::text_e4b::TextModel,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    config: Gemma3nConfig,
}

impl GemmaModel {
    /// Load a Gemma 4 E4B model from HuggingFace Hub.
    pub fn load(model_id: &str) -> Result<Self> {
        eprintln!("Loading Gemma4 E4B model: {model_id}");

        let config: Gemma3nConfig = loader::load_config(model_id)?;
        let text_cfg = &config.text_config;
        eprintln!(
            "  layers={}, heads={}, kv_heads={}, hidden={}, head_dim={}",
            text_cfg.num_hidden_layers,
            text_cfg.num_attention_heads,
            text_cfg.num_key_value_heads,
            text_cfg.hidden_size,
            text_cfg.head_dim,
        );
        eprintln!(
            "  altup_inputs={}, laurel_rank={}, ple_dim={}",
            text_cfg.altup_num_inputs, text_cfg.laurel_rank, text_cfg.hidden_size_per_layer_input,
        );

        let tokenizer = loader::load_tokenizer(model_id)?;
        eprintln!(
            "  tokenizer loaded (vocab={})",
            tokenizer.get_vocab_size(false)
        );

        let (vb, device) = loader::load_weights(model_id, DType::BF16)?;

        // E4B TextModel::new(vb) does vb.pp("model") internally,
        // but actual tensor keys are "model.language_model.*".
        // Rename to insert "language_model." after "model.".
        let vb = vb.rename_f(|name| {
            if let Some(rest) = name.strip_prefix("model.") {
                format!("model.language_model.{rest}")
            } else {
                name.to_string()
            }
        });
        let model = gemma4::text_e4b::TextModel::new(&config.text_config, vb)?;
        eprintln!("  model loaded on {device:?}");

        Ok(Self {
            model,
            tokenizer,
            device,
            config,
        })
    }

    pub fn config(&self) -> &Gemma3nConfig {
        &self.config
    }

    pub fn text_config(
        &self,
    ) -> &candle_transformers::models::gemma4::config_e4b::Gemma3nTextConfig {
        &self.config.text_config
    }

    pub fn tokenizer(&self) -> &tokenizers::Tokenizer {
        &self.tokenizer
    }

    /// Run forward and capture KV tensors for TurboQuant comparison.
    pub fn forward_capture_kv(&mut self, input_ids: &[u32]) -> Result<candle_core::Tensor> {
        let input_tensor = candle_core::Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        self.model.clear_kv_cache();
        let (logits, _kv) = self.model.forward_capture_kv(&input_tensor, 0)?;
        Ok(logits)
    }

    pub fn device(&self) -> &candle_core::Device {
        &self.device
    }

    /// Enable GPU TurboQuant compressed KV cache for all attention layers.
    /// Only compiled when the `turboquant` feature is on (forwards to the
    /// matching `candle-transformers/turboquant` cfg block).
    #[cfg(feature = "turboquant")]
    pub fn enable_turboquant(
        &mut self,
        bits: u32,
        _n_layers: usize,
        _n_kv_heads: usize,
        _head_dim: usize,
    ) {
        self.model
            .enable_turboquant_gpu(bits, 42, 1000)
            .expect("failed to enable TurboQuant GPU");
        eprintln!("  TurboQuant GPU enabled: {bits}-bit");
    }

    /// Encode text to token IDs (without adding special tokens).
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
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

        // Prefill
        let input_tensor = Tensor::new(input_ids, &self.device)?.unsqueeze(0)?;
        let logits = self.model.forward(&input_tensor, 0)?;
        let mut next_token = sample_token(&logits, temperature, top_p)?;
        generated.push(next_token);

        // Decode
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

    /// Apply Gemma 4 chat template and tokenize without running the model.
    /// Used by the server to report exact `prompt_tokens`.
    pub fn build_chat_input(&self, messages: &[(String, String)]) -> Result<Vec<u32>> {
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

        Ok(input_ids)
    }

    /// Apply Gemma 4 chat template and generate a response.
    ///
    /// Template format:
    /// ```text
    /// <bos><|turn>user\n{content}<turn|>\n<|turn>model\n
    /// ```
    pub fn chat(
        &mut self,
        messages: &[(String, String)],
        max_new_tokens: usize,
        temperature: f32,
    ) -> Result<String> {
        let input_ids = self.build_chat_input(messages)?;

        eprintln!(
            "  prompt: {} tokens, first 30: {:?}",
            input_ids.len(),
            &input_ids[..input_ids.len().min(30)]
        );

        let output_ids = self.generate(&input_ids, max_new_tokens, temperature, 0.9)?;
        eprintln!(
            "  generated {} tokens, first 20: {:?}",
            output_ids.len(),
            &output_ids[..output_ids.len().min(20)]
        );
        self.decode(&output_ids)
    }
}

/// Check if token is an end-of-sequence token for Gemma 4.
fn is_eos(token: u32) -> bool {
    // 1 = <eos>, 106 = <turn|> (end of turn)
    matches!(token, 1 | 106)
}

// Gemma 4 special token IDs
const BOS_TOKEN: u32 = 2; // <bos>
const TURN_START: u32 = 105; // <|turn>
const TURN_END: u32 = 106; // <turn|>
