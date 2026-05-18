pub use candle_transformers::models::gemma4::config_e4b::Gemma3nConfig;
/// Re-export Candle's model configs.
pub use candle_transformers::models::qwen2::Config as Qwen2Config;

/// Qwen2.5-0.5B config (for fast testing).
pub fn qwen2_5_0_5b() -> Qwen2Config {
    Qwen2Config {
        vocab_size: 151936,
        hidden_size: 896,
        intermediate_size: 4864,
        num_hidden_layers: 24,
        num_attention_heads: 14,
        num_key_value_heads: 2,
        max_position_embeddings: 32768,
        sliding_window: 32768,
        max_window_layers: 21,
        tie_word_embeddings: true,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    }
}

/// Qwen2.5-1.5B config.
pub fn qwen2_5_1_5b() -> Qwen2Config {
    Qwen2Config {
        vocab_size: 151936,
        hidden_size: 1536,
        intermediate_size: 8960,
        num_hidden_layers: 28,
        num_attention_heads: 12,
        num_key_value_heads: 2,
        max_position_embeddings: 32768,
        sliding_window: 32768,
        max_window_layers: 21,
        tie_word_embeddings: true,
        rope_theta: 1_000_000.0,
        rms_norm_eps: 1e-6,
        use_sliding_window: false,
        hidden_act: candle_nn::Activation::Silu,
    }
}
