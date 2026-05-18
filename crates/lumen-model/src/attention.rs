/// TurboQuant attention integration point.
///
/// candle's built-in Qwen2 attention (standard KV caching).
/// The TurboQuant KV cache compression is ready in lumen-core/cache,
/// and will be hooked into a custom attention layer in a future iteration.
///
/// The hook point is in `candle_transformers::models::qwen2::Attention::forward()`:
/// - After Q, K, V projection and RoPE application
/// - Before KV cache update
/// - Replace `self.kv_cache = Some((k, v))` with TurboQuant compression
/// - Replace matmul attention with `cache.attend_keys()` + `cache.attend_values()`
///
/// This module is a placeholder for that future custom attention layer.
