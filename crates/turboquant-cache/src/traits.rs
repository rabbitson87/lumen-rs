/// KV cache abstraction for TurboQuant-compressed attention.
///
/// Implemented by `SimpleCache` (contiguous memory, no paging). A vLLM-style
/// `PagedCache` was the intended second implementor; task 007 measured what
/// paging would reclaim on this workload (under 1% of process memory) and the
/// paged crate was deleted instead — see `docs/maintainer-workflow.md` §9.
/// The trait stays an abstraction because the seam is cheap, not because a
/// second implementation is pending.
pub trait KVCache: Send + Sync {
    /// Store KV vectors for a given layer during prefill/decode.
    ///
    /// `keys` and `values` are `[n_tokens, head_dim]` row-major f32 slices.
    /// The implementation compresses and stores them.
    fn store(&mut self, layer: usize, head: usize, keys: &[f32], values: &[f32], n_tokens: usize);

    /// Compute attention scores between a query and all cached keys for a layer/head.
    ///
    /// `query` is `[head_dim]`. Returns scores of length `seq_len(layer, head)`.
    fn attend_keys(&self, layer: usize, head: usize, query: &[f32]) -> Vec<f32>;

    /// Compute weighted sum of cached values using attention weights.
    ///
    /// `weights` has length `seq_len(layer, head)`. Returns `[head_dim]` output.
    fn attend_values(&self, layer: usize, head: usize, weights: &[f32]) -> Vec<f32>;

    /// Number of cached tokens for a given layer/head.
    fn seq_len(&self, layer: usize, head: usize) -> usize;

    /// Clear all cached data.
    fn clear(&mut self);

    /// Drop the tail of the cached sequence so each `(layer, head)` pair is
    /// left with at most `n_keep` tokens. Used by speculative decoding to
    /// roll back state after partially-accepted verify batches. No-op if
    /// the current length is already ≤ `n_keep`.
    fn truncate(&mut self, layer: usize, n_keep: usize);
}
