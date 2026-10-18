//! Qwen3.8-Flash-Next (`qwen4_exp`) on MLX.
//!
//! A 48-layer hybrid: three Gated-DeltaNet layers for every full-attention layer
//! (which adds a block-sparse "QSA" indexer once the context outgrows its
//! budget), a sparse MoE block after each (512 routed experts, top 10, plus a
//! shared expert), a residual stream that is `hc_count` parallel streams mixed
//! by learned gates ("hyper-connections"), and an n-gram embedding injected
//! before one early layer.
//!
//! The routed experts are 70% of the file and the n-gram table is another 30%;
//! neither has to be resident. Experts go through [`crate::offload`], the table
//! through [`ngram_table`].
//!
//! The reference implementation this follows is the MLX port vendored by
//! Slotstream (`Tools/reference/qwen4_exp.py`); `tests/` checks the port against
//! it on a small random-weights replica of the checkpoint.

pub mod config;
pub mod model;
pub mod ngram;
pub mod ngram_table;
pub(crate) mod runner;

pub use config::{LayerKind, Qwen4ExpConfig};

/// Whether `dir` holds a Flash-Next checkpoint (its `config.json` says `qwen4_exp`).
pub(crate) fn is_qwen4_exp(dir: &std::path::Path) -> bool {
    std::fs::read_to_string(dir.join("config.json"))
        .ok()
        .and_then(|text| serde_json::from_str::<serde_json::Value>(&text).ok())
        .and_then(|v| {
            v.get("model_type")
                .and_then(|t| t.as_str().map(str::to_owned))
        })
        .is_some_and(|t| t == "qwen4_exp")
}
pub use model::{LoadOptions, NgramSource, Qwen4ExpCache, Qwen4ExpModel};

#[cfg(test)]
mod tests;
