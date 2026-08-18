//! Print the link-time flag registry as JSON — the bridge between the
//! in-source `lumen_flags::flag!` declarations and `cargo xtask flags`.
//!
//! linkme collects `FlagDesc` entries from every crate linked into this
//! binary, so the output covers lumen-mlx's flags (and, transitively, any
//! dependency that declares some). Built with `mlx-native` because most flags
//! live inside that gate; a flag declared but feature-gated off is invisible
//! here, which is the correct answer — it is invisible to the server too.
//!
//! ```text
//! cargo run -p lumen-mlx --features mlx-native --example dump_flags
//! ```

fn main() {
    // Link anchor — load-bearing, not decorative. Without at least one
    // referenced symbol from lumen-mlx, the linker pulls nothing from its
    // rlib and the registry prints EMPTY (observed, not theorized). One
    // reference suffices to pull the crate today, but that is a linker
    // behavior, not a contract — which is why `cargo xtask flags --check`
    // also diffs this dump against a source grep for `env: "LUMEN_` and
    // fails if the linker ever starts dropping entries.
    lumen_mlx::flags_link_anchor();

    // The registry is static; nothing needs the GPU or a model. One JSON
    // object per line so xtask can parse without a serde dependency on the
    // exact schema.
    for d in lumen_flags::registry_sorted() {
        println!(
            "{}",
            serde_json::json!({
                "env": d.env,
                "default": d.default,
                "kind": d.kind.as_str(),
                "declared_in": d.declared_in,
                "doc": d.doc.trim(),
            })
        );
    }
}
