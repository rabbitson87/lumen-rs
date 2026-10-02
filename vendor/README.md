# Vendored crates

Third-party code kept in this repository because lumen-rs changes it. Each
directory keeps its upstream license and notices unchanged, and lists what was
changed in its `PATCHES.md`.

| Crate | Upstream | License | Why it is here |
|---|---|---|---|
| `fastokens` 0.3.2 | <https://github.com/crusoecloud/fastokens> | Apache-2.0 (bundles PCRE2, BSD-3, statically linked) | Its split cache tokenized a string differently depending on the previous call on the same thread; fixed here. Used by `lumen-mlx` behind `LUMEN_FASTOKENS`. |

These are not workspace members (`exclude` in the root `Cargo.toml`), so the
workspace's lints do not apply to them. `cargo fmt --all` does format them, as
it does every local path dependency; upstream's code was already formatted the
same way. Their own tests run with
`cargo test --manifest-path vendor/<crate>/Cargo.toml`.
