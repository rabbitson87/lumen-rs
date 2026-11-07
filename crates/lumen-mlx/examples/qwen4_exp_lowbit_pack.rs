//! Build a low-bit copy of a Qwen3.8-Flash-Next checkpoint's routed experts, next to its shards.
//!
//! ```text
//! cargo run --release -p lumen-mlx --features mlx-native --example qwen4_exp_lowbit_pack -- \
//!     <model_dir> [--bits 2] [--group 64] [--recipe lsq|minmax]
//! ```
//!
//! Prints how close each projection stayed, in dB. The copy is another model, not a cache of this
//! one: nothing reads it unless asked to.

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    use anyhow::{Context, bail};
    use lumen_mlx::qwen4_exp::Qwen4ExpConfig;
    use lumen_mlx::{LowbitSpec, Recipe, build_lowbit_pack};
    use std::path::PathBuf;

    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        let v = args.get(i + 1).cloned();
        args.drain(i..(i + 2).min(args.len()));
        v
    };
    let bits: u32 = take("--bits")
        .map_or(Ok(2), |v| v.parse())
        .context("--bits")?;
    let group: usize = take("--group")
        .map_or(Ok(64), |v| v.parse())
        .context("--group")?;
    let recipe = Recipe::parse(take("--recipe").as_deref().unwrap_or("lsq"))?;
    let [dir] = args.as_slice() else {
        bail!("usage: <model_dir> [--bits 2] [--group 64] [--recipe lsq|minmax]");
    };
    let dir = PathBuf::from(dir);
    let cfg = Qwen4ExpConfig::load(&dir)?;
    let base = "language_model.model.layers.0.mlp.switch_mlp";
    let src_bits = cfg.quant.for_base(base).bits as u32;
    let path = build_lowbit_pack(&dir, src_bits, LowbitSpec::new(bits, group, recipe)?)?;
    println!("{}", path.display());
    Ok(())
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("qwen4_exp_lowbit_pack needs --features mlx-native");
}
