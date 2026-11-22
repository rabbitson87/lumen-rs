//! Validates an `inkling2_loop` checkpoint directory without loading weights:
//! `config.json`, every shard header, the tensor manifest and the packed shapes.
//! Prints the tensor totals, the layer-kind table and the expert quantization.
//! Exits non-zero on the first error.
//!
//! `cargo run -q -p lumen-mlx --example inkling2_manifest -- <checkpoint dir>`

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, ensure};
use lumen_mlx::inkling2::{
    Inkling2Config, LayerKind, Manifest, Part, QuantTable, TensorInfo, check_packed_shapes,
    read_header,
};

fn main() -> Result<()> {
    let dir = PathBuf::from(
        std::env::args_os()
            .nth(1)
            .context("usage: inkling2_manifest <checkpoint dir>")?,
    );
    let cfg = Inkling2Config::load(&dir)?;

    let mut per_file: BTreeMap<String, usize> = BTreeMap::new();
    let mut tensors: Vec<TensorInfo> = Vec::new();
    for shard in safetensors_in(&dir)? {
        let headers = read_header(&shard)?;
        let name = shard
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or_default()
            .to_string();
        per_file.insert(name, headers.len());
        tensors.extend(headers);
    }
    let manifest = Manifest::build(&cfg, &tensors)?;
    let quant = QuantTable::from_block(&cfg.quantization)?;
    let packed = check_packed_shapes(&tensors, &quant, cfg.text_config.n_routed_experts)?;

    println!("checkpoint {}", dir.display());
    println!("tensors per file:");
    for (file, n) in &per_file {
        println!("  {file:<36} {n:>5}");
    }
    println!("  {:<36} {:>5}", "total", tensors.len());
    println!(
        "  text {}, vision {}, mtp {}",
        manifest.count(Part::Text),
        manifest.count(Part::Vision),
        manifest.count(Part::Mtp)
    );

    let text = &cfg.text_config;
    let core = text.loop_config;
    println!();
    println!(
        "layer kinds (num_hidden_layers {}, core {}..{}):",
        text.num_hidden_layers, core.core_start, core.core_end
    );
    let kinds = (0..text.num_hidden_layers).map(|layer| {
        let kind = match cfg.layer_kind(layer) {
            LayerKind::Local => "local",
            LayerKind::Global => "global",
        };
        if cfg.is_core(layer) {
            format!("{kind}, core")
        } else {
            kind.to_string()
        }
    });
    for (first, last, label) in ranges(kinds) {
        println!("  {:<10} {label}", range_label(first, last));
    }

    println!();
    println!("expert quantization (w13 / w2):");
    let experts = (0..text.num_hidden_layers).map(|layer| {
        let w13 = expert_label(&quant, layer, "w13");
        let w2 = expert_label(&quant, layer, "w2");
        format!("{w13} / {w2}")
    });
    for (first, last, label) in ranges(experts) {
        println!("  {:<10} {label}", range_label(first, last));
    }

    println!();
    println!(
        "packed shapes: {} quantized modules ok (affine {}, nvfp4 {})",
        packed.affine + packed.nvfp4,
        packed.affine,
        packed.nvfp4
    );
    println!("OK");
    Ok(())
}

/// Every `.safetensors` file directly in `dir`, sorted. Errors when there is none.
fn safetensors_in(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut paths = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("read {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "safetensors") {
            paths.push(path);
        }
    }
    ensure!(
        !paths.is_empty(),
        "no .safetensors files in {}",
        dir.display()
    );
    paths.sort();
    Ok(paths)
}

/// One expert projection's spec, with whether it is an override or the default.
fn expert_label(quant: &QuantTable, layer: usize, proj: &str) -> String {
    let module = format!("model.layers.{layer}.mlp.experts.{proj}");
    let spec = quant.resolve(&module);
    let source = if quant.is_override(&module) {
        "override"
    } else {
        "default"
    };
    format!(
        "{proj} {} {}-bit g{} ({source})",
        spec.mode.name(),
        spec.bits,
        spec.group_size
    )
}

/// Consecutive layers with the same label, as `(first, last, label)`.
fn ranges(labels: impl Iterator<Item = String>) -> Vec<(usize, usize, String)> {
    let mut rows: Vec<(usize, usize, String)> = Vec::new();
    for (layer, label) in labels.enumerate() {
        match rows.last_mut() {
            Some(row) if row.2 == label => row.1 = layer,
            _ => rows.push((layer, layer, label)),
        }
    }
    rows
}

fn range_label(first: usize, last: usize) -> String {
    if first == last {
        first.to_string()
    } else {
        format!("{first}..={last}")
    }
}
