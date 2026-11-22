//! Integration tests for the `inkling2_loop` parsing layer, on the checked-in
//! fixtures (`config.json`, `tensors.json`). Negative cases mutate the parsed
//! fixture in memory; the truncation case writes a temporary shard.

use std::path::{Path, PathBuf};

use lumen_mlx::inkling2::{
    Inkling2Config, LayerKind, Manifest, PackedSummary, Part, Placement, QuantMode, QuantSpec,
    QuantTable, Role, TensorInfo, check_packed_shapes, read_header,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    tensors: Vec<TensorInfo>,
}

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/inkling2_loop")
}

fn fixture() -> (Inkling2Config, Vec<TensorInfo>) {
    let dir = fixture_dir();
    let cfg = Inkling2Config::load(&dir).expect("config.json validates");
    let text = std::fs::read_to_string(dir.join("tensors.json")).expect("tensors.json is readable");
    let parsed: Fixture = serde_json::from_str(&text).expect("tensors.json parses");
    (cfg, parsed.tensors)
}

fn manifest(cfg: &Inkling2Config, tensors: &[TensorInfo]) -> Manifest {
    Manifest::build(cfg, tensors).expect("the fixture maps")
}

fn quant(cfg: &Inkling2Config) -> QuantTable {
    QuantTable::from_block(&cfg.quantization).expect("quantization parses")
}

fn spec(mode: QuantMode, bits: u32, group_size: u32) -> QuantSpec {
    QuantSpec {
        mode,
        bits,
        group_size,
    }
}

fn layer_range(first: usize, end: usize) -> Vec<usize> {
    (first..end).collect()
}

#[test]
fn config_parses_and_validates() {
    let (cfg, _) = fixture();
    assert_eq!(cfg.text_config.num_hidden_layers, 24);
    assert_eq!(cfg.text_config.local_layer_ids.len(), 20);
    assert_eq!(cfg.global_layer_ids(), vec![5, 11, 17, 23]);
    assert_eq!(cfg.text_config.n_routed_experts, 128);
    let core = cfg.text_config.loop_config;
    assert_eq!((core.core_start, core.core_end), (6, 18));
    assert_eq!(cfg.vision_config.num_hidden_layers, 23);
    assert_eq!(cfg.mtp_config.num_hidden_layers, 3);
}

#[test]
fn quant_table_matches_the_config() {
    let (cfg, _) = fixture();
    let table = quant(&cfg);
    let affine4 = spec(QuantMode::Affine, 4, 64);
    let affine8 = spec(QuantMode::Affine, 8, 64);
    let nvfp4 = spec(QuantMode::Nvfp4, 4, 16);
    assert_eq!(table.default_spec(), affine4);
    assert_eq!(table.resolve("model.embed"), affine8);
    assert_eq!(table.resolve("lm_head"), affine8);
    for layer in 0..cfg.text_config.num_hidden_layers {
        let base = format!("model.layers.{layer}");
        let experts = if layer <= 19 { nvfp4 } else { affine4 };
        for proj in ["w13", "w2"] {
            let module = format!("{base}.mlp.experts.{proj}");
            assert_eq!(table.resolve(&module), experts, "{module}");
            assert_eq!(table.is_override(&module), layer <= 19, "{module}");
        }
        assert_eq!(table.resolve(&format!("{base}.attn.o_proj")), affine8);
        match cfg.layer_kind(layer) {
            LayerKind::Local => {
                assert_eq!(table.resolve(&format!("{base}.attn.in_proj")), affine8);
            }
            LayerKind::Global => {
                for proj in ["q_proj", "k_proj", "v_proj"] {
                    let module = format!("{base}.attn.{proj}");
                    assert_eq!(table.resolve(&module), affine8, "{module}");
                }
            }
        }
    }
}

#[test]
fn manifest_maps_all_1152_fixture_tensors_and_none_is_unmapped() {
    let (cfg, tensors) = fixture();
    assert_eq!(tensors.len(), 1152);
    let m = manifest(&cfg, &tensors);
    assert_eq!(m.entries().len(), 1152);
    assert_eq!(m.count(Part::Text), 676);
    assert_eq!(m.count(Part::Vision), 422);
    assert_eq!(m.count(Part::Text) + m.count(Part::Vision), 1098);
    assert_eq!(m.count(Part::Mtp), 54);
}

#[test]
fn loop_roles_are_on_exactly_the_core_layers() {
    let (cfg, tensors) = fixture();
    let m = manifest(&cfg, &tensors);
    let core = layer_range(6, 18);
    for role in [Role::LoopDown, Role::LoopGainForget, Role::LoopMessageQuery] {
        assert_eq!(m.layers_with(role), core, "{}", role.key());
    }
}

#[test]
fn attn_input_query_is_on_23_layers_and_absent_on_layer_0() {
    let (cfg, tensors) = fixture();
    let m = manifest(&cfg, &tensors);
    let layers = m.layers_with(Role::AttnInputQuery);
    assert_eq!(layers.len(), 23);
    assert!(!layers.contains(&0));
    assert_eq!(layers, layer_range(1, 24));
}

#[test]
fn global_layers_have_k_static() {
    let (cfg, tensors) = fixture();
    let m = manifest(&cfg, &tensors);
    let globals = cfg.global_layer_ids();
    assert_eq!(m.layers_with(Role::AttnKStatic), globals);
    assert_eq!(m.layers_with(Role::AttnVStatic), globals);
}

#[test]
fn packed_shapes_pass_for_every_quantized_module() {
    let (cfg, tensors) = fixture();
    let table = quant(&cfg);
    let summary = check_packed_shapes(&tensors, &table, cfg.text_config.n_routed_experts)
        .expect("every quantized module is consistent");
    // 20 in_proj + 24 o_proj + 12 q/k/v + 48 experts + embed + lm_head = 106 modules.
    // Experts of layers 0..=19 are nvfp4 (40 modules); the rest are affine (66).
    assert_eq!(
        summary,
        PackedSummary {
            affine: 66,
            nvfp4: 40
        }
    );
}

#[test]
fn an_unknown_key_is_an_error_that_names_the_key() {
    let (cfg, mut tensors) = fixture();
    let key = "model.layers.3.attn.mystery.weight";
    tensors.push(TensorInfo {
        name: key.to_string(),
        dtype: "BF16".to_string(),
        shape: vec![1],
        file: "model-00001-of-00004.safetensors".to_string(),
    });
    let err = Manifest::build(&cfg, &tensors).expect_err("an unknown key is rejected");
    let text = err.to_string();
    assert!(
        text.contains("unknown tensor key") && text.contains(key),
        "{text}"
    );
}

#[test]
fn deleting_layer_5_q_proj_is_an_error() {
    let (cfg, mut tensors) = fixture();
    tensors.retain(|t| !t.name.starts_with("model.layers.5.attn.q_proj."));
    let err = Manifest::build(&cfg, &tensors).expect_err("a missing q_proj is rejected");
    let text = err.to_string();
    assert!(
        text.contains("layer 5") && text.contains("is missing model.layers.5.attn.q_proj."),
        "{text}"
    );
}

#[test]
fn attn_in_proj_on_global_layer_5_is_an_error() {
    let (cfg, mut tensors) = fixture();
    tensors.push(TensorInfo {
        name: "model.layers.5.attn.in_proj.weight".to_string(),
        dtype: "U32".to_string(),
        shape: vec![8600, 576],
        file: "model-00001-of-00004.safetensors".to_string(),
    });
    let err = Manifest::build(&cfg, &tensors).expect_err("a local-only role on a global layer");
    let text = err.to_string();
    assert!(
        text.contains("model.layers.5.attn.in_proj.weight is on layer 5")
            && text.contains("which may not carry it"),
        "{text}"
    );
}

fn find_mut<'a>(tensors: &'a mut [TensorInfo], name: &str) -> &'a mut TensorInfo {
    tensors
        .iter_mut()
        .find(|t| t.name == name)
        .unwrap_or_else(|| panic!("the fixture has {name}"))
}

type Edit = Box<dyn Fn(&mut Vec<TensorInfo>)>;

/// Each case breaks one layout rule of the fixture, and the error must be that rule's.
#[test]
fn each_layout_rule_rejects_its_own_violation() {
    const IN_PROJ: &str = "model.layers.0.attn.in_proj.weight";
    let once = Role::ALL
        .iter()
        .copied()
        .find(|r| r.part() == Part::Mtp && r.placement() == Placement::Once)
        .expect("an MTP role placed once");
    let cases: Vec<(&str, Edit, &str)> = vec![
        (
            "an MTP tensor in a main shard",
            Box::new(|t| {
                find_mut(t, "fuse_weight").file = "model-00001-of-00004.safetensors".into();
            }),
            "but sits in",
        ),
        (
            "a text layer past the last one",
            Box::new(|t| {
                let mut extra = find_mut(t, IN_PROJ).clone();
                extra.name = "model.layers.24.attn.in_proj.weight".into();
                t.push(extra);
            }),
            "layer 24 is not below the 24",
        ),
        (
            "a missing tensor that appears once",
            Box::new(move |t| t.retain(|x| x.name != once.key())),
            "missing required tensor",
        ),
        (
            "one tensor listed in two shards",
            Box::new(|t| {
                let mut copy = find_mut(t, IN_PROJ).clone();
                copy.file = "model-00002-of-00004.safetensors".into();
                t.push(copy);
            }),
            "repeats the role",
        ),
        (
            "a zero-padded layer index",
            Box::new(|t| find_mut(t, IN_PROJ).name = "model.layers.00.attn.in_proj.weight".into()),
            "unknown tensor key \"model.layers.00.attn.in_proj.weight\"",
        ),
    ];
    let (cfg, tensors) = fixture();
    for (what, edit, expected) in cases {
        let mut broken = tensors.clone();
        edit(&mut broken);
        let err = Manifest::build(&cfg, &broken).expect_err(what);
        assert!(err.to_string().contains(expected), "{what}: {err}");
    }
}

/// Each case breaks one packed-shape rule of one quantized module, and the
/// error must name that module and that rule.
#[test]
fn each_packed_shape_rule_rejects_its_own_violation() {
    const NVFP4: &str = "model.layers.0.mlp.experts.w13";
    const AFFINE: &str = "model.layers.20.mlp.experts.w13";
    fn part(module: &str, suffix: &str, dtype: &str, shape: &[usize]) -> TensorInfo {
        TensorInfo {
            name: format!("{module}.{suffix}"),
            dtype: dtype.to_string(),
            shape: shape.to_vec(),
            file: "model-00001-of-00004.safetensors".to_string(),
        }
    }
    let cases: Vec<(&str, &str, Edit, &str)> = vec![
        (
            "weight not packed",
            AFFINE,
            Box::new(|t| find_mut(t, &format!("{AFFINE}.weight")).dtype = "BF16".into()),
            "expected U32",
        ),
        (
            "scales and weight differ in a leading dim",
            AFFINE,
            Box::new(|t| find_mut(t, &format!("{AFFINE}.scales")).shape[0] += 1),
            "differ in their leading dims",
        ),
        (
            "affine scales not BF16",
            AFFINE,
            Box::new(|t| find_mut(t, &format!("{AFFINE}.scales")).dtype = "F32".into()),
            "affine scales are F32",
        ),
        (
            "affine biases of another shape",
            AFFINE,
            Box::new(|t| find_mut(t, &format!("{AFFINE}.biases")).shape[0] += 1),
            "affine biases must be",
        ),
        (
            "affine module with a global_scale",
            AFFINE,
            Box::new(|t| t.push(part(AFFINE, "global_scale", "F32", &[128]))),
            "affine module has a global_scale",
        ),
        (
            "nvfp4 scales not U8",
            NVFP4,
            Box::new(|t| find_mut(t, &format!("{NVFP4}.scales")).dtype = "BF16".into()),
            "nvfp4 scales are BF16",
        ),
        (
            "nvfp4 module with biases",
            NVFP4,
            Box::new(|t| t.push(part(NVFP4, "biases", "BF16", &[128, 2304, 144]))),
            "nvfp4 module has biases",
        ),
        (
            "nvfp4 global_scale of another shape",
            NVFP4,
            Box::new(|t| find_mut(t, &format!("{NVFP4}.global_scale")).shape = vec![127]),
            "nvfp4 global_scale must be",
        ),
    ];
    let (cfg, tensors) = fixture();
    let table = quant(&cfg);
    for (what, module, edit, expected) in cases {
        let mut broken = tensors.clone();
        edit(&mut broken);
        let err =
            check_packed_shapes(&broken, &table, cfg.text_config.n_routed_experts).expect_err(what);
        let text = err.to_string();
        assert!(
            text.contains(module) && text.contains(expected),
            "{what}: {text}"
        );
    }
}

#[test]
fn an_override_that_names_no_quantized_module_is_rejected() {
    let (mut cfg, tensors) = fixture();
    let module = "model.layers.0.attn.nothing";
    cfg.quantization.insert(
        module.to_string(),
        serde_json::json!({"bits": 8, "group_size": 64, "mode": "affine"}),
    );
    let table = quant(&cfg);
    let err = check_packed_shapes(&tensors, &table, cfg.text_config.n_routed_experts)
        .expect_err("an override of a module that is not there");
    let text = err.to_string();
    assert!(
        text.contains(module) && text.contains("names no quantized module"),
        "{text}"
    );
}

/// A shard file: the u64 length, the JSON header, then `body` zero bytes.
fn shard_bytes(header: &str, body: usize) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.resize(bytes.len() + body, 0);
    bytes
}

#[test]
fn a_truncated_shard_fails_the_header_check() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("model-00009-of-00009.safetensors");
    let header = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    std::fs::write(&path, shard_bytes(header, 16)).expect("write shard");
    let tensors = read_header(&path).expect("the complete shard reads");
    assert_eq!(tensors.len(), 1);
    assert_eq!(tensors[0].file, "model-00009-of-00009.safetensors");

    let complete_len = 8 + header.len() as u64 + 16;
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&path)
        .expect("reopen shard");
    file.set_len(complete_len - 8).expect("truncate shard");
    let err = read_header(&path).expect_err("a truncated shard is rejected");
    assert!(
        err.to_string().contains("truncated: tensor w ends at byte"),
        "{err}"
    );
}

/// The length prefix is checked against the file before the header buffer is
/// allocated, so a corrupt prefix asks for at most the file's own size.
#[test]
fn a_length_prefix_past_the_end_of_the_file_is_rejected_before_reading() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("model-00009-of-00009.safetensors");
    let header = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    let mut bytes = shard_bytes(header, 16);
    let file_len = bytes.len() as u64;
    bytes[..8].copy_from_slice(&file_len.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write shard");
    let err = read_header(&path).expect_err("a prefix past the end is rejected");
    assert!(
        err.to_string()
            .contains(&format!("header declares {file_len} bytes")),
        "{err}"
    );
}

/// A prefix over the 100 MB cap is rejected before the file size is consulted,
/// so even a multi-GB shard cannot be read whole as its header.
#[test]
fn a_length_prefix_over_the_cap_is_rejected() {
    let dir = tempfile::tempdir().expect("temp dir");
    let path = dir.path().join("model-00009-of-00009.safetensors");
    let header = r#"{"w":{"dtype":"F32","shape":[4],"data_offsets":[0,16]}}"#;
    let mut bytes = shard_bytes(header, 16);
    bytes[..8].copy_from_slice(&100_000_001u64.to_le_bytes());
    std::fs::write(&path, &bytes).expect("write shard");
    let err = read_header(&path).expect_err("a prefix over the cap is rejected");
    assert!(
        err.to_string()
            .contains("header declares 100000001 bytes, over the 100000000-byte cap"),
        "{err}"
    );
}
