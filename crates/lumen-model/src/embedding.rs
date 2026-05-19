//! In-process embedding model — Qwen3-Embedding-style.
//!
//! Loads a Qwen3 transformer via `candle_transformers::models::qwen3` and
//! exposes `embed(texts) -> Vec<Vec<f32>>` for OpenAI-shaped
//! `/v1/embeddings`. No Python, no subprocess — runs in the same process
//! as the chat backend, sharing the Metal device.
//!
//! Supports **two checkpoint layouts**:
//!   1. Plain bf16/f16 safetensors (`*.weight` only).
//!   2. **MLX 8-bit quantized** (`*.weight` uint32-packed + `*.scales`
//!      + `*.biases` f16, per-group). Default path keeps the 8-bit
//!      packed weights resident on-GPU and runs a fused dequant+matmul
//!      Metal kernel (`affine8_matmul_bf16`). Memory ~600 MB (vs ~1.2 GB
//!      with eager dequant). Opt-out via `LUMEN_EMBEDDING_QUANT_KERNEL=0`
//!      to fall back to CPU dequant + standard bf16 inference.
//!
//! Pooling: **last-token** (Qwen3-Embedding standard). The hidden state
//! at the final input position is taken as the sentence embedding.
//!
//! Output: **L2-normalized** float32 vectors — cosine == dot product.
//!
//! Batching: inputs are right-padded into a single causal forward;
//! pooling reads each row at `original_seq_len - 1`. Padding sits past
//! every real token so causal masking shields the pooled position
//! from it.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;

use crate::qwen3_stateless::StatelessQwen3;
use hf_hub::Repo;
use hf_hub::api::sync::{Api, ApiBuilder};
use lumen_metal::affine8_gpu::{Affine8Context, Affine8Weight};
use lumen_metal::affine8_linear::Affine8Linear;
use memmap2::Mmap;
use rayon::prelude::*;
use safetensors::SafeTensors;
use serde::Deserialize;

use crate::loader::get_device;

fn quant_kernel_enabled() -> bool {
    std::env::var("LUMEN_EMBEDDING_QUANT_KERNEL")
        .map(|v| v != "0")
        .unwrap_or(true)
}

/// Is this safetensors key a per-layer attention/MLP projection that
/// `StatelessQwen3::new_quantized` expects from the projections map?
fn is_layer_projection(key: &str) -> bool {
    if !key.starts_with("model.layers.") {
        return false;
    }
    matches!(
        key.rsplit('.').next(),
        Some("q_proj")
            | Some("k_proj")
            | Some("v_proj")
            | Some("o_proj")
            | Some("gate_proj")
            | Some("up_proj")
            | Some("down_proj")
    )
}

#[derive(Debug)]
pub struct EmbeddingBatch {
    pub embeddings: Vec<Vec<f32>>,
    pub prompt_tokens: u32,
}

pub struct EmbeddingModel {
    model: StatelessQwen3,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
    dim: usize,
    max_seq_len: usize,
    model_id: String,
}

impl EmbeddingModel {
    pub fn load(model_id_or_path: &str) -> Result<Self> {
        eprintln!("[embedding] loading {model_id_or_path}");
        let t0 = std::time::Instant::now();

        let files = resolve_model_files(model_id_or_path)?;

        let config_str = std::fs::read_to_string(&files.config_path)
            .with_context(|| format!("read config: {}", files.config_path.display()))?;
        let raw_cfg: RawConfig = serde_json::from_str(&config_str).context("parse config.json")?;
        let config = raw_cfg.to_qwen3_config();
        let dim = config.hidden_size;
        let max_seq_len = config.max_position_embeddings;

        let device = get_device()?;
        // Always materialize weights as bf16 — Qwen3-Embedding-0.6B trained
        // in bf16, candle handles bf16 efficiently on Metal.
        let dtype = DType::BF16;

        let use_quant_kernel = raw_cfg.quantization.is_some() && quant_kernel_enabled();
        let loaded = load_weights_into(
            &files.safetensors_paths,
            dtype,
            &device,
            raw_cfg.quantization.as_ref(),
            use_quant_kernel,
        )
        .context("load weights")?;

        let model = match loaded {
            LoadedWeights::Plain(tensors) => {
                let vb = VarBuilder::from_tensors(tensors, dtype, &device);
                StatelessQwen3::new(&config, vb).context("instantiate stateless qwen3 (plain)")?
            }
            LoadedWeights::Quantized {
                tensors,
                projections,
            } => {
                eprintln!(
                    "[embedding] using fused 8-bit quant Metal kernel ({} projections)",
                    projections.len()
                );
                let vb = VarBuilder::from_tensors(tensors, dtype, &device);
                StatelessQwen3::new_quantized(&config, vb, projections)
                    .context("instantiate stateless qwen3 (quant)")?
            }
        };

        let tokenizer = tokenizers::Tokenizer::from_file(&files.tokenizer_path).map_err(|e| {
            anyhow!(
                "tokenizer from_file({}): {e}",
                files.tokenizer_path.display()
            )
        })?;

        eprintln!(
            "[embedding] loaded model_id={model_id_or_path} dim={dim} dtype={dtype:?} device={device:?} in {:.1}s",
            t0.elapsed().as_secs_f32()
        );

        Ok(Self {
            model,
            tokenizer,
            device,
            dim,
            max_seq_len,
            model_id: model_id_or_path.to_string(),
        })
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Encode `texts` and return `(L2-normalized embeddings, total prompt tokens)`.
    ///
    /// Hot path is one batched forward + a fully GPU-resident
    /// gather/normalize/copy-out so the only host sync per request is the
    /// single final `to_vec2`. Set `LUMEN_EMBEDDING_TIMING=1` for stage
    /// timing on stderr.
    pub fn embed(&mut self, texts: &[String]) -> Result<EmbeddingBatch> {
        if texts.is_empty() {
            return Ok(EmbeddingBatch {
                embeddings: vec![],
                prompt_tokens: 0,
            });
        }

        let timing = std::env::var("LUMEN_EMBEDDING_TIMING").is_ok();
        let t_tok = std::time::Instant::now();

        // Tokenize via batch API — `tokenizers` parallelizes internally.
        // `add_special_tokens=true` so the BPE appends `<|endoftext|>`
        // (id 151643) at every sequence's tail; that token's hidden state
        // is what Qwen3-Embedding pools.
        let inputs: Vec<&str> = texts.iter().map(String::as_str).collect();
        let encs = self
            .tokenizer
            .encode_batch(inputs, true)
            .map_err(|e| anyhow!("tokenizer encode_batch: {e}"))?;

        let b = texts.len();
        let mut lens: Vec<usize> = Vec::with_capacity(b);
        let mut prompt_tokens: u32 = 0;
        let mut max_len = 1usize;
        let mut all_ids: Vec<Vec<u32>> = Vec::with_capacity(b);
        for enc in &encs {
            let mut ids: Vec<u32> = enc.get_ids().to_vec();
            if ids.is_empty() {
                ids.push(0);
            }
            if ids.len() > self.max_seq_len {
                ids.truncate(self.max_seq_len);
            }
            prompt_tokens = prompt_tokens.saturating_add(ids.len() as u32);
            max_len = max_len.max(ids.len());
            lens.push(ids.len());
            all_ids.push(ids);
        }

        // Right-pad with token id 0. Causal masking shields the last-real-
        // token pooling site from the padding tail.
        let mut padded: Vec<u32> = Vec::with_capacity(b * max_len);
        for ids in &all_ids {
            padded.extend_from_slice(ids);
            padded.resize(padded.len() + (max_len - ids.len()), 0);
        }
        let dt_tok = t_tok.elapsed();

        // Stateless model — no kv cache to clear.
        let t_fwd = std::time::Instant::now();
        let input = Tensor::from_vec(padded, (b, max_len), &self.device)?;
        let hidden = self.model.forward(&input)?; // (b, max_len, hidden) bf16
        let dt_fwd = t_fwd.elapsed();

        // Gather last-real-position rows on device via one `gather` call.
        // candle gather wants the index tensor to match the input rank with
        // a value per output element along the gather dim. For
        // hidden: (b, L, H) and target (b, 1, H), index shape is (b, 1, H)
        // and each row's value = lens[i] - 1, broadcast across H.
        let t_pool = std::time::Instant::now();
        let h_dim = self.dim;
        let mut idx_vec: Vec<u32> = Vec::with_capacity(b * h_dim);
        for &len in &lens {
            let pos = (len.saturating_sub(1)) as u32;
            idx_vec.extend(std::iter::repeat(pos).take(h_dim));
        }
        let idx = Tensor::from_vec(idx_vec, (b, 1, h_dim), &self.device)?;
        let last = hidden.gather(&idx, 1)?.squeeze(1)?; // (b, H) bf16

        // L2 normalize on device: x / max(||x||_2, eps), keepdim broadcast.
        let last_f32 = last.to_dtype(DType::F32)?;
        let norm = last_f32.sqr()?.sum_keepdim(1)?.sqrt()?; // (b, 1)
        let eps = Tensor::new(&[[1e-12_f32]], &self.device)?; // (1, 1)
        let safe = norm.broadcast_maximum(&eps)?;
        let normalized = last_f32.broadcast_div(&safe)?; // (b, H)

        // Single host hop for the whole batch.
        let embeddings: Vec<Vec<f32>> = normalized.to_vec2()?;
        let dt_pool = t_pool.elapsed();

        if timing {
            eprintln!(
                "[embed-timing] b={b} max_len={max_len} tok={:?} fwd={:?} pool={:?}",
                dt_tok, dt_fwd, dt_pool
            );
        }

        Ok(EmbeddingBatch {
            embeddings,
            prompt_tokens,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Config
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct QuantConfig {
    group_size: usize,
    bits: usize,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    head_dim: usize,
    #[serde(default)]
    attention_bias: bool,
    num_key_value_heads: usize,
    max_position_embeddings: usize,
    #[serde(default)]
    sliding_window: Option<usize>,
    #[serde(default)]
    max_window_layers: Option<usize>,
    #[serde(default)]
    tie_word_embeddings: bool,
    rope_theta: f64,
    rms_norm_eps: f64,
    #[serde(default)]
    use_sliding_window: bool,
    #[serde(default = "default_hidden_act")]
    hidden_act: String,
    #[serde(default)]
    quantization: Option<QuantConfig>,
}

fn default_hidden_act() -> String {
    "silu".into()
}

impl RawConfig {
    fn to_qwen3_config(&self) -> qwen3::Config {
        let act = match self.hidden_act.as_str() {
            "silu" => candle_nn::Activation::Silu,
            "gelu" => candle_nn::Activation::Gelu,
            other => panic!("unsupported hidden_act for Qwen3-Embedding: {other}"),
        };
        qwen3::Config {
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            attention_bias: self.attention_bias,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            sliding_window: self.sliding_window,
            max_window_layers: self.max_window_layers.unwrap_or(self.num_hidden_layers),
            tie_word_embeddings: self.tie_word_embeddings,
            rope_theta: self.rope_theta,
            rms_norm_eps: self.rms_norm_eps,
            use_sliding_window: self.use_sliding_window,
            hidden_act: act,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Files: HF-Hub repo id OR local path
// ─────────────────────────────────────────────────────────────────────────────

struct ResolvedFiles {
    config_path: PathBuf,
    tokenizer_path: PathBuf,
    safetensors_paths: Vec<PathBuf>,
}

fn resolve_model_files(model_id_or_path: &str) -> Result<ResolvedFiles> {
    let looks_local = model_id_or_path.starts_with('/') || model_id_or_path.starts_with("./");
    if looks_local {
        let root = PathBuf::from(model_id_or_path);
        let config_path = root.join("config.json");
        let tokenizer_path = root.join("tokenizer.json");
        if !config_path.exists() {
            return Err(anyhow!("no config.json at {}", config_path.display()));
        }
        if !tokenizer_path.exists() {
            return Err(anyhow!("no tokenizer.json at {}", tokenizer_path.display()));
        }
        // Prefer the sharded index when present; otherwise scan for *.safetensors.
        let mut safetensors_paths: Vec<PathBuf> = Vec::new();
        let index_path = root.join("model.safetensors.index.json");
        if index_path.exists() {
            #[derive(Deserialize)]
            struct Index {
                weight_map: HashMap<String, String>,
            }
            let idx: Index = serde_json::from_str(&std::fs::read_to_string(&index_path)?)?;
            let mut unique: std::collections::BTreeSet<String> =
                idx.weight_map.values().cloned().collect();
            for name in unique.iter() {
                safetensors_paths.push(root.join(name));
            }
            unique.clear();
        } else {
            for entry in std::fs::read_dir(&root)? {
                let p = entry?.path();
                if p.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                    safetensors_paths.push(p);
                }
            }
        }
        safetensors_paths.sort();
        if safetensors_paths.is_empty() {
            return Err(anyhow!("no safetensors at {}", root.display()));
        }
        Ok(ResolvedFiles {
            config_path,
            tokenizer_path,
            safetensors_paths,
        })
    } else {
        let api = hf_api()?;
        let repo = api.repo(Repo::model(model_id_or_path.to_string()));
        let config_path = repo.get("config.json").context("download config.json")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("download tokenizer.json")?;
        let info = repo.info().context("repo info")?;
        let mut safetensors_names: Vec<String> = info
            .siblings
            .iter()
            .map(|s| s.rfilename.clone())
            .filter(|n| n.ends_with(".safetensors"))
            .collect();
        safetensors_names.sort();
        let mut safetensors_paths = Vec::new();
        for name in &safetensors_names {
            safetensors_paths.push(repo.get(name).with_context(|| format!("download {name}"))?);
        }
        Ok(ResolvedFiles {
            config_path,
            tokenizer_path,
            safetensors_paths,
        })
    }
}

fn hf_api() -> Result<Api> {
    let mut b = ApiBuilder::new();
    if let Ok(tok) = std::env::var("HF_TOKEN") {
        b = b.with_token(Some(tok));
    }
    b.build().context("hf_hub api init")
}

// ─────────────────────────────────────────────────────────────────────────────
// Weight loading: plain bf16/f16 OR MLX 8-bit dequant
// ─────────────────────────────────────────────────────────────────────────────

enum LoadedWeights {
    Plain(HashMap<String, Tensor>),
    Quantized {
        tensors: HashMap<String, Tensor>,
        projections: HashMap<String, Affine8Linear>,
    },
}

fn load_weights_into(
    paths: &[PathBuf],
    target_dtype: DType,
    device: &Device,
    quant: Option<&QuantConfig>,
    use_quant_kernel: bool,
) -> Result<LoadedWeights> {
    if let Some(qc) = quant {
        if use_quant_kernel {
            return load_quant_kernel(paths, target_dtype, device, qc).map(
                |(tensors, projections)| LoadedWeights::Quantized {
                    tensors,
                    projections,
                },
            );
        }
    }
    Ok(LoadedWeights::Plain(load_weight_map(
        paths,
        target_dtype,
        device,
        quant,
    )?))
}

fn load_quant_kernel(
    paths: &[PathBuf],
    target_dtype: DType,
    device: &Device,
    qc: &QuantConfig,
) -> Result<(HashMap<String, Tensor>, HashMap<String, Affine8Linear>)> {
    if qc.bits != 8 {
        return Err(anyhow!(
            "fused quant kernel only supports 8-bit (got bits={}); \
             set LUMEN_EMBEDDING_QUANT_KERNEL=0 to fall back to eager dequant",
            qc.bits
        ));
    }
    // Affine8Weight buffers must live on a `lumen_metal::MetalContext`,
    // which compiles its own pipeline and owns its device queue. We use a
    // single shared `Affine8Context` for all 196 projections so the
    // shader compiles once.
    let ctx = Arc::new(Affine8Context::new().context("Affine8Context::new")?);

    let mut mmaps: Vec<Arc<Mmap>> = Vec::with_capacity(paths.len());
    for p in paths {
        let file = File::open(p).with_context(|| format!("open {}", p.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        mmaps.push(Arc::new(mmap));
    }
    let mut all: HashMap<String, ViewMeta> = HashMap::new();
    for (i, mm) in mmaps.iter().enumerate() {
        let st = SafeTensors::deserialize(mm).context("parse safetensors header")?;
        for k in st.names() {
            all.insert(
                k.to_string(),
                ViewMeta {
                    file_idx: i,
                    key: k.to_string(),
                },
            );
        }
    }

    // Partition (weight, scales, biases) triples into:
    //   - layer projections → Affine8Linear (kept on-GPU, fused kernel)
    //   - everything else (embed_tokens) → CPU dequant → bf16 Tensor
    let weight_keys: Vec<String> = all
        .keys()
        .filter(|k| k.ends_with(".weight") && all.contains_key(&k.replace(".weight", ".scales")))
        .cloned()
        .collect();

    let mut projection_paths: Vec<String> = Vec::new();
    let mut dequant_paths: Vec<String> = Vec::new();
    for wkey in &weight_keys {
        let prefix = wkey.trim_end_matches(".weight").to_string();
        if is_layer_projection(&prefix) {
            projection_paths.push(prefix);
        } else {
            // embed_tokens etc. — dequant to bf16 so candle's `embedding`
            // helper can lookup rows.
            dequant_paths.push(wkey.clone());
        }
    }
    eprintln!(
        "[embedding] quant load: {} layer projections (GPU-resident packed) + {} eager-dequant weights",
        projection_paths.len(),
        dequant_paths.len()
    );

    // Parallel CPU dequant for the non-projection (eager) weights.
    let dequanted: Vec<(String, Vec<half::bf16>, (usize, usize))> = dequant_paths
        .par_iter()
        .map(|wkey| -> Result<_> {
            let skey = wkey.replace(".weight", ".scales");
            let bkey = wkey.replace(".weight", ".biases");
            let w_view = open_view(&mmaps, &all[wkey])?;
            let s_view = open_view(&mmaps, &all[&skey])?;
            let b_view = open_view(&mmaps, &all[&bkey])?;
            let (buf, shape) = dequant_mlx_8bit_to_buf(
                w_view.0,
                &w_view.1,
                s_view.0,
                &s_view.1,
                b_view.0,
                qc.group_size,
            )
            .with_context(|| format!("dequant {wkey}"))?;
            Ok((wkey.clone(), buf, shape))
        })
        .collect::<Result<Vec<_>>>()?;

    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (key, buf, shape) in dequanted {
        let t = Tensor::from_vec(buf, shape, &Device::Cpu)?.to_device(device)?;
        tensors.insert(key, t);
    }

    // Plain weights (norms — just `.weight` with no scales/biases sibling).
    for (k, meta) in &all {
        if k.ends_with(".scales") || k.ends_with(".biases") {
            continue;
        }
        if tensors.contains_key(k) {
            continue;
        }
        if projection_paths.iter().any(|p| k == &format!("{p}.weight")) {
            continue;
        }
        let (raw, shape, sdtype) = open_view(&mmaps, meta)?;
        let tensor = tensor_from_view(raw, &shape, sdtype, target_dtype, device)
            .with_context(|| format!("decode {k}"))?;
        tensors.insert(k.clone(), tensor);
    }

    // Build GPU-resident Affine8Linear for each layer projection.
    let mut projections: HashMap<String, Affine8Linear> = HashMap::new();
    for prefix in projection_paths {
        let wkey = format!("{prefix}.weight");
        let skey = format!("{prefix}.scales");
        let bkey = format!("{prefix}.biases");
        let w_view = open_view(&mmaps, &all[&wkey])?;
        let s_view = open_view(&mmaps, &all[&skey])?;
        let b_view = open_view(&mmaps, &all[&bkey])?;

        // weight shape: (out, in/4) u32 ; in = packs_per_row * 4
        let out_features = w_view.1[0];
        let packs_per_row = w_view.1[1];
        let in_features = packs_per_row * 4;

        let packed_u32 = bytes_to_u32_vec(w_view.0);
        // safetensors stores scales/biases as f16; kernel expects bf16.
        let scales_u16 = f16_bytes_to_bf16_bits(s_view.0);
        let biases_u16 = f16_bytes_to_bf16_bits(b_view.0);

        let weight = Affine8Weight::from_host(
            &ctx.ctx,
            &packed_u32,
            &scales_u16,
            &biases_u16,
            out_features,
            in_features,
        )
        .with_context(|| format!("upload Affine8Weight {prefix}"))?;
        projections.insert(prefix, Affine8Linear::new(weight, ctx.clone()));
    }

    Ok((tensors, projections))
}

fn bytes_to_u32_vec(bytes: &[u8]) -> Vec<u32> {
    let n = bytes.len() / 4;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(u32::from_le_bytes([
            bytes[4 * i],
            bytes[4 * i + 1],
            bytes[4 * i + 2],
            bytes[4 * i + 3],
        ]));
    }
    v
}

fn bytes_to_u16_vec(bytes: &[u8]) -> Vec<u16> {
    let n = bytes.len() / 2;
    let mut v = Vec::with_capacity(n);
    for i in 0..n {
        v.push(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
    }
    v
}

/// MLX checkpoints store quant `scales` / `biases` as **f16** (IEEE
/// half-precision, 5-bit exponent). Our Metal kernel reads those buffers
/// as **bf16** (8-bit exponent) — different bit layouts. Convert at load
/// time so the kernel sees the right values without an extra in-kernel
/// branch.
fn f16_bytes_to_bf16_bits(bytes: &[u8]) -> Vec<u16> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let f16 = half::f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
        let bf16 = half::bf16::from_f32(f16.to_f32());
        out.push(bf16.to_bits());
    }
    out
}

fn load_weight_map(
    paths: &[PathBuf],
    target_dtype: DType,
    device: &Device,
    quant: Option<&QuantConfig>,
) -> Result<HashMap<String, Tensor>> {
    // Open all safetensors files via mmap and accumulate (name, view) pairs.
    let mut mmaps: Vec<Arc<Mmap>> = Vec::with_capacity(paths.len());
    for p in paths {
        let file = File::open(p).with_context(|| format!("open {}", p.display()))?;
        let mmap = unsafe { Mmap::map(&file)? };
        mmaps.push(Arc::new(mmap));
    }

    // Build a global name → (file_idx, key) map.
    // We can't keep TensorView refs across function boundaries because of
    // borrow lifetimes, so we instead snapshot the metadata up front.
    let mut all: HashMap<String, ViewMeta> = HashMap::new();
    for (i, mm) in mmaps.iter().enumerate() {
        let st = SafeTensors::deserialize(mm).context("parse safetensors header")?;
        for k in st.names() {
            all.insert(
                k.to_string(),
                ViewMeta {
                    file_idx: i,
                    key: k.to_string(),
                },
            );
        }
    }

    let mut out: HashMap<String, Tensor> = HashMap::new();

    if let Some(qc) = quant {
        if qc.bits != 8 {
            return Err(anyhow!(
                "only 8-bit MLX dequant is implemented (got bits={})",
                qc.bits
            ));
        }
        // Pass 1: dequantize every (weight, scales, biases) triple.
        //
        // CPU dequant runs in parallel via rayon (each output row is an
        // independent slice). Device upload happens sequentially after —
        // Metal's command queue is single-threaded anyway, and serializing
        // here keeps thread-safety reasoning trivial.
        let weight_keys: Vec<String> = all
            .keys()
            .filter(|k| {
                k.ends_with(".weight") && all.contains_key(&k.replace(".weight", ".scales"))
            })
            .cloned()
            .collect();

        let dequanted: Vec<(String, Vec<half::bf16>, (usize, usize))> = weight_keys
            .par_iter()
            .map(|wkey| -> Result<_> {
                let skey = wkey.replace(".weight", ".scales");
                let bkey = wkey.replace(".weight", ".biases");
                let w_view = open_view(&mmaps, &all[wkey])?;
                let s_view = open_view(&mmaps, &all[&skey])?;
                let b_view = open_view(&mmaps, &all[&bkey])?;
                let (buf, shape) = dequant_mlx_8bit_to_buf(
                    w_view.0,
                    &w_view.1,
                    s_view.0,
                    &s_view.1,
                    b_view.0,
                    qc.group_size,
                )
                .with_context(|| format!("dequant {wkey}"))?;
                Ok((wkey.clone(), buf, shape))
            })
            .collect::<Result<Vec<_>>>()?;

        for (key, buf, shape) in dequanted {
            let t = Tensor::from_vec(buf, shape, &Device::Cpu)?.to_device(device)?;
            out.insert(key, t);
        }
        // Pass 2: copy everything that wasn't part of a quant triple (norms,
        // any plain weights) cast to target dtype.
        for (k, meta) in &all {
            if k.ends_with(".scales") || k.ends_with(".biases") {
                continue;
            }
            if out.contains_key(k) {
                continue;
            }
            let (raw, shape, sdtype) = open_view(&mmaps, meta)?;
            let tensor = tensor_from_view(raw, &shape, sdtype, target_dtype, device)
                .with_context(|| format!("decode {k}"))?;
            out.insert(k.clone(), tensor);
        }
    } else {
        // Plain (non-MLX-quantized) checkpoint.
        for (k, meta) in &all {
            let (raw, shape, sdtype) = open_view(&mmaps, meta)?;
            let tensor = tensor_from_view(raw, &shape, sdtype, target_dtype, device)
                .with_context(|| format!("decode {k}"))?;
            out.insert(k.clone(), tensor);
        }
    }

    Ok(out)
}

fn open_view<'a>(
    mmaps: &'a [Arc<Mmap>],
    meta: &ViewMeta,
) -> Result<(&'a [u8], Vec<usize>, safetensors::Dtype)> {
    let st = SafeTensors::deserialize(&mmaps[meta.file_idx])?;
    let v = st
        .tensor(&meta.key)
        .map_err(|e| anyhow!("tensor view {}: {e}", meta.key))?;
    let shape: Vec<usize> = v.shape().to_vec();
    let dtype = v.dtype();
    // SAFETY: the data pointer comes from an mmap held alive by `mmaps` for
    // the duration of the caller's use. SafeTensors::tensor returns a view
    // that borrows from the deserialized object, but the underlying bytes
    // live in the mmap. We re-borrow them with the mmap's lifetime.
    let bytes_ptr = v.data().as_ptr();
    let bytes_len = v.data().len();
    let bytes: &'a [u8] = unsafe { std::slice::from_raw_parts(bytes_ptr, bytes_len) };
    Ok((bytes, shape, dtype))
}

// Workaround: ViewMeta defined inside load_weight_map can't escape; mirror
// it here for open_view's signature.
struct ViewMeta {
    file_idx: usize,
    key: String,
}

fn tensor_from_view(
    raw: &[u8],
    shape: &[usize],
    sdtype: safetensors::Dtype,
    target: DType,
    device: &Device,
) -> Result<Tensor> {
    use safetensors::Dtype as SDt;
    let cpu = Device::Cpu;
    let t = match sdtype {
        SDt::BF16 => {
            let n = raw.len() / 2;
            let mut buf = Vec::with_capacity(n);
            for i in 0..n {
                buf.push(half::bf16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]));
            }
            Tensor::from_vec(buf, shape, &cpu)?
        }
        SDt::F16 => {
            let n = raw.len() / 2;
            let mut buf = Vec::with_capacity(n);
            for i in 0..n {
                buf.push(half::f16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]));
            }
            Tensor::from_vec(buf, shape, &cpu)?
        }
        SDt::F32 => {
            let n = raw.len() / 4;
            let mut buf = Vec::with_capacity(n);
            for i in 0..n {
                buf.push(f32::from_le_bytes([
                    raw[4 * i],
                    raw[4 * i + 1],
                    raw[4 * i + 2],
                    raw[4 * i + 3],
                ]));
            }
            Tensor::from_vec(buf, shape, &cpu)?
        }
        other => return Err(anyhow!("unsupported safetensors dtype: {:?}", other)),
    };
    Ok(t.to_dtype(target)?.to_device(device)?)
}

/// MLX 8-bit dequant (CPU-side, pure compute):
///   w[o,i] = byte(weight_u32[o,i/4], i%4) * scales[o,i/g] + biases[o,i/g]
///
/// Inner row loop is parallelized via rayon: each output row is independent,
/// so `out_bf16.par_chunks_mut(in)` distributes rows across cores. On a
/// 12-P-core M3 Max this approaches the bytes/sec memory-read ceiling
/// (mmap-resident weight + scale/bias inputs).
///
/// Returns the raw bf16 buffer + shape. Device upload happens at the
/// call-site so all Metal interaction stays on one thread.
fn dequant_mlx_8bit_to_buf(
    w_raw: &[u8],
    w_shape: &[usize],
    s_raw: &[u8],
    s_shape: &[usize],
    b_raw: &[u8],
    group_size: usize,
) -> Result<(Vec<half::bf16>, (usize, usize))> {
    if w_shape.len() != 2 || s_shape.len() != 2 {
        return Err(anyhow!(
            "unexpected MLX quant shape: w={:?} s={:?}",
            w_shape,
            s_shape
        ));
    }
    let out_features = w_shape[0];
    let packs_per_row = w_shape[1];
    let in_features = packs_per_row * 4;
    if in_features % group_size != 0 {
        return Err(anyhow!(
            "in_features ({in_features}) not divisible by group_size ({group_size})"
        ));
    }
    let groups_per_row = in_features / group_size;
    if s_shape[0] != out_features || s_shape[1] != groups_per_row {
        return Err(anyhow!(
            "scales shape mismatch: got {:?}, expected ({}, {})",
            s_shape,
            out_features,
            groups_per_row
        ));
    }

    let mut out_bf16: Vec<half::bf16> = vec![half::bf16::ZERO; out_features * in_features];

    // Parallel over output rows. Each row's work touches disjoint output
    // bytes + disjoint slices of w_raw / s_raw / b_raw, so no contention.
    out_bf16
        .par_chunks_mut(in_features)
        .enumerate()
        .for_each(|(o, row_out)| {
            let s_off = o * groups_per_row;
            let w_off = o * packs_per_row;
            for p in 0..packs_per_row {
                let bi = (w_off + p) * 4;
                let pack =
                    u32::from_le_bytes([w_raw[bi], w_raw[bi + 1], w_raw[bi + 2], w_raw[bi + 3]]);
                let base_i = p * 4;
                for byte_idx in 0..4 {
                    let i = base_i + byte_idx;
                    let q = ((pack >> (8 * byte_idx)) & 0xFF) as f32;
                    let g = i / group_size;
                    let sbi = (s_off + g) * 2;
                    let scale = half::f16::from_le_bytes([s_raw[sbi], s_raw[sbi + 1]]).to_f32();
                    let bias = half::f16::from_le_bytes([b_raw[sbi], b_raw[sbi + 1]]).to_f32();
                    row_out[i] = half::bf16::from_f32(q * scale + bias);
                }
            }
        });

    Ok((out_bf16, (out_features, in_features)))
}
