//! In-process embedding model — Qwen3-Embedding-style.
//!
//! Loads a Qwen3 transformer via `candle_transformers::models::qwen3` and
//! exposes `embed(texts) -> Vec<Vec<f32>>` for OpenAI-shaped
//! `/v1/embeddings`. No Python, no subprocess — runs in the same process
//! as the chat backend, sharing the Metal device.
//!
//! Supports **four checkpoint layouts**:
//!   1. Plain bf16/f16 safetensors (`*.weight` only).
//!   2. **MLX AFFINE 8-bit** (`*.weight` uint32-packed, 4 bytes/u32 +
//!      `*.scales` + `*.biases` f16, per-group, group_size=64). Default
//!      path keeps the 8-bit packed weights resident on-GPU and runs a
//!      fused dequant+matmul Metal kernel (`affine8_matmul_bf16`).
//!      Memory ~600 MB (vs ~1.2 GB with eager dequant). Opt-out via
//!      `LUMEN_EMBEDDING_QUANT_KERNEL=0` to fall back to CPU dequant +
//!      standard bf16 inference.
//!   3. **MLX AFFINE 4-bit** (`*.weight` uint32-packed, 8 nibbles/u32
//!      + `*.scales` + `*.biases` f16, per-group, group_size=64). Default
//!      path uses the fused `affine4_qmv_fast_bf16in_bf16out` Metal kernel
//!      (`Affine4Linear`) — packed weights stay GPU-resident, no eager
//!      dequant. Memory footprint ~½ of 8-bit on disk AND in RAM (e.g.
//!      4B-4bit-DWQ: ~2.3 GB resident vs ~4.4 GB for 8-bit). Falls back
//!      to CPU eager dequant when `LUMEN_EMBEDDING_QUANT_KERNEL=0` or
//!      when a projection shape misses qmv_fast's `in % 512 == 0 &&
//!      out % 8 == 0` requirement.
//!   4. **MLX MXFP8 (OCP)** — `mode="mxfp8"`, bits=8, group_size=32.
//!      `*.weight` is uint32-packed E4M3 (4 bytes/u32 again) + `*.scales`
//!      as raw u8 E8M0 bytes, NO `.biases` (E4M3 sign bit carries the
//!      sign). Runs on the fused `mxfp8_qmv_fast_bf16` Metal kernel
//!      (`Mxfp8Linear`) — packed weights stay GPU-resident. GPU↔CPU parity
//!      validated bit-identical at random inputs. Used by e.g.
//!      `mlx-community/Qwen3-Embedding-4B-mxfp8`.
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
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3;

use crate::qwen3_stateless::{QuantProj, StatelessQwen3};
use hf_hub::Repo;
use hf_hub::api::sync::{Api, ApiBuilder};
use lumen_metal::affine4_gpu::{Affine4Context, Affine4Weight};
use lumen_metal::affine4_linear::Affine4Linear;
use lumen_metal::affine8_gpu::{Affine8Context, Affine8Weight};
use lumen_metal::affine8_linear::Affine8Linear;
use lumen_metal::mxfp8_gpu::{Mxfp8Context, Mxfp8Weight, e4m3_to_f32, e8m0_to_f32};
use lumen_metal::mxfp8_linear::Mxfp8Linear;
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
                    "[embedding] using fused {}-bit quant Metal kernel ({} projections)",
                    raw_cfg.quantization.as_ref().map(|q| q.bits).unwrap_or(0),
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
    /// MLX quantization mode. Absent / `"affine"` → AFFINE (uint + zp + scale,
    /// f16 scale/bias, group_size=64 typical). `"mxfp8"` → OCP MXFP8
    /// (E4M3 elements + E8M0 byte scales, group_size=32, no biases).
    /// Any other value is rejected at load time so a future format isn't
    /// silently mis-loaded as AFFINE.
    #[serde(default)]
    mode: Option<String>,
}

impl QuantConfig {
    /// Normalized mode string (`"affine"` is the default when the field is
    /// missing — matches MLX's pre-mxfp4 convention).
    fn mode_str(&self) -> &str {
        self.mode.as_deref().unwrap_or("affine")
    }
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
        projections: HashMap<String, QuantProj>,
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
        // Three fused dequant+matmul kernels live behind `QuantProj`:
        //   - AFFINE 4-bit / 8-bit (Qwen3-Embedding-0.6B-8bit,
        //     Qwen3-Embedding-{4B,8B}-4bit-DWQ): `mode` absent / "affine".
        //   - MXFP8 (OCP, Qwen3-Embedding-4B-mxfp8): `mode == "mxfp8"`,
        //     bits=8, group_size=32, no biases (E4M3 elements + E8M0 scales).
        //
        // Any other (bits, mode) combination drops to the CPU eager-dequant
        // branch below; truly unsupported modes are rejected explicitly in
        // `load_quant_kernel` so a future format isn't silently misloaded.
        let mode = qc.mode_str();
        let kernel_eligible = use_quant_kernel
            && match (qc.bits, mode) {
                (4 | 8, "affine") => true,
                (8, "mxfp8") => true,
                _ => false,
            };
        if kernel_eligible {
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
) -> Result<(HashMap<String, Tensor>, HashMap<String, QuantProj>)> {
    let mode = qc.mode_str();
    let is_mxfp8 = mode == "mxfp8";

    // Lazy per-mode context construction. Each compiles its Metal pipelines
    // once, then is reused across all ~196 projections in the model.
    let ctx8: Option<Arc<Affine8Context>> = if qc.bits == 8 && !is_mxfp8 {
        Some(Arc::new(
            Affine8Context::new().context("Affine8Context::new")?,
        ))
    } else {
        None
    };
    let ctx4: Option<Arc<Affine4Context>> = if qc.bits == 4 && !is_mxfp8 {
        Some(Arc::new(
            Affine4Context::new().context("Affine4Context::new")?,
        ))
    } else {
        None
    };
    let ctx_mxfp8: Option<Arc<Mxfp8Context>> = if is_mxfp8 {
        Some(Arc::new(Mxfp8Context::new().context("Mxfp8Context::new")?))
    } else {
        None
    };

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

    // A `.weight` is part of a quant triple iff it has a `.scales` sibling.
    // (MXFP8 omits `.biases`; AFFINE includes it.)
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
        "[embedding] quant load (bits={}, mode={}): {} layer projections (GPU-resident packed) + {} eager-dequant weights",
        qc.bits,
        mode,
        projection_paths.len(),
        dequant_paths.len()
    );

    // Parallel CPU dequant for the non-projection (eager) weights.
    // embed_tokens is the only such weight on Qwen3-Embedding.
    let dequanted: Vec<(String, Vec<half::bf16>, (usize, usize))> = dequant_paths
        .par_iter()
        .map(|wkey| -> Result<_> {
            let skey = wkey.replace(".weight", ".scales");
            let w_view = open_view(&mmaps, &all[wkey])?;
            let s_view = open_view(&mmaps, &all[&skey])?;
            let (buf, shape) = if is_mxfp8 {
                dequant_mlx_mxfp8_to_buf(w_view.0, &w_view.1, s_view.0, &s_view.1, qc.group_size)
            } else {
                let bkey = wkey.replace(".weight", ".biases");
                let b_view = open_view(&mmaps, &all[&bkey])?;
                if qc.bits == 8 {
                    dequant_mlx_8bit_to_buf(
                        w_view.0,
                        &w_view.1,
                        s_view.0,
                        &s_view.1,
                        s_view.2,
                        b_view.0,
                        b_view.2,
                        qc.group_size,
                    )
                } else {
                    dequant_mlx_4bit_to_buf(
                        w_view.0,
                        &w_view.1,
                        s_view.0,
                        &s_view.1,
                        s_view.2,
                        b_view.0,
                        b_view.2,
                        qc.group_size,
                    )
                }
            }
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

    // Build GPU-resident projection wrappers. The on-disk layout determines
    // the variant: AFFINE has `.weight` + `.scales` + `.biases`; MXFP8 has
    // `.weight` + `.scales` (u8 E8M0) only.
    let mut projections: HashMap<String, QuantProj> = HashMap::new();
    for prefix in projection_paths {
        let wkey = format!("{prefix}.weight");
        let skey = format!("{prefix}.scales");
        let w_view = open_view(&mmaps, &all[&wkey])?;
        let s_view = open_view(&mmaps, &all[&skey])?;

        // MLX packs (32 / bits) logical weights per u32 — 4 E4M3 / uint8
        // bytes for bits=8 (both AFFINE and MXFP8), 8 nibbles for bits=4.
        // weight.shape[1] is the packed last-dim; logical
        // `in_features = packed_last_dim * (32/bits)`.
        let out_features = w_view.1[0];
        let packs_per_row = w_view.1[1];
        let in_features = packs_per_row * (32 / qc.bits);
        let packed_u32 = bytes_to_u32_vec(w_view.0);

        let proj = if is_mxfp8 {
            // MXFP8: scales are raw u8 E8M0 bytes — no f16 decode.
            anyhow::ensure!(
                s_view.2 == safetensors::Dtype::U8,
                "mxfp8 expects U8 scales for {prefix}, got {:?}",
                s_view.2
            );
            let ctx = ctx_mxfp8
                .as_ref()
                .expect("ctx_mxfp8 must exist when mode=mxfp8");
            let weight =
                Mxfp8Weight::from_host(&ctx.ctx, &packed_u32, s_view.0, out_features, in_features)
                    .with_context(|| format!("upload Mxfp8Weight {prefix}"))?;
            QuantProj::M8(Mxfp8Linear::new(weight, ctx.clone()))
        } else {
            // AFFINE: scales/biases are F16 (original builds) OR BF16
            // (e.g. DWQ rebuilds) on disk; kernel expects bf16. Dispatch on
            // the safetensors dtype — see `scales_to_bf16_bits` for the
            // bug that motivated this branch (4B-4bit-DWQ ships BF16 and
            // produced all-NaN embeddings when read as F16).
            let bkey = format!("{prefix}.biases");
            let b_view = open_view(&mmaps, &all[&bkey])?;
            let scales_u16 = scales_to_bf16_bits(s_view.0, s_view.2)
                .with_context(|| format!("scales {prefix}"))?;
            let biases_u16 = scales_to_bf16_bits(b_view.0, b_view.2)
                .with_context(|| format!("biases {prefix}"))?;
            if qc.bits == 8 {
                let ctx = ctx8.as_ref().expect("ctx8 must exist for bits=8");
                let weight = Affine8Weight::from_host(
                    &ctx.ctx,
                    &packed_u32,
                    &scales_u16,
                    &biases_u16,
                    out_features,
                    in_features,
                )
                .with_context(|| format!("upload Affine8Weight {prefix}"))?;
                QuantProj::A8(Affine8Linear::new(weight, ctx.clone()))
            } else {
                let ctx = ctx4.as_ref().expect("ctx4 must exist for bits=4");
                let weight = Affine4Weight::from_host(
                    &ctx.ctx,
                    &packed_u32,
                    &scales_u16,
                    &biases_u16,
                    out_features,
                    in_features,
                )
                .with_context(|| format!("upload Affine4Weight {prefix}"))?;
                QuantProj::A4(Affine4Linear::new(weight, None, ctx.clone()))
            }
        };
        projections.insert(prefix, proj);
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

/// Normalize the quant `scales` / `biases` byte slice to bf16 bits, which
/// is what the Metal kernel + CPU reference both consume.
///
/// MLX checkpoints disagree on the on-disk dtype:
///   - Original AFFINE builds (e.g. `Qwen3-Embedding-0.6B-8bit`) use
///     **F16** (IEEE half, 5-bit exponent).
///   - DWQ builds (e.g. `Qwen3-Embedding-4B-4bit-DWQ`) use **BF16**
///     already.
///
/// Both formats are 16-bit but their exponent widths differ — reading a
/// BF16 byte pair as F16 yields garbage (NaN/Inf on most values), which
/// is the bug that made `Qwen3-Embedding-4B-4bit-DWQ` emit all-NaN
/// embeddings before this branch existed. Dispatch on the safetensors
/// dtype enum so we get correct values for both.
fn scales_to_bf16_bits(bytes: &[u8], dtype: safetensors::Dtype) -> Result<Vec<u16>> {
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    match dtype {
        safetensors::Dtype::F16 => {
            for i in 0..n {
                let f = half::f16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]);
                out.push(half::bf16::from_f32(f.to_f32()).to_bits());
            }
        }
        safetensors::Dtype::BF16 => {
            for i in 0..n {
                out.push(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]));
            }
        }
        other => {
            return Err(anyhow!(
                "scales/biases dtype must be F16 or BF16, got {other:?}"
            ));
        }
    }
    Ok(out)
}

/// Decode one 16-bit scalar at byte offset `byte_off` to f32. Handles both
/// F16 and BF16 on-disk dtypes (mirrors [`scales_to_bf16_bits`]). Used by
/// the host-side eager dequant helpers that walk the scales buffer
/// element by element instead of pre-converting.
fn read_scale_f32(bytes: &[u8], byte_off: usize, dtype: safetensors::Dtype) -> f32 {
    let lo = bytes[byte_off];
    let hi = bytes[byte_off + 1];
    match dtype {
        safetensors::Dtype::F16 => half::f16::from_le_bytes([lo, hi]).to_f32(),
        safetensors::Dtype::BF16 => half::bf16::from_le_bytes([lo, hi]).to_f32(),
        _ => f32::NAN,
    }
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
        if qc.bits != 4 && qc.bits != 8 {
            return Err(anyhow!(
                "only 4-bit and 8-bit MLX dequant are implemented (got bits={})",
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
                let (buf, shape) = if qc.bits == 8 {
                    dequant_mlx_8bit_to_buf(
                        w_view.0,
                        &w_view.1,
                        s_view.0,
                        &s_view.1,
                        s_view.2,
                        b_view.0,
                        b_view.2,
                        qc.group_size,
                    )
                } else {
                    dequant_mlx_4bit_to_buf(
                        w_view.0,
                        &w_view.1,
                        s_view.0,
                        &s_view.1,
                        s_view.2,
                        b_view.0,
                        b_view.2,
                        qc.group_size,
                    )
                }
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

/// MLX 4-bit dequant (CPU-side, pure compute):
///   nibble_value(p, k) = (weight_u32[o, p] >> (4*k)) & 0xF      // k in 0..8
///   w[o, p*8 + k]      = nibble_value(p, k) * scales[o, i/g] + biases[o, i/g]
///
/// Packing: 8 nibbles per u32, little-endian (`k=0` is the low nibble).
/// `weight` shape: `(out_features, in_features / 8)` u32; logical
/// `in_features = packed_last_dim * 8`. `scales`/`biases` shape:
/// `(out_features, in_features / group_size)` f16. Group size is independent
/// of pack size — `group_size=64` with `bits=4` means 2 groups per u32
/// chunk of 32 logical weights.
///
/// Output is bf16 so it can flow into the standard candle matmul path
/// (we don't have a fused 4-bit Metal kernel — only 8-bit). Memory cost
/// is the same as bf16 inference; only the on-disk + initial-load
/// footprint is smaller.
fn dequant_mlx_4bit_to_buf(
    w_raw: &[u8],
    w_shape: &[usize],
    s_raw: &[u8],
    s_shape: &[usize],
    s_dtype: safetensors::Dtype,
    b_raw: &[u8],
    b_dtype: safetensors::Dtype,
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
    let in_features = packs_per_row * 8;
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

    // Parallel over output rows — same access pattern as the 8-bit path.
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
                let base_i = p * 8;
                for nib_idx in 0..8 {
                    let i = base_i + nib_idx;
                    let q = ((pack >> (4 * nib_idx)) & 0xF) as f32;
                    let g = i / group_size;
                    let sbi = (s_off + g) * 2;
                    let scale = read_scale_f32(s_raw, sbi, s_dtype);
                    let bias = read_scale_f32(b_raw, sbi, b_dtype);
                    row_out[i] = half::bf16::from_f32(q * scale + bias);
                }
            }
        });

    Ok((out_bf16, (out_features, in_features)))
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
    s_dtype: safetensors::Dtype,
    b_raw: &[u8],
    b_dtype: safetensors::Dtype,
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
                    let scale = read_scale_f32(s_raw, sbi, s_dtype);
                    let bias = read_scale_f32(b_raw, sbi, b_dtype);
                    row_out[i] = half::bf16::from_f32(q * scale + bias);
                }
            }
        });

    Ok((out_bf16, (out_features, in_features)))
}

/// CPU dequant for MLX MXFP8 (OCP). Mirrors [`dequant_mlx_8bit_to_buf`] but:
///   - Each packed byte is OCP E4M3 instead of AFFINE uint8 + zp + scale.
///   - Scales are raw u8 E8M0 bytes (NOT f16), one per group of 32.
///   - No biases (MXFP8 is symmetric — E4M3 sign bit carries the sign).
///
/// Used for the `embed_tokens` weight (the only non-projection quantized
/// tensor on Qwen3-Embedding) so candle's `embedding` row-lookup helper
/// sees a plain bf16 matrix. All projection weights stay GPU-resident in
/// the MXFP8 packed form via [`Mxfp8Linear`].
fn dequant_mlx_mxfp8_to_buf(
    w_raw: &[u8],
    w_shape: &[usize],
    s_raw: &[u8],
    s_shape: &[usize],
    group_size: usize,
) -> Result<(Vec<half::bf16>, (usize, usize))> {
    if w_shape.len() != 2 || s_shape.len() != 2 {
        return Err(anyhow!(
            "unexpected MLX mxfp8 shape: w={:?} s={:?}",
            w_shape,
            s_shape
        ));
    }
    if group_size != 32 {
        return Err(anyhow!("mxfp8 requires group_size=32 (got {group_size})"));
    }
    let out_features = w_shape[0];
    let packs_per_row = w_shape[1];
    let in_features = packs_per_row * 4;
    if !in_features.is_multiple_of(group_size) {
        return Err(anyhow!(
            "in_features ({in_features}) not divisible by group_size ({group_size})"
        ));
    }
    let groups_per_row = in_features / group_size;
    if s_shape[0] != out_features || s_shape[1] != groups_per_row {
        return Err(anyhow!(
            "mxfp8 scales shape mismatch: got {:?}, expected ({}, {})",
            s_shape,
            out_features,
            groups_per_row
        ));
    }

    let mut out_bf16: Vec<half::bf16> = vec![half::bf16::ZERO; out_features * in_features];

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
                    let raw_byte = ((pack >> (8 * byte_idx)) & 0xFF) as u8;
                    let g = i / group_size;
                    let scale = e8m0_to_f32(s_raw[s_off + g]);
                    let val = e4m3_to_f32(raw_byte) * scale;
                    row_out[i] = half::bf16::from_f32(val);
                }
            }
        });

    Ok((out_bf16, (out_features, in_features)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quantize a known weight matrix to MLX 4-bit packing and verify the
    /// dequant function recovers it (within bf16 rounding tolerance).
    #[test]
    fn dequant_mlx_4bit_roundtrip() {
        // 1 row × 16 logical weights = 2 packed u32. group_size=8 ⇒ 2 groups.
        let out_features = 1usize;
        let in_features = 16usize;
        let group_size = 8usize;
        let groups_per_row = in_features / group_size;
        let packs_per_row = in_features / 8;

        let s_vals = [1.0f32, 2.0f32];
        let b_vals = [0.0f32, -3.0f32];

        let mut w_raw = vec![0u8; packs_per_row * 4];
        let mut s_raw = vec![0u8; out_features * groups_per_row * 2];
        let mut b_raw = vec![0u8; out_features * groups_per_row * 2];
        for p in 0..packs_per_row {
            let mut pack: u32 = 0;
            for k in 0..8 {
                let nib = ((p * 8 + k) & 0xF) as u32;
                pack |= nib << (4 * k);
            }
            w_raw[p * 4..p * 4 + 4].copy_from_slice(&pack.to_le_bytes());
        }
        for g in 0..groups_per_row {
            let sb = half::f16::from_f32(s_vals[g]).to_le_bytes();
            let bb = half::f16::from_f32(b_vals[g]).to_le_bytes();
            s_raw[g * 2..g * 2 + 2].copy_from_slice(&sb);
            b_raw[g * 2..g * 2 + 2].copy_from_slice(&bb);
        }

        let (buf, shape) = dequant_mlx_4bit_to_buf(
            &w_raw,
            &[out_features, packs_per_row],
            &s_raw,
            &[out_features, groups_per_row],
            safetensors::Dtype::F16,
            &b_raw,
            safetensors::Dtype::F16,
            group_size,
        )
        .expect("dequant ok");
        assert_eq!(shape, (out_features, in_features));

        let got: Vec<f32> = buf.iter().map(|x| x.to_f32()).collect();
        let expected: Vec<f32> = (0..in_features)
            .map(|i| {
                let nib = (i & 0xF) as f32;
                let g = i / group_size;
                nib * s_vals[g] + b_vals[g]
            })
            .collect();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 0.05,
                "idx {i}: got {g} expected {e} (bf16 rounding tol)"
            );
        }
    }

    #[test]
    fn dequant_mlx_4bit_rejects_misaligned_group() {
        // in_features = packs_per_row * 8 = 8, group_size = 10 → not divisible.
        let w_raw = [0u8; 4];
        let s_raw = [0u8; 2];
        let b_raw = [0u8; 2];
        let err = dequant_mlx_4bit_to_buf(
            &w_raw,
            &[1, 1],
            &s_raw,
            &[1, 1],
            safetensors::Dtype::F16,
            &b_raw,
            safetensors::Dtype::F16,
            10,
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("not divisible by group_size"),
            "expected divisibility error, got: {err}"
        );
    }
}
