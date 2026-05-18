use anyhow::{Context, Result};
use candle_core::{DType, Device};
use candle_nn::VarBuilder;
use hf_hub::Repo;
use hf_hub::api::sync::{Api, ApiBuilder};

/// Create an HF Hub API client, with token from `HF_TOKEN` env var or token file.
fn api() -> Result<Api> {
    let builder = ApiBuilder::new();
    let builder = if let Ok(token) = std::env::var("HF_TOKEN") {
        eprintln!("  using HF_TOKEN from environment");
        builder.with_token(Some(token))
    } else {
        builder
    };
    builder.build().context("failed to create HF Hub API")
}

/// Load a model config from HuggingFace Hub and deserialize to the given type.
pub fn load_config<T: serde::de::DeserializeOwned>(model_id: &str) -> Result<T> {
    let api = api()?;
    let repo = api.repo(Repo::model(model_id.to_string()));
    let config_path = repo
        .get("config.json")
        .context("failed to download config.json")?;
    let config_str = std::fs::read_to_string(&config_path)?;
    let config: T = serde_json::from_str(&config_str)?;
    Ok(config)
}

/// Load model weights from HuggingFace Hub as a VarBuilder.
///
/// Uses memory-mapped safetensors for efficient loading.
/// Returns (VarBuilder, Device) — device is Metal on Apple Silicon, CPU fallback.
pub fn load_weights(model_id: &str, dtype: DType) -> Result<(VarBuilder<'static>, Device)> {
    let device = get_device()?;

    let api = api()?;
    let repo = api.repo(Repo::model(model_id.to_string()));

    // Discover safetensors files via repo info
    let repo_info = repo.info().context("failed to fetch repo info")?;
    let mut st_filenames: Vec<String> = repo_info
        .siblings
        .iter()
        .map(|s| s.rfilename.clone())
        .filter(|name| name.ends_with(".safetensors"))
        .collect();
    st_filenames.sort();

    if st_filenames.is_empty() {
        anyhow::bail!("no safetensors files found for {model_id}");
    }

    eprintln!("  downloading {} safetensors file(s)", st_filenames.len());
    let mut safetensors_paths = Vec::with_capacity(st_filenames.len());
    for filename in &st_filenames {
        let path = repo
            .get(filename)
            .with_context(|| format!("failed to download {filename}"))?;
        safetensors_paths.push(path);
    }

    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&safetensors_paths, dtype, &device)
            .context("failed to load safetensors")?
    };

    Ok((vb, device))
}

/// Get the best available device.
///
/// Set `DEVICE=cpu` to force CPU. Otherwise tries Metal on Apple Silicon.
pub fn get_device() -> Result<Device> {
    if std::env::var("DEVICE").as_deref() == Ok("cpu") {
        eprintln!("Using CPU device (DEVICE=cpu)");
        return Ok(Device::Cpu);
    }
    #[cfg(feature = "metal")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                eprintln!("Using Metal device");
                return Ok(device);
            }
            Err(e) => {
                eprintln!("Metal unavailable ({e}), falling back to CPU");
            }
        }
    }
    Ok(Device::Cpu)
}

/// Load tokenizer from HuggingFace Hub or, when `model_id` looks like a local
/// path (absolute or starts with `./`), load directly from `<model_id>/tokenizer.json`.
pub fn load_tokenizer(model_id: &str) -> Result<tokenizers::Tokenizer> {
    let looks_local = model_id.starts_with('/') || model_id.starts_with("./");
    if looks_local {
        let local_tok = std::path::Path::new(model_id).join("tokenizer.json");
        if local_tok.exists() {
            return tokenizers::Tokenizer::from_file(&local_tok)
                .map_err(|e| anyhow::anyhow!("local tokenizer {}: {e}", local_tok.display()));
        }
    }
    let api = api()?;
    let repo = api.repo(Repo::model(model_id.to_string()));
    let tokenizer_path = repo
        .get("tokenizer.json")
        .context("failed to download tokenizer.json")?;
    let tokenizer =
        tokenizers::Tokenizer::from_file(&tokenizer_path).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(tokenizer)
}
