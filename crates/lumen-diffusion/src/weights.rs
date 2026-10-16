//! The safetensors reader the VAE, the DiT and the text encoder load through.
//!
//! A checkpoint is one `.safetensors` file (the klein VAE and DiT) or a
//! directory of shards keyed by `model.safetensors.index.json` (the dev DiT and
//! the text encoder). Float tensors come back as `f32` whatever they are stored
//! as — BF16, F16 or F32 — and the packed 4-bit weights as `u32`.
//!
//! Every header is checked when its file is opened, so a truncated or malformed
//! shard is an error at load rather than a slice panic halfway through building
//! a model. The parsing is plain Rust and is tested without a Metal device;
//! only the `Array`-returning half needs `mlx-native`.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};

/// One tensor's header entry. `start..end` indexes the whole file.
struct TensorMeta {
    dtype: String,
    shape: Vec<i32>,
    start: usize,
    end: usize,
}

/// One safetensors file: its parsed header and its bytes.
struct Shard {
    metas: HashMap<String, TensorMeta>,
    blob: Vec<u8>,
}

impl Shard {
    fn load(path: &Path) -> Result<Self> {
        let blob = std::fs::read(path).with_context(|| format!("reading safetensors {path:?}"))?;
        Self::parse(blob).with_context(|| format!("loading safetensors {path:?}"))
    }

    /// `u64` LE header length, JSON header, then the tensor bytes.
    fn parse(blob: Vec<u8>) -> Result<Self> {
        let Some(len_bytes) = blob.first_chunk::<8>() else {
            bail!("safetensors file too short");
        };
        let hlen = u64::from_le_bytes(*len_bytes);
        let header_end = usize::try_from(hlen)
            .ok()
            .and_then(|h| h.checked_add(8))
            .filter(|&end| end <= blob.len())
            .ok_or_else(|| anyhow!("safetensors header length {hlen} exceeds file size"))?;
        let header: serde_json::Value = serde_json::from_slice(&blob[8..header_end])
            .context("parsing safetensors JSON header")?;
        let obj = header
            .as_object()
            .ok_or_else(|| anyhow!("safetensors header is not an object"))?;
        let data_len = blob.len() - header_end;

        let mut metas = HashMap::new();
        for (name, v) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = v
                .get("dtype")
                .and_then(|d| d.as_str())
                .ok_or_else(|| anyhow!("tensor {name} missing dtype"))?
                .to_string();
            let shape: Vec<i32> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| anyhow!("tensor {name} missing shape"))?
                .iter()
                .map(|d| d.as_i64().map(|x| x as i32))
                .collect::<Option<_>>()
                .ok_or_else(|| anyhow!("tensor {name} shape not all ints"))?;
            let offs = v
                .get("data_offsets")
                .and_then(|o| o.as_array())
                .ok_or_else(|| anyhow!("tensor {name} missing data_offsets"))?;
            let offset = |i: usize| {
                offs.get(i)
                    .and_then(|o| o.as_u64())
                    .and_then(|o| usize::try_from(o).ok())
            };
            let (Some(start), Some(end)) = (offset(0), offset(1)) else {
                bail!("tensor {name} data_offsets are not two integers");
            };
            if start > end || end > data_len {
                bail!(
                    "tensor {name} data_offsets {start}..{end} fall outside the \
                     {data_len}-byte data section"
                );
            }
            metas.insert(
                name.clone(),
                TensorMeta {
                    dtype,
                    shape,
                    start: header_end + start,
                    end: header_end + end,
                },
            );
        }
        Ok(Self { metas, blob })
    }
}

/// A checkpoint: one safetensors file, or every shard its index names.
pub(crate) struct Weights {
    shards: Vec<Shard>,
    /// Tensor name → index into `shards`; `None` for a single file.
    index: Option<HashMap<String, usize>>,
}

impl Weights {
    /// A single `.safetensors` file, or a directory of shards (which must hold
    /// `model.safetensors.index.json`).
    pub(crate) fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            Self::open_sharded(path)
        } else {
            Self::open_file(path)
        }
    }

    /// A single `.safetensors` file.
    pub(crate) fn open_file(path: impl AsRef<Path>) -> Result<Self> {
        Ok(Self {
            shards: vec![Shard::load(path.as_ref())?],
            index: None,
        })
    }

    /// Every shard named by `<dir>/model.safetensors.index.json`.
    pub(crate) fn open_sharded(dir: impl AsRef<Path>) -> Result<Self> {
        let dir = dir.as_ref();
        let index_path = dir.join("model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(&index_path).with_context(|| format!("reading index {index_path:?}"))?,
        )
        .context("parsing model.safetensors.index.json")?;
        let weight_map = index
            .get("weight_map")
            .and_then(|m| m.as_object())
            .ok_or_else(|| anyhow!("index.json missing weight_map"))?;

        // Distinct shard filenames in deterministic order.
        let mut files: Vec<&str> = weight_map.values().filter_map(|v| v.as_str()).collect();
        files.sort_unstable();
        files.dedup();
        let shards = files
            .iter()
            .map(|f| Shard::load(&dir.join(f)))
            .collect::<Result<Vec<_>>>()?;
        let index = weight_map
            .iter()
            .filter_map(|(k, v)| {
                let si = files.binary_search(&v.as_str()?).ok()?;
                Some((k.clone(), si))
            })
            .collect();
        Ok(Self {
            shards,
            index: Some(index),
        })
    }

    /// True iff the checkpoint has a tensor with this exact name.
    pub(crate) fn has(&self, name: &str) -> bool {
        match &self.index {
            None => self.shards[0].metas.contains_key(name),
            Some(index) => index.contains_key(name),
        }
    }

    fn tensor(&self, name: &str) -> Result<(&TensorMeta, &[u8])> {
        let shard = match &self.index {
            None => &self.shards[0],
            Some(index) => {
                let &si = index
                    .get(name)
                    .ok_or_else(|| anyhow!("tensor not in index: {name}"))?;
                &self.shards[si]
            }
        };
        let m = shard
            .metas
            .get(name)
            .ok_or_else(|| anyhow!("tensor not found: {name}"))?;
        Ok((m, &shard.blob[m.start..m.end]))
    }

    /// A BF16, F16 or F32 tensor's values as `f32`, and its shape.
    pub(crate) fn f32_data(&self, name: &str) -> Result<(Vec<f32>, &[i32])> {
        let (m, bytes) = self.tensor(name)?;
        let data = match m.dtype.as_str() {
            "BF16" => decode(name, "BF16", bytes, |c: [u8; 2]| {
                half::bf16::from_bits(u16::from_le_bytes(c)).to_f32()
            })?,
            "F16" => decode(name, "F16", bytes, |c: [u8; 2]| {
                half::f16::from_bits(u16::from_le_bytes(c)).to_f32()
            })?,
            "F32" => decode(name, "F32", bytes, f32::from_le_bytes)?,
            other => bail!("unsupported dtype {other} for tensor {name}"),
        };
        Ok((data, &m.shape))
    }

    /// A `U32` tensor's values (the packed 4-bit weights), and its shape.
    pub(crate) fn u32_data(&self, name: &str) -> Result<(Vec<u32>, &[i32])> {
        let (m, bytes) = self.tensor(name)?;
        if m.dtype != "U32" {
            bail!("expected U32 for {name}, got {}", m.dtype);
        }
        Ok((decode(name, "U32", bytes, u32::from_le_bytes)?, &m.shape))
    }
}

/// Little-endian `N`-byte elements, refusing a byte length `N` does not divide.
fn decode<const N: usize, T>(
    name: &str,
    dtype: &str,
    bytes: &[u8],
    f: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>> {
    if !bytes.len().is_multiple_of(N) {
        bail!(
            "{dtype} tensor {name} has {} bytes, not a multiple of {N}",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(N)
        .map(|c| f(c.try_into().expect("chunks_exact yields N bytes")))
        .collect())
}

#[cfg(feature = "mlx-native")]
mod mlx {
    use anyhow::{Context, Result};
    use mlx_rs::Array;

    use super::Weights;
    use crate::ops::qlinear::{Linear, QuantLinear};

    impl Weights {
        /// A float tensor as an `f32` [`Array`] in its stored shape, not
        /// evaluated.
        pub(crate) fn f32(&self, name: &str) -> Result<Array> {
            let (data, shape) = self.f32_data(name)?;
            Ok(Array::from_slice(&data, shape))
        }

        /// A `U32` tensor as stored, not evaluated.
        pub(crate) fn u32(&self, name: &str) -> Result<Array> {
            let (data, shape) = self.u32_data(name)?;
            Ok(Array::from_slice(&data, shape))
        }

        /// A float tensor (a bias, a norm weight) as-is, evaluated.
        pub(crate) fn vec(&self, name: &str) -> Result<Array> {
            let v = self.f32(name)?;
            v.eval().with_context(|| format!("eval vec {name}"))?;
            Ok(v)
        }

        /// A linear weight, diffusers `[out,in]` → `[in,out]` for `matmul`,
        /// evaluated.
        pub(crate) fn linear_weight(&self, name: &str) -> Result<Array> {
            let w = self
                .f32(name)?
                .transpose_axes(&[1, 0])
                .with_context(|| format!("linear weight transpose {name}"))?;
            w.eval()
                .with_context(|| format!("eval linear weight {name}"))?;
            Ok(w)
        }

        /// A conv weight, PyTorch `[Cout,Cin,kH,kW]` → mlx `[Cout,kH,kW,Cin]`,
        /// evaluated.
        pub(crate) fn conv_weight(&self, name: &str) -> Result<Array> {
            let w = self
                .f32(name)?
                .transpose_axes(&[0, 2, 3, 1])
                .with_context(|| format!("conv weight transpose {name}"))?;
            w.eval()
                .with_context(|| format!("eval conv weight {name}"))?;
            Ok(w)
        }

        /// A projection: mlx affine 4-bit iff a `<prefix>.scales` sibling
        /// exists, else the dense `<prefix>.weight` through
        /// [`Self::linear_weight`]. The packed `{weight,scales,biases}` triple
        /// comes back as stored, not evaluated.
        pub(crate) fn linear(&self, prefix: &str) -> Result<Linear> {
            if self.has(&format!("{prefix}.scales")) {
                Ok(Linear::Quant(QuantLinear {
                    weight: self.u32(&format!("{prefix}.weight"))?,
                    scales: self.f32(&format!("{prefix}.scales"))?,
                    biases: self.f32(&format!("{prefix}.biases"))?,
                }))
            } else {
                Ok(Linear::Dense(
                    self.linear_weight(&format!("{prefix}.weight"))?,
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lumen_testkit::faults::{Corruption, TensorSpec, build_safetensors, corrupt};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A fresh scratch directory under the system temp dir, unique to this
    /// process and call, removed when dropped.
    struct Scratch(PathBuf);

    impl Scratch {
        fn new(stem: &str) -> Self {
            static NEXT: AtomicU32 = AtomicU32::new(0);
            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            let dir = std::env::temp_dir()
                .join(format!("lumen-diffusion-{stem}-{}-{n}", std::process::id()));
            std::fs::create_dir_all(&dir).expect("create scratch dir");
            Self(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn spec(name: &'static str, dtype: &'static str, shape: &[usize], data: Vec<u8>) -> TensorSpec {
        TensorSpec {
            name,
            dtype,
            shape: shape.to_vec(),
            data,
        }
    }

    fn le<const N: usize>(xs: impl IntoIterator<Item = [u8; N]>) -> Vec<u8> {
        xs.into_iter().flatten().collect()
    }

    /// One tensor per supported dtype, with values exact in every format.
    fn every_dtype() -> Vec<u8> {
        let vals = [1.0f32, -2.5, 0.5, 3.0];
        build_safetensors(&[
            spec("f32", "F32", &[2, 2], le(vals.map(f32::to_le_bytes))),
            spec(
                "bf16",
                "BF16",
                &[4],
                le(vals.map(|v| half::bf16::from_f32(v).to_bits().to_le_bytes())),
            ),
            spec(
                "f16",
                "F16",
                &[1, 4],
                le(vals.map(|v| half::f16::from_f32(v).to_bits().to_le_bytes())),
            ),
            spec(
                "packed",
                "U32",
                &[2],
                le([7u32, u32::MAX].map(u32::to_le_bytes)),
            ),
        ])
    }

    #[test]
    fn a_single_file_decodes_every_dtype() {
        let dir = Scratch::new("single");
        let path = dir.0.join("model.safetensors");
        std::fs::write(&path, every_dtype()).unwrap();
        let w = Weights::open_file(&path).unwrap();

        let want = vec![1.0f32, -2.5, 0.5, 3.0];
        assert_eq!(w.f32_data("f32").unwrap(), (want.clone(), &[2, 2][..]));
        assert_eq!(w.f32_data("bf16").unwrap(), (want.clone(), &[4][..]));
        assert_eq!(w.f32_data("f16").unwrap(), (want, &[1, 4][..]));
        assert_eq!(w.u32_data("packed").unwrap(), (vec![7, u32::MAX], &[2][..]));

        assert!(w.has("bf16") && !w.has("missing"));
        let err = |r: Result<_>| format!("{:#}", r.expect_err("must fail"));
        assert!(err(w.f32_data("packed").map(|_| ())).contains("unsupported dtype U32"));
        assert!(err(w.u32_data("f32").map(|_| ())).contains("expected U32"));
        assert!(err(w.f32_data("missing").map(|_| ())).contains("tensor not found"));
    }

    /// The index decides which shard a tensor is read from, and whether it
    /// exists at all: a tensor present in a shard but absent from the index is
    /// not part of the checkpoint.
    #[test]
    fn a_sharded_dir_reads_each_tensor_from_its_indexed_shard() {
        let dir = Scratch::new("sharded");
        let a = build_safetensors(&[
            spec("x", "F32", &[1], le([1.5f32.to_le_bytes()])),
            spec("unindexed", "F32", &[1], le([9.0f32.to_le_bytes()])),
        ]);
        let b = build_safetensors(&[spec(
            "y",
            "F32",
            &[2],
            le([2.0f32, 4.0].map(f32::to_le_bytes)),
        )]);
        std::fs::write(dir.0.join("a.safetensors"), a).unwrap();
        std::fs::write(dir.0.join("b.safetensors"), b).unwrap();
        std::fs::write(
            dir.0.join("model.safetensors.index.json"),
            r#"{"metadata":{},"weight_map":{"x":"a.safetensors","y":"b.safetensors"}}"#,
        )
        .unwrap();

        let w = Weights::open(&dir.0).unwrap();
        assert_eq!(w.f32_data("x").unwrap(), (vec![1.5], &[1][..]));
        assert_eq!(w.f32_data("y").unwrap(), (vec![2.0, 4.0], &[2][..]));
        assert!(w.has("y") && !w.has("unindexed"));
        let e = w.f32_data("unindexed").expect_err("not in the index");
        assert!(format!("{e:#}").contains("not in index"), "{e:#}");

        let empty = Scratch::new("no-index");
        let e = Weights::open_sharded(&empty.0).err().expect("no index");
        assert!(format!("{e:#}").contains("reading index"), "{e:#}");
    }

    /// Every truncation, a hostile header length and a garbled header are
    /// errors — none of them a panic, which is what the three loaders this
    /// replaced did on some of these inputs.
    #[test]
    fn malformed_files_are_errors_not_panics() {
        let good = every_dtype();
        assert!(Shard::parse(good.clone()).is_ok());
        for n in 0..good.len() {
            assert!(
                Shard::parse(corrupt(&good, Corruption::TruncateAt(n))).is_err(),
                "truncated to {n} of {} bytes parsed",
                good.len()
            );
        }
        for c in [
            Corruption::HeaderLen(u64::MAX),
            Corruption::HeaderLen(good.len() as u64),
            Corruption::GarbageHeader,
        ] {
            assert!(Shard::parse(corrupt(&good, c)).is_err(), "{c:?} parsed");
        }

        // Offsets past the data section, and an element size that does not
        // divide the bytes: both well-formed headers describing bad content.
        let mut lying = build_safetensors(&[spec("t", "F32", &[2], vec![0; 8])]);
        lying.truncate(lying.len() - 4);
        let e = Shard::parse(lying).err().expect("offsets past the end");
        assert!(format!("{e:#}").contains("data_offsets"), "{e:#}");

        let dir = Scratch::new("odd");
        let path = dir.0.join("odd.safetensors");
        std::fs::write(
            &path,
            build_safetensors(&[spec("t", "BF16", &[2], vec![0; 3])]),
        )
        .unwrap();
        let e = Weights::open_file(&path)
            .unwrap()
            .f32_data("t")
            .expect_err("odd BF16");
        assert!(format!("{e:#}").contains("not a multiple of 2"), "{e:#}");
    }
}
