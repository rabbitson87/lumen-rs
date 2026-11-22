//! Header reader for one `.safetensors` shard: a little-endian u64 header
//! length, then the JSON table of tensors (the `__metadata__` entry is skipped).
//! It checks the container layout only. Tensor data is never read.
//!
//! Truncation: a shard must be exactly `8 + header_len + max(data_offsets end)`
//! bytes. A tensor whose data runs past the end of the file is reported by name.

use std::fs::File;
use std::io::Read;
use std::path::Path;

use anyhow::{Context, Result, anyhow, ensure};
use serde::Deserialize;
use serde_json::{Map, Value};

/// One tensor as its shard's header declares it. `file` is the shard's file name.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    /// The checkpoint's own dtype string: `U32`, `BF16`, `F32`, `U8`.
    pub dtype: String,
    pub shape: Vec<usize>,
    pub file: String,
}

#[derive(Deserialize)]
struct RawTensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [u64; 2],
}

/// Bytes before the JSON header: its length as a little-endian u64.
const LEN_PREFIX: u64 = 8;
const METADATA_KEY: &str = "__metadata__";

/// Reads the header of the shard at `path`. Tensors come back sorted by name.
pub fn read_header(path: &Path) -> Result<Vec<TensorInfo>> {
    let file_name = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("{}: file name is not UTF-8", path.display()))?;
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let file_len = file
        .metadata()
        .with_context(|| format!("stat {}", path.display()))?
        .len();
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes)
        .with_context(|| format!("{file_name}: truncated: shorter than its length prefix"))?;
    let header_len = u64::from_le_bytes(len_bytes);
    ensure!(
        header_len <= file_len.saturating_sub(LEN_PREFIX),
        "{file_name}: truncated: header declares {header_len} bytes, the file has {file_len}"
    );
    let mut header = vec![0u8; usize::try_from(header_len)?];
    file.read_exact(&mut header)
        .with_context(|| format!("{file_name}: read header"))?;
    parse_header(file_name, file_len, &header)
}

/// Parses the JSON `header` of a shard named `file` whose total length is `file_len`.
fn parse_header(file: &str, file_len: u64, header: &[u8]) -> Result<Vec<TensorInfo>> {
    let table: Map<String, Value> = serde_json::from_slice(header)
        .with_context(|| format!("{file}: header is not a JSON object"))?;
    let data_start = LEN_PREFIX + u64::try_from(header.len())?;
    ensure!(
        data_start <= file_len,
        "{file}: truncated: header ends at byte {data_start}, the file has {file_len}"
    );
    let mut accounted = data_start;
    let mut tensors = Vec::with_capacity(table.len());
    for (name, value) in table {
        if name == METADATA_KEY {
            continue;
        }
        let raw: RawTensor =
            serde_json::from_value(value).with_context(|| format!("{file}: tensor {name}"))?;
        let [begin, end] = raw.data_offsets;
        ensure!(
            begin <= end,
            "{file}: tensor {name}: data_offsets [{begin}, {end}] run backwards"
        );
        let stop = data_start
            .checked_add(end)
            .filter(|&stop| stop <= file_len)
            .ok_or_else(|| {
                anyhow!(
                    "{file}: truncated: tensor {name} ends at byte {}, the file has {file_len}",
                    data_start.saturating_add(end)
                )
            })?;
        accounted = accounted.max(stop);
        tensors.push(TensorInfo {
            name,
            dtype: raw.dtype,
            shape: raw.shape,
            file: file.to_string(),
        });
    }
    ensure!(
        accounted == file_len,
        "{file}: {} trailing bytes after the last tensor, which the header does not declare",
        file_len - accounted
    );
    tensors.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(tensors)
}

#[cfg(test)]
mod tests {
    use super::*;

    const TWO: &str = r#"{"__metadata__":{"format":"pt"},"b.weight":{"dtype":"BF16","shape":[2,2],"data_offsets":[0,8]},"a.scales":{"dtype":"U8","shape":[4],"data_offsets":[8,12]}}"#;

    /// Parses `header` as a shard whose data section is `body` bytes long.
    fn parse(header: &str, body: u64) -> Result<Vec<TensorInfo>> {
        let file_len = LEN_PREFIX + header.len() as u64 + body;
        parse_header("s.safetensors", file_len, header.as_bytes())
    }

    #[test]
    fn tensors_come_back_sorted_and_the_metadata_entry_is_skipped() {
        let tensors = parse(TWO, 12).expect("a complete shard parses");
        let names: Vec<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        assert_eq!(names, ["a.scales", "b.weight"]);
        assert_eq!(tensors[1].dtype, "BF16");
        assert_eq!(tensors[1].shape, [2, 2]);
        assert_eq!(tensors[0].file, "s.safetensors");
    }

    #[test]
    fn one_byte_short_is_reported_as_truncated() {
        let err = parse(TWO, 11).expect_err("a short shard is rejected");
        assert!(err.to_string().contains("truncated"), "{err}");
    }

    #[test]
    fn one_byte_too_many_is_reported_as_trailing() {
        let err = parse(TWO, 13).expect_err("trailing bytes are rejected");
        assert!(err.to_string().contains("trailing"), "{err}");
    }

    #[test]
    fn backwards_offsets_are_rejected() {
        let header = r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[4,0]}}"#;
        let err = parse(header, 4).expect_err("backwards offsets are rejected");
        assert!(err.to_string().contains("run backwards"), "{err}");
    }
}
