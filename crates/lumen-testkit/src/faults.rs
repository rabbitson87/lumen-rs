//! Fault-injection fixtures (005 Phase 3).
//!
//! SQLite's discipline here is the loop, not the fixture: corrupt the input at
//! *every* offset — truncate after byte 1, then 2, then 3 — and require the
//! same typed failure every time, because the bug is never at the offset you
//! thought to test by hand. These helpers build small valid artifacts in
//! memory and enumerate systematic corruptions of them, so a sweep needs no
//! downloaded fixture and no I/O beyond a tempdir.
//!
//! The invariant every consumer asserts is the same three-part one:
//! a typed `Err` (never a panic, unwrap, abort or hang), an error message that
//! names what was wrong, and **the process still works afterwards** — the
//! sweep re-runs a known-good input after every corrupt one, which is the
//! moral equivalent of SQLite's post-failure `integrity_check`.
//!
//! Nothing here depends on the crates under test (`lumen-testkit` is their
//! dev-dependency); everything is bytes and `serde_json::Value`.

use serde_json::Value;

// ───────────────────────── safetensors builder ─────────────────────────

/// One tensor entry for [`build_safetensors`].
pub struct TensorSpec {
    pub name: &'static str,
    /// safetensors dtype string (`"F32"`, `"BF16"`, …). Not validated here —
    /// an *invalid* dtype is a corruption a caller may want to build.
    pub dtype: &'static str,
    pub shape: Vec<usize>,
    /// Raw little-endian payload. Its length is trusted, not derived, so a
    /// caller can deliberately build a length/shape mismatch.
    pub data: Vec<u8>,
}

/// Serialize a valid `.safetensors` byte image: `u64` LE header length, JSON
/// header mapping names to `{dtype, shape, data_offsets}`, then the packed
/// data section. Small by construction (a few tensors of a few KB) so a
/// truncation sweep over every stride stays fast.
pub fn build_safetensors(tensors: &[TensorSpec]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for t in tensors {
        let end = offset + t.data.len();
        header.insert(
            t.name.to_string(),
            serde_json::json!({
                "dtype": t.dtype,
                "shape": t.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let header_bytes = serde_json::to_vec(&Value::Object(header)).expect("header serializes");
    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset);
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for t in tensors {
        out.extend_from_slice(&t.data);
    }
    out
}

/// A ready-made two-tensor file: `"weight"` (f32 4×4) and `"bias"` (f32 4).
/// Deterministic contents so a test failure names reproducible bytes.
pub fn minimal_safetensors() -> Vec<u8> {
    let w: Vec<u8> = (0..16u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let b: Vec<u8> = (0..4u32)
        .flat_map(|i| (i as f32 / 2.0).to_le_bytes())
        .collect();
    build_safetensors(&[
        TensorSpec {
            name: "weight",
            dtype: "F32",
            shape: vec![4, 4],
            data: w,
        },
        TensorSpec {
            name: "bias",
            dtype: "F32",
            shape: vec![4],
            data: b,
        },
    ])
}

// ───────────────────────── byte-level corruption ─────────────────────────

/// Byte-level corruptions applicable to any length-prefixed binary format
/// (safetensors, LKV1). Semantic corruptions (wrong dtype, shape/len
/// mismatch) are built directly via [`build_safetensors`] instead — they are
/// valid *bytes* describing invalid *content*.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Corruption {
    /// Keep only the first `n` bytes. The sweep case: applied at every stride.
    TruncateAt(usize),
    /// Overwrite the leading `u64` length prefix. `u64::MAX` here is the
    /// allocation-bomb probe — a reader that trusts it will try to allocate
    /// 16 EiB and die, which is a defect, not an environment problem.
    HeaderLen(u64),
    /// Flip one byte at `at` (XOR 0xFF) — the single-bit-rot case.
    FlipByte(usize),
    /// Replace the JSON header region with the same number of `b'#'` bytes:
    /// length prefix stays consistent, content is garbage.
    GarbageHeader,
}

/// Apply `c` to a copy of `bytes`. Panics only on out-of-range offsets, which
/// is a bug in the *test*, not the code under test.
pub fn corrupt(bytes: &[u8], c: Corruption) -> Vec<u8> {
    let mut out = bytes.to_vec();
    match c {
        Corruption::TruncateAt(n) => {
            assert!(n <= out.len(), "TruncateAt({n}) beyond len {}", out.len());
            out.truncate(n);
        }
        Corruption::HeaderLen(v) => {
            assert!(out.len() >= 8, "no header to overwrite");
            out[..8].copy_from_slice(&v.to_le_bytes());
        }
        Corruption::FlipByte(at) => {
            assert!(at < out.len(), "FlipByte({at}) beyond len {}", out.len());
            out[at] ^= 0xFF;
        }
        Corruption::GarbageHeader => {
            assert!(out.len() >= 8, "no header to garble");
            let hlen = u64::from_le_bytes(out[..8].try_into().unwrap()) as usize;
            let end = (8 + hlen).min(out.len());
            out[8..end].fill(b'#');
        }
    }
    out
}

/// Truncation offsets worth sweeping for a file of `len` bytes: every
/// `stride`, plus a ±1 neighborhood around each `boundary` (header edges,
/// section starts) — the off-by-one positions where length checks actually
/// break. Sorted, deduped, all `< len` (truncating at `len` is the intact
/// file).
pub fn truncation_offsets(len: usize, stride: usize, boundaries: &[usize]) -> Vec<usize> {
    let mut offs: Vec<usize> = (0..len).step_by(stride.max(1)).collect();
    for &b in boundaries {
        for d in [b.wrapping_sub(1), b, b + 1] {
            if d < len {
                offs.push(d);
            }
        }
    }
    offs.sort_unstable();
    offs.dedup();
    offs
}

// ───────────────────────── config mutation ─────────────────────────

/// Structured mutations of a `config.json`-shaped value. The alphabet comes
/// from failures that shipped, not from imagination: `NullValue` is the
/// JGOS-31B incident (`num_experts: null` hard-failed a `usize` field before
/// any migration could run), and `UnknownModelType` is what every
/// not-yet-supported checkpoint looks like on day one.
#[derive(Debug, Clone)]
pub enum ConfigMutation {
    /// Delete the key at `path` (dot-separated).
    RemoveKey(&'static str),
    /// Set the key at `path` to JSON `null` — the JGOS-31B failure shape.
    NullValue(&'static str),
    /// Replace the value at `path` with a string where a number is expected.
    StringWhereNumber(&'static str),
    /// Negate the number at `path` (dims, counts — anything `usize`-ish).
    NegativeNumber(&'static str),
    /// Set `model_type` to a value no loader recognizes.
    UnknownModelType,
    /// Chop the serialized JSON at byte `n` — a partial download.
    TruncateJson(usize),
}

/// Apply `m` to a copy of `config`, returning the mutated JSON **text** (the
/// form loaders actually read). `TruncateJson` necessarily returns invalid
/// JSON; everything else returns valid JSON describing an invalid config —
/// the more dangerous half, because it gets past the syntax check.
pub fn mutate_config(config: &Value, m: &ConfigMutation) -> String {
    fn at<'v>(
        root: &'v mut Value,
        path: &str,
    ) -> Option<(&'v mut serde_json::Map<String, Value>, String)> {
        let mut parts: Vec<&str> = path.split('.').collect();
        let leaf = parts.pop()?.to_string();
        let mut cur = root;
        for p in parts {
            cur = cur.get_mut(p)?;
        }
        cur.as_object_mut().map(|m| (m, leaf))
    }

    let mut v = config.clone();
    match m {
        ConfigMutation::RemoveKey(path) => {
            let (map, leaf) = at(&mut v, path).unwrap_or_else(|| panic!("no path {path}"));
            assert!(map.remove(&leaf).is_some(), "RemoveKey: {path} was absent");
        }
        ConfigMutation::NullValue(path) => {
            let (map, leaf) = at(&mut v, path).unwrap_or_else(|| panic!("no path {path}"));
            map.insert(leaf, Value::Null);
        }
        ConfigMutation::StringWhereNumber(path) => {
            let (map, leaf) = at(&mut v, path).unwrap_or_else(|| panic!("no path {path}"));
            map.insert(leaf, Value::String("not-a-number".into()));
        }
        ConfigMutation::NegativeNumber(path) => {
            let (map, leaf) = at(&mut v, path).unwrap_or_else(|| panic!("no path {path}"));
            let n = map
                .get(&leaf)
                .and_then(Value::as_i64)
                .unwrap_or_else(|| panic!("NegativeNumber: {path} is not an integer"));
            map.insert(leaf, Value::from(-n.abs().max(1)));
        }
        ConfigMutation::UnknownModelType => {
            v.as_object_mut().expect("config is an object").insert(
                "model_type".into(),
                Value::String("model-from-the-future".into()),
            );
        }
        ConfigMutation::TruncateJson(n) => {
            let s = serde_json::to_string_pretty(&v).expect("serialize");
            return s[..(*n).min(s.len())].to_string();
        }
    }
    serde_json::to_string_pretty(&v).expect("serialize")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn built_safetensors_is_internally_consistent() {
        let bytes = minimal_safetensors();
        let hlen = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        let header: Value = serde_json::from_slice(&bytes[8..8 + hlen]).expect("header is JSON");
        let w = &header["weight"];
        assert_eq!(w["dtype"], "F32");
        assert_eq!(w["shape"], serde_json::json!([4, 4]));
        // data_offsets must cover exactly the data section.
        let total: usize = 8 + hlen + 16 * 4 + 4 * 4;
        assert_eq!(bytes.len(), total);
    }

    #[test]
    fn corruption_ops_do_what_they_say() {
        let base = minimal_safetensors();
        assert_eq!(corrupt(&base, Corruption::TruncateAt(10)).len(), 10);
        let bad = corrupt(&base, Corruption::HeaderLen(u64::MAX));
        assert_eq!(u64::from_le_bytes(bad[..8].try_into().unwrap()), u64::MAX);
        let flipped = corrupt(&base, Corruption::FlipByte(9));
        assert_ne!(flipped[9], base[9]);
        assert_eq!(flipped.len(), base.len());
        let garbled = corrupt(&base, Corruption::GarbageHeader);
        assert!(serde_json::from_slice::<Value>(&garbled[8..garbled.len() - 80]).is_err());
    }

    #[test]
    fn truncation_offsets_cover_boundaries_with_neighbors() {
        let offs = truncation_offsets(100, 32, &[8, 50]);
        for want in [0, 7, 8, 9, 32, 49, 50, 51, 64, 96] {
            assert!(offs.contains(&want), "missing offset {want}: {offs:?}");
        }
        assert!(offs.iter().all(|&o| o < 100));
    }

    #[test]
    fn config_mutations_apply() {
        let cfg = serde_json::json!({
            "model_type": "qwen3_5",
            "text_config": {"num_hidden_layers": 32, "hidden_size": 4096},
        });
        let out = mutate_config(
            &cfg,
            &ConfigMutation::NullValue("text_config.num_hidden_layers"),
        );
        assert!(out.contains("\"num_hidden_layers\": null"));
        let out = mutate_config(&cfg, &ConfigMutation::RemoveKey("text_config.hidden_size"));
        assert!(!out.contains("hidden_size"));
        let out = mutate_config(
            &cfg,
            &ConfigMutation::NegativeNumber("text_config.hidden_size"),
        );
        assert!(out.contains("-4096"));
        let out = mutate_config(&cfg, &ConfigMutation::UnknownModelType);
        assert!(out.contains("model-from-the-future"));
        let out = mutate_config(&cfg, &ConfigMutation::TruncateJson(30));
        assert!(
            serde_json::from_str::<Value>(&out).is_err(),
            "truncated JSON must not parse"
        );
    }
}
