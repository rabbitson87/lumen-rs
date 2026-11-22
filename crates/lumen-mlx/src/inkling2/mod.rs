//! Parsing layer for the `inkling2_loop` checkpoint (`Inkling-2-Mini-oQ4e-mtp`).
//!
//! What it does: reads `config.json` and the header of every `.safetensors`
//! shard, checks the config's internal consistency, resolves the quantization
//! of each module, checks the packed shapes of every quantized module, and maps
//! every tensor name to a [`Role`] with the layer layout checked against the config.
//!
//! What it does not do: no weights are loaded, there is no forward pass, and
//! MLX is not called. What a tensor means (what a loop, a "query" or a static
//! key computes) is unverified: needs the reference. Names, shapes, dtypes and
//! config fields are the only evidence used.

mod config;
mod manifest;
mod quant;
mod safetensors_header;

pub use config::{Inkling2Config, LayerKind, LoopConfig, MtpConfig, TextConfig, VisionConfig};
pub use manifest::{Entry, Manifest, Part, Placement, Role};
pub use quant::{PackedSummary, QuantMode, QuantSpec, QuantTable, check_packed_shapes};
pub use safetensors_header::{TensorInfo, read_header};
