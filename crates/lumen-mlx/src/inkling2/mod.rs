//! The `inkling2_loop` checkpoint (`Inkling-2-Mini-oQ4e-mtp`): parsing and the text model.
//!
//! Parsing (ungated, no MLX): reads `config.json` and the header of every `.safetensors`
//! shard, checks the config's internal consistency, resolves the quantization of each module,
//! checks the packed shapes of every quantized module, and maps every tensor name to a [`Role`]
//! with the layer layout checked against the config. [`TextArgs`] reads the text model's
//! hyper-parameters and the schedule of layer visits.
//!
//! The text model (`mlx-native`): [`Inkling2Model`] loads the weights and runs the forward pass,
//! ported from oMLX's Inkling 2 (Apache-2.0) and held to it by the parity tests against a tiny
//! replica built with that reference. Not here yet: chat serving, the vision tower, the draft
//! head.

mod chat;
mod config;
mod manifest;
#[cfg(feature = "mlx-native")]
mod model;
#[cfg(feature = "mlx-native")]
mod ops;
mod quant;
mod safetensors_header;
#[cfg(all(test, feature = "mlx-native"))]
mod tests;
mod text_config;

pub use chat::{DEFAULT_EFFORT, Effort, render_prompt};
pub use config::{Inkling2Config, LayerKind, LoopConfig, MtpConfig, TextConfig, VisionConfig};
pub use manifest::{Entry, Manifest, Part, Placement, Role};
#[cfg(feature = "mlx-native")]
pub use model::{Inkling2Cache, Inkling2Model, LoadOptions};
pub use quant::{PackedSummary, QuantMode, QuantSpec, QuantTable, check_packed_shapes};
pub use safetensors_header::{TensorInfo, read_header};
pub use text_config::{AttnDims, LoopArgs, TextArgs};
