//! The `inkling2_loop` checkpoint (`Inkling-2-Mini-oQ4e-mtp`): parsing and the text model.
//!
//! Parsing (ungated, no MLX): reads `config.json` and the header of every `.safetensors`
//! shard, checks the config's internal consistency, resolves the quantization of each module,
//! checks the packed shapes of every quantized module, and maps every tensor name to a [`Role`]
//! with the layer layout checked against the config. [`TextArgs`] reads the text model's
//! hyper-parameters and the schedule of layer visits.
//!
//! Chat prompts and replies (ungated): [`render_prompt`] renders the checkpoint's chat template,
//! [`ResponseParser`] reads what the model generates into reasoning, answer and tool calls.
//!
//! The text model (`mlx-native`): [`Inkling2Model`] loads the weights and runs the forward pass,
//! ported from oMLX's Inkling 2 (Apache-2.0) and held to it by the parity tests against a tiny
//! replica built with that reference. [`Inkling2Backend`] serves it: a prompt from chat turns, a
//! generation loop and the caches of the last conversation kept for the next request. Not here
//! yet: the vision tower and the draft head.

#[cfg(feature = "mlx-native")]
mod backend;
mod chat;
mod config;
mod manifest;
#[cfg(feature = "mlx-native")]
mod model;
#[cfg(feature = "mlx-native")]
mod ops;
mod pixels;
mod quant;
mod response;
mod safetensors_header;
#[cfg(all(test, feature = "mlx-native"))]
mod tests;
mod text_config;
#[cfg(feature = "mlx-native")]
mod vision;
#[cfg(all(test, feature = "mlx-native"))]
mod vision_tests;

#[cfg(feature = "mlx-native")]
pub use backend::{Inkling2Backend, RunStats, named_effort, turns_from_pairs};
pub use chat::{
    DEFAULT_EFFORT, Effort, IMAGE_MESSAGE_FRAME_TOKENS, image_message_tokens, render_prompt,
    render_prompt_with_images,
};
pub use config::{Inkling2Config, LayerKind, LoopConfig, MtpConfig, TextConfig, VisionConfig};
pub use manifest::{Entry, Manifest, Part, Placement, Role};
#[cfg(feature = "mlx-native")]
pub use model::{ConvSnapshot, ImageRows, Inkling2Cache, Inkling2Model, LoadOptions};
pub use pixels::{
    BLOCK, MAX_IMAGE_TOKENS, PreparedImage, prepare as prepare_image,
    token_count as image_token_count,
};
pub use quant::{PackedSummary, QuantMode, QuantSpec, QuantTable, check_packed_shapes};
pub use response::{Markers, ResponseParser};
pub use safetensors_header::{TensorInfo, read_header};
pub use text_config::{AttnDims, LoopArgs, TextArgs};
#[cfg(feature = "mlx-native")]
pub use vision::{VisionArgs, VisionTower};
