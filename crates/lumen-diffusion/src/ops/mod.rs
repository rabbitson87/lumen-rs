//! Net-new MLX ops required by the FLUX.2 diffusion path.
//!
//! These fill the gaps the autoregressive LLM stack never needed: the VAE wants
//! 2D convolution and group normalization, the DiT wants multi-axis (4D) RoPE
//! and adaptive layer-norm modulation. Each op is a thin wrapper over the
//! patched mlx-rs fork plus a self-verifying parity test.

pub mod conv2d;
pub mod groupnorm;
pub mod layernorm;
pub mod linear;
pub mod linear_nobias;
pub mod qlinear;
pub mod rope4d;
pub mod silu;
pub mod timestep_embed;
