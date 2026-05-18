//! MPSGraph FFI integration (Phase 2 — PoC scaffold).
//!
//! Wraps `objc2-metal-performance-shaders-graph` for compiled subgraph execution.
//! Goal: compress N small dispatches (e.g. ssm_loop, RmsNorm + matmul + narrow)
//! into a single Apple-stitched Metal shader via MPSGraphExecutable.
//!
//! ## Design (Option A — Phase 1.3 decision)
//! - **mxfp4 matmul stays outside MPSGraph** (custom Metal shader nodes are not
//!   supported). The graph contains only RmsNorm / arithmetic / shape / sdpa
//!   operations.
//! - The graph is built once at warmup, compiled to `MPSGraphExecutable`,
//!   then executed on every forward pass with input/output [`MTLBuffer`]s.
//! - Zero-copy: `MPSGraphTensorData` wraps the same `objc2-metal::MTLBuffer`
//!   the rest of the codebase uses.
//!
//! ## Status
//! - Scaffold only — `MpsGraphContext::new` and a tiny smoke test verify the
//!   FFI compiles and links. PoC 1 (RmsNorm + matmul + narrow) lives on a
//!   follow-up commit.

#![cfg(feature = "mpsgraph")]

mod builder;
mod context;
mod executable;
#[cfg(feature = "model-integration")]
mod rms_norm;
mod tensor;

pub use builder::{MatmulGraph2D, RmsNormBf16OutGraph, RmsNormGraph};
pub use context::{MpsGraphContext, MpsGraphError};
pub use executable::{compile, encode_and_commit, run_sync};
#[cfg(feature = "model-integration")]
pub use rms_norm::{MpsRmsNorm, MpsRmsNormBf16Out};
pub use tensor::{shape_from_dims, tensor_data_from_buffer};
