//! Where MLX finds its Metal kernel library (`mlx.metallib`).
//!
//! The binary carries the library inside itself, and MLX loads it from disk on
//! first use — first from beside the binary, then (the lumen-rs MLX fork) from
//! an app bundle's `Contents/Resources`, and last from a path fixed when MLX was
//! compiled, which only exists on the machine that built it.
//!
//! Unpacking beside the binary is right for a bare `lumen-server`, and wrong
//! inside `Lumen.app`: there the binary sits in the signed, sealed
//! `Contents/MacOS`, and the write either breaks the seal or — when the app runs
//! from its read-only disk image, or Gatekeeper has translocated it — fails, at
//! which point MLX aborts with "Failed to load the default metallib". The
//! release bundle therefore ships the library in `Contents/Resources`, and
//! [`plan`] leaves the bundle alone when that copy is the one this binary
//! carries.

use std::path::{Path, PathBuf};

/// What to do about the kernel library before the first MLX call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Plan {
    /// The app bundle ships this binary's library; MLX loads it from there.
    UseBundled,
    /// This binary's library already sits beside it.
    AlreadyColocated,
    /// Unpack the embedded library beside the binary.
    Unpack,
}

/// Decide from the sizes on disk (`None` = absent) and the embedded size.
///
/// A size match stands in for identity, as it always has here: the library is
/// ~130 MB and its size moves with every kernel change. The bundled copy wins
/// only when nothing contradicts it beside the binary — MLX searches beside the
/// binary first, so a stale copy there would be loaded instead.
pub fn plan(colocated: Option<u64>, bundled: Option<u64>, embedded: u64) -> Plan {
    match (colocated, bundled) {
        (Some(c), _) if c == embedded => Plan::AlreadyColocated,
        (None, Some(b)) if b == embedded => Plan::UseBundled,
        _ => Plan::Unpack,
    }
}

/// The library beside the binary in `exe_dir`.
pub fn colocated_path(exe_dir: &Path) -> PathBuf {
    exe_dir.join("mlx.metallib")
}

/// The library an app bundle ships, for a binary in its `Contents/MacOS`.
pub fn bundled_path(exe_dir: &Path) -> PathBuf {
    exe_dir.join("../Resources/mlx.metallib")
}
