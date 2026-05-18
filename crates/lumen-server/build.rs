//! Locate mlx-sys's built `mlx.metallib` and expose its absolute path via
//! the `LUMEN_MLX_METALLIB_PATH` env var so `main.rs` can `include_bytes!`
//! it into the lumen-server binary. This lets us ship a single self-
//! contained executable instead of forcing operators to copy metallib
//! alongside the binary.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    // Active only under `mlx-native` — Qwen / fallback builds don't need
    // the Metal kernel library.
    let mlx_native = std::env::var_os("CARGO_FEATURE_MLX_NATIVE").is_some();
    if !mlx_native {
        // Provide an empty placeholder so `include_bytes!` doesn't fail —
        // the unpack code path is gated on the same feature in main.rs.
        let placeholder = std::env::var_os("OUT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join("empty.metallib");
        if !placeholder.exists() {
            std::fs::write(&placeholder, b"").ok();
        }
        println!(
            "cargo:rustc-env=LUMEN_MLX_METALLIB_PATH={}",
            placeholder.display()
        );
        return;
    }

    let target_dir = locate_target_dir();
    let candidate = find_metallib(&target_dir).expect(
        "lumen-server build.rs: mlx.metallib not found under target/. \
         Build mlx-sys first (`cargo build -p mlx-sys`) or check that \
         `--features mlx-native` was passed to a prior build.",
    );
    println!("cargo:rerun-if-changed={}", candidate.display());
    println!(
        "cargo:rustc-env=LUMEN_MLX_METALLIB_PATH={}",
        candidate.display()
    );
}

fn locate_target_dir() -> PathBuf {
    // OUT_DIR is `<target>/<profile>/build/<pkg-hash>/out` — climb to <target>.
    let out_dir = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .expect("OUT_DIR not set by Cargo");
    // out_dir/../../../.. = <target>
    let mut p = out_dir.clone();
    for _ in 0..4 {
        if let Some(parent) = p.parent() {
            p = parent.to_path_buf();
        }
    }
    p
}

fn find_metallib(target_dir: &PathBuf) -> Option<PathBuf> {
    // mlx-sys writes to:
    //   <target>/<profile>/build/mlx-sys-<hash>/out/build/lib/mlx.metallib
    // Walk that subtree and pick the newest match.
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    walk(target_dir, &mut best, 8);
    best.map(|(_, p)| p)
}

fn walk(
    dir: &PathBuf,
    best: &mut Option<(std::time::SystemTime, PathBuf)>,
    depth_budget: usize,
) {
    if depth_budget == 0 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk(&path, best, depth_budget - 1);
        } else if path.file_name().and_then(|s| s.to_str()) == Some("mlx.metallib") {
            if let Ok(meta) = entry.metadata() {
                if let Ok(mtime) = meta.modified() {
                    let candidate = (mtime, path.clone());
                    match best {
                        None => *best = Some(candidate),
                        Some((cur_mtime, _)) if *cur_mtime < mtime => {
                            *best = Some(candidate)
                        }
                        _ => {}
                    }
                }
            }
        }
    }
}
