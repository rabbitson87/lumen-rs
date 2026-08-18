//! Locate mlx-sys's built `mlx.metallib` and expose its absolute path via
//! the `LUMEN_MLX_METALLIB_PATH` env var so `main.rs` can `include_bytes!`
//! it into the lumen-server binary. This lets us ship a single self-
//! contained executable instead of forcing operators to copy metallib
//! alongside the binary.

use std::path::{Path, PathBuf};

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

    // Search both `<workspace>/target/` (root) AND `<workspace>/target/<triple>/`
    // (cross-compile dir) because mlx-sys's CMake can land the artifact under
    // either depending on host vs target match and the CMake version.
    let (target_triple_dir, target_root_dir) = locate_target_dirs();
    let candidate = find_metallib(&target_triple_dir).or_else(|| find_metallib(&target_root_dir));

    let candidate = candidate.unwrap_or_else(|| {
        // Print diagnostics so the next CI failure (if any) shows what mlx-sys
        // actually produced — instead of just "not found" with no clue where.
        eprintln!("--- lumen-server build.rs diagnostics ---");
        eprintln!("search root 1 (triple): {}", target_triple_dir.display());
        eprintln!("search root 2 (target): {}", target_root_dir.display());
        list_mlx_sys_dirs(&target_triple_dir);
        list_mlx_sys_dirs(&target_root_dir);
        panic!(
            "lumen-server build.rs: mlx.metallib not found under {} or {}. \
             Build mlx-sys first (`cargo build -p mlx-sys --features mlx-native`) \
             or check that mlx's MLX_BUILD_METAL=ON wasn't suppressed.",
            target_triple_dir.display(),
            target_root_dir.display()
        )
    });
    println!("cargo:rerun-if-changed={}", candidate.display());
    println!(
        "cargo:rustc-env=LUMEN_MLX_METALLIB_PATH={}",
        candidate.display()
    );
}

/// Returns `(target/<triple>/, target/)`. When the build is invoked without
/// `--target`, both end up being the same `target/` root — the dedup happens
/// naturally because we walk the second only if the first misses.
fn locate_target_dirs() -> (PathBuf, PathBuf) {
    let out_dir = std::env::var_os("OUT_DIR")
        .map(PathBuf::from)
        .expect("OUT_DIR not set by Cargo");
    // OUT_DIR layouts:
    //   non-target: target/<profile>/build/<pkg>/out          (4 climbs -> target/)
    //   with triple: target/<triple>/<profile>/build/<pkg>/out (4 climbs -> target/<triple>)
    let triple_dir = climb(&out_dir, 4);
    // One more climb gets us to target/ even in the triple case. If we were
    // already at target/ root, this would go to the workspace root — handled
    // by the search itself (mlx-sys dirs only exist under target/).
    let root_dir = climb(&triple_dir, 1);
    (triple_dir, root_dir)
}

fn climb(start: &Path, levels: usize) -> PathBuf {
    let mut p = start.to_path_buf();
    for _ in 0..levels {
        if let Some(parent) = p.parent() {
            p = parent.to_path_buf();
        }
    }
    p
}

fn find_metallib(target_dir: &PathBuf) -> Option<PathBuf> {
    // mlx-sys writes the metallib somewhere under
    //   <target>/<profile>/build/mlx-sys-<hash>/out/build/...
    // The exact depth depends on the mlx CMake layout (sometimes
    // `out/build/lib/mlx.metallib`, sometimes
    // `out/build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib`).
    // Walk a generous depth so cross-compiled CI builds and any future
    // CMake layout change still find it.
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    walk(target_dir, &mut best, 16);
    best.map(|(_, p)| p)
}

fn walk(dir: &PathBuf, best: &mut Option<(std::time::SystemTime, PathBuf)>, depth_budget: usize) {
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
            // GUARD: only ever embed mlx-sys's *own* build output. Never a
            // colocated copy such as `target/<profile>/mlx.metallib`, which
            // `ensure_metallib_colocated` writes next to the binary at runtime.
            // That copy is always the newest file after any run, so without
            // this filter `find_metallib` (newest-mtime wins) would re-embed
            // it on the next build — a feedback loop that perpetuates any
            // stale or incomplete metallib (e.g. one assembled during nvfp4
            // GPU-kernel dev that dropped mlx's full steel_attention set,
            // breaking Gemma 4 prefill with "function … not found in library").
            // mlx-sys's real artifacts always live under `…/mlx-sys-<hash>/out/…`.
            if !path.to_string_lossy().contains("mlx-sys-") {
                continue;
            }
            if let Ok(meta) = entry.metadata() {
                if let Ok(mtime) = meta.modified() {
                    let candidate = (mtime, path.clone());
                    match best {
                        None => *best = Some(candidate),
                        Some((cur_mtime, _)) if *cur_mtime < mtime => *best = Some(candidate),
                        _ => {}
                    }
                }
            }
        }
    }
}

/// Diagnostic — locate any mlx-sys-* build directories under `root` and dump
/// their out/build/ layout so a failed CI run shows what mlx-sys actually
/// produced (vs guessing). Bounded to 3 levels of recursion under `root`.
fn list_mlx_sys_dirs(root: &Path) {
    let Ok(rd) = std::fs::read_dir(root.join("release/build")) else {
        // Try other common profile dirs if `release` isn't there.
        for prof in ["debug", "release"] {
            let p = root.join(prof).join("build");
            if p.is_dir() {
                list_mlx_sys_dirs_inner(&p);
            }
        }
        return;
    };
    list_mlx_sys_dirs_inner(&root.join("release/build"));
    drop(rd);
}

fn list_mlx_sys_dirs_inner(build_dir: &PathBuf) {
    let Ok(rd) = std::fs::read_dir(build_dir) else {
        return;
    };
    for entry in rd.flatten() {
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("mlx-sys-") {
            eprintln!("found mlx-sys dir: {}", entry.path().display());
            let out_dir = entry.path().join("out");
            if !out_dir.is_dir() {
                eprintln!("  (no out/ subdir — build script likely didn't run)");
                continue;
            }
            // Dump top of the out/ tree
            dump_tree(&out_dir, 0, 4);
        }
    }
}

fn dump_tree(dir: &PathBuf, depth: usize, budget: usize) {
    if budget == 0 {
        return;
    }
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    let indent = "  ".repeat(depth + 1);
    let mut entries: Vec<_> = rd.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries.iter().take(40) {
        let path = entry.path();
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if path.is_dir() {
            eprintln!("{indent}{name_str}/");
            dump_tree(&path, depth + 1, budget - 1);
        } else if name_str.ends_with(".metallib")
            || name_str.contains("metal")
            || name_str.ends_with(".a")
            || name_str.ends_with(".dylib")
        {
            eprintln!("{indent}{name_str}");
        }
    }
}
