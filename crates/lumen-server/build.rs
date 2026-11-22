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

    // mlx-sys reports the library it built (`links = "mlx"` → DEP_MLX_METALLIB),
    // and declaring `links` is also what makes Cargo run this script after
    // mlx-sys's. Without that ordering a fresh build ran this script while MLX
    // was still compiling, found nothing, and failed — CI pre-built mlx-sys to
    // get around it, compiling MLX twice. The search below is the fallback for
    // an mlx-sys without the metadata.
    println!("cargo:rerun-if-env-changed=DEP_MLX_METALLIB");
    if let Some(reported) = std::env::var_os("DEP_MLX_METALLIB")
        .map(PathBuf::from)
        .filter(|p| p.exists())
    {
        println!("cargo:rerun-if-changed={}", reported.display());
        println!(
            "cargo:rustc-env=LUMEN_MLX_METALLIB_PATH={}",
            reported.display()
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
    // OUT_DIR layouts (4 climbs):
    //   older cargo, no triple:  target/<profile>/build/<pkg>/out                -> target/
    //   older cargo, triple:     target/<triple>/<profile>/build/<pkg>/out       -> target/<triple>
    //   cargo 1.100+, no triple: target/<profile>/build/<pkg>/<hash>/out         -> target/<profile>
    //   cargo 1.100+, triple:    target/<triple>/<profile>/build/<pkg>/<hash>/out -> target/<triple>/<profile>
    // Either way the first root holds mlx-sys's build dirs, which the walk
    // finds at any depth.
    let triple_dir = climb(&out_dir, 4);
    // One more climb is a wider second root. From target/ it reaches the
    // workspace root, which the search also handles: only paths under an
    // mlx-sys build dir are taken.
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

fn find_metallib(target_dir: &Path) -> Option<PathBuf> {
    // mlx-sys writes the metallib somewhere under
    //   <target>/<profile>/build/mlx-sys-<hash>/out/build/...   (older cargo)
    //   <target>/<profile>/build/mlx-sys/<hash>/out/build/...   (cargo 1.100+)
    // The exact depth depends on the mlx CMake layout (sometimes
    // `out/build/lib/mlx.metallib`, sometimes
    // `out/build/_deps/mlx-build/mlx/backend/metal/kernels/mlx.metallib`).
    // Walk a generous depth so cross-compiled CI builds and any future
    // CMake layout change still find it.
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    walk(target_dir, target_dir, &mut best, 16);
    best.map(|(_, p)| p)
}

/// Walks `dir`, keeping the newest `mlx.metallib` that sits in one of mlx-sys's
/// build dirs below `root`.
fn walk(
    root: &Path,
    dir: &Path,
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
            walk(root, &path, best, depth_budget - 1);
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
            // mlx-sys's real artifacts always live under one of its build dirs.
            // Only the part below `root` counts, so a target dir that is
            // itself named like one (CARGO_TARGET_DIR=…/mlx-sys-cache) cannot
            // let the colocated copy through.
            let under_mlx_sys = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .components()
                .any(|c| mlx_sys_dir(&c.as_os_str().to_string_lossy()).is_some());
            if !under_mlx_sys {
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

/// How a directory under `build/` holds mlx-sys's output. Both the metallib
/// search and the diagnostics below read the layout from here, so a new cargo
/// layout is a change in one place.
enum MlxSysDir {
    /// `mlx-sys-<hash>/`: the directory is one build (older cargo).
    OneBuild,
    /// `mlx-sys/`: one `<hash>/` per build below it (cargo 1.100+).
    PerHash,
}

fn mlx_sys_dir(name: &str) -> Option<MlxSysDir> {
    if name == "mlx-sys" {
        Some(MlxSysDir::PerHash)
    } else if name.starts_with("mlx-sys-") {
        Some(MlxSysDir::OneBuild)
    } else {
        None
    }
}

/// Diagnostic — locate mlx-sys's build directories under `root`
/// (`build/mlx-sys-<hash>/`, or `build/mlx-sys/<hash>/` on cargo 1.100+) and
/// dump their out/ layout so a failed CI run shows what mlx-sys actually
/// produced (vs guessing). Bounded to 4 levels below each out/.
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
        let dirs: Vec<PathBuf> = match mlx_sys_dir(&name_str) {
            Some(MlxSysDir::OneBuild) => vec![entry.path()],
            Some(MlxSysDir::PerHash) => std::fs::read_dir(entry.path())
                .map(|hashes| {
                    hashes
                        .flatten()
                        .map(|h| h.path())
                        .filter(|p| p.is_dir())
                        .collect()
                })
                .unwrap_or_default(),
            None => continue,
        };
        for dir in dirs {
            eprintln!("found mlx-sys dir: {}", dir.display());
            let out_dir = dir.join("out");
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
