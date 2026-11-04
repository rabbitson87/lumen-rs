//! How the release app bundle and the server binary agree on where MLX's
//! kernel library lives. See `lumen_server::metallib`.

use lumen_server::metallib::{Plan, bundled_path, colocated_path, plan};
use std::path::Path;

/// The release app ran its server from a read-only disk image (or a
/// Gatekeeper-translocated copy) and died: it tried to unpack `mlx.metallib`
/// into the bundle's `Contents/MacOS`, the write failed, and MLX aborted with
/// "Failed to load the default metallib". When the bundle ships this binary's
/// library, the bundle must be left alone.
#[test]
fn a_bundled_library_is_used_without_writing_into_the_bundle() {
    let embedded = 131_005_534;
    assert_eq!(plan(None, Some(embedded), embedded), Plan::UseBundled);
    // A bare binary (no bundle) still unpacks beside itself, as before.
    assert_eq!(plan(None, None, embedded), Plan::Unpack);
    // A copy already beside the binary is kept — MLX looks there first.
    assert_eq!(plan(Some(embedded), None, embedded), Plan::AlreadyColocated);
    assert_eq!(
        plan(Some(embedded), Some(embedded), embedded),
        Plan::AlreadyColocated
    );
}

/// A bundled copy that is not this binary's (a different size) is not trusted,
/// and neither is a matching bundled copy shadowed by a stale one beside the
/// binary, which MLX would load first.
#[test]
fn a_mismatched_library_is_never_relied_on() {
    let embedded = 131_005_534;
    assert_eq!(plan(None, Some(embedded - 1), embedded), Plan::Unpack);
    assert_eq!(plan(Some(1), Some(embedded), embedded), Plan::Unpack);
}

/// The paths are the ones the MLX fork searches: beside the binary, and the
/// bundle's `Contents/Resources` for a binary in `Contents/MacOS`.
#[test]
fn the_paths_match_where_mlx_looks() {
    let macos = Path::new("/Applications/Lumen.app/Contents/MacOS");
    assert_eq!(
        colocated_path(macos),
        Path::new("/Applications/Lumen.app/Contents/MacOS/mlx.metallib")
    );
    assert_eq!(
        bundled_path(macos),
        Path::new("/Applications/Lumen.app/Contents/MacOS/../Resources/mlx.metallib")
    );
}

/// The release workflow is the other half: it has to put this binary's library
/// in the bundle's `Contents/Resources`, or `UseBundled` never applies and the
/// app falls back to writing into its own `Contents/MacOS`.
#[test]
fn the_release_workflow_ships_the_library_where_mlx_looks() {
    let workflow = std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/release.yml"),
    )
    .expect("read .github/workflows/release.yml");
    assert!(
        workflow.contains("--write-metallib crates/lumen-app/binaries/mlx.metallib"),
        "the workflow must stage the server's own metallib next to the sidecar"
    );
    assert!(
        workflow.contains(r#""resources":{"binaries/mlx.metallib":"mlx.metallib"}"#),
        "the workflow must map the staged metallib into Contents/Resources"
    );
}

/// MLX's kernel library is compiled for one macOS version and does not load on
/// an older one. The release job pins that version with
/// `MACOSX_DEPLOYMENT_TARGET`; the bundle has to declare the same floor, or the
/// app installs on a macOS whose server then cannot load its kernels — which is
/// what a bundle claiming 11.0 around a metallib built for the runner's SDK
/// (26.5) did.
#[test]
fn the_bundle_declares_the_macos_its_kernels_were_built_for() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let workflow = std::fs::read_to_string(root.join(".github/workflows/release.yml"))
        .expect("read release.yml");
    let pinned = workflow
        .lines()
        .find_map(|l| l.trim().strip_prefix("MACOSX_DEPLOYMENT_TARGET:"))
        .map(|v| v.trim().trim_matches('"').to_string())
        .expect("the release job pins MACOSX_DEPLOYMENT_TARGET");
    let conf: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("crates/lumen-app/tauri.conf.json"))
            .expect("read tauri.conf.json"),
    )
    .expect("parse tauri.conf.json");
    let declared = conf["bundle"]["macOS"]["minimumSystemVersion"]
        .as_str()
        .expect("bundle.macOS.minimumSystemVersion");
    assert_eq!(
        declared, pinned,
        "tauri.conf.json minimumSystemVersion must equal the release job's MACOSX_DEPLOYMENT_TARGET"
    );
}

/// The pinned `rev` of the mlx-rs fork in a crate's manifest line for `dep`.
fn pinned_rev(manifest: &str, dep: &str) -> Option<String> {
    manifest
        .lines()
        .find(|l| l.trim_start().starts_with(&format!("{dep} = {{")))
        .and_then(|l| l.split("rev = \"").nth(1))
        .and_then(|r| r.split('"').next())
        .map(str::to_string)
}

/// A fresh `cargo build -p lumen-server` used to fail: build.rs looked for the
/// metallib while MLX was still compiling, because nothing ordered it after
/// mlx-sys's build script. mlx-sys declares `links = "mlx"`, which orders the
/// build scripts of its *direct* dependents and hands them DEP_MLX_METALLIB —
/// so the server must depend on mlx-sys itself, at the same rev lumen-mlx uses
/// (another rev would be a second MLX build, and a different library).
#[test]
fn the_server_depends_on_mlx_sys_directly_at_lumen_mlx_s_rev() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let server = std::fs::read_to_string(root.join("Cargo.toml")).expect("read server manifest");
    let mlx = std::fs::read_to_string(root.join("../lumen-mlx/Cargo.toml"))
        .expect("read lumen-mlx manifest");
    assert!(
        server.contains("\"dep:mlx-sys\""),
        "the mlx-native feature must enable the direct mlx-sys dependency"
    );
    let server_rev = pinned_rev(&server, "mlx-sys").expect("server pins mlx-sys");
    assert_eq!(
        Some(server_rev),
        pinned_rev(&mlx, "mlx-sys"),
        "lumen-server and lumen-mlx must pin the same mlx-sys"
    );
}

/// The release workflow, as text.
fn release_workflow() -> String {
    std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.github/workflows/release.yml"),
    )
    .expect("read release.yml")
}

/// v0.12.0 never shipped: the job handed the notary credentials to the build,
/// and with the Apple Developer Program membership lapsed Apple answered 401 —
/// after the build and the code signature had succeeded. Releases are now built
/// with no Apple certificate and not notarized, so cutting one needs no
/// membership: nothing in the workflow may read an Apple credential.
#[test]
fn the_release_builds_without_apple_credentials() {
    let workflow = release_workflow();
    let reads: Vec<&str> = workflow
        .lines()
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter(|l| l.contains("secrets.APPLE_") || l.contains("MACOS_NOTARIZE"))
        .collect();
    assert!(
        reads.is_empty(),
        "the release must build without Apple credentials, but reads {reads:?}"
    );
}

/// An unnotarized download is blocked on its first launch until its quarantine
/// is cleared. The release page and the README give the same command for it,
/// naming the app where the DMG installs it; the update notes, which the app
/// shows when it offers an update, stay free of it (tauri-action's
/// `releaseBody` would become those notes).
#[test]
fn the_release_page_and_the_readme_give_the_same_first_launch_command() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let conf: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("crates/lumen-app/tauri.conf.json"))
            .expect("read tauri.conf.json"),
    )
    .expect("parse tauri.conf.json");
    let app = conf["productName"].as_str().expect("productName");
    let command = format!("xattr -dr com.apple.quarantine /Applications/{app}.app");
    let workflow = release_workflow();
    let notes_step = workflow
        .split("- name: Put the install notes on the release")
        .nth(1)
        .expect("the step that writes the install notes");
    assert!(
        notes_step.contains(&command),
        "the release notes must give `{command}`"
    );
    let readme = std::fs::read_to_string(root.join("README.md")).expect("read README.md");
    assert!(readme.contains(&command), "README.md must give `{command}`");
    assert!(
        !workflow.contains("releaseBody:"),
        "install notes as tauri-action's releaseBody would become the update notes"
    );
}

/// Without a Developer ID certificate Tauri left only the linker's signature on
/// the executables, which promises a bundle seal that is not there: macOS calls
/// such a download "damaged" and offers only Move to Trash. An ad-hoc identity
/// seals the bundle, so an unnotarized build can still be opened once its
/// quarantine is cleared.
#[test]
fn a_build_without_a_certificate_is_still_validly_signed() {
    let conf: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../lumen-app/tauri.conf.json"),
        )
        .expect("read tauri.conf.json"),
    )
    .expect("parse tauri.conf.json");
    assert_eq!(conf["bundle"]["macOS"]["signingIdentity"], "-");
}
