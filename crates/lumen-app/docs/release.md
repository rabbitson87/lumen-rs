# Releasing Lumen

The app ships from **GitHub Releases** with the **Tauri auto-updater** doing the
heavy lifting. Each release atomically updates the `.app` bundle **and** the
sidecar `lumen-server` binary so users can't end up with a UI/server version
mismatch.

## One-time setup (per maintainer machine)

### 1. Generate the signing keypair

The updater verifies every download against an Ed25519 public key embedded in
the app. Generate the keypair once and **keep the private key offline** (it
lives outside the repo — only the GitHub Actions runner sees it, via a secret).

```bash
cargo install tauri-cli --version "^2"  # if not already installed
# Verify install — should print 2.x.y
cargo tauri --version
# Generate keypair (note: `cargo tauri ...`, not standalone `tauri ...` —
# cargo-tauri is a cargo subcommand, not a top-level binary on PATH).
cargo tauri signer generate -w ~/.tauri/lumen.key
```

This produces:
- `~/.tauri/lumen.key` — **private** key (PEM). Treat like an SSH key.
- `~/.tauri/lumen.key.pub` — public key (base64). Goes into `tauri.conf.json`.

Paste the base64 line from the `.pub` file into
[crates/lumen-app/tauri.conf.json](../tauri.conf.json) →
`plugins.updater.pubkey`, replacing `REPLACE_ME_WITH_GENERATED_ED25519_PUBKEY`.

Commit this — the **public** key in the repo is correct and required for the
auto-updater to validate signatures.

### 2. Store the private key as a GitHub Actions secret

In the GitHub repo: Settings → Secrets and variables → Actions → New secret.

- Name: `TAURI_SIGNING_PRIVATE_KEY`
- Value: the full contents of `~/.tauri/lumen.key`

Also add:
- `TAURI_SIGNING_PRIVATE_KEY_PASSWORD` — if you set a passphrase during generation.

## Cutting a release

### 1. Bump the version

Three places must agree on the version string:

```bash
# crates/lumen-app/Cargo.toml         → [package].version = "0.2.0"
# crates/lumen-app/tauri.conf.json    → "version": "0.2.0"
# crates/lumen-app/frontend/package.json → "version": "0.2.0"
```

If a config-schema change landed, also bump `CURRENT_SCHEMA_VERSION` in
[`config.rs`](../src/config.rs) and add a migration step.

### 2. Build & sign locally (manual flow)

```bash
# From repo root
cd crates/lumen-app

# Build the lumen-server binary for the target triple.
# Apple Silicon only — MLX (mlx-sys) refuses to build on x86_64, so there
# is no Intel Mac target.
TARGET=aarch64-apple-darwin
cargo build -p lumen-server --release --target "$TARGET"

# Tauri's sidecar feature requires the binary at a specific name.
mkdir -p binaries
cp ../../target/$TARGET/release/lumen-server \
   binaries/lumen-server-$TARGET

# Build the .app bundle. The --config flag injects the sidecar binding so the
# default tauri.conf.json stays clean for `cargo tauri dev`.
cargo tauri build --target "$TARGET" --config '{"bundle":{"externalBin":["binaries/lumen-server"]}}'
```

Outputs:
- `target/$TARGET/release/bundle/macos/Lumen.app` — signable artifact
- `target/$TARGET/release/bundle/dmg/Lumen_0.2.0_<arch>.dmg`
- `target/$TARGET/release/bundle/macos/Lumen.app.tar.gz.sig` — updater signature

### 3. Generate the update manifest

The Tauri updater fetches a JSON file describing the latest release. For the
GitHub Releases pattern (referenced by `tauri.conf.json::plugins.updater.endpoints`):

```json
{
  "version": "0.2.0",
  "notes": "## 0.2.0 — 2026-06-01\n\n- New ADVANCED card with spec-decoding toggle\n- Fixed: KV cache memory limit not applying on M3 Max",
  "pub_date": "2026-06-01T12:00:00Z",
  "platforms": {
    "darwin-aarch64": {
      "signature": "<contents of Lumen.app.tar.gz.sig>",
      "url": "https://github.com/rabbitson87/lumen-rs/releases/download/v0.2.0/Lumen_0.2.0_aarch64.app.tar.gz"
    }
  }
}
```

Save as `latest.json`.

### 4. Tag + publish to GitHub Releases

```bash
git tag v0.2.0
git push origin v0.2.0
gh release create v0.2.0 \
  target/aarch64-apple-darwin/release/bundle/macos/Lumen.app.tar.gz \
  target/aarch64-apple-darwin/release/bundle/macos/Lumen.app.tar.gz.sig \
  target/aarch64-apple-darwin/release/bundle/dmg/Lumen_0.2.0_aarch64.dmg \
  latest.json \
  --notes-file CHANGELOG.md
```

Existing v0.1.0 users will see the new version next time they hit **Update →
Check for updates** in the app (or on next launch once you flip the policy to
auto-check).

## Automated flow (GitHub Actions sketch)

Drop the following at `.github/workflows/release.yml`:

```yaml
name: Release
on:
  push:
    tags: ['v*']
jobs:
  build:
    strategy:
      matrix:
        # Apple Silicon only — MLX upstream CMakeLists.txt errors out
        # on x86_64. An Intel matrix entry can't succeed with mlx-native.
        target: [aarch64-apple-darwin]
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          targets: ${{ matrix.target }}
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - name: Build server
        run: |
          cargo build -p lumen-server --release --target ${{ matrix.target }}
          mkdir -p crates/lumen-app/binaries
          cp target/${{ matrix.target }}/release/lumen-server \
             crates/lumen-app/binaries/lumen-server-${{ matrix.target }}
      - name: Install frontend deps
        run: cd crates/lumen-app/frontend && npm ci
      - uses: tauri-apps/tauri-action@v0
        env:
          TAURI_SIGNING_PRIVATE_KEY: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY }}
          TAURI_SIGNING_PRIVATE_KEY_PASSWORD: ${{ secrets.TAURI_SIGNING_PRIVATE_KEY_PASSWORD }}
        with:
          projectPath: crates/lumen-app
          tagName: ${{ github.ref_name }}
          releaseName: 'Lumen ${{ github.ref_name }}'
          releaseDraft: true
          prerelease: false
          args: --target ${{ matrix.target }} --config '{"bundle":{"externalBin":["binaries/lumen-server"]}}'
```

`tauri-action` handles signing the `.app.tar.gz`, uploading both bundle + sig to
the GitHub Release, and writing the `latest.json` manifest in the format the
updater expects. After the job completes, publish the draft release —
that flips the visibility for `releases/latest/download/latest.json`, which is
the URL baked into `tauri.conf.json`.

## Config schema migrations

Bump [`CURRENT_SCHEMA_VERSION`](../src/config.rs) and append a step to
`migrate_in_place`:

```rust
while cfg.schema_version < 2 {
    // v1 -> v2: rename `server.api_key: Option<String>` to
    // `server.api_keys: Vec<String>` for multi-key support.
    if let Some(k) = cfg.server.api_key.take() {
        cfg.server.api_keys.push(k);
    }
    cfg.schema_version = 2;
}
```

`load_or_default` automatically writes a `.toml.bak` next to the live config
before running the chain, so a botched migration is recoverable by hand.

## Rollback

If a release introduces a critical regression:

1. Mark the GitHub Release as **draft** (or delete it). This pulls
   `releases/latest/download/latest.json` back to the previous version.
2. Users who haven't auto-updated yet see no notification.
3. Users on the bad version stay on it — the updater never auto-downgrades. The
   recommended path is to cut **v0.2.1** with the fix, not roll back the
   binary.
4. If schema migrations ran on the bad version, the `.toml.bak` is the recovery
   artifact; surface a fix-up command on next release if needed.
