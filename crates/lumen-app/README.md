# lumen-app

Tauri 2.x desktop control plane for the `lumen-server` inference engine. A single-window
card-grid dashboard for model management, quantization settings, server lifecycle, and
live metrics — no chat UI in v1 (chat lives in third-party clients via `/v1/*`).

## Prerequisites

- Rust 1.85+ (workspace toolchain)
- Node.js 20+ and npm
- macOS 11+ (this scaffold targets Apple Silicon — Linux/Windows untested)
- `cargo install tauri-cli --version "^2"` (the `cargo tauri` subcommand)

## First-time setup

```bash
# 1. Install frontend deps
cd crates/lumen-app/frontend
npm install
cd ..

# 2. Build the inference server (the desktop app spawns this as a subprocess)
cargo build -p lumen-server --release

# 3. Generate placeholder icons (Tauri ships a CLI helper). Replace the source
#    PNG when you have a real brand mark.
mkdir -p icons
# Provide your own 1024x1024 PNG at /tmp/lumen-icon-source.png, then:
# cargo tauri icon /tmp/lumen-icon-source.png
```

## Dev loop

```bash
# From crates/lumen-app/
cargo tauri dev
```

`cargo tauri dev` starts vite on :5173, builds the Rust binary with `cargo run -p lumen-app`,
and opens the native window. Hot-reload works on the Svelte side; Rust changes require a
restart.

## Production bundle (.app + DMG)

```bash
cargo tauri build
```

Outputs land in `target/release/bundle/macos/Lumen.app` and `target/release/bundle/dmg/`.
For distribution you'll need code signing (`signingIdentity` in `tauri.conf.json::bundle.macOS`)
and notarization via `xcrun notarytool`.

## Architecture

```
┌─ Tauri main process (Rust) ──────────────────────────────────────┐
│  • spawns `lumen-server` as a subprocess via tokio::process     │
│  • streams stdout/stderr → frontend via Tauri events             │
│  • persists config to ~/Library/Application Support/ai.lumen.app │
│  • scans + downloads HF Hub models via hf-hub                    │
└────────────────────────────────────────────────────────────────┘
                       ▲  invoke() / event listen
                       ▼
┌─ Webview (Svelte 5) ────────────────────────────────────────────┐
│  • single-window dashboard, 6 cards always visible              │
│  • no routing, no tabs — everything editable in place           │
└────────────────────────────────────────────────────────────────┘
```

## Server binary resolution

The supervisor looks for `lumen-server` in this order:

1. `config.toml::server_binary_path` (explicit override)
2. Sibling to the running app binary (production .app: `Resources/lumen-server`)
3. `LUMEN_SERVER_BIN` env var
4. `lumen-server` on `$PATH`
5. Workspace target dir (`target/{release,debug}/lumen-server`) — dev fallback

## Config file

Lives at `~/Library/Application Support/ai.lumen.app/config.toml`. Edit manually or via
the UI; the file is rewritten on every UI mutation. The MODELS card's "Download" action
writes weights under `~/Library/Application Support/ai.lumen.app/models/<repo-id>/`.

## What's not in v1

- Chat UI — deferred. Third-party clients use `/v1/chat/completions` directly.
- Streaming metrics from the server — `server_metrics` returns `None` until `lumen-server`
  exposes a `/v1/metrics` endpoint.
- Windows / Linux builds.
- Auto-update — `tauri-plugin-updater` is wired up but disabled in v1.
