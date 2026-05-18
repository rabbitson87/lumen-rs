#!/usr/bin/env python3
"""Download integration-test fixtures from the HuggingFace Hub.

Several `crates/lumen-model/tests/fixtures/*.safetensors` files (≈3.4 GB
total) are not committed to git because of GitHub's 100 MB per-file
limit. This script fetches them from a HuggingFace Hub dataset repo so
the integration tests can run from a clean clone.

Setup:
  1. Upload the fixture files to a HF Hub dataset, e.g.
     `huggingface.co/<USER>/lumen-rs-fixtures`.
  2. Set `LUMEN_FIXTURES_REPO=<USER>/lumen-rs-fixtures` (or edit the
     `DEFAULT_REPO` constant below).
  3. `python scripts/fetch_fixtures.py`

The current implementation is a stub — `DEFAULT_REPO` is intentionally
unset so this script errors out instead of silently downloading from
the wrong place. Wire up the real repo once the fixtures are uploaded.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Default HF Hub dataset for the lumen-rs integration-test fixtures.
# Override with `LUMEN_FIXTURES_REPO=<USER>/<dataset>` at runtime.
DEFAULT_REPO: str | None = "hsng95/lumen-rs-fixtures"

# (file in the HF Hub dataset, destination path relative to repo root).
FIXTURES: list[tuple[str, str]] = [
    (
        "layer0_moe_weights.safetensors",
        "crates/lumen-model/tests/fixtures/layer0_moe_weights.safetensors",
    ),
    (
        "layer0_linear_attn_weights.safetensors",
        "crates/lumen-model/tests/fixtures/layer0_linear_attn_weights.safetensors",
    ),
    (
        "layer3_self_attn_weights.safetensors",
        "crates/lumen-model/tests/fixtures/layer3_self_attn_weights.safetensors",
    ),
    # Add additional fixtures here as they are uploaded.
]


def main() -> int:
    repo = os.environ.get("LUMEN_FIXTURES_REPO", DEFAULT_REPO)
    if not repo:
        print(
            "error: no HuggingFace dataset repo configured. Either set\n"
            "  LUMEN_FIXTURES_REPO=<USER>/<dataset-name>\n"
            "in the environment, or edit DEFAULT_REPO at the top of\n"
            "scripts/fetch_fixtures.py once the fixtures are uploaded.",
            file=sys.stderr,
        )
        return 2

    try:
        from huggingface_hub import hf_hub_download  # type: ignore
    except ImportError:
        print(
            "error: `huggingface_hub` not installed. Run:\n"
            "  pip install huggingface_hub",
            file=sys.stderr,
        )
        return 2

    root = Path(__file__).resolve().parent.parent
    for src, dst_rel in FIXTURES:
        dst = root / dst_rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {src} -> {dst_rel}")
        local = hf_hub_download(repo_id=repo, filename=src, repo_type="dataset")
        # Hard-link if same filesystem; copy otherwise.
        try:
            if dst.exists():
                dst.unlink()
            os.link(local, dst)
        except OSError:
            import shutil

            shutil.copy2(local, dst)
    print("[fetch] done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
