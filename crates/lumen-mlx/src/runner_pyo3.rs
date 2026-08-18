//! In-process MLX runner using PyO3. Imports `mlx_runner.MlxRunner` from the
//! crate's `python/` directory and invokes its methods directly through the
//! Python C API. Eliminates the IPC pipe overhead (~150 μs per call) of the
//! subprocess fallback.
//!
//! GIL semantics: each call wraps in `Python::attach`. Our engine drives a
//! single backend behind a `Mutex`, so the GIL doesn't introduce contention
//! it didn't already have — multiple sequences are still serialized on the
//! Python side either way (the same way `mlx_lm` itself runs).

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::{Context, Result, anyhow};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::{LoadInfo, ProbeRows, crate_python_dir};

/// Locate the workspace `.venv` directory. The embedded Python interpreter
/// doesn't inherit a virtualenv automatically — we have to add the venv's
/// `site-packages` to `sys.path` ourselves, otherwise `import mlx_lm` fails
/// even though we linked against `.venv/bin/python`'s libpython at compile time.
fn workspace_venv() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("LUMEN_MLX_VENV") {
        return Some(PathBuf::from(p));
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let candidate = crate_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join(".venv"))?;
    if candidate.exists() {
        Some(candidate)
    } else {
        None
    }
}

/// Find the venv's `site-packages` directory. Walks `lib/pythonX.Y/site-packages`
/// — the version dir is whatever the venv was created with.
fn venv_site_packages(venv: &PathBuf) -> Option<PathBuf> {
    let lib = venv.join("lib");
    if let Ok(entries) = std::fs::read_dir(&lib) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with("python") {
                let sp = entry.path().join("site-packages");
                if sp.exists() {
                    return Some(sp);
                }
            }
        }
    }
    None
}

/// Adds (1) the venv's `site-packages` and (2) our `python/` directory to
/// `sys.path`. Done exactly once per process; subsequent calls are no-ops.
///
/// Also rewires `sys.executable` and `multiprocessing.set_executable` to the
/// venv's real Python — without this, embedded PyO3's `sys.executable` points
/// at the host Rust binary, so anything calling `multiprocessing.spawn` or
/// `ProcessPoolExecutor` re-launches our binary, which re-runs `main()`,
/// which re-loads the model. Observed concretely: 5 processes spawned during
/// `MlxBackend::load`, 4 mlx_lm.load calls, ~95 GB RSS for a 35B model →
/// instant OOM on a 36 GB machine.
fn ensure_sys_path(py: Python<'_>) -> Result<()> {
    static DONE: OnceLock<()> = OnceLock::new();
    if DONE.get().is_some() {
        return Ok(());
    }

    let sys = py.import("sys").map_err(|e| anyhow!("import sys: {e}"))?;
    let path = sys.getattr("path").map_err(|e| anyhow!("sys.path: {e}"))?;

    // Rewire sys.executable to the venv's Python BEFORE the runner module
    // imports anything that might spawn (mlx_lm → huggingface_hub →
    // ProcessPoolExecutor). Without this, `multiprocessing.spawn` inherits
    // sys.executable pointing at the Rust binary and re-launches us.
    if let Some(venv) = workspace_venv() {
        let venv_python = venv.join("bin").join("python");
        if venv_python.exists() {
            let venv_python_str = venv_python
                .to_str()
                .ok_or_else(|| anyhow!("venv python path not utf-8"))?;
            // 1. Set in os.environ so `mlx_runner.py`'s _fix_embedded_python_executable
            //    sees it on import.
            // 2. Also set on sys + multiprocessing here directly as a belt-and-suspenders.
            let os_module = py.import("os").map_err(|e| anyhow!("import os: {e}"))?;
            let environ = os_module
                .getattr("environ")
                .map_err(|e| anyhow!("os.environ: {e}"))?;
            environ
                .set_item("LUMEN_MLX_PYTHON", venv_python_str)
                .map_err(|e| anyhow!("set LUMEN_MLX_PYTHON: {e}"))?;
            sys.setattr("executable", venv_python_str)
                .map_err(|e| anyhow!("sys.executable: {e}"))?;
            // multiprocessing import is lazy — guard it.
            if let Ok(mp) = py.import("multiprocessing") {
                let _ = mp.call_method1("set_executable", (venv_python_str,));
            }
            eprintln!("[mlx-pyo3] sys.executable = {}", venv_python_str);
        }
    }

    // 1. Venv site-packages — required so `import mlx_lm` resolves.
    if let Some(venv) = workspace_venv() {
        if let Some(sp) = venv_site_packages(&venv) {
            let sp_str = sp
                .to_str()
                .ok_or_else(|| anyhow!("venv site-packages path not utf-8: {}", sp.display()))?;
            path.call_method1("insert", (0i32, sp_str))
                .map_err(|e| anyhow!("sys.path.insert(site-packages): {e}"))?;
            eprintln!("[mlx-pyo3] sys.path += {}", sp_str);
        } else {
            eprintln!(
                "[mlx-pyo3] warning: venv site-packages not found under {} \
                 (set LUMEN_MLX_VENV to override; expected lib/pythonX.Y/site-packages)",
                venv.display()
            );
        }
    }

    // 2. Crate's python/ dir — where mlx_runner.py lives.
    let python_dir = crate_python_dir();
    let path_str = python_dir
        .to_str()
        .ok_or_else(|| anyhow!("python dir path not utf-8: {}", python_dir.display()))?;
    path.call_method1("insert", (0i32, path_str))
        .map_err(|e| anyhow!("sys.path.insert(python_dir): {e}"))?;
    eprintln!("[mlx-pyo3] sys.path += {}", path_str);

    DONE.set(()).ok();
    Ok(())
}

pub(crate) struct Pyo3Runner {
    /// Python `MlxRunner` instance held across calls. Holding `Py<PyAny>` is
    /// `Send + Sync` — actual access still requires the GIL via `bind(py)`.
    runner: Py<PyAny>,
}

impl Pyo3Runner {
    pub(crate) fn new() -> Result<Self> {
        let runner = Python::attach(|py| -> Result<Py<PyAny>> {
            ensure_sys_path(py)?;
            let module = py
                .import("mlx_runner")
                .context("import mlx_runner (check that python/mlx_runner.py exists and venv has mlx-lm installed)")?;
            let class = module
                .getattr("MlxRunner")
                .context("MlxRunner class not found in mlx_runner module")?;
            let instance = class.call0().context("instantiate MlxRunner()")?;
            Ok(instance.unbind())
        })?;
        Ok(Self { runner })
    }

    pub(crate) fn load(&mut self, model_id: &str) -> Result<LoadInfo> {
        Python::attach(|py| -> Result<LoadInfo> {
            let runner = self.runner.bind(py);
            let result = runner
                .call_method1("load", (model_id,))
                .map_err(|e| anyhow!("MlxRunner.load: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("load returned non-dict: {e}"))?;
            let eos_tokens: Vec<u32> = dict
                .get_item("eos_tokens")
                .map_err(|e| anyhow!("eos_tokens key: {e}"))?
                .map(|v| -> Result<Vec<u32>> {
                    let list = v
                        .cast::<PyList>()
                        .map_err(|e| anyhow!("eos_tokens not list: {e}"))?;
                    list.iter()
                        .map(|item| {
                            item.extract::<u32>()
                                .map_err(|e| anyhow!("eos token extract: {e}"))
                        })
                        .collect()
                })
                .transpose()?
                .unwrap_or_default();
            let vocab_size: usize = dict
                .get_item("vocab_size")
                .map_err(|e| anyhow!("vocab_size key: {e}"))?
                .map(|v| {
                    v.extract::<usize>()
                        .map_err(|e| anyhow!("vocab_size extract: {e}"))
                })
                .transpose()?
                .unwrap_or(0);
            Ok(LoadInfo {
                eos_tokens,
                vocab_size,
            })
        })
    }

    pub(crate) fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        Python::attach(|py| -> Result<(u32, usize)> {
            let runner = self.runner.bind(py);
            // Python int + list[int]; PyO3 converts &[u32] via ToPyObject.
            let toks_list = PyList::new(py, tokens).map_err(|e| anyhow!("PyList::new: {e}"))?;
            let result = runner
                .call_method1("prefill", (seq_id, toks_list))
                .map_err(|e| anyhow!("MlxRunner.prefill: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("prefill returned non-dict: {e}"))?;
            let next: u32 = dict
                .get_item("next_token")
                .map_err(|e| anyhow!("next_token: {e}"))?
                .ok_or_else(|| anyhow!("prefill: missing next_token"))?
                .extract()
                .map_err(|e| anyhow!("next_token extract: {e}"))?;
            let pos: usize = dict
                .get_item("position")
                .map_err(|e| anyhow!("position: {e}"))?
                .ok_or_else(|| anyhow!("prefill: missing position"))?
                .extract()
                .map_err(|e| anyhow!("position extract: {e}"))?;
            Ok((next, pos))
        })
    }

    pub(crate) fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        Python::attach(|py| -> Result<(u32, usize)> {
            let runner = self.runner.bind(py);
            let result = runner
                .call_method1("decode_step", (seq_id, last_token, position))
                .map_err(|e| anyhow!("MlxRunner.decode_step: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("decode_step returned non-dict: {e}"))?;
            let next: u32 = dict
                .get_item("next_token")
                .map_err(|e| anyhow!("next_token: {e}"))?
                .ok_or_else(|| anyhow!("decode_step: missing next_token"))?
                .extract()
                .map_err(|e| anyhow!("next_token extract: {e}"))?;
            let pos: usize = dict
                .get_item("position")
                .map_err(|e| anyhow!("position: {e}"))?
                .ok_or_else(|| anyhow!("decode_step: missing position"))?
                .extract()
                .map_err(|e| anyhow!("position extract: {e}"))?;
            Ok((next, pos))
        })
    }

    pub(crate) fn take_decode_stage_timings(&mut self) -> Result<Vec<(u64, u64, u64, u64)>> {
        Python::attach(|py| -> Result<Vec<(u64, u64, u64, u64)>> {
            let runner = self.runner.bind(py);
            let result = runner
                .call_method0("get_decode_stage_timings")
                .map_err(|e| anyhow!("MlxRunner.get_decode_stage_timings: {e}"))?;
            let list: Vec<(u64, u64, u64, u64)> = result
                .extract()
                .map_err(|e| anyhow!("decode_stage_timings extract: {e}"))?;
            Ok(list)
        })
    }

    pub(crate) fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        Python::attach(|py| -> Result<()> {
            let runner = self.runner.bind(py);
            runner
                .call_method1("remove_seq", (seq_id,))
                .map_err(|e| anyhow!("MlxRunner.remove_seq: {e}"))?;
            Ok(())
        })
    }

    pub(crate) fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        Python::attach(|py| -> Result<(u32, usize)> {
            let runner = self.runner.bind(py);
            let toks_list = PyList::new(py, tokens).map_err(|e| anyhow!("PyList::new: {e}"))?;
            let result = runner
                .call_method1("extend", (seq_id, toks_list))
                .map_err(|e| anyhow!("MlxRunner.extend: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("extend returned non-dict: {e}"))?;
            let next: u32 = dict
                .get_item("next_token")
                .map_err(|e| anyhow!("next_token: {e}"))?
                .ok_or_else(|| anyhow!("extend: missing next_token"))?
                .extract()
                .map_err(|e| anyhow!("next_token extract: {e}"))?;
            let pos: usize = dict
                .get_item("position")
                .map_err(|e| anyhow!("position: {e}"))?
                .ok_or_else(|| anyhow!("extend: missing position"))?
                .extract()
                .map_err(|e| anyhow!("position extract: {e}"))?;
            Ok((next, pos))
        })
    }

    pub(crate) fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        Python::attach(|py| -> Result<u64> {
            let runner = self.runner.bind(py);
            let result = runner
                .call_method1("snapshot_state", (seq_id,))
                .map_err(|e| anyhow!("MlxRunner.snapshot_state: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("snapshot_state returned non-dict: {e}"))?;
            let sid: u64 = dict
                .get_item("snapshot_id")
                .map_err(|e| anyhow!("snapshot_id: {e}"))?
                .ok_or_else(|| anyhow!("snapshot_state: missing snapshot_id"))?
                .extract()
                .map_err(|e| anyhow!("snapshot_id extract: {e}"))?;
            Ok(sid)
        })
    }

    pub(crate) fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        Python::attach(|py| -> Result<usize> {
            let runner = self.runner.bind(py);
            let result = runner
                .call_method1("restore_state", (seq_id, snapshot_id))
                .map_err(|e| anyhow!("MlxRunner.restore_state: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("restore_state returned non-dict: {e}"))?;
            let pos: usize = dict
                .get_item("position")
                .map_err(|e| anyhow!("position: {e}"))?
                .ok_or_else(|| anyhow!("restore_state: missing position"))?
                .extract()
                .map_err(|e| anyhow!("position extract: {e}"))?;
            Ok(pos)
        })
    }

    pub(crate) fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        Python::attach(|py| -> Result<()> {
            let runner = self.runner.bind(py);
            runner
                .call_method1("release_snapshot", (snapshot_id,))
                .map_err(|e| anyhow!("MlxRunner.release_snapshot: {e}"))?;
            Ok(())
        })
    }

    pub(crate) fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        Python::attach(|py| -> Result<(u64, usize)> {
            let runner = self.runner.bind(py);
            let result = runner
                .call_method1("snapshot_state_deep", (seq_id,))
                .map_err(|e| anyhow!("MlxRunner.snapshot_state_deep: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("snapshot_state_deep returned non-dict: {e}"))?;
            let sid: u64 = dict
                .get_item("snapshot_id")
                .map_err(|e| anyhow!("snapshot_id: {e}"))?
                .ok_or_else(|| anyhow!("snapshot_state_deep: missing snapshot_id"))?
                .extract()
                .map_err(|e| anyhow!("snapshot_id extract: {e}"))?;
            let pos: usize = dict
                .get_item("position")
                .map_err(|e| anyhow!("position: {e}"))?
                .ok_or_else(|| anyhow!("snapshot_state_deep: missing position"))?
                .extract()
                .map_err(|e| anyhow!("position extract: {e}"))?;
            Ok((sid, pos))
        })
    }

    pub(crate) fn fork_from_snapshot(
        &mut self,
        snapshot_id: u64,
        dst_seq_id: u64,
    ) -> Result<usize> {
        Python::attach(|py| -> Result<usize> {
            let runner = self.runner.bind(py);
            let result = runner
                .call_method1("fork_from_snapshot", (snapshot_id, dst_seq_id))
                .map_err(|e| anyhow!("MlxRunner.fork_from_snapshot: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("fork_from_snapshot returned non-dict: {e}"))?;
            let pos: usize = dict
                .get_item("position")
                .map_err(|e| anyhow!("position: {e}"))?
                .ok_or_else(|| anyhow!("fork_from_snapshot: missing position"))?
                .extract()
                .map_err(|e| anyhow!("position extract: {e}"))?;
            Ok(pos)
        })
    }

    pub(crate) fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
        Python::attach(|py| -> Result<ProbeRows> {
            let runner = self.runner.bind(py);
            let toks_list = PyList::new(py, tokens).map_err(|e| anyhow!("PyList::new: {e}"))?;
            let result = runner
                .call_method1("forward_probe", (seq_id, toks_list))
                .map_err(|e| anyhow!("MlxRunner.forward_probe: {e}"))?;
            let dict = result
                .cast::<PyDict>()
                .map_err(|e| anyhow!("forward_probe returned non-dict: {e}"))?;
            let row_argmaxes: Vec<u32> = dict
                .get_item("row_argmaxes")
                .map_err(|e| anyhow!("row_argmaxes: {e}"))?
                .ok_or_else(|| anyhow!("forward_probe: missing row_argmaxes"))?
                .extract()
                .map_err(|e| anyhow!("row_argmaxes extract: {e}"))?;
            let row_max_abs: Vec<f32> = dict
                .get_item("row_max_abs")
                .map_err(|e| anyhow!("row_max_abs: {e}"))?
                .ok_or_else(|| anyhow!("forward_probe: missing row_max_abs"))?
                .extract()
                .map_err(|e| anyhow!("row_max_abs extract: {e}"))?;
            let position: usize = dict
                .get_item("position")
                .map_err(|e| anyhow!("position: {e}"))?
                .ok_or_else(|| anyhow!("forward_probe: missing position"))?
                .extract()
                .map_err(|e| anyhow!("position extract: {e}"))?;
            Ok(ProbeRows {
                row_argmaxes,
                row_max_abs,
                // The Python probe returns argmax + max|logit| only. Left empty
                // rather than faked; consumers check the length.
                row_top2_gap: Vec::new(),
                position,
            })
        })
    }
}

// `Py<PyAny>` is Send + Sync but our runner is logically !Sync (mutable Python
// state). The MlxBackend owner is single-threaded behind a Mutex in the engine,
// so this matches existing usage. We don't add a manual !Sync because the
// underlying Py<PyAny> is fine; the discipline is at the MlxBackend level.
//
// Touch the unused PathBuf import:
#[allow(dead_code)]
fn _touch_pathbuf_import(_p: PathBuf) {}
