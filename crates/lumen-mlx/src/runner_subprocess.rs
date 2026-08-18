//! Subprocess JSON-RPC runner. Spawns `python mlx_runner.py` and pipes
//! newline-delimited JSON commands over stdin/stdout. Used as fallback when
//! `LUMEN_MLX_SUBPROCESS=1` is set.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, anyhow};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{LoadInfo, ProbeRows, crate_python_dir};

#[derive(Debug, Deserialize)]
struct RpcResponse {
    ok: bool,
    #[serde(default)]
    err: Option<String>,
    #[serde(flatten)]
    extra: Value,
}

fn default_runner_path() -> PathBuf {
    if let Ok(p) = std::env::var("LUMEN_MLX_RUNNER") {
        return PathBuf::from(p);
    }
    crate_python_dir().join("mlx_runner.py")
}

fn default_python_path() -> String {
    if let Ok(p) = std::env::var("LUMEN_MLX_PYTHON") {
        return p;
    }
    let crate_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let venv_python = crate_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join(".venv").join("bin").join("python"));
    if let Some(p) = venv_python
        && p.exists()
    {
        return p.to_string_lossy().into_owned();
    }
    "python3".to_string()
}

pub(crate) struct SubprocessRunner {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl SubprocessRunner {
    pub(crate) fn spawn() -> Result<Self> {
        let python = default_python_path();
        let runner = default_runner_path();
        if !runner.exists() {
            return Err(anyhow!(
                "MLX runner not found at {} (set LUMEN_MLX_RUNNER to override)",
                runner.display()
            ));
        }
        // NEGATIVE result (2026-04-30): tried `taskpolicy -a` to force higher
        // QoS — actually clamped throughput from 72 to 17 tok/s on M3 Max.
        // CLI binaries already get user-interactive QoS by default. Keep the
        // env var as an experimentation hatch but skip by default.
        let taskpolicy = std::env::var("LUMEN_MLX_TASKPOLICY").ok();
        let mut cmd = if let Some(ref policy) = taskpolicy {
            let mut c = Command::new("taskpolicy");
            for tok in policy.split_whitespace() {
                c.arg(tok);
            }
            c.arg(&python).arg(&runner);
            c
        } else {
            let mut c = Command::new(&python);
            c.arg(&runner);
            c
        };
        eprintln!(
            "[mlx-subprocess] spawning: {python} {} (taskpolicy={taskpolicy:?})",
            runner.display()
        );
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .with_context(|| format!("spawn {python} {}", runner.display()))?;
        let stdin = child.stdin.take().ok_or_else(|| anyhow!("no stdin"))?;
        let stdout = BufReader::new(child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?);
        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    fn call(&mut self, req: Value) -> Result<RpcResponse> {
        let line = serde_json::to_string(&req)?;
        self.stdin.write_all(line.as_bytes())?;
        self.stdin.write_all(b"\n")?;
        self.stdin.flush()?;
        let mut buf = String::new();
        let n = self.stdout.read_line(&mut buf)?;
        if n == 0 {
            return Err(anyhow!("mlx runner closed stdout (child exited?)"));
        }
        let resp: RpcResponse = serde_json::from_str(buf.trim_end())
            .with_context(|| format!("parse mlx response: {buf:?}"))?;
        if !resp.ok {
            return Err(anyhow!(
                "mlx runner: {}",
                resp.err.unwrap_or_else(|| "unknown".into())
            ));
        }
        Ok(resp)
    }

    pub(crate) fn load(&mut self, model_id: &str) -> Result<LoadInfo> {
        let resp = self.call(json!({"cmd": "load", "model_id": model_id}))?;
        let eos_tokens: Vec<u32> = resp
            .extra
            .get("eos_tokens")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as u32))
                    .collect()
            })
            .unwrap_or_default();
        let vocab_size = resp
            .extra
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .map(|n| n as usize)
            .unwrap_or(0);
        Ok(LoadInfo {
            eos_tokens,
            vocab_size,
        })
    }

    pub(crate) fn prefill(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        let resp = self.call(json!({
            "cmd": "prefill",
            "seq_id": seq_id,
            "tokens": tokens,
        }))?;
        let next = resp
            .extra
            .get("next_token")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("prefill: missing next_token"))? as u32;
        let pos = resp
            .extra
            .get("position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("prefill: missing position"))? as usize;
        Ok((next, pos))
    }

    pub(crate) fn decode_step(
        &mut self,
        seq_id: u64,
        last_token: u32,
        position: usize,
    ) -> Result<(u32, usize)> {
        let resp = self.call(json!({
            "cmd": "decode_step",
            "seq_id": seq_id,
            "last_token": last_token,
            "position": position,
        }))?;
        let next = resp
            .extra
            .get("next_token")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("decode_step: missing next_token"))? as u32;
        let pos = resp
            .extra
            .get("position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("decode_step: missing position"))? as usize;
        Ok((next, pos))
    }

    pub(crate) fn remove_seq(&mut self, seq_id: u64) -> Result<()> {
        let _ = self.call(json!({"cmd": "remove_seq", "seq_id": seq_id}))?;
        Ok(())
    }

    pub(crate) fn extend(&mut self, seq_id: u64, tokens: &[u32]) -> Result<(u32, usize)> {
        let resp = self.call(json!({
            "cmd": "extend",
            "seq_id": seq_id,
            "tokens": tokens,
        }))?;
        let next = resp
            .extra
            .get("next_token")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("extend: missing next_token"))? as u32;
        let pos = resp
            .extra
            .get("position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("extend: missing position"))? as usize;
        Ok((next, pos))
    }

    pub(crate) fn snapshot_state(&mut self, seq_id: u64) -> Result<u64> {
        let resp = self.call(json!({"cmd": "snapshot_state", "seq_id": seq_id}))?;
        let sid = resp
            .extra
            .get("snapshot_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("snapshot_state: missing snapshot_id"))?;
        Ok(sid)
    }

    pub(crate) fn restore_state(&mut self, seq_id: u64, snapshot_id: u64) -> Result<usize> {
        let resp = self.call(json!({
            "cmd": "restore_state",
            "seq_id": seq_id,
            "snapshot_id": snapshot_id,
        }))?;
        let pos = resp
            .extra
            .get("position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("restore_state: missing position"))? as usize;
        Ok(pos)
    }

    pub(crate) fn release_snapshot(&mut self, snapshot_id: u64) -> Result<()> {
        let _ = self.call(json!({"cmd": "release_snapshot", "snapshot_id": snapshot_id}))?;
        Ok(())
    }

    pub(crate) fn snapshot_state_deep(&mut self, seq_id: u64) -> Result<(u64, usize)> {
        let resp = self.call(json!({"cmd": "snapshot_state_deep", "seq_id": seq_id}))?;
        let sid = resp
            .extra
            .get("snapshot_id")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("snapshot_state_deep: missing snapshot_id"))?;
        let pos = resp
            .extra
            .get("position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("snapshot_state_deep: missing position"))?
            as usize;
        Ok((sid, pos))
    }

    pub(crate) fn fork_from_snapshot(
        &mut self,
        snapshot_id: u64,
        dst_seq_id: u64,
    ) -> Result<usize> {
        let resp = self.call(json!({
            "cmd": "fork_from_snapshot",
            "snapshot_id": snapshot_id,
            "dst_seq_id": dst_seq_id,
        }))?;
        let pos = resp
            .extra
            .get("position")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| anyhow!("fork_from_snapshot: missing position"))?
            as usize;
        Ok(pos)
    }

    pub(crate) fn forward_probe(&mut self, seq_id: u64, tokens: &[u32]) -> Result<ProbeRows> {
        let resp = self.call(json!({
            "cmd": "forward_probe",
            "seq_id": seq_id,
            "tokens": tokens,
        }))?;
        let row_argmaxes: Vec<u32> = resp
            .extra
            .get("row_argmaxes")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as u32))
                    .collect()
            })
            .ok_or_else(|| anyhow!("forward_probe: missing row_argmaxes"))?;
        let row_max_abs: Vec<f32> = resp
            .extra
            .get("row_max_abs")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_f64().map(|n| n as f32))
                    .collect()
            })
            .ok_or_else(|| anyhow!("forward_probe: missing row_max_abs"))?;
        let position =
            resp.extra
                .get("position")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| anyhow!("forward_probe: missing position"))? as usize;
        Ok(ProbeRows {
            row_argmaxes,
            row_max_abs,
            // The subprocess protocol carries argmax + max|logit| only. Left
            // empty rather than faked; consumers check the length.
            row_top2_gap: Vec::new(),
            position,
        })
    }
}

impl Drop for SubprocessRunner {
    fn drop(&mut self) {
        // Best-effort clean shutdown.
        let _ = self.call(json!({"cmd": "shutdown"}));
        std::thread::sleep(std::time::Duration::from_millis(100));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
