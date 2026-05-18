#![allow(dead_code)]

use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::Runner;

pub(crate) const TRANSCRIPT_SCHEMA_VERSION: u32 = 1;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct PrefillRecord {
    pub next_token: u32,
    pub position: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct DecodeRecord {
    pub input_token: u32,
    pub input_position: usize,
    pub next_token: u32,
    pub position: usize,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct RunnerTranscript {
    pub schema_version: u32,
    pub runner: String,
    pub model_id: String,
    pub seq_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub prefill: PrefillRecord,
    pub decode_steps: Vec<DecodeRecord>,
}

impl RunnerTranscript {
    pub(crate) fn compare_to_golden(&self, golden: &Self) -> Result<()> {
        if self == golden {
            return Ok(());
        }

        if self.schema_version != golden.schema_version {
            return Err(anyhow!(
                "schema_version mismatch: actual={} golden={}",
                self.schema_version,
                golden.schema_version
            ));
        }
        if self.model_id != golden.model_id {
            return Err(anyhow!(
                "model_id mismatch: actual={} golden={}",
                self.model_id,
                golden.model_id
            ));
        }
        if self.prompt_tokens != golden.prompt_tokens {
            return Err(anyhow!("prompt_tokens mismatch"));
        }
        if self.prefill != golden.prefill {
            return Err(anyhow!(
                "prefill mismatch: actual={:?} golden={:?}",
                self.prefill,
                golden.prefill
            ));
        }
        if self.decode_steps.len() != golden.decode_steps.len() {
            return Err(anyhow!(
                "decode step count mismatch: actual={} golden={}",
                self.decode_steps.len(),
                golden.decode_steps.len()
            ));
        }
        for (idx, (actual, expected)) in self
            .decode_steps
            .iter()
            .zip(golden.decode_steps.iter())
            .enumerate()
        {
            if actual != expected {
                return Err(anyhow!(
                    "decode step {idx} mismatch: actual={actual:?} golden={expected:?}"
                ));
            }
        }

        Ok(())
    }
}

pub(crate) fn load_runner_transcript(path: impl AsRef<Path>) -> Result<RunnerTranscript> {
    let path = path.as_ref();
    let json = std::fs::read_to_string(path)
        .map_err(|err| anyhow!("failed to read runner transcript {}: {err}", path.display()))?;
    serde_json::from_str(&json).map_err(|err| {
        anyhow!(
            "failed to parse runner transcript {}: {err}",
            path.display()
        )
    })
}

pub(crate) fn capture_runner_transcript<R: Runner + ?Sized>(
    runner: &mut R,
    model_id: &str,
    seq_id: u64,
    prompt_tokens: &[u32],
    decode_steps: usize,
) -> Result<RunnerTranscript> {
    let (mut last_token, mut position) = runner.prefill(seq_id, prompt_tokens)?;
    let prefill = PrefillRecord {
        next_token: last_token,
        position,
    };

    let mut records = Vec::with_capacity(decode_steps);
    for _ in 0..decode_steps {
        let input_token = last_token;
        let input_position = position;
        let (next_token, next_position) =
            runner.decode_step(seq_id, input_token, input_position)?;
        records.push(DecodeRecord {
            input_token,
            input_position,
            next_token,
            position: next_position,
        });
        last_token = next_token;
        position = next_position;
    }

    runner.remove_seq(seq_id)?;

    Ok(RunnerTranscript {
        schema_version: TRANSCRIPT_SCHEMA_VERSION,
        runner: runner.name().to_string(),
        model_id: model_id.to_string(),
        seq_id,
        prompt_tokens: prompt_tokens.to_vec(),
        prefill,
        decode_steps: records,
    })
}

pub(crate) fn compare_runner_to_golden_transcript<R: Runner + ?Sized>(
    runner: &mut R,
    golden: &RunnerTranscript,
) -> Result<()> {
    let actual = capture_runner_transcript(
        runner,
        &golden.model_id,
        golden.seq_id,
        &golden.prompt_tokens,
        golden.decode_steps.len(),
    )?;
    actual.compare_to_golden(golden)
}
