//! Print what the prefill memory guard predicts for this model on this machine, at a few
//! prompt lengths, without prefilling anything: the resident memory after load, the predicted
//! peak a prefill of each length adds, and the GPU working set the guard refuses past.
//!
//! Usage:
//!   MODEL_ID=~/models/Qwen3.6-27B-MTPLX-Speed \
//!     cargo run --release -p lumen-mlx --features mlx-native --example prefill_peak_estimate \
//!       -- 4096 32768 65536

use anyhow::Result;
use lumen_mlx::MlxBackend;

fn main() -> Result<()> {
    let model_id = std::env::var("MODEL_ID")?;
    let lengths: Vec<usize> = std::env::args()
        .skip(1)
        .filter_map(|a| a.parse().ok())
        .collect();
    let backend = MlxBackend::load(&model_id)?;
    let gb = |b: u64| b as f64 / 1e9;
    for len in lengths {
        match backend.prefill_peak(len) {
            Some(p) => println!(
                "tokens={len:>7}  resident={:.2} GB  prefill={:.2} GB  peak={:.2} GB  working_set={:.2} GB  refused={}",
                gb(p.resident),
                gb(p.transient),
                gb(p.resident + p.transient),
                gb(p.working_set),
                p.exceeds()
            ),
            None => println!("tokens={len:>7}  no prediction for this backend"),
        }
    }
    Ok(())
}
