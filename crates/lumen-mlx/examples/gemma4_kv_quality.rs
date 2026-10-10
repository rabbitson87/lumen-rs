//! Teacher-forced quality check for Gemma 4 attention changes (task 021).
//!
//! One fixed token sequence (this crate's source, in file order) goes through
//! the model; at every position the harness records the argmax and the gap
//! between the top two logits. Two conditions are then compared position by
//! position. A flip whose gap sits at the bottom of the gap distribution is a
//! broken tie, not a changed prediction — the protocol of `kv_bf16_quality`,
//! which is Qwen-only.
//!
//! * `--path prefill` — the sequence in `--chunk`-token forwards: windowed
//!   steel attention on the sliding layers, the materializing fallback on the
//!   global ones, and with quantized KV the quantized or dequantized prefill.
//! * `--path decode` — all but the last `--steps` tokens prefilled, those then
//!   fed one at a time: single-query attention.
//!
//! Conditions:
//! * `--ab dequant-prefill | fused-decode | none` — one process, A with the
//!   flag off and B with it on; `none` runs A twice (the determinism floor).
//! * `--save FILE` writes A's rows; `--compare FILE` compares A with rows a
//!   different build wrote (an MLX fork change).
//!
//! Quantized KV comes from the environment as in production, e.g.
//! `LUMEN_GEMMA4_QUANT_KV_MODE=on LUMEN_GEMMA4_QUANT_KV_SLIDING=1`.
//!
//! ```text
//! MODEL_ID=~/models/mlx-community--gemma-4-26b-a4b-it-4bit \
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!   --example gemma4_kv_quality -- --path prefill --ab none
//! ```

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("rebuild with --features mlx-native");
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    imp::run()
}

#[cfg(feature = "mlx-native")]
mod imp {
    use anyhow::{Context, Result, anyhow};
    use lumen_mlx::gemma4::{
        Gemma4ChatTemplate, NativeGemma4Model, set_quant_kv_fused_attn,
        set_quant_kv_prefill_dequant,
    };
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
    use mlx_rs::{Array, Dtype};
    use std::path::Path;

    /// Prediction at one position: the argmax and the top-1 − top-2 gap.
    #[derive(Clone, Copy)]
    struct Row {
        argmax: u32,
        gap: f32,
    }

    struct Args {
        path: String,
        tokens: usize,
        chunk: usize,
        steps: usize,
        ab: Option<String>,
        save: Option<String>,
        compare: Option<String>,
    }

    fn parse_args() -> Result<Args> {
        let mut a = Args {
            path: "prefill".into(),
            tokens: 8192,
            chunk: 256,
            steps: 512,
            ab: None,
            save: None,
            compare: None,
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let v = argv
                .get(i + 1)
                .ok_or_else(|| anyhow!("{} needs a value", argv[i]))?
                .clone();
            match argv[i].as_str() {
                "--path" => a.path = v,
                "--tokens" => a.tokens = v.parse().context("--tokens")?,
                "--chunk" => a.chunk = v.parse().context("--chunk")?,
                "--steps" => a.steps = v.parse().context("--steps")?,
                "--ab" => a.ab = Some(v),
                "--save" => a.save = Some(v),
                "--compare" => a.compare = Some(v),
                other => return Err(anyhow!("unknown argument {other:?}")),
            }
            i += 2;
        }
        Ok(a)
    }

    /// Argmax and top-two gap at every position of `[1, L, V]` logits.
    fn rows_of(logits: &Array) -> Result<Vec<Row>> {
        let logits = logits.as_dtype(Dtype::Float32)?;
        let order = mlx_rs::ops::argpartition_axis(&mlx_rs::ops::negative(&logits)?, 1, -1)?;
        let idx = order.index((Ellipsis, 0..2));
        let val = logits.take_along_axis(&idx, -1)?;
        let idx = idx.as_dtype(Dtype::Int32)?.reshape(&[-1])?;
        let val = val.reshape(&[-1])?;
        idx.eval()?;
        val.eval()?;
        let (idx, val) = (idx.as_slice::<i32>(), val.as_slice::<f32>());
        Ok(idx
            .chunks_exact(2)
            .zip(val.chunks_exact(2))
            .map(|(i, v)| {
                let top = if v[0] >= v[1] { 0 } else { 1 };
                Row {
                    argmax: i[top] as u32,
                    gap: (v[0] - v[1]).abs(),
                }
            })
            .collect())
    }

    fn run_prefill(model: &NativeGemma4Model, tokens: &[u32], chunk: usize) -> Result<Vec<Row>> {
        let mut cache = model.make_cache();
        let mut rows = Vec::with_capacity(tokens.len());
        for piece in tokens.chunks(chunk) {
            rows.extend(rows_of(&model.forward(piece, &mut cache)?)?);
        }
        Ok(rows)
    }

    fn run_decode(model: &NativeGemma4Model, tokens: &[u32], steps: usize) -> Result<Vec<Row>> {
        let split = tokens.len() - steps;
        let mut cache = model.make_cache();
        let mut rows = rows_of(&model.forward_last_token_chunked(&tokens[..split], &mut cache)?)?;
        for t in split..tokens.len() - 1 {
            rows.extend(rows_of(&model.forward(&tokens[t..t + 1], &mut cache)?)?);
        }
        Ok(rows)
    }

    fn run_condition(model: &NativeGemma4Model, tokens: &[u32], a: &Args) -> Result<Vec<Row>> {
        match a.path.as_str() {
            "prefill" => run_prefill(model, tokens, a.chunk),
            "decode" => run_decode(model, tokens, a.steps),
            other => Err(anyhow!("--path must be prefill or decode, got {other:?}")),
        }
    }

    /// Agreement of `b` with `a`, and where the flips sit in `a`'s gaps.
    fn report(label: &str, a: &[Row], b: &[Row]) {
        let n = a.len().min(b.len());
        let mut gaps: Vec<f32> = a[..n].iter().map(|r| r.gap).collect();
        gaps.sort_by(f32::total_cmp);
        let pct = |g: f32| gaps.partition_point(|&x| x < g) as f64 / n as f64 * 100.0;
        let p15 = gaps[(n as f64 * 0.015) as usize];
        let flips: Vec<f32> = (0..n)
            .filter(|&i| a[i].argmax != b[i].argmax)
            .map(|i| a[i].gap)
            .collect();
        let over = flips.iter().filter(|&&g| g > p15).count();
        let worst = flips.iter().copied().fold(0f32, f32::max);
        println!(
            "[{label}] agree {}/{n} ({:.3}%), flips {} — largest flip gap {worst:.3} \
             (percentile {:.2}), {over} above the 1.5th-percentile gap {p15:.3}",
            n - flips.len(),
            100.0 * (n - flips.len()) as f64 / n as f64,
            flips.len(),
            if flips.is_empty() { 0.0 } else { pct(worst) },
        );
    }

    fn save(path: &str, rows: &[Row]) -> Result<()> {
        let text: String = rows
            .iter()
            .map(|r| format!("{} {}\n", r.argmax, r.gap))
            .collect();
        std::fs::write(path, text).with_context(|| format!("write {path}"))
    }

    fn load(path: &str) -> Result<Vec<Row>> {
        std::fs::read_to_string(path)
            .with_context(|| format!("read {path}"))?
            .lines()
            .map(|l| {
                let (a, g) = l.split_once(' ').ok_or_else(|| anyhow!("bad row {l:?}"))?;
                Ok(Row {
                    argmax: a.parse()?,
                    gap: g.parse()?,
                })
            })
            .collect()
    }

    fn corpus_tokens(template: &Gemma4ChatTemplate, n: usize) -> Result<Vec<u32>> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<_> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "rs"))
            .collect();
        files.sort();
        let mut text = String::new();
        for f in files {
            text.push_str(&std::fs::read_to_string(&f)?);
        }
        let ids = template.encode_plain(&text)?;
        if ids.len() < n {
            return Err(anyhow!("corpus is {} tokens, {n} asked for", ids.len()));
        }
        Ok(ids[..n].to_vec())
    }

    pub fn run() -> Result<()> {
        let a = parse_args()?;
        let model_dir = std::env::var("MODEL_ID")
            .map_err(|_| anyhow!("set MODEL_ID to a Gemma 4 model directory"))?;
        let model = NativeGemma4Model::load(Path::new(&model_dir))?;
        let template = Gemma4ChatTemplate::from_dir(&model_dir)?;
        let tokens = corpus_tokens(&template, a.tokens)?;
        println!(
            "gemma4_kv_quality: path={} tokens={} chunk={} steps={} quant_kv={}",
            a.path,
            a.tokens,
            a.chunk,
            a.steps,
            std::env::var("LUMEN_GEMMA4_QUANT_KV_MODE").unwrap_or_else(|_| "off".into()),
        );
        let set = |flag: &str, on: bool| -> Result<()> {
            match flag {
                "dequant-prefill" => set_quant_kv_prefill_dequant(on),
                "fused-decode" => set_quant_kv_fused_attn(on),
                "none" => {}
                other => return Err(anyhow!("--ab: unknown flag {other:?}")),
            }
            Ok(())
        };
        if let Some(flag) = &a.ab {
            set(flag, false)?;
        }
        let base = run_condition(&model, &tokens, &a)?;
        if let Some(path) = &a.save {
            save(path, &base)?;
            println!("saved {} rows to {path}", base.len());
        }
        if let Some(path) = &a.compare {
            report(&format!("this build vs {path}"), &load(path)?, &base);
        }
        if let Some(flag) = &a.ab {
            set(flag, flag != "none")?;
            let other = run_condition(&model, &tokens, &a)?;
            report(&format!("{flag} on vs off"), &base, &other);
        }
        Ok(())
    }
}
