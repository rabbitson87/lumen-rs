//! How far does one quantization of Qwen3.8-Flash-Next move the model's predictions from another?
//! Teacher-forced over the same tokens, position by position, in two runs, since two builds of a
//! 125B model do not fit in memory together:
//!
//! ```text
//! # 1. The reference: its top 32 log-probs at every position, and the tokens, to a file.
//! cargo run --release -p lumen-mlx --features mlx-native --example qwen4_exp_quant_quality -- \
//!     record <reference_dir> <out.bin> <text files...> [--max-tokens 8192] [--budget-gb 19]
//! # 2. Another build over the recorded tokens, against that file.
//! cargo run --release -p lumen-mlx --features mlx-native --example qwen4_exp_quant_quality -- \
//!     compare <model_dir> <ref.bin> [--budget-gb 19]
//! ```
//!
//! `compare` reports, overall and by context depth: how often the model's top token is the
//! reference's, the KL divergence from the reference (over the reference's top 32 tokens plus the
//! rest of the vocabulary as one bucket, which can only understate it), and both models'
//! perplexity on the text itself. A build compared against its own record is the control: 100%
//! and a KL of 0 (the same chunks, so the same bits).
//!
//! Each text file is one sequence, cut to `--max-tokens`, fed in 1,024-token chunks (every chunk
//! reads the whole expert set, so a bigger one is faster). `compare --seqs N` compares only the
//! first N sequences.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use anyhow::{Context, Result, anyhow, bail};
    use lumen_mlx::qwen4_exp::Qwen4ExpModel;
    use mlx_rs::ops::indexing::{Ellipsis, IndexOp};
    use mlx_rs::{Array, Dtype};
    use tokenizers::Tokenizer;

    const MAGIC: &[u8; 4] = b"Q4QQ";
    const TOP: usize = 32;
    const CHUNK: usize = 1024;
    /// Rows of a chunk's logits turned into log-probs at a time: a whole chunk's as f32 would hold
    /// 1 GB (a 248K vocabulary) on top of the forward's own memory.
    const SLICE: usize = 256;
    const DEPTHS: [(usize, &str); 4] = [
        (1024, "0-1K"),
        (4096, "1K-4K"),
        (8192, "4K-8K"),
        (usize::MAX, "8K+"),
    ];

    /// The reference at one position: the log-prob of the text's next token, and its top tokens.
    struct RefRow {
        next_lp: f32,
        ids: [u32; TOP],
        lps: [f32; TOP],
    }

    struct Record {
        seqs: Vec<(Vec<u32>, Vec<RefRow>)>,
    }

    fn put_u32(w: &mut impl Write, v: u32) -> Result<()> {
        Ok(w.write_all(&v.to_le_bytes())?)
    }

    fn put_f32(w: &mut impl Write, v: f32) -> Result<()> {
        Ok(w.write_all(&v.to_le_bytes())?)
    }

    fn get_u32(r: &mut impl Read) -> Result<u32> {
        let mut b = [0; 4];
        r.read_exact(&mut b)?;
        Ok(u32::from_le_bytes(b))
    }

    fn get_f32(r: &mut impl Read) -> Result<f32> {
        Ok(f32::from_bits(get_u32(r)?))
    }

    impl Record {
        fn save(&self, path: &Path) -> Result<()> {
            let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
            w.write_all(MAGIC)?;
            put_u32(&mut w, TOP as u32)?;
            put_u32(&mut w, self.seqs.len() as u32)?;
            for (tokens, rows) in &self.seqs {
                put_u32(&mut w, tokens.len() as u32)?;
                for &t in tokens {
                    put_u32(&mut w, t)?;
                }
                for row in rows {
                    put_f32(&mut w, row.next_lp)?;
                    for &id in &row.ids {
                        put_u32(&mut w, id)?;
                    }
                    for &lp in &row.lps {
                        put_f32(&mut w, lp)?;
                    }
                }
            }
            Ok(w.flush()?)
        }

        fn load(path: &Path) -> Result<Self> {
            let mut r = std::io::BufReader::new(
                std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?,
            );
            let mut magic = [0; 4];
            r.read_exact(&mut magic)?;
            if &magic != MAGIC || get_u32(&mut r)? as usize != TOP {
                bail!("{} is not a record of this harness", path.display());
            }
            let n = get_u32(&mut r)?;
            let mut seqs = Vec::new();
            for _ in 0..n {
                let len = get_u32(&mut r)? as usize;
                let tokens = (0..len)
                    .map(|_| get_u32(&mut r))
                    .collect::<Result<Vec<_>>>()?;
                let mut rows = Vec::with_capacity(len - 1);
                for _ in 1..len {
                    let next_lp = get_f32(&mut r)?;
                    let mut ids = [0; TOP];
                    let mut lps = [0.0; TOP];
                    for id in &mut ids {
                        *id = get_u32(&mut r)?;
                    }
                    for lp in &mut lps {
                        *lp = get_f32(&mut r)?;
                    }
                    rows.push(RefRow { next_lp, ids, lps });
                }
                seqs.push((tokens, rows));
            }
            Ok(Self { seqs })
        }
    }

    /// The log-prob each position gives the token after it (`next[i]`, for the first
    /// `next.len()` positions).
    fn next_lps(lp: &Array, next: &[u32]) -> Result<Vec<f32>> {
        if next.is_empty() {
            return Ok(Vec::new());
        }
        let rows = lp.index((0..next.len() as i32, ..));
        let ids = Array::from_slice(next, &[next.len() as i32, 1]).as_dtype(Dtype::Int32)?;
        let v = rows.take_along_axis(&ids, -1)?.reshape(&[-1])?;
        v.eval()?;
        Ok(v.as_slice::<f32>().to_vec())
    }

    /// Each sequence fed in chunks; `per_slice(lp, start)` sees the log-probs over the vocabulary
    /// (`[rows, V]` f32) of the positions from token `start` on, `SLICE` rows at a time.
    fn feed(
        model: &Qwen4ExpModel,
        tokens: &[u32],
        mut per_slice: impl FnMut(&Array, usize) -> Result<()>,
    ) -> Result<()> {
        let vocab = model.config().vocab_size as i32;
        let mut cache = model.make_cache();
        for (c, piece) in tokens.chunks(CHUNK).enumerate() {
            let logits = model
                .forward(piece, &mut cache, false)?
                .reshape(&[piece.len() as i32, vocab])?;
            for s in (0..piece.len()).step_by(SLICE) {
                let e = (s + SLICE).min(piece.len());
                let rows = logits
                    .index((s as i32..e as i32, ..))
                    .as_dtype(Dtype::Float32)?;
                let lse = rows.logsumexp_axis(-1, true)?;
                per_slice(&mlx_rs::ops::subtract(&rows, &lse)?, c * CHUNK + s)?;
            }
        }
        Ok(())
    }

    fn record(
        model: &Qwen4ExpModel,
        dir: &Path,
        out: &Path,
        files: &[PathBuf],
        max: usize,
    ) -> Result<()> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let mut seqs = Vec::new();
        for f in files {
            let text =
                std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
            let mut tokens = tok
                .encode(text.as_str(), false)
                .map_err(|e| anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec();
            tokens.truncate(max);
            if tokens.len() < 2 {
                continue;
            }
            let t0 = Instant::now();
            let mut rows: Vec<RefRow> = Vec::with_capacity(tokens.len() - 1);
            feed(model, &tokens, |lp, start| {
                let end = (start + lp.shape()[0] as usize).min(tokens.len());
                let next = &tokens[start + 1..(end + 1).min(tokens.len())];
                let nl = next_lps(lp, next)?;
                let order = mlx_rs::ops::argpartition_axis(
                    &mlx_rs::ops::negative(lp)?,
                    TOP as i32 - 1,
                    -1,
                )?;
                let idx = order.index((Ellipsis, 0..TOP as i32));
                let val = lp.take_along_axis(&idx, -1)?;
                let idx = idx.as_dtype(Dtype::Int32)?.reshape(&[-1])?;
                let val = val.reshape(&[-1])?;
                idx.eval()?;
                val.eval()?;
                let (idx, val) = (idx.as_slice::<i32>(), val.as_slice::<f32>());
                for (p, &next_lp) in nl.iter().enumerate() {
                    let mut ids = [0; TOP];
                    let mut lps = [0.0; TOP];
                    for k in 0..TOP {
                        ids[k] = idx[p * TOP + k] as u32;
                        lps[k] = val[p * TOP + k];
                    }
                    rows.push(RefRow { next_lp, ids, lps });
                }
                Ok(())
            })?;
            eprintln!(
                "{}: {} tokens in {:.1} s",
                f.display(),
                tokens.len(),
                t0.elapsed().as_secs_f64()
            );
            seqs.push((tokens, rows));
        }
        Record { seqs }.save(out)?;
        eprintln!("recorded to {}", out.display());
        Ok(())
    }

    #[derive(Default, Clone, Copy)]
    struct Tally {
        n: usize,
        agree: usize,
        kl: f64,
        nll_ref: f64,
        nll: f64,
    }

    impl Tally {
        fn line(&self, label: &str) -> String {
            let n = self.n.max(1) as f64;
            format!(
                "{label:>7}: {:>7} positions, top-1 same {:6.2}%, KL {:.4} nats, perplexity {:.3} (reference {:.3}, {:+.2}%)",
                self.n,
                100.0 * self.agree as f64 / n,
                self.kl / n,
                (self.nll / n).exp(),
                (self.nll_ref / n).exp(),
                100.0 * ((self.nll - self.nll_ref) / n).exp_m1(),
            )
        }
    }

    /// KL(reference || model) with the reference's top tokens as they are and the rest of the
    /// vocabulary as one bucket on each side.
    fn kl_top(r: &RefRow, model_lps: &[f32]) -> f64 {
        let (mut kl, mut p_top, mut q_top) = (0.0f64, 0.0f64, 0.0f64);
        for k in 0..TOP {
            let (lp, lq) = (r.lps[k] as f64, model_lps[k] as f64);
            let p = lp.exp();
            kl += p * (lp - lq);
            p_top += p;
            q_top += lq.exp();
        }
        let (p_rest, q_rest) = ((1.0 - p_top).max(0.0), (1.0 - q_top).max(1e-12));
        if p_rest > 1e-12 {
            kl += p_rest * (p_rest.ln() - q_rest.ln());
        }
        kl
    }

    fn compare(model: &Qwen4ExpModel, rec: &Path, seqs: usize) -> Result<()> {
        let rec = Record::load(rec)?;
        let mut total = Tally::default();
        let mut by_depth = [Tally::default(); DEPTHS.len()];
        for (s, (tokens, rows)) in rec.seqs.iter().enumerate().take(seqs) {
            let t0 = Instant::now();
            let mut seq = Tally::default();
            feed(model, tokens, |lp, start| {
                let end = (start + lp.shape()[0] as usize).min(tokens.len());
                let next = &tokens[start + 1..(end + 1).min(tokens.len())];
                let nl = next_lps(lp, next)?;
                let n = nl.len();
                if n == 0 {
                    return Ok(());
                }
                let ref_ids: Vec<u32> = rows[start..start + n].iter().flat_map(|r| r.ids).collect();
                let ids =
                    Array::from_slice(&ref_ids, &[n as i32, TOP as i32]).as_dtype(Dtype::Int32)?;
                let rows_lp = lp.index((0..n as i32, ..));
                let at = rows_lp.take_along_axis(&ids, -1)?.reshape(&[-1])?;
                let top = mlx_rs::ops::indexing::argmax_axis(&rows_lp, -1, false)?;
                at.eval()?;
                top.eval()?;
                let (at, top) = (at.as_slice::<f32>(), top.as_slice::<u32>());
                for p in 0..n {
                    let r = &rows[start + p];
                    let best = (0..TOP)
                        .max_by(|&a, &b| r.lps[a].total_cmp(&r.lps[b]))
                        .map(|k| r.ids[k])
                        .unwrap_or(0);
                    let kl = kl_top(r, &at[p * TOP..(p + 1) * TOP]);
                    let pos = start + p;
                    let d = DEPTHS.iter().position(|(lim, _)| pos < *lim).unwrap_or(0);
                    for t in [&mut seq, &mut by_depth[d]] {
                        t.n += 1;
                        t.agree += usize::from(top[p] == best);
                        t.kl += kl;
                        t.nll_ref -= r.next_lp as f64;
                        t.nll -= nl[p] as f64;
                    }
                }
                Ok(())
            })?;
            eprintln!(
                "{}",
                seq.line(&format!("seq {s}")) + &format!(" [{:.1} s]", t0.elapsed().as_secs_f64())
            );
            total.n += seq.n;
            total.agree += seq.agree;
            total.kl += seq.kl;
            total.nll_ref += seq.nll_ref;
            total.nll += seq.nll;
        }
        for (t, (_, label)) in by_depth.iter().zip(DEPTHS) {
            if t.n > 0 {
                println!("{}", t.line(label));
            }
        }
        println!("{}", total.line("all"));
        Ok(())
    }

    pub fn main() -> Result<()> {
        let mut args: Vec<String> = std::env::args().skip(1).collect();
        let mut take = |flag: &str| -> Option<String> {
            let i = args.iter().position(|a| a == flag)?;
            let v = args.get(i + 1).cloned();
            args.drain(i..(i + 2).min(args.len()));
            v
        };
        let budget_gb: f64 = take("--budget-gb")
            .and_then(|v| v.parse().ok())
            .unwrap_or(19.0);
        let max: usize = take("--max-tokens")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8192);
        let seqs: usize = take("--seqs")
            .and_then(|v| v.parse().ok())
            .unwrap_or(usize::MAX);
        let usage =
            "usage: record <model_dir> <out.bin> <text files...> | compare <model_dir> <ref.bin>";
        let (mode, dir, file) = match args.as_slice() {
            [m, d, f, ..] => (m.clone(), PathBuf::from(d), PathBuf::from(f)),
            _ => bail!(usage),
        };
        let t0 = Instant::now();
        let budget = (budget_gb > 0.0).then_some((budget_gb * 1e9) as u64);
        let model = Qwen4ExpModel::load(&dir, budget).context("load")?;
        eprintln!(
            "loaded {} in {:.1} s",
            dir.display(),
            t0.elapsed().as_secs_f64()
        );
        match mode.as_str() {
            "record" => {
                let files: Vec<PathBuf> = args[3..].iter().map(PathBuf::from).collect();
                record(&model, &dir, &file, &files, max)
            }
            "compare" => compare(&model, &file, seqs),
            _ => bail!(usage),
        }
    }
}

#[cfg(feature = "mlx-native")]
fn main() -> anyhow::Result<()> {
    imp::main()
}

#[cfg(not(feature = "mlx-native"))]
fn main() {
    eprintln!("qwen4_exp_quant_quality needs --features mlx-native");
}
