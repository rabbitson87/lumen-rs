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
//! first N sequences. `compare --dump <out.tsv>` writes one row per position, for finding where a
//! figure comes from: the text token, its KL, both negative log-probs, whether top-1 matched.
//!
//! Natural text is full of positions where the model is unsure, so the numbers above read much
//! lower than what a build changes along the model's own reply. Two more modes measure that, for
//! the question "how often would the fast copy pick another token, and could it have known?":
//!
//! ```text
//! # 1. The checkpoint's greedy reply to each prompt file, one token at a time.
//! ... greedy <model_dir> <out.bin> <prompt files...> [--new 300] [--budget-gb 12]
//! # 2. The same history through a low-bit copy, position by position, to a TSV.
//! ... audit <model_dir> <ref.bin> <out.tsv> --low 2,64,lsq [--low-budget-gb 12] [--budget-gb 4]
//!         [--prefill low] [--rows 2]
//! ```
//!
//! A prompt file is the rendered prompt, template included. `audit` feeds the recorded tokens, so
//! every position has the same history as the record, and the state the copy builds carries from
//! one position to the next, as it would in a real decode. The prompt is fed by the checkpoint's
//! experts unless `--prefill low`. Without `--low` the audit reads the checkpoint itself, which is
//! the control: 100%. `--rows 2` feeds two tokens per pass, which is what a speculative verify
//! does, and is the noise floor of any agreement figure. Each TSV row is one position: the record's
//! top two tokens and margin, this run's, and the token's text.
//!
//! `compare --low 2,64,lsq` runs the natural-text comparison through a copy.

#[cfg(feature = "mlx-native")]
mod imp {
    use std::io::{Read, Write};
    use std::path::{Path, PathBuf};
    use std::time::Instant;

    use anyhow::{Context, Result, anyhow, bail};
    use lumen_mlx::LowbitSpec;
    use lumen_mlx::qwen4_exp::model::MtpMode;
    use lumen_mlx::qwen4_exp::{
        ExpertTier, LoadOptions, LowExpertOptions, NgramSource, Qwen4ExpCache, Qwen4ExpModel,
    };
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

    fn compare(
        model: &Qwen4ExpModel,
        dir: &Path,
        rec: &Path,
        seqs: usize,
        dump: Option<&Path>,
    ) -> Result<()> {
        let rec = Record::load(rec)?;
        let mut dump = dump
            .map(|path| -> Result<_> {
                let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
                    .map_err(|e| anyhow!("tokenizer: {e}"))?;
                let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
                writeln!(w, "seq\tpos\tkl\tnll_ref\tnll\tsame\ttext")?;
                Ok((tok, w))
            })
            .transpose()?;
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
                    if let Some((tok, w)) = &mut dump {
                        writeln!(
                            w,
                            "{s}\t{pos}\t{kl:.4}\t{:.4}\t{:.4}\t{}\t{}",
                            -r.next_lp,
                            -nl[p],
                            u8::from(top[p] == best),
                            field(tok, tokens[pos + 1]),
                        )?;
                    }
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

    const GREEDY_MAGIC: &[u8; 4] = b"Q4QG";

    /// A position's two best tokens by logit.
    #[derive(Clone, Copy)]
    struct Top2 {
        ids: [u32; 2],
        logits: [f32; 2],
    }

    impl Top2 {
        /// From `[V]` logits of one position.
        fn of(row: &Array) -> Result<Self> {
            let order = mlx_rs::ops::argpartition_axis(&mlx_rs::ops::negative(row)?, 1, -1)?;
            let idx = order.index(0..2);
            let val = row.take_along_axis(&idx, -1)?;
            let idx = idx.as_dtype(Dtype::Int32)?;
            idx.eval()?;
            val.eval()?;
            let (i, v) = (idx.as_slice::<i32>(), val.as_slice::<f32>());
            let (a, b) = if v[0] >= v[1] { (0, 1) } else { (1, 0) };
            Ok(Self {
                ids: [i[a] as u32, i[b] as u32],
                logits: [v[a], v[b]],
            })
        }

        fn margin(&self) -> f32 {
            self.logits[0] - self.logits[1]
        }
    }

    /// One prompt and the checkpoint's reply: position `i` picked `steps[i].ids[0]`.
    struct GreedySeq {
        prompt: Vec<u32>,
        steps: Vec<Top2>,
    }

    fn save_greedy(seqs: &[GreedySeq], path: &Path) -> Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        w.write_all(GREEDY_MAGIC)?;
        put_u32(&mut w, seqs.len() as u32)?;
        for s in seqs {
            put_u32(&mut w, s.prompt.len() as u32)?;
            for &t in &s.prompt {
                put_u32(&mut w, t)?;
            }
            put_u32(&mut w, s.steps.len() as u32)?;
            for t in &s.steps {
                for k in 0..2 {
                    put_u32(&mut w, t.ids[k])?;
                    put_f32(&mut w, t.logits[k])?;
                }
            }
        }
        Ok(w.flush()?)
    }

    fn load_greedy(path: &Path) -> Result<Vec<GreedySeq>> {
        let mut r = std::io::BufReader::new(
            std::fs::File::open(path).with_context(|| format!("open {}", path.display()))?,
        );
        let mut magic = [0; 4];
        r.read_exact(&mut magic)?;
        if &magic != GREEDY_MAGIC {
            bail!("{} is not a greedy record", path.display());
        }
        let n = get_u32(&mut r)?;
        let mut seqs = Vec::new();
        for _ in 0..n {
            let len = get_u32(&mut r)?;
            let prompt = (0..len)
                .map(|_| get_u32(&mut r))
                .collect::<Result<Vec<_>>>()?;
            let len = get_u32(&mut r)?;
            let mut steps = Vec::new();
            for _ in 0..len {
                let (id0, lg0) = (get_u32(&mut r)?, get_f32(&mut r)?);
                let (id1, lg1) = (get_u32(&mut r)?, get_f32(&mut r)?);
                steps.push(Top2 {
                    ids: [id0, id1],
                    logits: [lg0, lg1],
                });
            }
            seqs.push(GreedySeq { prompt, steps });
        }
        Ok(seqs)
    }

    /// The whole prompt but its last token, in chunks: the record and the audit cut it alike, so
    /// their states start from the same bits.
    fn feed_prompt(model: &Qwen4ExpModel, prompt: &[u32], cache: &mut Qwen4ExpCache) -> Result<()> {
        for piece in prompt[..prompt.len() - 1].chunks(CHUNK) {
            model.forward(piece, cache, true)?;
        }
        Ok(())
    }

    /// One position's logits as a `[V]` array.
    fn last_row(logits: &Array) -> Result<Array> {
        Ok(logits.reshape(&[-1])?)
    }

    fn greedy(
        model: &Qwen4ExpModel,
        dir: &Path,
        out: &Path,
        files: &[PathBuf],
        new: usize,
    ) -> Result<()> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let eos: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
            .iter()
            .filter_map(|t| tok.token_to_id(t))
            .collect();
        let mut seqs = Vec::new();
        for f in files {
            let text =
                std::fs::read_to_string(f).with_context(|| format!("read {}", f.display()))?;
            let prompt = tok
                .encode(text.as_str(), false)
                .map_err(|e| anyhow!("encode: {e}"))?
                .get_ids()
                .to_vec();
            if prompt.len() < 2 {
                continue;
            }
            let t0 = Instant::now();
            let mut cache = model.make_cache();
            feed_prompt(model, &prompt, &mut cache)?;
            let t1 = Instant::now();
            let mut steps = Vec::with_capacity(new);
            let mut last = prompt[prompt.len() - 1];
            while steps.len() < new {
                let logits = model.forward(&[last], &mut cache, true)?;
                let top = Top2::of(&last_row(&logits)?)?;
                steps.push(top);
                last = top.ids[0];
                if eos.contains(&last) {
                    break;
                }
            }
            eprintln!(
                "{}: {} prompt tokens in {:.1} s, {} replied at {:.2} tok/s",
                f.display(),
                prompt.len(),
                (t1 - t0).as_secs_f64(),
                steps.len(),
                steps.len() as f64 / t1.elapsed().as_secs_f64(),
            );
            seqs.push(GreedySeq { prompt, steps });
        }
        save_greedy(&seqs, out)?;
        eprintln!("recorded to {}", out.display());
        Ok(())
    }

    /// What the audit reads the experts through.
    #[derive(Clone, Copy)]
    struct AuditPlan {
        decode: ExpertTier,
        prefill: ExpertTier,
        rows: usize,
    }

    /// A TSV field: one token's text on one line.
    fn field(tok: &Tokenizer, id: u32) -> String {
        tok.decode(&[id], false)
            .unwrap_or_default()
            .replace('\\', "\\\\")
            .replace('\t', "\\t")
            .replace('\n', "\\n")
            .replace('\r', "\\r")
    }

    fn audit(
        model: &Qwen4ExpModel,
        dir: &Path,
        rec: &Path,
        out: &Path,
        plan: AuditPlan,
    ) -> Result<()> {
        let tok = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow!("tokenizer: {e}"))?;
        let seqs = load_greedy(rec)?;
        let mut w = std::io::BufWriter::new(std::fs::File::create(out)?);
        writeln!(
            w,
            "seq\tpos\tprompt\tref_top1\tref_top2\tref_margin\ttop1\ttop2\tmargin\tref_text\ttext"
        )?;
        let (mut agree, mut total) = (0usize, 0usize);
        for (s, seq) in seqs.iter().enumerate() {
            let t0 = Instant::now();
            let mut cache = model.make_cache();
            model.set_expert_tier(plan.prefill)?;
            feed_prompt(model, &seq.prompt, &mut cache)?;
            model.set_expert_tier(plan.decode)?;
            let t1 = Instant::now();
            // The token before position `i`: the prompt's last, then the record's own picks.
            let feed: Vec<u32> = std::iter::once(seq.prompt[seq.prompt.len() - 1])
                .chain(seq.steps.iter().map(|t| t.ids[0]))
                .take(seq.steps.len())
                .collect();
            let mut mine = Vec::with_capacity(seq.steps.len());
            for piece in feed.chunks(plan.rows) {
                let logits = model.forward(piece, &mut cache, false)?;
                let vocab = model.config().vocab_size as i32;
                let rows = logits.reshape(&[piece.len() as i32, vocab])?;
                for r in 0..piece.len() as i32 {
                    mine.push(Top2::of(&rows.index((r, ..)))?);
                }
            }
            let decoded = t1.elapsed().as_secs_f64();
            model.set_expert_tier(ExpertTier::Full)?;
            let mut same = 0;
            for (pos, (r, m)) in seq.steps.iter().zip(&mine).enumerate() {
                same += usize::from(r.ids[0] == m.ids[0]);
                writeln!(
                    w,
                    "{s}\t{pos}\t{}\t{}\t{}\t{:.4}\t{}\t{}\t{:.4}\t{}\t{}",
                    seq.prompt.len(),
                    r.ids[0],
                    r.ids[1],
                    r.margin(),
                    m.ids[0],
                    m.ids[1],
                    m.margin(),
                    field(&tok, r.ids[0]),
                    field(&tok, m.ids[0]),
                )?;
            }
            agree += same;
            total += seq.steps.len();
            eprintln!(
                "seq {s}: {} prompt tokens in {:.1} s; {same} of {} same; {:.2} tok/s",
                seq.prompt.len(),
                (t1 - t0).as_secs_f64(),
                seq.steps.len(),
                seq.steps.len() as f64 / decoded,
            );
        }
        w.flush()?;
        println!(
            "same token at {agree} of {total} positions = {:.2}%",
            100.0 * agree as f64 / total.max(1) as f64
        );
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
        let new: usize = take("--new").and_then(|v| v.parse().ok()).unwrap_or(300);
        let rows: usize = take("--rows").and_then(|v| v.parse().ok()).unwrap_or(1);
        let dump = take("--dump").map(PathBuf::from);
        let low_prefill = take("--prefill").as_deref() == Some("low");
        let low_budget_gb: f64 = take("--low-budget-gb")
            .and_then(|v| v.parse().ok())
            .unwrap_or(12.0);
        let low = take("--low").map(|v| LowbitSpec::parse(&v)).transpose()?;
        let usage = "usage: record <model_dir> <out.bin> <text files...> | \
             compare <model_dir> <ref.bin> | greedy <model_dir> <out.bin> <prompt files...> | \
             audit <model_dir> <ref.bin> <out.tsv>";
        let (mode, dir, file) = match args.as_slice() {
            [m, d, f, ..] => (m.clone(), PathBuf::from(d), PathBuf::from(f)),
            _ => bail!(usage),
        };
        let t0 = Instant::now();
        let budget = (budget_gb > 0.0).then_some((budget_gb * 1e9) as u64);
        let model = Qwen4ExpModel::load_with(
            &dir,
            LoadOptions {
                expert_budget_bytes: budget,
                ngram: NgramSource::Auto,
                mtp: MtpMode::Off,
                scores_budget_bytes: 0,
                low_experts: low.map(|spec| LowExpertOptions {
                    spec,
                    budget_bytes: (low_budget_gb * 1e9) as u64,
                }),
                expert_copy: None,
            },
        )
        .context("load")?;
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
            "compare" => {
                if low.is_some() {
                    model.set_expert_tier(ExpertTier::Low)?;
                }
                compare(&model, &dir, &file, seqs, dump.as_deref())
            }
            "greedy" => {
                let files: Vec<PathBuf> = args[3..].iter().map(PathBuf::from).collect();
                greedy(&model, &dir, &file, &files, new)
            }
            "audit" => {
                let out = PathBuf::from(args.get(3).ok_or_else(|| anyhow!(usage))?);
                let tier = |on| {
                    if on {
                        ExpertTier::Low
                    } else {
                        ExpertTier::Full
                    }
                };
                let plan = AuditPlan {
                    decode: tier(low.is_some()),
                    prefill: tier(low.is_some() && low_prefill),
                    rows: rows.max(1),
                };
                audit(&model, &dir, &file, &out, plan)?;
                if let Some(report) = model.low_offload_report() {
                    eprintln!("[low] {report}");
                }
                Ok(())
            }
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
