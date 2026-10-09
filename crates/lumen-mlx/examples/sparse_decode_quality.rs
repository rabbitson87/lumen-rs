//! Does page-sparse decode (`LUMEN_SPARSE_DECODE`) change what the model
//! answers at long context? Task 021, Stage 3 quality gate.
//!
//! Sparse decode only touches single-token decode steps, so the usual
//! teacher-forced harness (`kv_bf16_quality`, built on the prefill-shaped
//! `forward_probe`) would never exercise it. Everything here goes through
//! `decode_step`. Each context is prefilled once and snapshotted; the dense
//! and sparse conditions then decode from the same restored state, in one
//! process, with the flag pinned by `set_sparse_decode`.
//!
//! The haystack is this crate's own source, in file order, cut to the target
//! length: realistic code an agent would hold in context.
//!
//! * `teacher` — decode `--steps` tokens greedily under dense attention, then
//!   feed that same sequence through sparse decode one token at a time and
//!   count the positions whose next-token prediction matches.
//! * `niah` — a passphrase planted at 10/50/90% depth; pass when the answer
//!   contains it.
//! * `retrieval` — questions about `const` values and `fn` return types in
//!   the haystack ("what value is `NAME` set to?", "what does `f` return?"),
//!   exact match against the source.
//!
//! ```text
//! MODEL_ID=~/models/mlx-community--Qwen3.6-35B-A3B-mxfp4 \
//!   cargo run --release -p lumen-mlx --features mlx-native \
//!   --example sparse_decode_quality -- --lengths 16384,32768,65536
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
    use lumen_mlx::metal_memory::clear_cache;
    use lumen_mlx::{MlxBackend, MlxQwen35Backend, set_sparse_decode};
    use std::collections::BTreeMap;
    use std::path::Path;

    const NEEDLE: &str = "periwinkle-anchor-4721";

    struct Args {
        lengths: Vec<usize>,
        steps: usize,
        questions: usize,
        modes: Vec<String>,
    }

    fn parse_args() -> Result<Args> {
        let mut args = Args {
            lengths: vec![16384, 32768, 65536],
            steps: 256,
            questions: 60,
            modes: vec!["teacher".into(), "niah".into(), "retrieval".into()],
        };
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let mut i = 0;
        while i < argv.len() {
            let value = argv
                .get(i + 1)
                .ok_or_else(|| anyhow!("{} needs a value", argv[i]))?;
            match argv[i].as_str() {
                "--lengths" => {
                    args.lengths = value
                        .split(',')
                        .map(|s| s.trim().parse())
                        .collect::<Result<_, _>>()
                        .context("--lengths")?
                }
                "--steps" => args.steps = value.parse().context("--steps")?,
                "--questions" => args.questions = value.parse().context("--questions")?,
                "--modes" => args.modes = value.split(',').map(str::to_string).collect(),
                other => return Err(anyhow!("unknown argument {other:?}")),
            }
            i += 2;
        }
        Ok(args)
    }

    /// This crate's source, concatenated in file-name order.
    fn source_corpus() -> Result<String> {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut files: Vec<_> = std::fs::read_dir(&dir)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "rs"))
            .collect();
        files.sort();
        let mut text = String::new();
        for f in files {
            text.push_str(&format!(
                "\n// ===== file: {} =====\n",
                f.file_name().unwrap().to_string_lossy()
            ));
            text.push_str(&std::fs::read_to_string(&f)?);
        }
        Ok(text)
    }

    /// The first `tokens` tokens of `corpus`, as text.
    fn haystack(backend: &MlxQwen35Backend, corpus: &str, tokens: usize) -> Result<String> {
        let ids = backend.encode(corpus)?;
        if ids.len() < tokens {
            return Err(anyhow!(
                "corpus is only {} tokens, {tokens} asked for",
                ids.len()
            ));
        }
        backend.decode(&ids[..tokens])
    }

    /// Questions with a single exact answer in `text`, as (question, answer):
    /// the value of each `const` item and the return type of each `fn` whose
    /// name is unique there. Answers are short literals, paths or types.
    fn questions(text: &str, limit: usize) -> Vec<(String, String)> {
        // name -> answer, or None once the name is seen with two answers.
        let mut consts: BTreeMap<String, Option<String>> = BTreeMap::new();
        let mut fns: BTreeMap<String, Option<String>> = BTreeMap::new();
        let note = |map: &mut BTreeMap<String, Option<String>>, name: &str, answer: &str| {
            map.entry(name.to_string())
                .and_modify(|v| {
                    if v.as_deref() != Some(answer) {
                        *v = None;
                    }
                })
                .or_insert_with(|| Some(answer.to_string()));
        };
        let short = |s: &str| !s.is_empty() && s.len() <= 40 && !s.contains(['{', '\'']);
        for line in text.lines() {
            let line = line.trim();
            let line = ["pub(crate) ", "pub(super) ", "pub "]
                .iter()
                .find_map(|p| line.strip_prefix(p))
                .unwrap_or(line);
            if let Some(rest) = line.strip_prefix("const ")
                && let Some((name, rest)) = rest.split_once(':')
                && let Some((_, value)) = rest.split_once('=')
                && let Some(value) = value.trim().strip_suffix(';')
            {
                let (name, value) = (name.trim(), value.trim());
                if short(value)
                    && name
                        .chars()
                        .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
                {
                    note(&mut consts, name, value);
                }
            } else if let Some(rest) = line.strip_prefix("fn ")
                && let Some((name, rest)) = rest.split_once('(')
                && let Some((_, ret)) = rest.rsplit_once(") -> ")
                && let Some(ret) = ret.strip_suffix(" {")
            {
                let name = name.split('<').next().unwrap_or(name).trim();
                if short(ret) && !name.is_empty() {
                    note(&mut fns, name, ret.trim());
                }
            }
        }
        let consts = consts.into_iter().filter_map(|(n, v)| {
            v.map(|v| {
                (
                    format!("what value is the constant `{n}` set to? Answer with the value only."),
                    v,
                )
            })
        });
        let fns = fns.into_iter().filter_map(|(n, v)| {
            v.map(|v| {
                (
                    format!(
                        "what is the return type of the function `{n}`? Answer with the type only."
                    ),
                    v,
                )
            })
        });
        consts.chain(fns).take(limit).collect()
    }

    /// Chat tokens for `content` as one user turn, thinking off.
    fn chat(backend: &mut MlxQwen35Backend, content: &str) -> Result<Vec<u32>> {
        backend.build_chat_input(&[("user".to_string(), content.to_string())], false, None)
    }

    /// Greedy-decode up to `max` tokens after `first` (the token the prefill
    /// or extend returned), stopping early at a newline once something has
    /// been said.
    fn answer(
        backend: &mut MlxQwen35Backend,
        seq: u64,
        first: u32,
        pos: usize,
        max: usize,
    ) -> Result<String> {
        let (mut last, mut pos) = (first, pos);
        let mut out = vec![first];
        for _ in 1..max {
            let (tok, p) = backend.decode_step(seq, last, pos)?;
            out.push(tok);
            last = tok;
            pos = p;
            let text = backend.decode(&out)?;
            if text.trim().contains('\n') {
                break;
            }
        }
        backend.decode(&out)
    }

    /// The longest prefix every sequence in `seqs` shares.
    fn shared_prefix(seqs: &[Vec<u32>]) -> usize {
        let first = &seqs[0];
        (0..first.len())
            .take_while(|&i| seqs.iter().all(|s| s.get(i) == Some(&first[i])))
            .count()
    }

    fn teacher(backend: &mut MlxQwen35Backend, hay: &str, steps: usize) -> Result<(usize, usize)> {
        let prompt = chat(
            backend,
            &format!("{hay}\n\nSummarize what the code above does, file by file."),
        )?;
        let seq = 1;
        set_sparse_decode(false);
        let (first, pos0) = backend.prefill(seq, &prompt)?;
        let snap = backend.snapshot_state(seq)?;
        // Dense reference.
        let mut reference = vec![first];
        let (mut last, mut pos) = (first, pos0);
        for _ in 0..steps {
            let (tok, p) = backend.decode_step(seq, last, pos)?;
            reference.push(tok);
            last = tok;
            pos = p;
        }
        // Sparse, teacher-forced on the reference.
        backend.restore_state(seq, snap)?;
        set_sparse_decode(true);
        let mut pos = pos0;
        let mut agree = 0;
        for i in 0..steps {
            let (pred, p) = backend.decode_step(seq, reference[i], pos)?;
            agree += usize::from(pred == reference[i + 1]);
            pos = p;
        }
        set_sparse_decode(false);
        backend.release_snapshot(snap)?;
        backend.remove_seq(seq)?;
        let _ = clear_cache();
        Ok((agree, steps))
    }

    fn niah(backend: &mut MlxQwen35Backend, hay: &str, depth: f64) -> Result<(bool, bool)> {
        let mut cut = (hay.len() as f64 * depth) as usize;
        while !hay.is_char_boundary(cut) {
            cut += 1;
        }
        let planted = format!(
            "{}\n// The secret passphrase for the vault is {NEEDLE}.\n{}",
            &hay[..cut],
            &hay[cut..]
        );
        let prompt = chat(
            backend,
            &format!(
                "{planted}\n\nWhat is the secret passphrase for the vault? \
                 Answer with the passphrase only."
            ),
        )?;
        let seq = 2;
        set_sparse_decode(false);
        let (first, pos) = backend.prefill(seq, &prompt)?;
        let snap = backend.snapshot_state(seq)?;
        let dense = answer(backend, seq, first, pos, 24)?;
        backend.restore_state(seq, snap)?;
        set_sparse_decode(true);
        let sparse = answer(backend, seq, first, pos, 24)?;
        set_sparse_decode(false);
        backend.release_snapshot(snap)?;
        backend.remove_seq(seq)?;
        let _ = clear_cache();
        Ok((dense.contains(NEEDLE), sparse.contains(NEEDLE)))
    }

    fn retrieval(
        backend: &mut MlxQwen35Backend,
        hay: &str,
        questions_wanted: usize,
    ) -> Result<(usize, usize, usize)> {
        let items = questions(hay, questions_wanted);
        if items.is_empty() {
            return Err(anyhow!("no questions in the haystack"));
        }
        let prompts: Vec<Vec<u32>> = items
            .iter()
            .map(|(question, _)| chat(backend, &format!("{hay}\n\nIn the code above, {question}")))
            .collect::<Result<_>>()?;
        // Prefill the part every question shares once, less a few tokens so
        // each question's extend re-tokenizes its own boundary.
        let shared = shared_prefix(&prompts).saturating_sub(4);
        let seq = 3;
        set_sparse_decode(false);
        backend.prefill(seq, &prompts[0][..shared])?;
        // `restore_state` consumes its snapshot; a deep master can be forked
        // into a fresh sequence any number of times.
        let (snap, _) = backend.snapshot_state_deep(seq)?;
        backend.remove_seq(seq)?;
        let (mut dense_hits, mut sparse_hits) = (0, 0);
        let matches = |reply: &str, value: &str| {
            let reply = reply.trim().trim_matches('`').trim();
            reply == value || reply.starts_with(value)
        };
        let fork = 4;
        for ((_, value), prompt) in items.iter().zip(&prompts) {
            for sparse in [false, true] {
                backend.fork_from_snapshot(snap, fork)?;
                set_sparse_decode(sparse);
                let (first, pos) = backend.runner_extend(fork, &prompt[shared..])?;
                let reply = answer(backend, fork, first, pos, 16)?;
                backend.remove_seq(fork)?;
                if matches(&reply, value) {
                    if sparse {
                        sparse_hits += 1;
                    } else {
                        dense_hits += 1;
                    }
                }
            }
        }
        set_sparse_decode(false);
        backend.release_snapshot(snap)?;
        let _ = clear_cache();
        Ok((dense_hits, sparse_hits, items.len()))
    }

    pub fn run() -> Result<()> {
        let args = parse_args()?;
        let model_id = std::env::var("MODEL_ID")
            .map_err(|_| anyhow!("set MODEL_ID to a local Qwen 3.5/3.6 model directory"))?;
        let mut backend = MlxBackend::load(&model_id)?;
        let backend = backend
            .as_qwen35_mut()
            .ok_or_else(|| anyhow!("this harness drives the Qwen 3.5/3.6 backend"))?;
        let corpus = source_corpus()?;
        println!(
            "sparse_decode_quality: model={model_id} lengths={:?}",
            args.lengths
        );
        for &len in &args.lengths {
            let hay = haystack(backend, &corpus, len)?;
            let want = |m: &str| args.modes.iter().any(|x| x == m);
            if want("teacher") {
                let (agree, n) = teacher(backend, &hay, args.steps)?;
                println!(
                    "[teacher  ctx={len}] sparse matches dense at {agree}/{n} positions ({:.2}%)",
                    100.0 * agree as f64 / n as f64
                );
            }
            if want("niah") {
                for depth in [0.1, 0.5, 0.9] {
                    let (dense, sparse) = niah(backend, &hay, depth)?;
                    println!(
                        "[niah     ctx={len} depth={:.0}%] dense={} sparse={}",
                        depth * 100.0,
                        if dense { "found" } else { "MISSED" },
                        if sparse { "found" } else { "MISSED" },
                    );
                }
            }
            if want("retrieval") {
                let (dense, sparse, n) = retrieval(backend, &hay, args.questions)?;
                println!("[retrieve ctx={len}] exact match dense={dense}/{n} sparse={sparse}/{n}");
            }
        }
        Ok(())
    }
}
