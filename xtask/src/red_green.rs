//! Proves that every regression guard on this branch would have caught its
//! defect.
//!
//! A guard written *after* a fix is worth nothing until you show it fails
//! without the fix — and that is the easy mistake to make when the fix already
//! works. For each defect this reverts the fix in place, requires the named
//! guards to go RED, restores the source, and requires them GREEN again. A
//! guard that passes in both states is reported as VACUOUS and the run fails.
//!
//! Each entry also records the symptom the defect produced in production, so
//! `--list` doubles as the evidence for "this was already broken".

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const MLX: &str = "crates/lumen-mlx/src";
const SRV: &str = "crates/lumen-server/src";

/// A single in-place edit. Both sides must be non-empty: the reverse direction
/// searches for `replace`, and searching for an empty string matches
/// everywhere. Express a deletion as a sentinel comment instead.
struct Mutation {
    path: &'static str,
    find: &'static str,
    replace: &'static str,
}

struct Guard {
    package: &'static str,
    /// Full test path, matched with `--exact`.
    filter: &'static str,
}

struct Defect {
    name: &'static str,
    symptom: &'static str,
    revert: &'static [Mutation],
    guards: &'static [Guard],
    /// Sites the fix touches. A fix applied at two render paths is not a moved
    /// anchor; anything other than this count is.
    occurrences: usize,
    needs_checkpoint: bool,
    /// Extra `libtest` arguments, e.g. `--ignored`.
    extra: &'static [&'static str],
}

const fn mlx(filter: &'static str) -> Guard {
    Guard {
        package: "lumen-mlx",
        filter,
    }
}
const fn srv(filter: &'static str) -> Guard {
    Guard {
        package: "lumen-server",
        filter,
    }
}

static DEFECTS: &[Defect] = &[
    Defect {
        name: "lark-opener",
        symptom: "every streaming tool call died: `byte 'ÿ' fails parse`; and \
                  tool_choice=required silently fell back to free sampling",
        revert: &[Mutation {
            path: MLX,
            find: r#"        self.lazy_trigger
            .is_some_and(|t| t.token == token && !t.in_grammar)"#,
            replace: r#"        let _ = token;
        false"#,
        }],
        guards: &[
            mlx("grammar::tests::lazy_activation_does_not_feed_the_trigger_to_the_matcher"),
            mlx("grammar::tests::lazy_activation_leaves_the_matcher_at_the_start_of_the_body"),
            mlx("grammar::tests::eager_prefill_replay_skips_the_opener_and_parses_the_rest"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "json-whitespace",
        symptom: "response_format replies were pure indentation up to max_tokens",
        revert: &[Mutation {
            path: MLX,
            find: r#""whitespace_flexible": false,"#,
            replace: r#""whitespace_flexible": true,"#,
        }],
        guards: &[mlx(
            "grammar::tests::response_format_grammar_forbids_whitespace_runs",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "json-separator-space",
        symptom: "the `\": \"` residue leaked into string values: \
                  {\"city\": \": way more than you've gotten…\"}",
        revert: &[Mutation {
            path: MLX,
            find: r#""key_separator": ": ",
                "item_separator": ", ","#,
            replace: r#""key_separator": ":",
                "item_separator": ",","#,
        }],
        guards: &[mlx(
            "grammar::tests::response_format_grammar_keeps_the_separator_space",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "grammar-rule-names",
        symptom: "a tool named `날씨_조회` was refused, the grammar dropped, and the \
                  model invented `weather_lookup` — a tool nobody declared",
        revert: &[Mutation {
            path: MLX,
            find: r#"let body_rule_name = format!("tool_{i}_body");"#,
            replace: r#"let body_rule_name = format!("tool_{name}_body");"#,
        }],
        guards: &[mlx(
            "grammar::tests::lark_grammar_escapes_a_non_identifier_tool_name",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "grammar-literal-escaping",
        symptom: "a quote inside a tool name closed the Lark literal",
        revert: &[Mutation {
            path: MLX,
            find: r#"            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),"#,
            replace: r#"            '\\' => out.push('\\'),
            '"' => out.push('"'),"#,
        }],
        guards: &[mlx(
            "grammar::tests::lark_grammar_escapes_a_quote_in_a_tool_name",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "tool-name-scanner",
        symptom: "`call:bad call:good{x:1}` parsed as ONE tool named \
                  \"bad call:good\" — a name no client declared",
        revert: &[Mutation {
            path: MLX,
            find: r#"                if bytes[brace_start..].starts_with(b"call:") {
                    hit_next_call = true;
                    break;
                }
"#,
            replace: "                // xtask red-green: next-call boundary removed\n",
        }],
        guards: &[
            mlx("gemma4_response::imp::tests::body_parser_stops_a_name_at_the_next_opener"),
            mlx("gemma4_response::imp::tests::body_parser_skips_a_run_of_malformed_openers"),
            mlx(
                "gemma4_response::imp::tests::body_parser_stops_a_non_ascii_name_at_the_next_opener",
            ),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "args-unicode-keys",
        symptom: "`{도시:…}` arrived as `{Ã«Â\\u{8f}Â\\u{84}…}` and failed to parse",
        revert: &[Mutation {
            path: MLX,
            find: r#".find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))"#,
            replace: r#".find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))"#,
        }],
        guards: &[
            mlx("gemma4_response::imp::tests::args_to_json_quotes_a_non_ascii_bare_key"),
            mlx(
                "gemma4_response::imp::tests::args_to_json_handles_non_ascii_in_nested_and_array_positions",
            ),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "gemma-nonstreaming-grammar",
        symptom: "non-streaming tool_choice=required returned `迎get_weather`",
        revert: &[Mutation {
            path: MLX,
            find: r#"    !tools.is_empty()
        && !matches!(tool_choice, crate::chat_io::ResolvedToolChoice::None)
        && crate::gemma4_backend::imp::gemma4_grammar_lark_enabled()"#,
            replace: r#"    let _ = (tools, tool_choice);
    false"#,
        }],
        guards: &[mlx(
            "grammar_routing_regressions::tools_alone_must_route_through_the_grammar_aware_decode",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "qwen-first-token-mask",
        symptom: "tool_choice=required never enforced: the first generated token \
                  was argmaxed unmasked, and disagreement dropped the grammar",
        revert: &[Mutation {
            path: MLX,
            find: "    grammar_active && prompt_ids.len() > 1 && image_token != prompt_ids.last().copied()",
            replace: "    let _ = (grammar_active, prompt_ids, image_token);\n    false",
        }],
        guards: &[mlx(
            "grammar_routing_regressions::an_active_grammar_holds_the_last_prompt_token_back",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "tool-choice-none",
        symptom: "tool_choice=\"none\" was accepted and ignored; Qwen 3.6 called the \
                  tool anyway",
        revert: &[Mutation {
            path: SRV,
            find: r#"    if matches!(tool_choice, ResolvedToolChoice::None) {
        Vec::new()
    } else {
        tools
    }"#,
            replace: r#"    let _ = tool_choice;
    tools"#,
        }],
        guards: &[srv(
            "engine::tool_choice_none_withholds_tools::none_hides_them",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "anthropic-turn-images",
        symptom: "on /v1/messages a tool_result expanded one message into several \
                  turns, so every later image bound to the wrong turn",
        revert: &[Mutation {
            path: SRV,
            find: r#"                for _ in 0..tool_result_counts.get(i).copied().unwrap_or(0) {
                    out.push(Vec::new());
                }
"#,
            replace: "                // xtask red-green: tool-turn rows removed\n",
        }],
        guards: &[
            srv(
                "engine::anthropic_turn_image_alignment::tool_results_expand_one_message_into_several_turns",
            ),
            srv(
                "engine::anthropic_turn_image_alignment::a_textless_imageless_message_emits_no_turn",
            ),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "gemma-thought-channel",
        symptom: "response_format degenerated into repetition to max_tokens — the \
                  eager grammar masked `<|channel>` at step 0, leaving 3 legal tokens",
        revert: &[Mutation {
            path: MLX,
            find: "if !opts.enable_thinking\n                    && (opts.close_thought_channel || empty_thought_on_nothink())",
            replace: "if !opts.enable_thinking && empty_thought_on_nothink()",
        }],
        guards: &[mlx(
            "gemma4_chat::imp::tests::close_thought_channel_prefills_the_empty_block",
        )],
        occurrences: 2, // the flat renderer and the history renderer
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
];

/// The file each mutation edits. `Mutation::path` names the source *directory*
/// so the table reads compactly; the file is derived from the guard it backs.
fn file_for(defect: &Defect, m: &Mutation) -> PathBuf {
    let leaf = match (m.path, defect.name) {
        (_, "lark-opener")
        | (_, "json-whitespace")
        | (_, "json-separator-space")
        | (_, "grammar-rule-names")
        | (_, "grammar-literal-escaping") => "grammar.rs",
        (_, "tool-name-scanner") | (_, "args-unicode-keys") => "gemma4_response.rs",
        (_, "gemma-nonstreaming-grammar") | (_, "qwen-first-token-mask") => "lib.rs",
        (_, "gemma-thought-channel") => "gemma4_chat.rs",
        (_, "tool-choice-none") | (_, "anthropic-turn-images") => "engine.rs",
        _ => unreachable!("no file mapped for {}", defect.name),
    };
    root().join(m.path).join(leaf)
}

fn root() -> PathBuf {
    // CARGO_MANIFEST_DIR is `<workspace>/xtask`.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level below the workspace root")
        .to_path_buf()
}

/// Where pristine copies live while a mutation is applied.
///
/// This tool edits tracked source files, so an interrupted run must not be able
/// to leave one edited. [`Restore`]'s `Drop` covers normal returns and panics —
/// but not SIGTERM or SIGKILL, which is exactly what a CI timeout or an
/// impatient Ctrl-C sends. The journal survives those: the next run finds it and
/// puts the sources back before doing anything else.
fn journal_dir() -> PathBuf {
    root().join("target/xtask-red-green-journal")
}

/// Journal entries are named after the file they back up, with separators
/// escaped, so the directory listing is its own index — no format to parse and
/// nothing that can be half-written.
fn journal_entry(path: &Path) -> PathBuf {
    let rel = path.strip_prefix(root()).unwrap_or(path);
    journal_dir().join(rel.to_string_lossy().replace(['/', '\\'], "%"))
}

/// Put back anything a killed run left mutated. Returns how many files it
/// repaired.
fn repair_interrupted_run() -> std::io::Result<usize> {
    let dir = journal_dir();
    if !dir.exists() {
        return Ok(0);
    }
    let mut repaired = 0;
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let rel = entry.file_name().to_string_lossy().replace('%', "/");
        let target = root().join(&rel);
        let pristine = std::fs::read_to_string(entry.path())?;
        if std::fs::read_to_string(&target).ok().as_deref() != Some(pristine.as_str()) {
            std::fs::write(&target, &pristine)?;
            eprintln!("restored {rel} from an interrupted run");
            repaired += 1;
        }
        std::fs::remove_file(entry.path())?;
    }
    let _ = std::fs::remove_dir(&dir);
    Ok(repaired)
}

/// Restores every touched file when it goes out of scope, including on panic.
struct Restore {
    saved: Vec<(PathBuf, String)>,
}

impl Restore {
    fn new() -> Self {
        Self { saved: Vec::new() }
    }
    fn keep(&mut self, path: &Path) -> std::io::Result<String> {
        let text = std::fs::read_to_string(path)?;
        // Journal first, mutate second — the reverse order would leave a window
        // where a kill loses the original.
        std::fs::create_dir_all(journal_dir())?;
        std::fs::write(journal_entry(path), &text)?;
        self.saved.push((path.to_path_buf(), text.clone()));
        Ok(text)
    }
}

impl Drop for Restore {
    fn drop(&mut self) {
        for (path, text) in self.saved.iter().rev() {
            if let Err(e) = std::fs::write(path, text) {
                eprintln!("!! could not restore {}: {e}", path.display());
                eprintln!("!! run `git checkout -- {}`", path.display());
                continue;
            }
            let _ = std::fs::remove_file(journal_entry(path));
        }
        let _ = std::fs::remove_dir(journal_dir());
    }
}

#[derive(PartialEq)]
enum Verdict {
    Pass,
    Vacuous,
    AlreadyRed,
    Skip,
}

fn guards_pass(defect: &Defect) -> Result<bool, String> {
    for g in defect.guards {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(root())
            .args(["test", "-p", g.package, "--features", "mlx-native"]);
        if g.package == "lumen-mlx" {
            // lumen-server is a binary crate; it has no lib target.
            cmd.arg("--lib");
        }
        cmd.args([g.filter, "--", "--test-threads=1", "--exact"])
            .args(defect.extra);
        let out = cmd.output().map_err(|e| format!("spawn cargo: {e}"))?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        // A filter that matched nothing is a broken entry in this table, not a
        // pass — that mistake would silently turn the whole run green.
        if !stdout.contains("running 1 test") {
            return Err(format!("guard matched no test: {}", g.filter));
        }
        if !out.status.success() {
            return Ok(false);
        }
    }
    Ok(true)
}

fn apply(defect: &Defect, restore: &mut Restore) -> Result<(), String> {
    for m in defect.revert {
        if m.find.trim().is_empty() || m.replace.trim().is_empty() {
            return Err(format!(
                "{}: both sides of a mutation must be non-empty, or the reverse \
                 direction searches for the empty string. Use a sentinel comment \
                 instead of deleting.",
                defect.name
            ));
        }
        let path = file_for(defect, m);
        let text = restore
            .keep(&path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let n = text.matches(m.find).count();
        if n != defect.occurrences {
            return Err(format!(
                "{}: expected {} occurrence(s) in {}, found {n} — the source moved, \
                 update xtask/src/red_green.rs",
                defect.name,
                defect.occurrences,
                path.display()
            ));
        }
        std::fs::write(&path, text.replace(m.find, m.replace))
            .map_err(|e| format!("write {}: {e}", path.display()))?;
    }
    Ok(())
}

fn check(defect: &Defect) -> Result<Verdict, String> {
    if defect.needs_checkpoint && std::env::var_os("LUMEN_GEMMA4_MODEL_DIR").is_none() {
        return Ok(Verdict::Skip);
    }
    if !guards_pass(defect)? {
        return Ok(Verdict::AlreadyRed);
    }
    let went_red = {
        let mut restore = Restore::new();
        apply(defect, &mut restore)?;
        let red = !guards_pass(defect)?;
        // `restore` drops here, putting the source back even if the call above
        // returned early with an error.
        red
    };
    if !guards_pass(defect)? {
        return Err(format!(
            "{}: source not restored cleanly — run `git status` and revert by hand",
            defect.name
        ));
    }
    Ok(if went_red {
        Verdict::Pass
    } else {
        Verdict::Vacuous
    })
}

pub fn main(args: Vec<String>) -> ExitCode {
    // Before anything else, including --list: a tree left dirty by a killed run
    // must not survive a subsequent invocation, whatever that invocation was.
    match repair_interrupted_run() {
        Ok(0) => {}
        Ok(n) => eprintln!("repaired {n} file(s) left behind by an interrupted run\n"),
        Err(e) => {
            eprintln!("could not repair a previous run: {e}");
            eprintln!("check `git status` before trusting this run");
            return ExitCode::FAILURE;
        }
    }

    if args.iter().any(|a| a == "--list") {
        for d in DEFECTS {
            println!("{:<28} {}", d.name, d.symptom);
        }
        return ExitCode::SUCCESS;
    }

    let wanted: BTreeSet<&str> = args.iter().map(String::as_str).collect();
    for name in &wanted {
        if !DEFECTS.iter().any(|d| d.name == *name) {
            eprintln!("no such defect {name:?}; try `cargo xtask red-green --list`");
            return ExitCode::from(2);
        }
    }
    let todo: Vec<&Defect> = DEFECTS
        .iter()
        .filter(|d| wanted.is_empty() || wanted.contains(d.name))
        .collect();

    let mut results = Vec::new();
    for d in &todo {
        println!("→ {}", d.name);
        match check(d) {
            Ok(v) => {
                println!(
                    "  {}",
                    match v {
                        Verdict::Pass => "PASS",
                        Verdict::Vacuous => "NO-OP",
                        Verdict::AlreadyRed => "BROKEN",
                        Verdict::Skip => "SKIP",
                    }
                );
                results.push((v, *d));
            }
            Err(e) => {
                eprintln!("  ERROR {e}");
                return ExitCode::FAILURE;
            }
        }
    }

    let width = todo.iter().map(|d| d.name.len()).max().unwrap_or(0);
    println!("\n{}", "=".repeat(78));
    let mut bad = 0;
    for (v, d) in &results {
        let note = match v {
            Verdict::Pass => "red→green",
            Verdict::Vacuous => "GUARD IS VACUOUS",
            Verdict::AlreadyRed => "ALREADY RED",
            Verdict::Skip => "skipped (set LUMEN_GEMMA4_MODEL_DIR)",
        };
        println!("{:<width$}  {note}", d.name, width = width);
        if matches!(v, Verdict::Vacuous | Verdict::AlreadyRed) {
            bad += 1;
        }
    }
    let ok = results.iter().filter(|(v, _)| *v == Verdict::Pass).count();
    let skipped = results.iter().filter(|(v, _)| *v == Verdict::Skip).count();
    print!("\n{ok}/{} verified red→green", results.len());
    if skipped > 0 {
        print!(", {skipped} skipped");
    }
    println!();

    if bad > 0 {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}
