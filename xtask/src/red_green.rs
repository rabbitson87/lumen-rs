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
const DIF: &str = "crates/lumen-diffusion/src";
const SRV: &str = "crates/lumen-server/src";
const CORE: &str = "crates/lumen-core/src";

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
    /// `--features` value; empty selects the crate default.
    features: &'static str,
    /// Restrict to the lib target — skips building the crate's binaries and
    /// examples for a guard that only needs the library. A crate with no lib
    /// target must leave this false.
    lib_only: bool,
    /// Integration test target for `--test`; empty runs every target.
    test_target: &'static str,
    /// The Metal guards take minutes in a debug build and seconds in release.
    release: bool,
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
        features: "mlx-native",
        lib_only: true,
        test_target: "",
        release: false,
    }
}
const fn dif(filter: &'static str) -> Guard {
    Guard {
        package: "lumen-diffusion",
        filter,
        features: "mlx-native",
        lib_only: false,
        test_target: "",
        release: false,
    }
}
/// Guard on an integration-test target over the ungated tool-calling surface.
///
/// No feature and no GPU: `gemma4_tool_syntax` and `grammar` both compile under
/// `default = []`, which is the entire reason they were hoisted out of the
/// `mlx-native` gate — so this guard builds in seconds where an `mlx-native`
/// one takes minutes. Also used by the fault sweeps, which are pure
/// bytes-in/`Result`-out and equally GPU-free.
const fn mlx_ungated_test(target: &'static str, filter: &'static str) -> Guard {
    Guard {
        package: "lumen-mlx",
        filter,
        features: "",
        lib_only: false,
        test_target: target,
        release: false,
    }
}

/// Integration-test guard that DOES need `mlx-native`.
///
/// Both config parsers have since been hoisted to ungated `qwen35_config` /
/// `gemma4_config`, so their sweeps moved to `mlx_ungated_test`. What is left
/// here genuinely needs the feature — `NativeWeights` holds `Array`, so the
/// safetensors sweep cannot be built without MLX.
const fn mlx_native_test(target: &'static str, filter: &'static str) -> Guard {
    Guard {
        package: "lumen-mlx",
        filter,
        features: "mlx-native",
        lib_only: false,
        test_target: target,
        release: false,
    }
}

/// `lumen-server` grew a lib target so its request types could be reached from
/// tests and fuzzing; these guards live in `engine.rs`, which moved with it.
const fn srv(filter: &'static str) -> Guard {
    Guard {
        package: "lumen-server",
        filter,
        features: "mlx-native",
        lib_only: true,
        test_target: "",
        release: false,
    }
}

/// `lumen-mlx` lib guard that needs **no** feature — `grammar` is ungated
/// (pure llguidance + serde_json), so this builds in seconds where an
/// `mlx-native` lib guard takes minutes.
const fn core_mlx_lib(filter: &'static str) -> Guard {
    Guard {
        package: "lumen-mlx",
        filter,
        features: "",
        lib_only: true,
        test_target: "",
        release: false,
    }
}

/// `lumen-server` integration-test guard. Its lib target carries the request
/// types, so no feature is needed for the pure request-policy checks.
const fn srv_test(target: &'static str, filter: &'static str) -> Guard {
    Guard {
        package: "lumen-server",
        filter,
        features: "",
        lib_only: false,
        test_target: target,
        release: false,
    }
}

/// `lumen-core` is the FFI-free crate: no feature, no GPU, and the fastest
/// guards in the catalogue.
const fn core(filter: &'static str) -> Guard {
    Guard {
        package: "lumen-core",
        filter,
        features: "",
        lib_only: true,
        test_target: "",
        release: false,
    }
}

static DEFECTS: &[Defect] = &[
    Defect {
        name: "no-overlap-keyed-on-presence",
        symptom: "`LUMEN_MLX_NO_OVERLAP=0` — which reads as \"do not disable \
                  overlap\" — disabled it. The read was \
                  `env::var(..).is_err()`, keyed on the variable being SET at \
                  all, so every value including `0` and the empty string turned \
                  off overlap scheduling and slowed streaming. The comment two \
                  lines above it said `=1` restores the synchronous path",
        revert: &[Mutation {
            path: MLX,
            find: r#"        !no_overlap::get()"#,
            replace: r#"        std::env::var("LUMEN_MLX_NO_OVERLAP").is_err()"#,
        }],
        guards: &[mlx(
            "gemma4_backend::imp::tests::zero_does_not_disable_overlap",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "parallel-tool-calls-not-enforced",
        symptom: "`parallel_tool_calls: false` was accepted and then not applied \
                  — a three-city prompt returned three tool calls either way. \
                  Two wrong fixes before the right one: keying off the grammar \
                  FINISHING lands one token before the `<tool_call|>` closer, so \
                  the turn was cut mid-frame and the client got HTTP 200 with an \
                  empty message; hanging the check off the grammar STATE left it \
                  inert on the imatrix-AWQ family, where `grammar_factory()` \
                  returns None and three calls came back regardless",
        revert: &[Mutation {
            path: MLX,
            find: r#"        self.stops_after_first_call() && token == TOK_TOOL_CALL_CLOSE"#,
            replace: r#"        let _ = token;
        false // defect: the cap never fires"#,
        }],
        guards: &[core_mlx_lib(
            "grammar::tests::the_one_call_stop_fires_on_the_closer_and_needs_no_grammar",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "mtp-norm-double-fold",
        symptom: "the MTP head's RMSNorm `+1` fold was applied unconditionally, \
                  which is right for a raw HF snapshot and WRONG for every MTPLX \
                  Speed bundle — those ship the head pre-folded, so the load \
                  folded it twice. Invisible by construction: 1.37 + 1 is a \
                  bounded scale change, not the sign inversion the fold exists to \
                  prevent, so output stayed bit-exact lossless and only the \
                  accept rate fell. Measured 5 paired prompts per model, K=2 \
                  GEN=320 greedy: Qwen3.8-27B loses 0.055 accept on 5/5 prompts \
                  (t=12.1), Qwen3.6-27B 0.046 on 4/5. A single prompt cannot see \
                  it — the one prompt measured during the 3.8 port came out 0.018 \
                  the other way and was recorded as no signal",
        revert: &[Mutation {
            path: MLX,
            find: r#"    mean_of_means < MTP_NORM_RUNG_THRESHOLD"#,
            replace: r#"    let _ = mean_of_means;
    true // defect: fold every checkpoint, including the pre-folded ones"#,
        }],
        guards: &[
            mlx(
                "qwen3_5_mtp::norm_convention_tests::the_two_rungs_are_classified_and_are_far_from_the_threshold",
            ),
            mlx("qwen3_5_mtp::norm_convention_tests::the_boundary_is_pinned"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "effort-ungated-in-token-count",
        symptom: "`usage.prompt_tokens` counted a reasoning-effort sentence the \
                  prompt did not contain. The renderer asks `resolved_effort`, \
                  which drops the level on a checkpoint whose chat template never \
                  declares `reasoning_effort`; the token counter used the \
                  client's raw `ov.reasoning_effort` instead. Measured on \
                  Qwen3.5-9B: a `thinking: true` request prefilled 12 tokens and \
                  reported 54, and `reasoning_effort: low` reported 42. The same \
                  figure feeds the context guard, so a request near the limit \
                  could be refused for tokens it never had. Found by driving a \
                  real session, not by any gate",
        revert: &[Mutation {
            path: MLX,
            find: r#"            Self::Qwen35Family(m) if m.wants_reasoning_effort => ov.reasoning_effort,"#,
            replace: r#"            Self::Qwen35Family(_) => ov.reasoning_effort, // defect: ungated"#,
        }],
        guards: &[core_mlx_lib(
            "tests::effort_is_gated_on_the_checkpoint_declaring_it",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "mtp-drops-sampling-knobs",
        symptom: "the MTP speculative path rebuilt its `SamplingConfig` from a \
                  few scalars with `..default()`, silently dropping `top_k`, \
                  `min_p` and every penalty. MTP auto-enables on a checkpoint \
                  that ships an MTP head, so that was the LIVE decode path: \
                  measured on Qwen3.8-27B, `top_k: 1` at temperature 1.5 \
                  returned 3/3 distinct garbled replies instead of collapsing \
                  to the argmax, and `repeat_penalty: 1.8` changed nothing at \
                  all. Found immediately after wiring request temperature — \
                  fixing the entry points was not enough, because the value \
                  reached a second place that threw most of it away",
        revert: &[Mutation {
            path: MLX,
            find: r#"    (c.repeat_penalty - 1.0).abs() < 1e-6"#,
            replace: r#"    true || (c.repeat_penalty - 1.0).abs() < 1e-6 // defect: penalties silently dropped"#,
        }],
        guards: &[mlx(
            "tests::speculative_decode_refuses_sampling_it_cannot_reproduce",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "qwen-sampling-discarded",
        symptom: "`temperature` and `top_p` were accepted and ignored on the \
                  entire Qwen family — all four entry points on `MlxBackend` \
                  opened with `let _ = (top_p, temperature, ov)`, so decoding \
                  was greedy whatever the client sent, and no error said so. \
                  Measured on Qwen3.8-27B: `temperature: 1.5, top_p: 1.0` \
                  returned byte-identical text 4/4, with MTP on AND off. The \
                  doc comment above `chat` claimed the opposite (\"its sampling \
                  is configured via REPEAT_PENALTY env and request-level \
                  temperature\") — `REPEAT_PENALTY` was read nowhere on that \
                  path either. It survived because every test and every \
                  by-hand check ran at temperature 0, where correct and broken \
                  emit the same bytes",
        revert: &[Mutation {
            path: MLX,
            find: r#"            temperature: temperature.max(0.0),"#,
            replace: r#"            temperature: 0.0, // defect: request temperature discarded"#,
        }],
        guards: &[core_mlx_lib(
            "tests::a_request_asking_for_randomness_gets_a_sampler",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "tool-schema-uncounted-in-usage",
        symptom: "`usage.prompt_tokens` omitted the entire tool-schema block. \
                  The counter rendered through the tool-free `build_chat_input` \
                  while the request decoded through the tool-aware renderer, so \
                  the error was zero at zero tools and grew with the client's \
                  schema. Measured on Qwen3.8-27B with ONE tool declared: \
                  OpenAI reported 39 prompt tokens against a 279-token prefill, \
                  Anthropic 20 against 259 — 7x under. An agentic client \
                  shipping thirty tools is billed for a fraction of its prompt, \
                  and `guard_prompt_fits` admits on that same fraction, so a \
                  prompt over the context cap passes the guard and fails deeper \
                  in. Found by driving a real session; every gate was green",
        revert: &[Mutation {
            path: MLX,
            find: r#"                m.build_chat_input_with_tools(messages, thinking, tools, tool_choice, effort)
                    .map(|(ids, _prefill)| ids)"#,
            replace: r#"                { let _ = (tools, tool_choice); m.build_chat_input(messages, thinking, effort) } // defect: tool block uncounted"#,
        }],
        guards: &[core_mlx_lib(
            "tests::the_prompt_count_renders_the_tool_block_the_model_is_shown",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "anthropic-stream-zero-input-tokens",
        symptom: "the Anthropic streaming route reported `input_tokens: 0` for \
                  every request. `message_start` is the one place the format \
                  names the prompt size and it goes out before the first token, \
                  so the field was hardcoded `0` — under a comment in the `Done` \
                  arm claiming the real figure was \"surfaced in message_start \
                  above\", which it never was. An SDK that accumulates usage \
                  across the stream therefore billed a 289-token tool prompt as \
                  0, and unlike OpenAI there is no later event to correct it. \
                  Fixed by sending the count ahead of prefill as \
                  `StreamEvent::Start`",
        revert: &[Mutation {
            path: SRV,
            find: r#"        Some(StreamEvent::Start { prompt_tokens }) => (prompt_tokens, None),"#,
            replace: r#"        Some(StreamEvent::Start { prompt_tokens }) => { let _ = prompt_tokens; (0, None) } // defect: hardcoded zero"#,
        }],
        guards: &[srv(
            "routes::messages::tests::message_start_reports_the_prompt_size_the_engine_measured",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "undeclared-tool-name-forwarded",
        symptom: "a tool call named something the client never declared was \
                  forwarded verbatim, so the client looked up a function it does \
                  not have. Measured on Qwen3.8-27B: with `tool_choice=required` \
                  and `parallel_tool_calls` unset, the second call of the turn \
                  came back as `geget_weather` for a client that had declared \
                  only `get_weather`. The raw decode dump shows the MODEL wrote \
                  it — after the one-call-per-activation grammar released, the \
                  tail of the turn decoded unconstrained. `remap_tool_call_names` \
                  repaired the opposite direction (a name shorter than the \
                  declared one) and passed anything else straight through",
        revert: &[Mutation {
            path: SRV,
            find: r#"        calls.retain(declared);"#,
            replace: r#"        // defect: log it and forward it anyway"#,
        }],
        guards: &[
            srv("engine::tool_name_resolve_tests::an_unresolvable_name_is_dropped_not_forwarded"),
            srv("engine::tool_name_resolve_tests::requires_separator_boundary"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "qwen-parallel-tool-calls-not-consulted",
        symptom: "the WIRING half of the defect below, and the half no guard \
                  covered when it shipped. `ToolCalls::ExactlyOne` was resolved \
                  correctly and the parser counted completed calls correctly — \
                  both pure pieces were right the whole time. What was missing \
                  was `chat_with_tools_impl` asking either of them, so tests over \
                  the two pieces passed in both states. This guard drives the \
                  real decode loop over a scripted token stream, no model and no \
                  GPU, which is what makes the omission visible",
        revert: &[Mutation {
            path: MLX,
            find: r#"            if calls.must_stop_after_completed_calls(parser.completed_calls()) {"#,
            replace: r#"            if false {
                let _ = &calls; // defect: the decode loop never consults the cap"#,
        }],
        guards: &[core_mlx_lib(
            "tests::the_tool_decode_loop_consults_parallel_tool_calls",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "qwen-parallel-tool-calls-inert",
        symptom: "the fix above covered Gemma 4 only. `must_stop_after_call_closer` \
                  compares against a Gemma special-token id, and Qwen frames a call \
                  with the literal text `</tool_call>`, so on the whole Qwen family \
                  the cap could not fire and nothing said so — \
                  `ToolCalls::ExactlyOne` was built correctly, handed to the \
                  grammar builder (where the count is deliberately inert: the \
                  grammar is one-call-per-activation by construction) and never \
                  consulted by the decode loop. Measured on Qwen3.8-27B, \
                  `tool_choice=required` + `parallel_tool_calls=false` returned \
                  SEVEN identical calls",
        revert: &[Mutation {
            path: MLX,
            find: r#"        self.stops_after_first_call() && completed >= 1"#,
            replace: r#"        let _ = completed;
        false // defect: the cap never fires on the Qwen path"#,
        }],
        guards: &[
            core_mlx_lib(
                "grammar::tests::the_qwen_one_call_stop_fires_on_the_first_completed_call",
            ),
            core_mlx_lib(
                "qwen3_5_tools::tests::exactly_one_cuts_the_turn_where_one_or_more_keeps_decoding",
            ),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "parallel-tool-calls-ignored",
        symptom: "a client sending `parallel_tool_calls: false` got HTTP 200 and \
                  as many calls as the model produced — the field was never \
                  declared, and with no `deny_unknown_fields` serde dropped it \
                  silently, so nothing said the parameter had not been honoured",
        revert: &[Mutation {
            path: SRV,
            find: r#"            parallel_tool_calls: self.parallel_tool_calls,"#,
            replace: r#"            parallel_tool_calls: None, // defect: drop the client's request"#,
        }],
        guards: &[srv_test(
            "request_policy",
            "an_explicit_parallel_tool_calls_reaches_the_backend",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "scratch-path-collision",
        symptom: "three save/load round-trip tests wrote to a fixed name under \
                  /tmp (`tq_codebook_test.bin` and friends). Two checkouts \
                  testing at once — or one `cargo test` racing a `cargo xtask \
                  gate` — write the same file and read back each other's bytes; \
                  the loser reports a corrupted codebook, and it passes when \
                  rerun alone",
        revert: &[Mutation {
            path: CORE,
            find: r#"            let n = NEXT.fetch_add(1, Ordering::Relaxed);
            Self(
                std::env::temp_dir()
                    .join(format!("lumen-core-{stem}-{}-{n}.bin", std::process::id())),
            )"#,
            replace: r#"            let _ = NEXT.fetch_add(1, Ordering::Relaxed);
            Self(std::env::temp_dir().join(format!("lumen-core-{stem}.bin")))"#,
        }],
        guards: &[core("testpath::two_temp_paths_never_collide")],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
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
        name: "grammar-control-chars",
        symptom: "a tool whose name held an ASCII control character (`\\x00`, \
                  `\\x0b`, `\\x1b`, DEL …) made llguidance reject the whole \
                  grammar with `lexer error` — so the grammar was dropped and \
                  the model was left unconstrained, free to emit a tool nobody \
                  declared. Same end state as `grammar-rule-names`, reached \
                  through the escape table instead of the rule names",
        revert: &[Mutation {
            path: MLX,
            find: r#"            '\x00'..='\x1f' | '\x7f' => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
"#,
            replace: "            // xtask red-green: control-char escaping removed\n",
        }],
        // Both guards, because they fail for different reasons and a fix that
        // satisfied only one would still ship broken: the unit test builds a
        // real llguidance matcher (the production symptom), the replay asserts
        // the escape contract over the committed seeds (what the fuzzer sees).
        guards: &[
            mlx("grammar::tests::lark_grammar_escapes_control_chars_in_a_tool_name"),
            mlx_ungated_test("fuzz_corpus_replay", "replay_grammar_literals"),
        ],
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
        guards: &[
            mlx("grammar::tests::lark_grammar_escapes_a_quote_in_a_tool_name"),
            // The second guard is what makes the `grammar_literals` fuzz target
            // worth having. The unit test above pins one hand-written name; this
            // one asserts the escaping *contract* over every committed seed, and
            // it is the same assertion the soak runs — so a revert here proves
            // the fuzzer would have caught the original defect rather than
            // leaving that as a claim.
            mlx_ungated_test("fuzz_corpus_replay", "replay_grammar_literals"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "tool-name-scanner",
        symptom: "`call:bad call:good{x:1}` parsed as ONE tool named \
                  \"bad call:good\" — a name no client declared",
        // De-indented by four relative to the original entry: the parser moved
        // out of `gemma4_response`'s `mod imp` into the ungated
        // `gemma4_tool_syntax`, so it sits one block shallower.
        revert: &[Mutation {
            path: MLX,
            find: r#"            if bytes[brace_start..].starts_with(b"call:") {
                hit_next_call = true;
                break;
            }
"#,
            replace: "            // xtask red-green: next-call boundary removed\n",
        }],
        guards: &[
            mlx("gemma4_tool_syntax::tests::body_parser_stops_a_name_at_the_next_opener"),
            mlx("gemma4_tool_syntax::tests::body_parser_skips_a_run_of_malformed_openers"),
            mlx("gemma4_tool_syntax::tests::body_parser_stops_a_non_ascii_name_at_the_next_opener"),
            // The generated driver, registered here so `red-green` proves it
            // is not vacuous. Its hand-written siblings above encode the three
            // shapes someone thought of; this one walks 600 seeded streams
            // built against a declared tool set and asserts no parsed name
            // escapes it.
            mlx_ungated_test(
                "tool_surface_fuzz",
                "parser_survives_generated_tool_call_streams",
            ),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "safetensors-silent-truncation",
        symptom: "a shard truncated inside its DATA section loaded with NO error \
                  and returned wrong weights (4x4 f32 missing 16 bytes read 3.0 \
                  where the file wrote 15.0) — a partial download served \
                  plausible, wrong output forever",
        revert: &[Mutation {
            path: MLX,
            find: r#"                validate_safetensors_complete(&shard)?;
"#,
            replace: "                // xtask red-green: completeness guard removed\n",
        }],
        guards: &[mlx_native_test(
            "weights_faults",
            "data_section_truncation_is_rejected_not_silently_wrong",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "config-null-moe-fields",
        symptom: "a dense checkpoint spelling its absent MoE fields as \
                  `\"num_experts\": null` failed to load — `#[serde(default)]` \
                  covers a MISSING key, not an explicit null (the JGOS-31B shape)",
        revert: &[Mutation {
            path: MLX,
            find: r#"    #[serde(default, deserialize_with = "null_as_default")]
    pub num_experts: usize,"#,
            replace: r#"    #[serde(default)]
    pub num_experts: usize,"#,
        }],
        guards: &[mlx_ungated_test(
            "config_faults",
            "explicit_null_moe_fields_on_a_dense_config",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "qwen-nested-eos-token-id",
        symptom: "a checkpoint declaring `eos_token_id` only inside \
                  `text_config` (Qwen3.5-9B-MTPLX-Speed) got an EMPTY stop \
                  set — no error, just generation running past the turn \
                  boundary and emitting the next turn's `\\nuser\\n` header \
                  into the reply, and 4 identical tool calls where one was \
                  asked for",
        revert: &[Mutation {
            path: MLX,
            // Renaming rather than deleting: removing the field breaks
            // COMPILATION of the guard that reads it, and the harness reports
            // that as "guard matched no test" — which is not the same as RED.
            // A mutation has to change behaviour while keeping the tree
            // buildable, or it proves nothing about the guard.
            find: r#"        rename = "eos_token_id",
        deserialize_with = "deserialize_token_ids"
    )]
    pub eos_token_ids: Vec<u32>,
    pub model_type: String,"#,
            replace: r#"        rename = "eos_token_id_never_present",
        deserialize_with = "deserialize_token_ids"
    )]
    pub eos_token_ids: Vec<u32>,
    pub model_type: String,"#,
        }],
        guards: &[mlx_ungated_test(
            "qwen35_config_validate",
            "eos_token_id_is_read_from_either_level",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "gemma4-config-null-moe-fields",
        symptom: "the JGOS-31B shape was still live in Gemma 4: a dense \
                  checkpoint spelling `\"num_experts\": null` hard-failed with \
                  `invalid type: null, expected usize at line N column M` — the \
                  fix was remembered as a Gemma 4 fix but had only ever landed \
                  in the Qwen parser",
        revert: &[Mutation {
            path: MLX,
            find: r#"    #[serde(default, deserialize_with = "crate::config_serde::null_as_default")]
    pub num_experts: usize,"#,
            replace: r#"    #[serde(default)]
    pub num_experts: usize,"#,
        }],
        guards: &[mlx_ungated_test(
            "gemma4_config_faults",
            "explicit_null_moe_fields_on_a_dense_config",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "prefill-budget-rounds-to-zero",
        symptom: "a positive-but-tiny `*_PREFILL_SCORES_GB` (under 1e-9, i.e. \
                  less than one byte) passed the `> 0.0` check and then cast to \
                  0 — the OOM guard was silently switched off, pinning every \
                  prompt to the 256-token floor while still logging a clamp as \
                  if it were working",
        revert: &[Mutation {
            path: MLX,
            find: r#"        .map(|g| (g * 1e9) as u64)
        .filter(|&b| b > 0)
        .unwrap_or(DEFAULT_SCORES_BUDGET_BYTES)"#,
            replace: r#"        .map(|g| (g * 1e9) as u64)
        .unwrap_or(DEFAULT_SCORES_BUDGET_BYTES)"#,
        }],
        guards: &[mlx_ungated_test(
            "prefill_budget_faults",
            "hostile_budget_env_values_never_hang_or_disable_the_guard",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "kv-disk-alloc-bomb",
        symptom: "a corrupt KV-disk record's u64 payload length was allocated \
                  unvalidated — a 280 TB request that ABORTS the process \
                  (allocation failure is not a catchable panic)",
        revert: &[Mutation {
            path: MLX,
            find: r#"        let data_len = read_u64(r)? as usize;
        if data_len != expected {
            bail!(
                "kv_disk: record declares {data_len} payload bytes but shape {shape:?} \
                 with dtype {dtype:?} implies {expected} (corrupt record)"
            );
        }
"#,
            replace: "        let data_len = read_u64(r)? as usize;\n",
        }],
        guards: &[mlx_ungated_test(
            "kv_disk_faults",
            "implausible_record_length_is_rejected_not_allocated",
        )],
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
            mlx("gemma4_tool_syntax::tests::args_to_json_quotes_a_non_ascii_bare_key"),
            mlx(
                "gemma4_tool_syntax::tests::args_to_json_handles_non_ascii_in_nested_and_array_positions",
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
        name: "causal-mask-coverage",
        symptom: "the sliding-window mask tests asserted an f32 dtype neither \
                  builder had produced for months; #[ignore]d, they panicked in \
                  silence instead of guarding the window clamp",
        revert: &[Mutation {
            path: MLX,
            find: r#"            let window_mask = linds
                .lt_device(&rinds_plus_w, &stream)"#,
            replace: r#"            let window_mask = linds
                .ge_device(&rinds_plus_w, &stream)"#,
        }],
        guards: &[
            mlx("native_attention::parity_tests::causal_mask_prefill_window_truncates_past_window"),
            mlx("native_attention::parity_tests::causal_mask_decode_with_offset_and_window"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &["--ignored"],
    },
    Defect {
        name: "causal-mask-builders-agree",
        symptom: "LUMEN_LEGACY_MASK_BUILDER is a live escape hatch, but nothing \
                  compared the two mask representations against each other",
        revert: &[Mutation {
            path: MLX,
            find: r#"            let valid_min_abs = std::cmp::max(window_min, cache_first_held_pos);"#,
            replace: r#"            let valid_min_abs = cache_first_held_pos;"#,
        }],
        guards: &[mlx(
            "native_attention::parity_tests::causal_mask_prefill_window_truncates_past_window",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &["--ignored"],
    },
    Defect {
        name: "rotating-cache-both-paths",
        symptom: "the rotating-cache growth test asserted cached_len == fetch, \
                  false for the default path since step-prealloc landed; and the \
                  legacy path was unreachable from any test (OnceLock over env)",
        // Drops a token from the legacy concat path. Before the thread-local
        // override this mutation was invisible: the flag is a OnceLock over an
        // env var, so every test in the binary ran the default path — and
        // before the content assertions, nothing compared what came back out.
        revert: &[Mutation {
            path: MLX,
            find: r#"                    let cached = self.offset;"#,
            replace: r#"                    let cached = self.offset.saturating_sub(1);"#,
        }],
        guards: &[mlx(
            "native_cache::lifecycle_tests::rotating_cache_growth_within_max_size",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &["--ignored"],
    },
    Defect {
        name: "flux-scheduler-invariants",
        symptom: "the scheduler test compared against /tmp/klein_sigmas.bin, a \
                  dev-session dump that no longer exists, so it failed on every \
                  machine",
        revert: &[Mutation {
            path: DIF,
            find: r#"        let s = exp_mu / (exp_mu + (1.0 / t - 1.0));"#,
            replace: r#"        let s = exp_mu / (exp_mu + (1.0 / t + 1.0));"#,
        }],
        guards: &[dif("scheduler::tests::shift_multiplies_the_odds_by_exp_mu")],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "flux-left-padding",
        symptom: "the encoder reads the tail of the window, so right-padding \
                  would feed it padding and drop the prompt — untested because \
                  the only coverage needed a 24GB tokenizer",
        revert: &[Mutation {
            path: DIF,
            find: r#"    let mut out = vec![pad; len - ids.len()];
    out.extend_from_slice(&ids);"#,
            replace: r#"    let mut out = ids.clone();
    out.extend(std::iter::repeat_n(pad, len - ids.len()));"#,
        }],
        guards: &[dif(
            "tokenizer::tests::padding_is_on_the_left_and_preserves_order",
        )],
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
        | (_, "grammar-literal-escaping")
        | (_, "grammar-control-chars") => "grammar.rs",
        // Both defects live in the tool-call body grammar, which moved out of
        // `gemma4_response`'s feature-gated `mod imp` so it can be tested and
        // fuzzed without `mlx-native`.
        (_, "tool-name-scanner") | (_, "args-unicode-keys") => "gemma4_tool_syntax.rs",
        (_, "kv-disk-alloc-bomb") => "kv_disk.rs",
        (_, "config-null-moe-fields") | (_, "qwen-nested-eos-token-id") => "qwen35_config.rs",
        (_, "safetensors-silent-truncation") => "qwen3_5_moe.rs",
        (_, "mtp-norm-double-fold") => "qwen3_5_mtp.rs",
        (_, "prefill-budget-rounds-to-zero") => "prefill_budget.rs",
        (_, "gemma4-config-null-moe-fields") => "gemma4_config.rs",
        (_, "gemma-nonstreaming-grammar")
        | (_, "qwen-first-token-mask")
        | (_, "qwen-parallel-tool-calls-not-consulted")
        | (_, "effort-ungated-in-token-count")
        | (_, "tool-schema-uncounted-in-usage")
        | (_, "qwen-sampling-discarded")
        | (_, "mtp-drops-sampling-knobs") => "lib.rs",
        (_, "gemma-thought-channel") => "gemma4_chat.rs",
        (_, "causal-mask-coverage") | (_, "causal-mask-builders-agree") => "native_attention.rs",
        (_, "rotating-cache-both-paths") => "native_cache.rs",
        (_, "flux-scheduler-invariants") => "scheduler.rs",
        (_, "flux-left-padding") => "tokenizer.rs",
        (_, "tool-choice-none")
        | (_, "anthropic-turn-images")
        | (_, "undeclared-tool-name-forwarded") => "engine.rs",
        (_, "anthropic-stream-zero-input-tokens") => "routes/messages.rs",
        // `TempPath` lives in `lumen-core`'s lib.rs rather than in a module of
        // its own: it is three lines of test scaffolding shared by three
        // round-trip tests, and a file for it would be more ceremony than code.
        (_, "scratch-path-collision") => "lib.rs",
        (_, "parallel-tool-calls-not-enforced") | (_, "qwen-parallel-tool-calls-inert") => {
            "grammar.rs"
        }
        (_, "no-overlap-keyed-on-presence") => "gemma4_backend.rs",
        (_, "parallel-tool-calls-ignored") => "types.rs",
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

/// The guards that currently FAIL, by filter.
///
/// Every guard runs — no short-circuit. "At least one guard went red" is too
/// weak a bar: a defect listing three guards where only one catches it would
/// report red→green while the other two are decoration. The red phase demands
/// that *each* listed guard fails, and names the ones that didn't.
fn failing_guards(defect: &Defect) -> Result<Vec<&'static str>, String> {
    let mut failing = Vec::new();
    for g in defect.guards {
        let mut cmd = Command::new("cargo");
        cmd.current_dir(root()).args(["test", "-p", g.package]);
        if !g.features.is_empty() {
            cmd.args(["--features", g.features]);
        }
        if g.release {
            cmd.arg("--release");
        }
        if g.lib_only {
            // lumen-server is a binary crate; it has no lib target.
            cmd.arg("--lib");
        }
        if !g.test_target.is_empty() {
            cmd.args(["--test", g.test_target]);
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
            failing.push(g.filter);
        }
    }
    Ok(failing)
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
    if !failing_guards(defect)?.is_empty() {
        return Ok(Verdict::AlreadyRed);
    }
    let still_green = {
        let mut restore = Restore::new();
        apply(defect, &mut restore)?;
        let failing = failing_guards(defect)?;
        // `restore` drops here, putting the source back even if the call above
        // returned early with an error.
        defect
            .guards
            .iter()
            .map(|g| g.filter)
            .filter(|f| !failing.contains(f))
            .collect::<Vec<_>>()
    };
    if !failing_guards(defect)?.is_empty() {
        return Err(format!(
            "{}: source not restored cleanly — run `git status` and revert by hand",
            defect.name
        ));
    }
    if still_green.is_empty() {
        Ok(Verdict::Pass)
    } else {
        for f in &still_green {
            eprintln!("  VACUOUS guard (green with the defect reintroduced): {f}");
        }
        Ok(Verdict::Vacuous)
    }
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
