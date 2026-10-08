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
/// fastokens, vendored with lumen-rs's fixes (vendor/fastokens/PATCHES.md).
const FASTOKENS: &str = "vendor/fastokens/src/pre_tokenizers";
const WORKFLOWS: &str = ".github/workflows";
const APP: &str = "crates/lumen-app";
const SRV_CRATE: &str = "crates/lumen-server";

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

/// `lumen-server` lib guard that loads a real checkpoint. Release, because
/// loading ~16 GB of weights in a debug build spends minutes before the test
/// starts; pair it with `needs_checkpoint` so the run skips without one.
const fn srv_checkpoint(filter: &'static str) -> Guard {
    Guard {
        release: true,
        ..srv(filter)
    }
}

/// `lumen-server` integration test over a real checkpoint (it may start the
/// server binary itself). Release, as `srv_checkpoint`.
const fn srv_checkpoint_test(target: &'static str, filter: &'static str) -> Guard {
    Guard {
        release: true,
        ..srv_test(target, filter)
    }
}

/// `lumen-mlx` lib guard over a real checkpoint: release, since a debug forward
/// over thousands of tokens takes minutes. Pair with `needs_checkpoint`.
const fn mlx_checkpoint(filter: &'static str) -> Guard {
    Guard {
        release: true,
        ..mlx(filter)
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
        name: "replay-drops-think-block",
        symptom: "`session_id` never reused a single KV token on the Qwen path. \
                  A `thinking: false` generation prompt ends with \
                  `<think>\\n\\n</think>\\n\\n` and the reply follows it, but the \
                  replayed assistant turn was rendered without that block — so \
                  the new prompt was not a token-prefix of what the model had \
                  been fed, `prompt.starts_with(stored)` failed at the first \
                  assistant token, and every turn cold-prefilled the entire \
                  conversation while reporting nothing. Measured on \
                  Qwen3.8-27B: a replayed turn occupied 16 tokens of framing \
                  where generation produced 2. It was also a prompt-fidelity \
                  bug — 3.8's template defaults `preserve_thinking` to true, so \
                  lumen was rendering the non-default branch. With a \
                  6324-token system prompt the fix takes turn 2 from a 48 s \
                  cold prefill to 0.44 s",
        revert: &[Mutation {
            path: MLX,
            find: r#"    template.contains("preserve_thinking is undefined")"#,
            replace: r#"    { let _ = template; false } // defect: replay drops the think block"#,
        }],
        guards: &[mlx(
            "tests::a_replayed_assistant_turn_matches_what_the_model_was_fed",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "reasoning-trace-has-no-wire-field",
        symptom: "a `thinking: true` session could not reuse one KV token. The \
                  generation prompt stops at an open `<think>` block and the \
                  model writes the trace itself, so the tokens in the cache \
                  contain reasoning — but the request type had nowhere to put \
                  it, so the replayed turn re-rendered with an empty block and \
                  diverged from its own KV at that point. The trace was not \
                  even unavailable: the server emits it as \
                  `ChatMessageResponse.reasoning`, so a client handing our own \
                  response object back was returning it and being ignored. \
                  Fixed by reading the three spellings that reach us — \
                  `reasoning_content` (DeepSeek/vLLM, and Qwen's own template's \
                  field name), `reasoning` (ours), and the `<think>` envelope \
                  already inside `content` — and folding them into the one \
                  representation the renderers read",
        revert: &[
            Mutation {
                path: SRV,
                find: r#"    if !role.eq_ignore_ascii_case("assistant") {"#,
                replace: "    if true {\n        // defect: the trace never reaches the prompt",
            },
            Mutation {
                path: MLX,
                find: "            let (reasoning, visible) = \
                       crate::chat_io::split_reasoning_envelope(content);",
                replace: "            let (reasoning, visible) = (\"\", content.as_str()); \
                          // defect: trace dropped",
            },
        ],
        guards: &[
            srv("types::reasoning_round_trip::the_deepseek_field_name_is_accepted"),
            srv("types::reasoning_round_trip::the_field_lumen_emits_is_a_field_lumen_reads"),
            mlx("tests::a_replayed_thinking_turn_matches_what_the_model_was_fed"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "anthropic-thinking-block-rejected",
        symptom: "the Anthropic Messages API *requires* a client to replay the \
                  thinking blocks of any assistant turn it sends back once \
                  extended thinking is on. `AnthropicContentBlock` is \
                  deliberately exhaustive, and `thinking` was not one of its \
                  variants — so a client doing exactly what the spec demands \
                  had the whole request rejected with `unknown variant \
                  \"thinking\"`, not just the block dropped. Fixed by accepting \
                  `thinking` (folded into the reasoning envelope so the trace \
                  reaches the prompt) and `redacted_thinking` (accepted and \
                  ignored — the payload is encrypted, and the client had no \
                  choice about receiving it), while an unrecognized type still \
                  fails",
        revert: &[Mutation {
            path: SRV,
            find: "    Thinking {\n        #[serde(default)]\n        thinking: String,",
            replace: "    #[serde(rename = \"thinking_defect_replay\")]\n    Thinking {\n        \
                      #[serde(default)]\n        thinking: String,",
        }],
        guards: &[srv(
            "types::reasoning_round_trip::an_anthropic_thinking_block_is_accepted_and_kept",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "qwen-thinking-trace-served-as-the-answer",
        symptom: "with thinking on, the Qwen *prompt* opens the `<think>` block \
                  and the model writes its trace with no opener of its own, \
                  closing with a bare `</think>`. The parser started in \
                  `Visible` and so never entered the thinking state at all. \
                  Measured on Qwen3.8-27B: a `thinking:true` request came back \
                  with **no `reasoning` field**, the raw chain-of-thought as \
                  `content`, and the stray close tag sitting in the middle of \
                  the answer — `\"We need answer user's simple request: … \
                  red.\\n</think>\\n\\nRed\"`. The default is that `content` \
                  carries the visible answer alone, so every thinking-on turn \
                  ever served violated it. The state cannot be recovered after \
                  the fact: visible text streams as it arrives, so by the time \
                  the close tag shows up the trace has already gone out as \
                  content deltas",
        revert: &[Mutation {
            path: MLX,
            find: "        Self {\n            state: State::Thinking,\n            \
                   in_think: true,\n            ..Self::new()\n        }",
            replace: "        Self::new() // defect: the prompt's open block is invisible",
        }],
        guards: &[mlx(
            "qwen3_5_tools::tests::a_prompt_opened_think_block_is_reasoning_not_visible_text",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "qwen-plain-path-never-split-reasoning",
        symptom: "the no-tools Qwen path runs no parser at all — it returned \
                  the whole decode as `visible` with `reasoning: String::new()` \
                  hardcoded, on both the streaming and non-streaming entry \
                  points. So the trace reached the client as the answer even \
                  though the tool path had separated it correctly for turns \
                  that happened to declare a tool. Fixed with a splitter that \
                  routes trace bytes to `BackendStreamEvent::Reasoning` as they \
                  arrive, holding back any tail that could still grow into \
                  `</think>` across a decode-step boundary. Gating it on the \
                  thinking flag is load-bearing: a thinking-off reply carries \
                  no close tag and is byte-indistinguishable from a trace that \
                  ran out of budget, so an ungated split answered `content: \
                  \"\"`, `reasoning: \"Red\"` — caught by hand against the \
                  running server, after the whole suite was green",
        revert: &[
            Mutation {
                path: MLX,
                find: "    let Some(idx) = text.find(\"</think>\") else {\n        \
                       return (text, \"\");",
                replace: "    let Some(idx) = text.find(\"\\u{0}never\") else {\n        \
                          return (\"\", text);",
            },
            Mutation {
                path: MLX,
                find: "            phase: if thinking_open {\n                Phase::Trace",
                replace: "            phase: if false {\n                Phase::Trace",
            },
        ],
        guards: &[
            mlx("qwen3_5_tools::tests::the_plain_path_splits_a_prompt_opened_trace"),
            mlx("qwen3_5_tools::tests::the_streaming_splitter_holds_a_tag_split_across_chunks"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "stray-think-close-reaches-the-answer",
        symptom: "a reasoning-first checkpoint sometimes closes a `<think>` \
                  block it never opened, even on a thinking-OFF turn where the \
                  prompt handed it an already-closed one. Measured on \
                  Qwen3.8-27B: `'Blue\\n</think>\\n\\nBlue'` — never at \
                  temperature 0 (0/16), rarely when sampling (1/32 at 0.8, 2/6 \
                  at the 0.7 default), so it is the model's doing and not a \
                  prompt defect: the rendered generation prompt ends with a \
                  closed `<think>\\n\\n</think>\\n\\n`, checked by rendering it. \
                  The tool-aware parser has dropped an unbalanced close since \
                  it was written, so the identical reply came back one way with \
                  tools attached and another way without — the plain path \
                  handed the raw delimiter to the client. Dropping rather than \
                  re-reading it as a delimiter is deliberate: the text before \
                  it cannot be moved to `reasoning` on the streaming surface \
                  because it has already gone out as content deltas, so \
                  splitting the batch surface alone would make the two \
                  disagree instead. A balanced pair the model quotes is left \
                  alone, and the depth is carried across deltas because such a \
                  pair almost always spans decode steps",
        revert: &[Mutation {
            path: MLX,
            find: "                out.push_str(&rest[..c]);\n                if *depth > 0 {",
            replace: "                out.push_str(&rest[..c]);\n                \
                      if true { // defect: the stray tag survives",
        }],
        guards: &[
            mlx("qwen3_5_tools::tests::a_stray_close_tag_never_reaches_the_answer"),
            mlx("qwen3_5_tools::tests::the_streaming_splitter_drops_a_stray_close_the_same_way"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "inference-error-drops-its-cause",
        symptom: "an inference failure reached the client naming only its \
                  outermost context. A Metal out-of-memory arrived as \
                  `\"inference error: native mlx-rs runner: prefill forward \
                  (seq_id=3)\"` — true, and useless: the link that says the GPU \
                  ran out of memory was one level down and anyhow's plain \
                  `Display` prints only the head of the chain. All three routes \
                  formatted with `{e}` independently, so the loss was in three \
                  places at once. With `{e:#}` the same failure now carries \
                  `[METAL] Command buffer execution failed: Insufficient Memory \
                  (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)` all \
                  the way to the client, on the batch and streaming surfaces of \
                  both APIs. Folded into one shared helper so a fourth route \
                  cannot quietly reintroduce the lossy form",
        revert: &[Mutation {
            path: SRV,
            find: "    format!(\"inference error: {e:#}\")",
            replace: "    format!(\"inference error: {e}\") // defect: cause dropped",
        }],
        guards: &[srv(
            "types::inference_error_carries_its_cause::the_root_cause_survives_into_the_message",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "session-reuse-needs-a-nonstandard-field",
        symptom: "KV reuse on the Qwen plain-chat path was reachable only \
                  through `session_id` — a Lumen extension that neither the \
                  OpenAI nor the Anthropic request defines, so no stock client \
                  sends it and every one of them re-prefilled the entire \
                  conversation on every turn. The tool path had auto-keyed \
                  itself from the system hash for ages, and the Gemma arm did \
                  too; only Qwen's flat path still demanded the field. Fixed by \
                  letting the prompt identify its own conversation: a session's \
                  stored tokens are what the model was fed plus what it \
                  generated, so a session whose tokens are a strict prefix of \
                  this prompt IS this conversation one turn earlier. That is \
                  also the gate reuse was always guarded by, so a guessed key \
                  costs a prefill and never an answer — verified by output: the \
                  auto path is byte-identical to the explicit-`session_id` path \
                  on the same conversation. Measured on Qwen3.8-27B with no \
                  `session_id` anywhere: turn 1 4.55 s, then 0.41 s and 0.40 s",
        revert: &[Mutation {
            path: MLX,
            find: "        .filter_map(|(k, s)| s.reusable_len(prompt_ids).map(|n| (k, n)))",
            replace: "        .filter_map(|(k, s)| { let _ = (k, s, prompt_ids); None::<(&String, usize)> })",
        }],
        guards: &[mlx(
            "tests::a_prompt_finds_its_own_conversation_without_being_told_which",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "anthropic-output-drops-thinking-block",
        symptom: "the Anthropic route emitted no `thinking` content block, on \
                  either surface — the non-streaming assembler simply never \
                  built one, and the SSE loop discarded `ReasoningDelta` under \
                  a comment saying proper wiring was 'deferred'. So an \
                  Anthropic client could not see the trace, and — the part that \
                  costs something — could not hand it back, which is what a \
                  thinking-enabled conversation needs to extend its KV instead \
                  of re-prefilling every turn. Measured on Qwen3.8-27B once the \
                  block existed: a client echoing our own `content[]` back gets \
                  `reuse cached=321 suffix=15` in 283 ms where one that drops \
                  the block gets `divergent prompt` and a 2237 ms re-prefill. \
                  Gated on the request's own thinking flag, which on this API \
                  is the client's alone, so a caller that never opted in sees \
                  exactly the `content[]` it saw before",
        revert: &[Mutation {
            path: SRV,
            find: "    if emit_thinking && !reasoning.is_empty() {",
            replace: "    if false && !reasoning.is_empty() { // defect: the trace is dropped",
        }],
        guards: &[srv(
            "engine::anthropic_thinking_blocks::the_trace_comes_first_then_text_then_tools",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "anthropic-stream-block-indices-pinned",
        symptom: "Anthropic identifies every streaming delta by its index into \
                  the final `content[]`, and the SSE loop hardcoded them: text \
                  was pinned to `index:0` by a `const` prefix and tool blocks \
                  started at 1. That was correct only while nothing could \
                  precede the text — a `thinking` block ahead of it shifts every \
                  later index by one, and a delta carrying the wrong index is \
                  still well-formed JSON, so the failure is a client quietly \
                  assembling one block's bytes into another. Fixed by giving the \
                  indices an owner (`BlockStream`) that emits the block \
                  lifecycle frames and can be read back by a test, which the \
                  loop writing straight to a socket could not be",
        revert: &[Mutation {
            path: SRV,
            find: "        self.next_index = idx.saturating_add(1);",
            replace: "        self.next_index = 0; // defect: every block claims index 0",
        }],
        guards: &[
            srv(
                "routes::messages::tests::a_thinking_block_takes_index_zero_and_pushes_the_rest_along",
            ),
            srv(
                "routes::messages::tests::without_a_trace_text_is_still_index_zero_and_tools_start_at_one",
            ),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "qwen-tool-stream-drops-the-trace",
        symptom: "the tool-aware Qwen parser accumulated the reasoning for \
                  `finish()` but never emitted it as it arrived, so the \
                  non-streaming answer carried the trace and the streaming one \
                  had none at all. Found against the running server while \
                  verifying the Anthropic thinking block: a thinking-enabled \
                  request WITH tools streamed `text@0, tool_use@1` and nothing \
                  else, where the same request without tools streamed \
                  `thinking@0, text@1`. The tool path is the agentic one, so \
                  this was the case where it mattered most — and it cost the \
                  OpenAI route its `delta.reasoning` on exactly the same turns",
        revert: &[Mutation {
            path: MLX,
            find: "                events.push(Qwen35ParseEvent::Reasoning(chunk));",
            replace: "                let _ = &chunk; // defect: kept but never streamed",
        }],
        guards: &[mlx(
            "qwen3_5_tools::tests::the_tool_parser_streams_the_trace_instead_of_only_keeping_it",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "tools-replay-doubles-think-block",
        symptom: "the tool-calling renderer prefixed its own empty `<think>` \
                  block to whatever the assistant turn's text already was. With \
                  `LUMEN_REASONING_IN_CONTENT=1` — or any client echoing that \
                  content back — the trace is already in there, so the turn \
                  rendered as `<think>\\n\\n</think>\\n\\n<think>\\n…`, two \
                  nested blocks in a shape no Qwen template emits, and the \
                  reasoning landed in the visible-text slot. Fixed by splitting \
                  the envelope out and rendering the trace inside the one block \
                  the template defines",
        revert: &[Mutation {
            path: MLX,
            find: "    let (reasoning, visible) = split_reasoning_envelope(content);",
            replace: "    let (reasoning, visible) = (\"\", content); // defect: block not split",
        }],
        guards: &[mlx(
            "qwen3_5_tools::tests::a_replayed_tool_turn_carries_the_trace_it_was_given",
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
        name: "tool-history-uncounted",
        symptom: "a request carrying tool history — prior assistant \
                  `tool_calls`, `role:\"tool\"` results, Anthropic \
                  `tool_use`/`tool_result` — was counted from its flattened \
                  `(role, content)` pairs while it decoded from `ChatTurn`s. \
                  Qwen's flat renderer drops tool turns and calls, the \
                  Anthropic flattening drops the blocks, and Gemma's rejects \
                  role `tool` (chars/4 fallback). Documented as a turn-framing \
                  gap of \"tens of tokens\"; measured on Qwen3.5-9B it was the \
                  whole tool result — a 34.8K-token prefill reported as 32.7K — \
                  in `usage.prompt_tokens` and in what `guard_prompt_fits` \
                  admitted on. Found by the per-request `[tokenize]` line in \
                  task 016",
        revert: &[Mutation {
            path: MLX,
            find: r#"                m.build_chat_input_with_tools_from_history(
                    turns,
                    thinking,
                    tools,
                    tool_choice,
                    effort,
                )
                .map(|(ids, _prefill)| ids)"#,
            replace: r#"                { let flat: Vec<(String, String)> = turns.iter().map(|t| match t { crate::chat_io::ChatTurn::System(s) => ("system".to_string(), s.to_string()), crate::chat_io::ChatTurn::User(s) => ("user".to_string(), s.to_string()), crate::chat_io::ChatTurn::Assistant { text, .. } => ("assistant".to_string(), text.to_string()), crate::chat_io::ChatTurn::Tool { content, .. } => ("tool".to_string(), content.to_string()) }).collect(); m.build_chat_input_with_tools(&flat, thinking, tools, tool_choice, effort).map(|(ids, _prefill)| ids) } // defect: tool history flattened"#,
        }],
        guards: &[core_mlx_lib(
            "tests::the_history_count_renders_the_tool_turns_the_model_is_shown",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "anthropic-batch-guard-omits-images",
        symptom: "a non-streaming /v1/messages request was admitted on its \
                  rendered text alone: every image's placeholder run (~280 \
                  soft tokens each on Gemma 4) slipped past the prompt cap that \
                  OpenAI and Anthropic streaming enforce, and was added only \
                  afterwards, to `usage` — so the figure a request was admitted \
                  on was not the figure it reported. Found in task 016",
        revert: &[Mutation {
            path: SRV,
            find: ") + image_tokens;\n            guard_prompt_fits(&self.backend, prompt_tokens)?;",
            replace: ") + image_tokens;\n            // defect: the guard sees the text alone\n            guard_prompt_fits(&self.backend, prompt_tokens - image_tokens)?;",
        }],
        guards: &[srv_checkpoint(
            "engine::anthropic_batch_image_admission::the_batch_guard_admits_on_the_count_it_reports",
        )],
        occurrences: 2, // the flat branch and the tool-history branch
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "server-binds-every-interface",
        symptom: "lumen-app's host scope (\"localhost (127.0.0.1)\" by default) \
                  reached the server as LUMEN_HOST and was never read: every \
                  launch bound 0.0.0.0, putting an API with no auth on the \
                  network, while the README promised a 127.0.0.1 default",
        revert: &[Mutation {
            path: SRV,
            find: "        .unwrap_or(\"127.0.0.1\");",
            replace: "        .unwrap_or(\"0.0.0.0\"); // defect: every interface",
        }],
        guards: &[srv("access::tests::the_listener_defaults_to_loopback")],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "api-key-not-enforced",
        symptom: "lumen-app's API key reached the server as LUMEN_API_KEY and was \
                  never read: a setup the app showed as key-protected answered \
                  every request without one",
        revert: &[Mutation {
            path: SRV,
            find: "        if path == \"/health\" || method == \"OPTIONS\" {",
            replace: "        if true {\n            // defect: the key is never checked",
        }],
        guards: &[srv(
            "access::tests::a_configured_key_is_required_everywhere_but_health_and_preflight",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "cors-setting-ignored",
        symptom: "lumen-app's CORS scope reached the server as LUMEN_CORS and was \
                  never read: no response carried CORS headers, so a browser \
                  client failed under every setting, \"all\" included",
        revert: &[Mutation {
            path: SRV,
            find: "            Self::All => Some(\"*\".to_string()),",
            replace: "            Self::All => None, // defect: CORS never applied",
        }],
        guards: &[srv("access::tests::all_and_off_ignore_the_origin")],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "completions-unguarded",
        symptom: "/v1/completions was the one surface with no prompt-size guard: \
                  a raw prompt of any size went straight to prefill, where an \
                  oversized prompt is a Metal out-of-memory instead of a refusal",
        revert: &[Mutation {
            path: SRV,
            find: "        guard_prompt_fits(&self.backend, prompt_tokens)?;\n\n        let ov = req.sampling_overrides();\n        let output_ids = self.backend.generate(",
            replace: "        // defect: no guard\n\n        let ov = req.sampling_overrides();\n        let output_ids = self.backend.generate(",
        }],
        guards: &[srv_checkpoint(
            "engine::completion_admission::a_raw_prompt_over_the_cap_is_refused",
        )],
        occurrences: 1,
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "prompt-refusal-reported-as-server-error",
        symptom: "a prompt the size guard refused went out as HTTP 500 on every \
                  batch route — a server error, which the OpenAI and Anthropic \
                  SDKs retry, resending a prompt that can only be refused again; \
                  the Anthropic body even said invalid_request_error",
        revert: &[Mutation {
            path: SRV,
            find: "        400\n    } else {\n        500\n    }",
            replace: "        500 // defect: refusals reported as server errors\n    } else {\n        500\n    }",
        }],
        guards: &[srv(
            "engine::prompt_refusal_status::a_refused_prompt_goes_out_as_a_client_error",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "gemma-cached-stream-thought-channel",
        symptom: "a streaming response_format request that carried a system \
                  message — so a prefix-cache key — degenerated on Gemma 4: \
                  both cached streaming routes rendered the prompt with the \
                  thought channel open while the JSON grammar masked from token \
                  0. The gemma-thought-channel fix had reached the uncached \
                  routes only, and the token counter assumed the channel closed",
        revert: &[Mutation {
            path: MLX,
            find: "            let close_thought_channel = response_schema.is_some();",
            replace: "            let close_thought_channel = false; // defect: channel left open",
        }],
        // The end-to-end test (`engine::gemma_structured_stream`) went green
        // with the defect once sliding-layer attention changed numerics: an
        // open channel under a JSON grammar degenerates or not depending on
        // rounding. The guard checks the ids both cached routes render.
        guards: &[mlx_checkpoint(
            "gemma4_backend::imp::tests::a_cached_route_closes_the_thought_channel_for_a_schema",
        )],
        occurrences: 1, // `cached_prompt_and_prefill`, shared by both cached routes
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "temperature-zero-is-a-coin-flip",
        symptom: "the same temperature-0 request gave 2-3 different answers \
                  across runs of one Gemma 4 binary: its default repeat \
                  penalty keeps such requests off the greedy path, the sampler \
                  scaled by 1/1e-5 and drew, and an exact bf16 tie at the top \
                  was settled by an RNG seeded from the clock",
        revert: &[Mutation {
            path: CORE,
            find: "    if cfg.temperature <= 0.0 {\n        return argmax_lowest(logits);\n    }\n\n    // Temperature scaling before softmax.\n    let t = cfg.temperature;",
            replace: "    // defect: temperature 0 drawn from the RNG\n\n    // Temperature scaling before softmax.\n    let t = cfg.temperature.max(1e-5);",
        }],
        guards: &[core(
            "sampling::tests::temperature_zero_breaks_a_tie_the_same_way_on_every_seed",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "greedy-decode-drops-the-grammar",
        symptom: "on Gemma 4 a grammar was applied only by the sampled decode \
                  branch, so a greedy request (temperature 0, penalties off — \
                  `repeat_penalty: 1.0` from the client is enough) decoded with \
                  no mask: response_format and forced tool calls came back as \
                  free text",
        revert: &[Mutation {
            path: MLX,
            find: "                    .or_else(|| grammar.is_some().then(SamplingConfig::default));",
            replace: "                    ; // defect: greedy decode ignores the grammar",
        }],
        guards: &[srv_checkpoint(
            "engine::gemma_greedy_grammar::a_greedy_request_still_gets_its_schema",
        )],
        occurrences: 1,
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "gemma-boundary-prefill-unchunked",
        symptom: "a 35.8K-token Gemma 4 prompt failed: the prefix cache's \
                  boundary snapshot was prefilled in one forward, and a global \
                  layer's attention scores asked Metal for 41 GB; the snapshot \
                  had been stored before anything was evaluated, so every later \
                  request with that system prompt failed too, until restart",
        revert: &[Mutation {
            path: MLX,
            find: "                .forward_last_token_chunked(&prompt[..boundary], cache)",
            replace: "                .forward_last_token(&prompt[..boundary], cache) // defect: one pass",
        }],
        guards: &[srv_checkpoint(
            "engine::gemma_chunked_prefill::a_long_prompt_is_prefilled_in_chunks",
        )],
        occurrences: 1,
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "gemma-batch-prefill-unchunked",
        symptom: "every non-streaming Gemma 4 request prefilled its prompt in \
                  one forward — only the streaming decode loop chunked — so a \
                  long batch prompt materialized whole-prompt attention scores \
                  and failed where the same prompt streamed fine",
        revert: &[Mutation {
            path: MLX,
            find: "                self.forward_last_token_chunked(prompt_ids, cache)",
            replace: "                self.forward_last_token(prompt_ids, cache) // defect: one pass",
        }],
        guards: &[srv_checkpoint(
            "engine::gemma_chunked_prefill::a_long_prompt_is_prefilled_in_chunks",
        )],
        occurrences: 1,
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "gemma-drop-leaves-boundary-snapshot",
        symptom: "DELETE /v1/prefix-cache/{key} on Gemma 4 removed the \
                  full-prompt snapshot but not its system-boundary sibling, so \
                  the next request with the key forked the boundary right back \
                  — the route could not evict the entry a failed prefill left",
        revert: &[Mutation {
            path: MLX,
            find: "            let boundary = self.prefix_caches.remove(&Self::sys_key(key)).is_some();",
            replace: "            let boundary = false; // defect: the boundary snapshot survives",
        }],
        guards: &[srv_checkpoint(
            "engine::gemma_prefix_cache_drop::dropping_a_key_drops_its_boundary_snapshot_too",
        )],
        occurrences: 1,
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "streams-delivered-in-one-burst",
        symptom: "every streaming response, OpenAI and Anthropic, reached the \
                  client in one burst when generation finished — 80 Qwen 9B \
                  deltas all at 2,894 ms of a 2,894 ms stream, Anthropic's \
                  message_start included — so time to first token was the \
                  whole generation. The engine ran as a tokio task that never \
                  yields, and the SSE writer it woke waited on the engine's own \
                  worker until the request was done",
        revert: &[Mutation {
            path: SRV,
            find: "        std::thread::Builder::new()\n            .name(\"lumen-engine\".into())\n            .spawn(move || {\n                tokio::runtime::Builder::new_current_thread()\n                    .enable_all()\n                    .build()\n                    .expect(\"engine runtime\")\n                    .block_on(self.run(rx))\n            })?;",
            replace: "        tokio::spawn(async move { self.run(rx).await }); // defect: engine on a tokio worker",
        }],
        // An in-process engine test stayed green with the defect: the stall was
        // the server runtime's scheduling, so the guard drives the real binary.
        guards: &[srv_checkpoint_test(
            "streaming_delivery",
            "the_server_streams_tokens_while_it_generates",
        )],
        occurrences: 1,
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "gemma-batch-grammar-skips-prefix-cache",
        symptom: "every non-streaming Gemma 4 request with tools or response_format \
                  took a route that never touched the prefix cache, so an agent — \
                  which sends tools on every turn — paid a cold prefill each time \
                  (~19-20 s at 16K tokens) where the same request streamed from \
                  the cache",
        revert: &[Mutation {
            path: MLX,
            find: "                    // The grammar masks generated tokens only, so the prompt\n                    // can still come from the prefix cache.\n                    let key = session_id",
            replace: "                    // defect: grammar requests skip the prefix cache\n                    let key: Option<String> = None;\n                    let _ = session_id",
        }],
        guards: &[srv_checkpoint(
            "engine::gemma_batch_grammar_prefix_cache::a_batch_tool_request_leaves_a_snapshot_for_the_next_turn",
        )],
        occurrences: 2, // the flat route and the history route
        needs_checkpoint: true,
        extra: &["--ignored"],
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
            find: "        let ordered = if idx == buf {\n            old.clone()",
            replace: "        let ordered = if idx == buf {\n            slice_axis2(old, 0, idx.saturating_sub(1) as i32)? // defect: drops a token",
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
        // One site since both renderers share `generation_prompt_ids` (task 016).
        occurrences: 1,
        needs_checkpoint: true,
        extra: &["--ignored"],
    },
    Defect {
        name: "rotating-cache-trims-against-offset",
        symptom: "Gemma 4 prompts prefilled in chunks lost in-window context \
                  from the third chunk on. The sliding layers' rotating cache \
                  trimmed against `offset`, every token ever pushed, instead of \
                  the keys it held, so once a chunk had been trimmed the slice \
                  ran past the buffer; a ring wrapped by decode was never put \
                  back in temporal order either, and the quantized variants kept \
                  only `max_size - S` old keys. With 512-token chunks at 3,000 \
                  tokens the last-token logits fell to cosine 0.89 against mlx-lm",
        revert: &[Mutation {
            path: MLX,
            find: "        let trim = (held + 1).saturating_sub(max_size);",
            replace: "        let trim = (offset + 1).saturating_sub(max_size); // defect: trims against offset",
        }],
        guards: &[mlx(
            "native_cache::lifecycle_tests::rotating_cache_holds_what_mlx_lm_holds",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &["--ignored"],
    },
    Defect {
        name: "windowed-kernel-used-unchecked",
        symptom: "the windowed steel kernel is on by default, and mlx-c fetches \
                  the MLX fork by branch: a build cached from before \
                  rabbitson87/mlx 8a2587df / 23b42543 keeps a kernel that read \
                  the wrong K/V blocks, and Gemma 4 long prompts came back as \
                  garbage with nothing logged. The load-time self-check compares \
                  the kernel with explicit-mask attention first and falls back \
                  (it caught a second, causal-mask bug in the fork this way)",
        revert: &[Mutation {
            path: MLX,
            find: "            if worst.is_nan() || worst > 0.05 {",
            replace: "            if false && worst > 0.0 { // defect: kernel trusted unchecked",
        }],
        guards: &[mlx(
            "gemma4_moe::imp::tests::the_windowed_kernel_self_check_tells_right_from_wrong",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &["--ignored"],
    },
    Defect {
        name: "fastokens-split-cache-reads-past-the-prefix",
        symptom: "fastokens 0.3.2 as published gave the same string different ids \
                  depending on what the thread encoded before: on every Qwen \
                  tokenizer, 5000 bytes of text plus 700 spaces split the run as \
                  59 + 2 tokens right after the same text with an `x` after the \
                  spaces, where a fresh encode and HF give one. Its split cache \
                  reused regex matches that bytes past the shared prefix decide \
                  (`\\s+(?!\\S)` gives a space back when a non-space follows). \
                  Task 016's parity run caught it before lumen encoded with it",
        revert: &[Mutation {
            path: FASTOKENS,
            find: "                let limit = reuse_limit(bytes, common_len);\n                let reuse_count = cache.prev_matches.partition_point(|&(_, end)| end <= limit);",
            replace: "                // defect: upstream's reuse condition\n                let reuse_count = cache.prev_matches.partition_point(|&(_, end)| end < common_len);",
        }],
        guards: &[core_mlx_lib(
            "text_tokenizer::tests::fastokens_ids_do_not_depend_on_the_previous_call",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "fastokens-nfc-newer-unicode",
        symptom: "with LUMEN_FASTOKENS on, text holding a combining mark newer \
                  than Unicode 9 encoded to other ids than HF's: `a`, U+1DF6, \
                  U+0301 became `á`, U+1DF6 under fastokens' ICU normalizer and \
                  stayed as written under HF's Unicode 9 tables. Task 016's \
                  code-point sweep showed four such marks; the tables differ on \
                  193 characters, and text holding one goes to HF",
        revert: &[Mutation {
            path: MLX,
            find: ".filter(|_| fastokens_encode::get() && !normalization_differs(text));",
            replace: ".filter(|_| fastokens_encode::get()); // defect: no normalization fallback",
        }],
        guards: &[core_mlx_lib(
            "text_tokenizer::tests::text_the_normalizers_disagree_on_is_encoded_by_hf",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "chat-turn-header-defeats-session-reuse",
        symptom: "plain multi-turn chat on Qwen 3.5 / 3.6 never reused its \
                  session: a turn ends in a generation header \
                  (`<|im_start|>assistant\\n<think>…`) that the next prompt does \
                  not reproduce — the template drops the `<think>` block from \
                  replayed assistant turns, and on 3.8 so does any client that \
                  does not return the trace — so the session's tokens were never \
                  a prefix of the next prompt and every turn prefilled the whole \
                  conversation again. Measured on Qwen3.5-9B with an 11.5K-token \
                  system prompt: ~25 s a turn, on main too. Fixed by leaving a \
                  rollback point at the conversation boundary and resuming from \
                  it: turn two 26.0 s -> 0.4 s, replies byte-identical",
        revert: &[Mutation {
            path: MLX,
            find: "        let kept = self.rollback_len?;",
            replace: "        let kept = self.rollback_len.filter(|_| false)?; // defect: no resume",
        }],
        guards: &[
            core_mlx_lib("tests::the_next_chat_turn_resumes_instead_of_prefilling_again"),
            core_mlx_lib("tests::a_session_is_found_again_through_its_rollback_point"),
        ],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "rollback-cut-inside-a-moe-chunk",
        symptom: "on a mixture-of-experts model a rollback point cut inside a \
                  prefill chunk changes the answer: MLX's GatherQMM sorts the \
                  routed rows by expert and tiles across them, so a row's result \
                  depends on which rows share its call. Measured on \
                  Qwen3.6-35B-A3B: a 32-row piece changed a greedy reply after \
                  ~30 words, and a turn resumed from a 128-row-floored cut still \
                  differed from a cold prefill. Such models mark only on chunk \
                  boundaries, where the pieces are the bulk pass's own chunks",
        revert: &[Mutation {
            path: MLX,
            find: "            let inside = grid.min_piece.map(|floor| {",
            replace: "            let inside = grid.min_piece.or(Some(MIN_BULK_PIECE)).map(|floor| { // defect",
        }],
        guards: &[core_mlx_lib(
            "session_feed::tests::a_mixture_of_experts_marks_only_on_chunk_boundaries",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "rollback-cut-leaves-a-short-piece",
        symptom: "cutting the prefill right at the conversation boundary leaves \
                  the generation header — ten rows on Qwen — as a piece of its \
                  own, and below ~32 rows MLX computes quantized projections with \
                  its vector kernel (`get_qmv_batch_limit`), which sums in a \
                  different order from the bulk chunk those rows sat in: the turn \
                  that places the point is no longer the turn without it. The \
                  planner keeps every piece cut inside a chunk at 32+ rows",
        revert: &[Mutation {
            path: MLX,
            find: "                let at = boundary.min(cell_end.saturating_sub(floor));",
            replace: "                let at = boundary; // defect: cuts at the header",
        }],
        guards: &[core_mlx_lib(
            "session_feed::tests::a_cut_never_changes_which_kernels_a_row_goes_through",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "bundled-server-writes-its-metallib-into-the-app",
        symptom: "the release app's server could not start on a user's Mac: it \
                  unpacked mlx.metallib next to itself, inside the signed \
                  Lumen.app/Contents/MacOS, and when the app ran from its \
                  read-only DMG (or a Gatekeeper-translocated copy) the write \
                  failed and MLX aborted with \"Failed to load the default \
                  metallib\" (exit 255) — on the build machine MLX's compiled-in \
                  path to the build tree hid it. The bundle now ships the library \
                  in Contents/Resources, the MLX fork looks there (97685a09), and \
                  the server leaves the bundle alone when that copy is its own",
        revert: &[Mutation {
            path: SRV,
            find: "        (None, Some(b)) if b == embedded => Plan::UseBundled,",
            replace: "        (None, Some(b)) if b == embedded && false => Plan::UseBundled, // defect",
        }],
        guards: &[srv_test(
            "packaging",
            "a_bundled_library_is_used_without_writing_into_the_bundle",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "release-bundle-ships-no-metallib",
        symptom: "the release workflow bundled the lumen-server sidecar without \
                  MLX's kernel library, so the server had to write it into the \
                  signed app at run time — and could not on a read-only volume. \
                  The workflow stages the server's own copy \
                  (`--write-metallib`) and maps it into Contents/Resources",
        revert: &[Mutation {
            path: WORKFLOWS,
            find: r#","resources":{"binaries/mlx.metallib":"mlx.metallib"}"#,
            replace: r#","resources":{}"#,
        }],
        guards: &[srv_test(
            "packaging",
            "the_release_workflow_ships_the_library_where_mlx_looks",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "bundle-claims-an-older-macos-than-its-kernels",
        symptom: "the release app declared macOS 11.0 while its MLX kernel \
                  library was compiled for the CI runner's SDK \
                  (air64-apple-macosx26.5 on the last build): MLX passes no \
                  deployment target, so `metal` defaulted to the SDK, and a Metal \
                  library does not load on an older macOS than it was built for. \
                  The release job now pins MACOSX_DEPLOYMENT_TARGET=14.0 (MLX's \
                  own minimum) and the bundle declares the same floor",
        revert: &[Mutation {
            path: APP,
            find: "\"minimumSystemVersion\": \"14.0\",",
            replace: "\"minimumSystemVersion\": \"11.0\",",
        }],
        guards: &[srv_test(
            "packaging",
            "the_bundle_declares_the_macos_its_kernels_were_built_for",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "server-build-races-mlx-for-the-metallib",
        symptom: "a fresh `cargo build -p lumen-server` failed: its build.rs \
                  searched the target directory for mlx.metallib while mlx-sys \
                  was still running CMake, found nothing, and panicked — Cargo \
                  orders a build script after a dependency's only when that \
                  dependency declares `links` and is a direct dependency. CI \
                  pre-built mlx-sys to get around it, as a differently-featured \
                  unit, so MLX compiled twice. mlx-sys now declares \
                  `links = \"mlx\"` (rabbitson87/mlx-rs cee2f18a) and the server \
                  depends on it directly, receiving DEP_MLX_METALLIB",
        revert: &[Mutation {
            path: SRV_CRATE,
            find: "    \"dep:mlx-sys\",",
            replace: "    # defect: mlx-sys not a direct dependency",
        }],
        guards: &[srv_test(
            "packaging",
            "the_server_depends_on_mlx_sys_directly_at_lumen_mlx_s_rev",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "release-requires-notarization",
        symptom: "the release job handed the notary credentials to the build on \
                  every run, so with the Apple Developer Program membership \
                  lapsed it failed at notarization (401) after building and \
                  signing — no release at all. Notarization is now opt-in \
                  (repository variable MACOS_NOTARIZE)",
        revert: &[Mutation {
            path: WORKFLOWS,
            find: "          # The Apple signing / notarization variables come from the step above.",
            replace: "          APPLE_ID: ${{ secrets.APPLE_ID }} # defect: always notarize",
        }],
        guards: &[srv_test(
            "packaging",
            "the_release_builds_without_notarization_unless_asked",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
    },
    Defect {
        name: "unsigned-bundle-reads-as-damaged",
        symptom: "built without a Developer ID certificate, the app carried only \
                  the linker's ad-hoc signature on its executables, which claims a \
                  bundle seal that was never written (`codesign --verify --strict`: \
                  'code has no resources but signature indicates they must be \
                  present'); macOS reports such a download as damaged, with no \
                  Open Anyway. The bundle is now signed ad-hoc by default",
        revert: &[Mutation {
            path: APP,
            find: "\"signingIdentity\": \"-\"",
            replace: "\"signingIdentity\": null",
        }],
        guards: &[srv_test(
            "packaging",
            "a_build_without_a_certificate_is_still_validly_signed",
        )],
        occurrences: 1,
        needs_checkpoint: false,
        extra: &[],
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
        | (_, "tool-history-uncounted")
        | (_, "qwen-sampling-discarded")
        | (_, "mtp-drops-sampling-knobs")
        | (_, "replay-drops-think-block")
        | (_, "session-reuse-needs-a-nonstandard-field")
        | (_, "gemma-batch-grammar-skips-prefix-cache") => "lib.rs",
        // One defect, two files: the wire had no field for the trace *and* the
        // renderer had nowhere to put one. Reverting either half is enough to
        // lose the KV, so both are mutated together.
        (SRV, "reasoning-trace-has-no-wire-field") => "types.rs",
        (MLX, "reasoning-trace-has-no-wire-field") => "lib.rs",
        (_, "anthropic-thinking-block-rejected") | (_, "inference-error-drops-its-cause") => {
            "types.rs"
        }
        (_, "tools-replay-doubles-think-block")
        | (_, "qwen-plain-path-never-split-reasoning")
        | (_, "qwen-thinking-trace-served-as-the-answer")
        | (_, "qwen-tool-stream-drops-the-trace")
        | (_, "stray-think-close-reaches-the-answer") => "qwen3_5_tools.rs",
        (_, "anthropic-output-drops-thinking-block") => "engine.rs",
        (_, "anthropic-stream-block-indices-pinned") => "routes/messages.rs",
        (_, "gemma-thought-channel") => "gemma4_chat.rs",
        (_, "gemma-cached-stream-thought-channel") | (_, "greedy-decode-drops-the-grammar") => {
            "gemma4_backend.rs"
        }
        (_, "temperature-zero-is-a-coin-flip") => "sampling.rs",
        (_, "gemma-boundary-prefill-unchunked") | (_, "gemma-drop-leaves-boundary-snapshot") => {
            "gemma4_backend.rs"
        }
        (_, "gemma-batch-prefill-unchunked") => "gemma4_moe.rs",
        (_, "causal-mask-coverage") | (_, "causal-mask-builders-agree") => "native_attention.rs",
        (_, "rotating-cache-both-paths") => "native_cache.rs",
        (_, "flux-scheduler-invariants") => "scheduler.rs",
        (_, "flux-left-padding") => "tokenizer.rs",
        (_, "tool-choice-none")
        | (_, "anthropic-turn-images")
        | (_, "anthropic-batch-guard-omits-images")
        | (_, "completions-unguarded")
        | (_, "streams-delivered-in-one-burst")
        | (_, "undeclared-tool-name-forwarded") => "engine.rs",
        (_, "server-binds-every-interface")
        | (_, "api-key-not-enforced")
        | (_, "cors-setting-ignored") => "access.rs",
        (_, "prompt-refusal-reported-as-server-error") => "types.rs",
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
        (_, "fastokens-split-cache-reads-past-the-prefix") => "split.rs",
        (_, "fastokens-nfc-newer-unicode") => "text_tokenizer.rs",
        (_, "rotating-cache-trims-against-offset") => "native_cache.rs",
        (_, "windowed-kernel-used-unchecked") => "gemma4_moe.rs",
        (_, "chat-turn-header-defeats-session-reuse") => "lib.rs",
        (_, "rollback-cut-inside-a-moe-chunk") | (_, "rollback-cut-leaves-a-short-piece") => {
            "session_feed.rs"
        }
        (_, "bundled-server-writes-its-metallib-into-the-app") => "metallib.rs",
        (_, "release-bundle-ships-no-metallib") => "release.yml",
        (_, "bundle-claims-an-older-macos-than-its-kernels") => "tauri.conf.json",
        (_, "server-build-races-mlx-for-the-metallib") => "Cargo.toml",
        (_, "release-requires-notarization") => "release.yml",
        (_, "unsigned-bundle-reads-as-damaged") => "tauri.conf.json",
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
