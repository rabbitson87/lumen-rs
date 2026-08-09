#!/usr/bin/env python3
"""Red/green harness for the tool-grammar and tool-call-parsing regressions.

Every defect fixed on this branch has guards in the test suite. This proves the
guards actually guard: for each defect it reverts the fix in place, asserts the
named tests turn RED, restores the source, and asserts they are GREEN again.

A test that passes both before and after a fix proves nothing, and that is the
easy mistake to make when writing a regression test after the fact. This is the
check for it.

Each entry records the symptom the defect produced in production, so the table
doubles as the changelog's evidence.

    python3 tools/red_green.py                 # all defects
    python3 tools/red_green.py --list          # names only
    python3 tools/red_green.py lark-opener     # one defect

Defects whose guard needs a checkpoint are skipped unless
LUMEN_GEMMA4_MODEL_DIR is set; they are reported as SKIP, never as PASS.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLX = "crates/lumen-mlx/src"
SRV = "crates/lumen-server/src"


@dataclass
class Defect:
    """One fixed defect, the symptom it caused, and how to bring it back."""

    name: str
    symptom: str
    # (path, find, replace) — `find` must occur exactly once.
    revert: list[tuple[str, str, str]]
    # Tests that must fail once reverted. Substring filters for `cargo test`.
    guards: list[str]
    package: str = "lumen-mlx"
    # How many sites the fix touches. A fix applied at two render paths is not
    # a moved anchor, but anything OTHER than this count is.
    occurrences: int = 1
    needs_checkpoint: bool = False
    extra_args: list[str] = field(default_factory=list)


DEFECTS: list[Defect] = [
    Defect(
        name="lark-opener",
        symptom="every streaming tool call died: `byte 'ÿ' fails parse`; and "
        "tool_choice=required silently fell back to free sampling",
        revert=[
            (
                f"{MLX}/grammar.rs",
                """        self.lazy_trigger
            .is_some_and(|t| t.token == token && !t.in_grammar)""",
                """        let _ = token;
        false""",
            )
        ],
        guards=[
            "grammar::tests::lazy_activation_does_not_feed_the_trigger_to_the_matcher",
            "grammar::tests::lazy_activation_leaves_the_matcher_at_the_start_of_the_body",
            "grammar::tests::eager_prefill_replay_skips_the_opener_and_parses_the_rest",
        ],
    ),
    Defect(
        name="json-whitespace",
        symptom="response_format replies were pure indentation up to max_tokens",
        revert=[
            (
                f"{MLX}/grammar.rs",
                '"whitespace_flexible": false,',
                '"whitespace_flexible": true,',
            )
        ],
        guards=["grammar::tests::response_format_grammar_forbids_whitespace_runs"],
    ),
    Defect(
        name="json-separator-space",
        symptom='the `": "` residue leaked into string values: '
        '{"city": ": way more than you\'ve gotten…"}',
        revert=[
            (
                f"{MLX}/grammar.rs",
                '"key_separator": ": ",\n                "item_separator": ", ",',
                '"key_separator": ":",\n                "item_separator": ",",',
            )
        ],
        guards=["grammar::tests::response_format_grammar_keeps_the_separator_space"],
    ),
    Defect(
        name="grammar-rule-names",
        symptom="a tool named `날씨_조회` was refused, the grammar dropped, and the "
        "model invented `weather_lookup` — a tool nobody declared",
        revert=[
            (
                f"{MLX}/grammar.rs",
                'let body_rule_name = format!("tool_{i}_body");',
                'let body_rule_name = format!("tool_{name}_body");',
            )
        ],
        guards=["grammar::tests::lark_grammar_escapes_a_non_identifier_tool_name"],
    ),
    Defect(
        name="grammar-literal-escaping",
        symptom="a quote inside a tool name closed the Lark literal",
        revert=[
            (
                f"{MLX}/grammar.rs",
                """            '\\\\' => out.push_str("\\\\\\\\"),
            '"' => out.push_str("\\\\\\""),""",
                """            '\\\\' => out.push('\\\\'),
            '"' => out.push('"'),""",
            )
        ],
        guards=["grammar::tests::lark_grammar_escapes_a_quote_in_a_tool_name"],
    ),
    Defect(
        name="tool-name-scanner",
        symptom="`call:bad call:good{x:1}` parsed as ONE tool named "
        '"bad call:good" — a name no client declared',
        revert=[
            (
                f"{MLX}/gemma4_response.rs",
                """                if bytes[brace_start..].starts_with(b"call:") {
                    hit_next_call = true;
                    break;
                }
""",
                "                // red_green.py: next-call boundary removed\n",
            )
        ],
        guards=[
            "gemma4_response::imp::tests::body_parser_stops_a_name_at_the_next_opener",
            "gemma4_response::imp::tests::body_parser_skips_a_run_of_malformed_openers",
            "gemma4_response::imp::tests::body_parser_stops_a_non_ascii_name_at_the_next_opener",
        ],
    ),
    Defect(
        name="args-unicode-keys",
        symptom="`{도시:…}` arrived as `{Ã«Â\\u{8f}Â\\u{84}…}` and failed to parse",
        revert=[
            (
                f"{MLX}/gemma4_response.rs",
                ".find(|c: char| !(c.is_alphanumeric() || c == '_' || c == '-'))",
                ".find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == '-'))",
            )
        ],
        guards=[
            "gemma4_response::imp::tests::args_to_json_quotes_a_non_ascii_bare_key",
            "gemma4_response::imp::tests::args_to_json_handles_non_ascii_in_nested_and_array_positions",
        ],
    ),
    Defect(
        name="gemma-nonstreaming-grammar",
        symptom="non-streaming tool_choice=required returned `迎get_weather`",
        revert=[
            (
                f"{MLX}/lib.rs",
                """    !tools.is_empty()
        && !matches!(tool_choice, crate::chat_io::ResolvedToolChoice::None)
        && crate::gemma4_backend::imp::gemma4_grammar_lark_enabled()""",
                """    let _ = (tools, tool_choice);
    false""",
            )
        ],
        guards=[
            "grammar_routing_regressions::tools_alone_must_route_through_the_grammar_aware_decode"
        ],
    ),
    Defect(
        name="qwen-first-token-mask",
        symptom="tool_choice=required never enforced: the first generated token "
        "was argmaxed unmasked, and disagreement dropped the grammar",
        revert=[
            (
                f"{MLX}/lib.rs",
                "    grammar_active && prompt_ids.len() > 1 && image_token != prompt_ids.last().copied()",
                "    let _ = (grammar_active, prompt_ids, image_token);\n    false",
            )
        ],
        guards=[
            "grammar_routing_regressions::an_active_grammar_holds_the_last_prompt_token_back"
        ],
    ),
    Defect(
        name="tool-choice-none",
        symptom='tool_choice="none" was accepted and ignored; Qwen 3.6 called the '
        "tool anyway",
        revert=[
            (
                f"{SRV}/engine.rs",
                """    if matches!(tool_choice, ResolvedToolChoice::None) {
        Vec::new()
    } else {
        tools
    }""",
                """    let _ = tool_choice;
    tools""",
            )
        ],
        guards=["engine::tool_choice_none_withholds_tools::none_hides_them"],
        package="lumen-server",
    ),
    Defect(
        name="anthropic-turn-images",
        symptom="on /v1/messages a tool_result expanded one message into several "
        "turns, so every later image bound to the wrong turn",
        revert=[
            (
                f"{SRV}/engine.rs",
                """                for _ in 0..tool_result_counts.get(i).copied().unwrap_or(0) {
                    out.push(Vec::new());
                }
""",
                "                // red_green.py: tool-turn rows removed\n",
            )
        ],
        guards=[
            "engine::anthropic_turn_image_alignment::tool_results_expand_one_message_into_several_turns",
            "engine::anthropic_turn_image_alignment::a_textless_imageless_message_emits_no_turn",
        ],
        package="lumen-server",
    ),
    Defect(
        name="gemma-thought-channel",
        symptom="response_format degenerated into repetition to max_tokens — the "
        "eager grammar masked `<|channel>` at step 0, leaving 3 legal tokens",
        revert=[
            (
                f"{MLX}/gemma4_chat.rs",
                "if !opts.enable_thinking\n                    && (opts.close_thought_channel || empty_thought_on_nothink())",
                "if !opts.enable_thinking && empty_thought_on_nothink()",
            )
        ],
        guards=["gemma4_chat::imp::tests::close_thought_channel_prefills_the_empty_block"],
        occurrences=2,  # the flat renderer and the history renderer
        needs_checkpoint=True,
        extra_args=["--ignored"],
    ),
]


def run_tests(defect: Defect) -> bool:
    """True when every guard passes."""
    for guard in defect.guards:
        cmd = ["cargo", "test", "-p", defect.package, "--features", "mlx-native"]
        if defect.package == "lumen-mlx":
            cmd.append("--lib")  # lumen-server is a binary crate: no lib target
        cmd += [guard, "--", "--test-threads=1", "--exact"] + defect.extra_args
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        # `--exact` with a full path; a guard that matched nothing is a broken
        # entry in this table, not a pass.
        if "running 1 test" not in r.stdout:
            print(f"      !! guard did not match exactly one test: {guard}")
            return False
        if r.returncode != 0:
            return False
    return True


def apply(defect: Defect, forward: bool) -> None:
    for rel, find, replace in defect.revert:
        if not find.strip() or not replace.strip():
            raise SystemExit(
                f"{defect.name}: both sides of a mutation must be non-empty, or the "
                "reverse direction searches for the empty string. Use a sentinel "
                "comment instead of deleting."
            )
        p = ROOT / rel
        text = p.read_text(encoding="utf-8")
        a, b = (find, replace) if forward else (replace, find)
        n = text.count(a)
        if n != defect.occurrences:
            raise SystemExit(
                f"{defect.name}: expected {defect.occurrences} occurrence(s) in "
                f"{rel}, found {n}.\n"
                f"The source moved — update tools/red_green.py.\n---\n{a}\n---"
            )
        p.write_text(text.replace(a, b), encoding="utf-8")


def check(defect: Defect) -> str:
    if defect.needs_checkpoint and "LUMEN_GEMMA4_MODEL_DIR" not in __import__("os").environ:
        return "SKIP"
    if not run_tests(defect):
        return "BROKEN"  # red before we touched anything
    apply(defect, forward=True)
    try:
        went_red = not run_tests(defect)
    finally:
        apply(defect, forward=False)
    if not run_tests(defect):
        raise SystemExit(f"{defect.name}: source not restored cleanly — fix by hand")
    return "PASS" if went_red else "NO-OP"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="defects to check (default: all)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for d in DEFECTS:
            print(f"{d.name:<28} {d.symptom}")
        return 0

    todo = [d for d in DEFECTS if not args.names or d.name in args.names]
    if not todo:
        print(f"no such defect; try --list")
        return 2

    results = []
    for d in todo:
        print(f"→ {d.name}", flush=True)
        verdict = check(d)
        results.append((verdict, d))
        print(f"  {verdict}", flush=True)

    print("\n" + "=" * 78)
    width = max(len(d.name) for _, d in results)
    bad = 0
    for verdict, d in results:
        mark = {"PASS": "red→green", "NO-OP": "GUARD IS VACUOUS",
                "SKIP": "skipped (no checkpoint)", "BROKEN": "ALREADY RED"}[verdict]
        print(f"{d.name:<{width}}  {mark}")
        if verdict in ("NO-OP", "BROKEN"):
            bad += 1
    ok = sum(1 for v, _ in results if v == "PASS")
    skip = sum(1 for v, _ in results if v == "SKIP")
    print(f"\n{ok}/{len(results)} verified red→green"
          + (f", {skip} skipped" if skip else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
