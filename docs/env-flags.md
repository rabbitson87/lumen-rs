# Env flags

GENERATED — do not edit. Regenerate with `cargo xtask flags --docs`;
`cargo xtask flags --check` fails when this file is stale.

Source of truth is the `lumen_flags::flag!` declaration next to the code
each flag gates; this file is its projection. Parse rule for every flag:
unset → default, `0` / `false` / `off` / `no` (any case) → off, any other
value → on.

| Env | Default | Kind | Declared in |
|---|---|---|---|
| `LUMEN_CONTENT_AS_TEXT` | on | Behavior | `lumen_mlx::text_tokenizer::content_as_text` |
| `LUMEN_FASTOKENS` | off | Optimization | `lumen_mlx::text_tokenizer::fastokens_encode` |
| `LUMEN_GEMMA4_FUSE_DENSE_MLP` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_dense_mlp` |
| `LUMEN_GEMMA4_FUSE_EXPERTS` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_experts` |
| `LUMEN_GEMMA4_FUSE_LAYER_EPILOGUE` | off | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_layer_epilogue` |
| `LUMEN_GEMMA4_FUSE_NORM_ROUTING_EXPERTS` | off | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_norm_routing_experts` |
| `LUMEN_GEMMA4_FUSE_PRE_POST_NORM_DENSE_MLP` | off | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_pre_post_norm_dense_mlp` |
| `LUMEN_GEMMA4_FUSE_ROUTER` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_router` |
| `LUMEN_GEMMA4_FUSE_ROUTING_EXPERTS` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_routing_experts` |
| `LUMEN_GEMMA4_FUSE_SOFTCAP` | on | Optimization | `lumen_mlx::gemma4_moe::imp::fuse_softcap` |
| `LUMEN_GEMMA4_PREFILL_CHUNK_LOG` | off | Diagnostic | `lumen_mlx::gemma4_moe::imp::prefill_chunk_log` |
| `LUMEN_GEMMA4_QUANT_KV_FUSED_ATTN` | on | Behavior | `lumen_mlx::gemma4_moe::imp::quant_kv_fused_attn` |
| `LUMEN_GEMMA4_QUANT_KV_PREFILL_DEQUANT` | on | Behavior | `lumen_mlx::gemma4_moe::imp::quant_kv_prefill_dequant` |
| `LUMEN_JOINT_PREFILL` | on | Optimization | `lumen_mlx::joint_prefill` |
| `LUMEN_KV_DISK` | on | Optimization | `lumen_mlx::kv_disk::kv_disk_tier` |
| `LUMEN_MEMORY_PLAN_RECLAIMABLE` | on | Behavior | `lumen_mlx::qwen4_exp::runner::plan_reclaimable` |
| `LUMEN_MLX_AUTO_SESSION` | on | Behavior | `lumen_mlx::session::auto_session_enabled` |
| `LUMEN_MLX_KV_BF16` | on | Behavior | `lumen_mlx::qwen3_5_moe::imp::kv_bf16` |
| `LUMEN_MLX_MEMORY_EVICT` | on | Behavior | `lumen_mlx::session::memory_evict` |
| `LUMEN_MLX_NO_OVERLAP` | off | Optimization | `lumen_mlx::gemma4_backend::imp::no_overlap` |
| `LUMEN_MLX_SESSION_ROLLBACK` | on | Optimization | `lumen_mlx::session::session_rollback_enabled` |
| `LUMEN_MOE_EVICTION` | off | Optimization | `lumen_mlx::offload::slot_policy::moe_eviction` |
| `LUMEN_MOE_IO_THROTTLE_GBPS` | off | Diagnostic | `lumen_mlx::offload::io::io_throttle` |
| `LUMEN_MOE_OVERLAP` | off | Optimization | `lumen_mlx::offload::overlap` |
| `LUMEN_MOE_PROBE_LAYER_SYNC` | off | Diagnostic | `lumen_mlx::native_moe::imp::moe_layer_sync_probe` |
| `LUMEN_MOE_TRACE` | off | Diagnostic | `lumen_mlx::offload::trace::moe_trace` |
| `LUMEN_NATIVE_ACT_BF16` | off | Behavior | `lumen_mlx::qwen3_5_moe::imp::act_bf16` |
| `LUMEN_NATIVE_ALLOC_REUSE` | on | Optimization | `lumen_mlx::qwen3_5_moe::imp::alloc_reuse` |
| `LUMEN_NATIVE_CACHED_STREAM` | off | Optimization | `lumen_mlx::native_quant::imp::cached_stream` |
| `LUMEN_NATIVE_COMPILE` | on | Behavior | `lumen_mlx::native_ssm::imp::ssm_compile` |
| `LUMEN_NATIVE_COMPILE_ROUTING` | off | Optimization | `lumen_mlx::native_moe::imp::routing_compile` |
| `LUMEN_NATIVE_CONV_SLICE` | off | Optimization | `lumen_mlx::qwen3_5_moe::imp::conv_slice` |
| `LUMEN_NATIVE_DECODE_QMV` | on | Optimization | `lumen_mlx::native_qmv::decode_qmv` |
| `LUMEN_NATIVE_DECODE_QMV_ROWS` | on | Behavior | `lumen_mlx::native_qmv::decode_qmv_rows` |
| `LUMEN_NATIVE_FUSED_MOE_DECODE` | on | Optimization | `lumen_mlx::qwen3_5_moe::imp::fused_moe_decode` |
| `LUMEN_NATIVE_FUSED_MOE_ROUTER` | on | Optimization | `lumen_mlx::native_moe_decode::fused_router` |
| `LUMEN_NATIVE_FUSE_GATE_UP` | off | Optimization | `lumen_mlx::qwen3_5_moe::imp::fuse_gate_up_flag` |
| `LUMEN_NATIVE_FUSE_LINATTN_IN` | on | Optimization | `lumen_mlx::qwen3_5_moe::imp::fuse_linattn_in` |
| `LUMEN_NATIVE_FUSE_MOE_GATE_UP` | off | Optimization | `lumen_mlx::qwen3_5_moe::imp::fuse_moe_gate_up` |
| `LUMEN_NATIVE_FUSE_SIGMOID_MUL` | off | Optimization | `lumen_mlx::native_kernels::imp::sigmoid_mul_fuse` |
| `LUMEN_NATIVE_FUSE_SWIGLU` | on | Optimization | `lumen_mlx::native_moe::imp::swiglu_fuse` |
| `LUMEN_NATIVE_KV_STEP_PREALLOC` | on | Optimization | `lumen_mlx::native_cache::imp::kv_step_prealloc` |
| `LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE` | on | Optimization | `lumen_mlx::qwen3_5_moe::imp::linear_attn_scale_fuse` |
| `LUMEN_NATIVE_PIPELINE_DECODE` | on | Optimization | `lumen_mlx::runner_native::imp::pipeline_decode` |
| `LUMEN_NATIVE_RMS_NORM_GATED_FUSED` | off | Optimization | `lumen_mlx::native_ssm::imp::rms_norm_gated_fused` |
| `LUMEN_NATIVE_TIMING` | off | Diagnostic | `lumen_mlx::native_runtime::imp::fine_timing` |
| `LUMEN_NATIVE_WIDEN_SCALES` | on | Optimization | `lumen_mlx::qwen3_5_moe::imp::widen_scales` |
| `LUMEN_PREFIX_EXACT_REUSE` | on | Behavior | `lumen_mlx::prefix_cache::exact_reuse` |
| `LUMEN_QWEN35_QUANT_KV` | off | Behavior | `lumen_mlx::qwen3_5_moe::imp::quant_kv` |
| `LUMEN_QWEN35_REASONING_EFFORT` | on | Behavior | `lumen_mlx::reasoning_effort_enabled` |
| `LUMEN_QWEN4_BUILD_AHEAD` | off | Optimization | `lumen_mlx::qwen4_exp::model::build_ahead` |
| `LUMEN_QWEN4_KV_BF16` | on | Behavior | `lumen_mlx::qwen4_exp::config::kv_bf16` |
| `LUMEN_QWEN4_MEM_LOG` | off | Diagnostic | `lumen_mlx::qwen4_exp::runner::mem_log` |
| `LUMEN_QWEN4_STAGE_SYNC` | off | Diagnostic | `lumen_mlx::decode_probe::stage_sync` |
| `LUMEN_QWEN4_TOKEN_COUNTERS` | off | Diagnostic | `lumen_mlx::decode_probe::token_counters` |
| `LUMEN_QWEN4_WIDEN_SCALES` | on | Optimization | `lumen_mlx::qwen4_exp::model::widen_scales` |
| `LUMEN_SPARSE_DECODE` | off | Behavior | `lumen_mlx::native_attention::imp::sparse_decode` |
| `LUMEN_TOKENIZE_MEMO` | on | Optimization | `lumen_mlx::text_tokenizer::tokenize_memo` |

## Details

### `LUMEN_CONTENT_AS_TEXT`

*Behavior, default on.*

Encode what a client sent — messages, tool results, tool arguments and
 definitions — as ordinary text, even where it spells an added token
 such as `<|im_end|>` or `<tool_call>`; only the chat template's own
 markers become control tokens. Off, a tool result holding
 `<|im_end|>\n<|im_start|>system\n…` ends the turn and opens a system
 turn of its own: HF matches added tokens inside raw text, and its
 `encode_special_tokens` leaves the ones not marked special matched
 (Qwen's `<think>`, `<tool_call>`, `<tool_response>`). A prompt whose
 client text spells no added token encodes to the same ids either way.
 Covers lumen's own renderers and Gemma's opt-in upstream-template path
 (`LUMEN_USE_JINJA_RENDERER=1`), where the template sees the client's
 spellings as plain text and so no longer acts on them; a raw
 `/v1/completions` prompt is the client's to write.

### `LUMEN_FASTOKENS`

*Optimization, default off.*

Encode with the vendored fastokens (`vendor/fastokens`) instead of HF
 `tokenizers`: BPE only, 10-16x faster on long prompts. Exact by
 construction of the check, not by trust: the engine is built at load
 only if it returns HF's ids for every probe string, with and without
 special tokens, and text the two would normalize differently (193
 characters newer than HF's Unicode 9 tables) or a per-call error goes
 to HF. Decoding stays on HF. Off by default: with the memo on, a warm
 turn encodes in ~1 ms, so what it saves is one cold encode — 15-20 ms
 at 30K tokens, 60-95 ms at 107K — against a prefill of seconds, and
 loading the tokenizer takes 0.3-0.9 s longer.

### `LUMEN_GEMMA4_FUSE_DENSE_MLP`

*Optimization, default on.*

Fuse the dense-MLP chain. Output-identical.

### `LUMEN_GEMMA4_FUSE_EXPERTS`

*Optimization, default on.*

Fuse the expert gather + GLU. Output-identical.

### `LUMEN_GEMMA4_FUSE_LAYER_EPILOGUE`

*Optimization, default off.*

Fuse the per-layer epilogue. Default OFF — opt-in; land path is enable,
 measure, promote if positive.

### `LUMEN_GEMMA4_FUSE_NORM_ROUTING_EXPERTS`

*Optimization, default off.*

Fold the pre/post norms into the fused routing+experts path. Default OFF —
 opt-in; land path is enable, measure, promote if positive.

### `LUMEN_GEMMA4_FUSE_PRE_POST_NORM_DENSE_MLP`

*Optimization, default off.*

Fold the pre/post RMSNorm into the dense-MLP fusion. Default OFF — opt-in;
 land path is enable, measure, promote if positive.

### `LUMEN_GEMMA4_FUSE_ROUTER`

*Optimization, default on.*

Fuse the MoE router into one dispatch. Output-identical.

### `LUMEN_GEMMA4_FUSE_ROUTING_EXPERTS`

*Optimization, default on.*

Fuse routing into the expert dispatch, replacing the two-slot
 routing_tail + experts path. LANDED 2026-05-17, bit-identical with the
 baseline (greedy tokens match exactly); 3-pair cool-state A/B at 8K
 showed +2.2% throughput (53.35 +/- 0.5 -> 54.5 +/- 0.7 tok/s).

### `LUMEN_GEMMA4_FUSE_SOFTCAP`

*Optimization, default on.*

Fuse the attention logit softcap. Output-identical.

### `LUMEN_GEMMA4_PREFILL_CHUNK_LOG`

*Diagnostic, default off.*

Log each chunked-prefill chunk's wall time and MLX memory (active,
 peak) to stderr, as `LUMEN_QWEN35_PREFILL_CHUNK_LOG` does for Qwen.
 Chunk time against cache offset is how much of prefill attention
 costs as the context grows. The last chunk is evaluated too, so the
 returned logits are no longer lazy; every chunk already ends in an
 eval, so no barrier is added inside a forward.

### `LUMEN_GEMMA4_QUANT_KV_FUSED_ATTN`

*Behavior, default on.*

Decode over a quantized KV cache (single-token steps) through the
 fused kernel in `native_quant_attention`, which dequantizes inside
 its loop, instead of `quantized_matmul` · softmax ·
 `quantized_matmul`. Falls back to the three-op path for layouts the
 kernel does not read.

 Default on (task 021): 2.3× faster at 8 bits on a global layer at
 64K keys, even at 4 bits; end-to-end decode −7.7% at 32K.
 Teacher-forced decode against bf16 KV it agrees as well or better
 than the three-op path: 96.68% vs 95.12% at 4 bits, 97.07% vs 96.88%
 at 8. `=0` restores the three-op path.

### `LUMEN_GEMMA4_QUANT_KV_PREFILL_DEQUANT`

*Behavior, default on.*

Run quantized-KV prefill (queries of more than one token) through
 the bf16 attention kernels: dequantize the cached K/V, then dispatch
 as the bf16 caches do (the windowed kernel on sliding layers, causal
 SDPA on full ones). The quantized path builds `Q·Kᵀ` explicitly with
 `quantized_matmul`; this one lets the sliding layers skip K blocks
 outside the window and drop their scores tensor. Different kernels,
 so not bit-identical.

 Default on (task 021): prefill −10.0% at 16K, −12.7% at 32K (paired
 per-chunk t −28 / −5.9). Teacher-forced over 8,192 tokens against
 bf16 KV it matches the quantized path it replaces: 90.64% vs 90.88%
 at 4 bits, 95.40% vs 94.74% at 8 (`examples/gemma4_kv_quality.rs`).
 `=0` restores the quantized matmuls.

### `LUMEN_JOINT_PREFILL`

*Optimization, default on.*

When requests are waiting while one is served, compute the prefills of the ones that
 extend a prompt snapshot in memory (Flash-Next agent rounds: tools requests whose
 conversation the prefix cache holds) together with it, in one pass over the layers, so
 each layer's experts are read from the SSD once for all of them instead of once per
 request. Each request then takes its own result, computed with its own chunks from its
 own snapshot, so no output changes. Only feeds that fit one prefill group together
 (8,192 tokens) and the memory plan without releasing anything are joined; anything else
 prefills on its own as before. Three 6-round agent sessions at once on a 2.9 GB pool:
 530 s against 609 s, every reply the same (`docs/flash-next.md`).

### `LUMEN_KV_DISK`

*Optimization, default on.*

The disk KV tier (`kv_disk.rs`): prompt and conversation states, and
 Flash-Next agents' system-prompt heads, written under
 `LUMEN_KV_DISK_DIR` (default `~/.cache/lumen/kv`) and read back after a
 restart or after memory pressure dropped them, instead of prefilling
 them again. A state read back is the cold computation exactly, so only
 speed changes.

 On by default, within three limits: `LUMEN_KV_DISK_MAX_GB` (default
 10; least recently used entries go past it; `0` = unlimited),
 `LUMEN_KV_DISK_TTL_SECS` (default 2592000, 30 days, since an entry was
 last used; `0` = entries never expire) and `LUMEN_KV_DISK_MIN_FREE_GB`
 (default 10: no writes while the store's volume has less than that
 free, though stored entries stay and are still read; `0` = no floor).
 `LUMEN_KV_DISK=0` (or `false`, `off`, `no`) turns the tier off.

### `LUMEN_MEMORY_PLAN_RECLAIMABLE`

*Behavior, default on.*

Plan the expert pool within the memory that is free right now, not only within the GPU's
 working set: the working set's share is lowered to what this process holds plus the pages
 the rest of the machine gives up without being compressed or swapped (free, file-backed
 and purgeable ones), less a headroom of 2 GB or 5% of RAM, whichever is larger. The
 pool cannot change while the server runs, so a server started while other apps hold
 memory keeps a smaller pool (and decodes slower) until it is restarted; the log says so.
 `0` plans from the working set alone, as before, and the system makes room for the pool by
 compressing other apps and writing them to swap.

### `LUMEN_MLX_AUTO_SESSION`

*Behavior, default on.*

Reuse a conversation's KV without the client naming a session.

 Neither the OpenAI nor the Anthropic request has a conversation id, and
 `session_id` is a Lumen extension no stock client sends — so without
 this, every spec-conformant client re-prefilled its whole conversation
 on every turn. ON matches the prompt against live sessions by token
 prefix instead, which needs no cooperation and cannot be wrong: the same
 `starts_with` gate guards the reuse itself.

 **`Behavior`, not `Optimization`.** `session_reuse_reproduces_a_cold_prefill`
 runs three arms over the same tokens: `extend` reproduces a bulk prefill
 **bit-identically**, so the cache handoff is exact. What a session adds
 is the decode path in between: the tokens between two prompts advanced
 the cache one at a time (M=1) rather than in a bulk chunk (M=N), a
 different matmul, so their keys and values need not equal a prefill's in
 every bit, and a near-tie could flip. Measured with the arms fed the same
 tokens: all 48 tokens equal a cold prefill's on Qwen3.6-35B-A3B,
 Qwen3.6-27B and Qwen3.8-27B. (An earlier "flips at 44" was the test
 leaving out the reply's last token, which the session store also did:
 see `held_tokens`.)

 Each path is deterministic and stable across runs, but they are not
 guaranteed to be each other, so the equivalence matrix must never flip
 this. Not a new property — the explicit-`session_id` path has always had
 it, and this output is byte-identical to that path's.

### `LUMEN_MLX_KV_BF16`

*Behavior, default on.*

Store the full-attention KV cache in bf16 rather than f32.

 Task 007 measured the cache at ~67 KB per allocated slot on
 Qwen3.5-9B, which matches the f32 arithmetic exactly (8 full-attn
 layers × 4 KV heads × 256 head_dim × 2 for K+V × 4 bytes =
 64.0 KB). Halving it is the largest memory lever left after that
 task ruled out PagedAttention — worth ~1.5 GB at eight concurrent
 long-context sequences, against paging's 35 MB — and decode gets
 +1.6–4.1% faster because attention reads half the bytes.

 **`Behavior`, not `Optimization`: this is a numerics change.** The
 cast lands after k_norm and RoPE, so attention runs in bf16
 end-to-end and greedy output may differ from f32 — the equivalence
 matrix must never flip it expecting identical output. Default ON
 since the quality pass: 6,300 teacher-forced positions across two
 models, 99.73–99.83% top-1 agreement against an f32-vs-f32 control
 of exactly 100.000%, flat across context depth, every disagreement
 below the 1.5th percentile of the top1−top2 logit gap. `=0`
 restores f32. Harnesses: `examples/kv_bf16_ab.rs`,
 `examples/kv_bf16_quality.rs`.

### `LUMEN_MLX_MEMORY_EVICT`

*Behavior, default on.*

Drop the conversations a memory-planned runner cannot hold beside a new prompt.

 A runner that plans its memory (Flash-Next, whose expert pool takes what the GPU's memory
 limit leaves after one prompt's worth of KV) has no room for a second long conversation
 next to the one it is feeding. A session keeps its KV and a rollback point (two copies of
 the prompt's KV, ~2 GB at 16K tokens), and the tools path keeps a snapshot or two of every
 system prompt it has seen (~0.8 GB each at 12K tokens, without a bound). Past the memory
 limit MLX stops recycling its buffer cache, every allocation of every decode step becomes
 a fresh Metal buffer, and decode ran at half speed (3.8 against 7.7 tok/s, measured) for
 the second long prompt and every one after it; the prefill ran 15-35% slower too.

 ON, before a prompt is fed and only if it would not fit, drops the least recently used
 other sessions; then the prefix snapshots, a system prompt's head before any prompt's own
 and the least recently used first. A new prompt that cannot afford a head snapshot beside
 its own is prefilled the same way and keeps only its own. What the prompt continues is
 never dropped.

 **`Behavior`:** a dropped session is cold-prefilled when its conversation comes back,
 instead of reusing its KV (see `auto_session_enabled` for how those two differ), and a
 dropped snapshot is a prefix-cache miss. A request that does not reuse what was dropped
 gets the same text either way.

### `LUMEN_MLX_NO_OVERLAP`

*Optimization, default off.*

Disable overlap scheduling on the sampled decode path, restoring the
 original fully-synchronous order. Default OFF, i.e. overlap is on.

 Output-identical by construction: everything that affects the token
 stream — sampling, `all_tokens.push`, thinking-budget force-close and
 channel-block, the EOS check, the runaway check, the hard break —
 stays synchronous and in the original order. Only the parser advance,
 detokenisation and SSE send of the *previous* token are deferred, and
 the parser is consumed solely by `emit_token_event` / `finalize`.

### `LUMEN_MLX_SESSION_ROLLBACK`

*Optimization, default on.*

Leave a rollback point at each chat turn's conversation boundary, so
 the next turn can resume a session it does not exactly extend.

 Without it, plain multi-turn chat on Qwen 3.5 / 3.6 never reuses a
 session: the generation header a turn ends with is not in the next
 prompt (the template drops the `<think>` block from replayed turns), so
 every turn prefilled the whole conversation again — measured on
 Qwen3.5-9B with an 11.5K-token system prompt, about 25 s a turn. The
 same holds on 3.8 for any client that does not return the trace.

 **`Optimization`: the output does not change, by construction.** The
 point is placed by cutting the prefill, and `session_feed::plan_feed`
 only cuts on the bulk pass's own chunk grid, so every call is one of
 that pass's chunks. A resumed turn re-feeds what
 follows the point through the same bulk path, which is what a cold
 prefill of the longer prompt does — `extend` reproduces a bulk prefill
 bit-identically (`session_reuse_reproduces_a_cold_prefill`).

### `LUMEN_MOE_EVICTION`

*Optimization, default off.*

Which resident expert the SSD expert pool evicts when it is full, for Flash-Next style
 expert offload. Unset and `0` mean `bias` (the default); `clock` is plain CLOCK, and so is
 `1` (the flag switched on, as the flags matrix does: the policy before `bias`). `gclock` is CLOCK with a
 2-bit hit counter in place of the reference bit, and loads start at zero. `bias` keeps a
 decayed hit count per expert and a per-layer table of which experts followed which at the
 previous token, and evicts the sampled expert it expects least. In trace-driven
 simulation `gclock` gains 1.6 to 3.4 and `bias` 1.7 to 6.3 decode hit-rate points over
 CLOCK, at pools from 2 to 20 GB. Never changes the output, only which experts are
 resident and so how many are read from the SSD; any other value is an error when the
 pool is built.

### `LUMEN_MOE_IO_THROTTLE_GBPS`

*Diagnostic, default off.*

Diagnostic: pace the expert reads to the aggregate rate in this variable, in GB/s (`3`,
 `1.5`), to see how the offload behaves on a slower SSD than this machine's. Each read
 reserves its share of the rate across all reader threads, starts no earlier than its
 reservation and completes no earlier than its end. Only when the bytes arrive changes,
 never which bytes are read; `0` or unset reads at full speed.

### `LUMEN_MOE_OVERLAP`

*Optimization, default off.*

Keep the GPU busy while a decode step's missing experts are read from the SSD: as soon as
 the routing is known, submit the routed experts the pool already holds (and the shared
 expert), and gather the missing ones once their reads land. Each expert's output is the
 same per-row `gather_qmm` kernel over the same bytes either way, and the outputs are put
 back in the routing's order before today's score-weighted sum, so the output is
 bit-identical (`offload::exactness_tests`). Only calls of up to four rows that both hit
 and miss, with reads expected to take at least 0.5 ms; everything else (prefill, staging,
 `LUMEN_QWEN4_BUILD_AHEAD`) is unchanged. Measured on the real checkpoint (M3 Max, N=5,
 identical text): +4.0% at a 4.2 GB pool and +4.6% at 15 GB on this 6.5 GB/s SSD, -0.8%
 (noise) at 6.7 GB, and only +1.3% to +2.4% with reads paced to 3 or 1.5 GB/s, where the
 read is 5-10 ms and the hidden GPU work 0.1-0.2 ms per layer call. Kept opt-in.

### `LUMEN_MOE_PROBE_LAYER_SYNC`

*Diagnostic, default off.*

Task 023 Phase 0 probe: read the routed expert ids back to the CPU at
 every MoE layer while every expert stays resident. An exact SSD
 expert cache needs exactly this readback, so the decode-time delta
 against the default is the sync tax on its own. Output unchanged.

### `LUMEN_MOE_TRACE`

*Diagnostic, default off.*

Write a binary trace of the expert pool's layer calls (expert ids in request order, the
 engine's hit flags, request/pass markers, per-expert token counts for prefill calls and
 per-call timestamps) to the path in this variable, for replaying a workload against other
 eviction policies offline. Any value but `0` is the output path. Never changes the output
 or what is resident; costs one `Option` check per layer call when off.

### `LUMEN_NATIVE_ACT_BF16`

*Behavior, default off.*

Run each quantized projection on a bf16 activation (cast back to f32 after) when the
 activation is f32 and the scales are bf16/f16, so MLX takes its fast path (276 GB/s on
 M3 Max against 187 GB/s with an f32 activation, stock MLX 0.32.3, 17408x5120 4-bit).

 **`Behavior`, not `Optimization`: this is a numerics change.** The activation is
 rounded to bf16 before the matmul (the accumulation stays f32), as mlx-lm does
 throughout. Applies to every `ResolvedLinear::apply` -- the Qwen trunk and MTP head
 and Gemma 4 -- whose activation is f32 and scales are bf16/f16. A no-op on scales
 `LUMEN_NATIVE_WIDEN_SCALES` widened; it does act on the projections that keep bf16
 scales for `LUMEN_NATIVE_DECODE_QMV` (every trunk projection of a 4-bit group-64
 checkpoint), where it replaces those kernels.

 Measured on Qwen3.6-27B (M3 Max): 65.9 -> 62.0 ms/step against the widened default,
 with a cast before and after every matmul eating part of the gain. Teacher-forced
 top-1 agreement with the f32 activation over 6 seeds x ~675 positions
 (`examples/kv_bf16_quality.rs --act-bf16`; the f32-vs-f32 control is 100.000%): 27B
 99.31%, 9B 99.09%. Every flip is a near-tie (the largest top1-top2 gap sits at the
 3.4th / 3.9th percentile) and agreement does not fall with context depth, but it is
 below the 99.73% that the bf16 KV cache cleared before it became the default.
 Default off.

### `LUMEN_NATIVE_ALLOC_REUSE`

*Optimization, default on.*

Reuse per-layer scratch allocations across decode steps instead of
 reallocating. Default ON (LANDED 2026-05-11). Prerequisite for
 `LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE` (the fused-weight constants
 live in the reused `LinearAttnConstants`).

### `LUMEN_NATIVE_CACHED_STREAM`

*Optimization, default off.*

Share one `OnceLock` stream + sentinel across FFI quant ops instead
 of constructing `StreamGuard::gpu()` + `OwnedEmptyArray` per call.
 Default OFF — Lever 2 infrastructure, measured an **A/B wash**
 (2026-05-03). Kept because the per-call construction is the thing it
 would replace, not because it won.

 Stream plumbing only; the arithmetic is untouched, hence
 `Optimization`.

### `LUMEN_NATIVE_COMPILE`

*Behavior, default on.*

compute_g persistent compile cache. Default ON (LANDED Phase 3):
 wins σ-significantly across a 12-run 4-condition matrix
 (Δp50=−0.33 ms, Δtps=+1.96 vs legacy direct dispatch). `=0` falls
 back to the legacy direct-ops path.

 **`Behavior`, not `Optimization`:** the legacy ops round some gates
 differently from the compiled function, by one unit in the last place
 (three of 256 in `linattn_prework_gates_are_the_composed_gates_to_the_bit`
 at `=0`), so `=0` can change a near-tie. Gates the fused prework kernel
 computes (decode) are the compiled function's either way.

### `LUMEN_NATIVE_COMPILE_ROUTING`

*Optimization, default off.*

Compile-wrap the MoE routing graph. Default OFF (A/B WASH). (Parse
 note: previously only `=1` enabled this; the uniform rule now
 accepts any non-`"0"`.)

### `LUMEN_NATIVE_CONV_SLICE`

*Optimization, default off.*

Conv-state advance via `slice` (view/copy) instead of `take_axis`
 (gather kernel). Bit-identical for the s==1 decode regime — same
 indices, cheaper Metal kernel. Default OFF (A/B WASH).

### `LUMEN_NATIVE_DECODE_QMV`

*Optimization, default on.*

Run the matvecs of an affine 4- or 8-bit, group-32/64 projection whose activation is f32 and
 whose scales and biases are bf16 or f16 in kernels that read those scales as they are
 (`native_qmv::matvec`), instead of MLX's `quantized_matmul`, which casts them to f32 on every
 call, and keep such projections' scales in bf16 under `LUMEN_NATIVE_WIDEN_SCALES`. A
 single-token step runs the projections of one activation (linear attention's four input
 projections, attention's q/k/v, a dense MLP's gate and up) in one launch.

 A matvec is up to MLX's batch limit of rows (6 to 33, by GPU and size; 13-15 on an M3 Max):
 a decode step, a batch of decode steps, an MTP verify. Larger matmuls (prefill) stay on MLX,
 which casts the bf16 scales per call there; against a matmul that size the cast is noise.

 Bit-identical: the kernels run MLX's own matvec arithmetic for that many rows on this GPU
 (`qmv_fast` for one; for more, `qmv_wide` or `qmv_fast` as MLX would, or `qmv_fast` row by
 row under `LUMEN_NATIVE_DECODE_QMV_ROWS`, which this flag's off path then also runs row by
 row), its lane partition and order of adds, and widening a bf16 or f16 scale to f32 is
 exact. An `Optimization` flag. Default on; `=0` keeps MLX's matmul (and widens every affine
 projection's scales again).

 M3 Max, one matvec over projections larger than the system cache, best of 12: 17408x5120
 218 -> 200 us, 5120x17408 213 -> 200 us, 248320x2048 (lm_head) 1187 -> 1063 us against MLX
 on widened scales (MLX on bf16 activations and scales, the bound: 195, 196, 1066 us); a small
 projection (32x2048) is slower alone, 3.6 -> 6.1 us, which the shared launch removes.
 Pipelined greedy decode against MLX on the widened copies, alternating runs, text identical
 in every run: Qwen3.6-27B 61.4 / 61.6 / 61.1 -> 56.6 / 57.2 / 56.7 ms/token, footprint 18.51
 -> 17.21 GB; Qwen3.6-35B-A3B 9.3 / 9.3 / 9.3 -> 9.2 / 9.2 / 9.2 ms/token, 20.65 -> 20.52 GB.
 Qwen3.8-27B-MTPLX-Optimized (4-bit group 32, 8-bit group 64 layers), chat: 78.5 -> 70.3
 ms/token, 24.25 -> 22.0 GB; its 8-bit kernels gain the decode step nothing measurable but
 let its MTP verify read those layers once (lossless MTP 58.5 -> 47.8 ms/token).

### `LUMEN_NATIVE_DECODE_QMV_ROWS`

*Behavior, default on.*

Run a matvec of two to four vectors (an MTP verify, a few sequences decoded together, a
 prompt of a few tokens) as `qmv_fast`'s arithmetic for each vector, all of them in the same
 simdgroups, so each weight block is read once and dotted with every vector. Each row then
 equals the single-vector matvec of that row in every bit. MLX runs that many vectors as
 `qmv_wide` from GPU generation 15 on, whose lane partition differs.

 **`Behavior`, not `Optimization`**: the low bits differ from MLX's matvec of that many rows
 (not from a decode step). In exchange an MTP verify computes exactly what the decode steps it
 stands in for would, and a sequence decoded beside others exactly what it would alone.
 A projection the kernels do not read (a checkpoint's 8-bit layers, other group sizes, or
 every projection under `LUMEN_NATIVE_DECODE_QMV=0`) runs such matvecs on MLX one row at a
 time, which is what its decode step runs.

 M3 Max, 17408x5120, bench wall time: one vector 183 us; two, three, four 186 / 244 / 320 us
 here, 258 / 387 / 543 us as `qmv_wide` on the f32 activation. A Qwen3.6-27B forward of two,
 three, four tokens 66.6 / 89.0 / 108.7 ms, 87.0 / 126.5 / 165.8 off, one token 61.8. MTP K=1
 (lossless) serves chat at 42.5 ms a token against 56.1 without MTP. Default on; `=0` runs
 them as MLX would.

### `LUMEN_NATIVE_FUSED_MOE_DECODE`

*Optimization, default on.*

Run a single-token decode step's routed and shared experts in two Metal kernels
 (`native_moe_decode`) instead of about twenty-five MLX ops. Implies the stacked
 gate/up layout of `LUMEN_NATIVE_FUSE_MOE_GATE_UP`. Applies to affine 4-bit group-64
 layers with a top-8 router whose hidden and intermediate sizes are multiples of 512,
 behind an f32 activation and bf16 expert scales; prefill, SSD expert offload and other
 layouts keep the composed ops.

 Bit-identical to the composed ops: the kernels run MLX's own quantized matvec arithmetic
 (its lane partition, `load_vector`'s division and bf16 quad sum, `qdot`'s expression),
 round to bf16 where the composed block does, and add the eight routed outputs in the
 composed reduction's order. An `Optimization` flag. Default on; `=0` restores the
 composed ops.

 SSD expert offload serves the experts through the composed ops. Because the kernels
 equal them in every bit, an offloaded run equals the resident run on an affine
 checkpoint too: the weights-gated offload tests pass with the kernels on.

 Qwen3.6-35B-A3B affine, M3 Max, pipelined greedy decode, ABBA: 11.2 ms/token composed
 against 9.5 with the kernels (and the router kernels), the generated text identical
 byte for byte. Before the kernels were bit-identical they ran at 11.1 ms against 12.0
 with 814 of 819 bytes the same.

### `LUMEN_NATIVE_FUSED_MOE_ROUTER`

*Optimization, default on.*

Route a single-token step with a logits kernel (the expert and shared-gate logits) and the
 select (softmax, top-8, renormalisation, the shared sigmoid) inside the experts' gate/up
 kernel, instead of the twelve MLX ops, when `LUMEN_NATIVE_FUSED_MOE_DECODE` runs the experts
 in kernels and the router is an 8-bit group-64 gate over a multiple of 128 experts.

 Bit-identical to the composed router: the logits run MLX's matvec arithmetic, the softmax
 adds as MLX's kernel does, and the eight experts come in the order the composed sort leaves
 them (ascending probability, equal ones by index), so the routing, the scores and the
 block's output equal the composed ones in every bit. An `Optimization` flag. Default on;
 `=0` keeps the composed router.

 Qwen3.6-35B-A3B affine, M3 Max, pipelined greedy decode, ABBA: 11.9 ms composed, 11.1 ms
 with the expert kernels only, 10.5 ms with the router kernels too (-0.6 ms; the router alone
 is 31 us a layer, 1.24 ms a token, in the composed ops); folding the select into the
 gate/up kernel took another 0.1 ms.

### `LUMEN_NATIVE_FUSE_GATE_UP`

*Optimization, default off.*

Fuse `gate_proj` + `up_proj` into one QMV when their quant params
 match. Default OFF and **tested negative**: an A/B on 27B / M3 Max
 showed no win (separate 85.7 ms vs fused 89.2 ms/tok). Real decode
 pipelines the matmuls and is bandwidth-bound, so fusing same-byte
 matmuls saves nothing and the extra split op costs a little. The
 isolation bench overstated dispatch overhead because it `eval()`s
 every call, which defeats pipelining.

 Kept as a future-hardware lever. Row-concat is bit-identical, which
 is why this is an `Optimization` rather than a `Behavior` flag.

### `LUMEN_NATIVE_FUSE_LINATTN_IN`

*Optimization, default on.*

One kernel for a decode token's linear-attention prework: eight dependent dispatches
 (concat, slice, conv1d, SiLU, two RMS norms, sigmoid, compiled `compute_g`) become one.

 Bit-identical to the composed ops in f32, the dtype the model runs this in: the kernel
 repeats MLX's own sums in MLX's own order (its `Sigmoid` for SiLU, `rms_single_row`
 for the norm, `precise::rsqrt`, `LogAddExp` for the softplus of `compute_g`), and the
 tests compare bits at the A3B shape. M3 Max, ABBA over four runs each, generated text
 byte-identical with the flag on and off: Qwen3.6-35B-A3B affine 10.5 -> 10.1 ms/token,
 A3B mxfp4 11.2 -> 10.4, Qwen3.5-9B 27.9 -> 27.3, Qwen3.6-27B 67 -> 63. Only a decode
 token (not prefill, not under MTP capture, not under the timing probes) takes it.
 Default on; `=0` restores the composed ops.

### `LUMEN_NATIVE_FUSE_MOE_GATE_UP`

*Optimization, default off.*

Stack each MoE layer's routed-expert `gate_proj` and `up_proj` along the output axis
 at load, so the experts' gate and up outputs come from one `gather_qmm` (split in
 half afterwards) instead of two.

 Every output row is the same reduction either way, so the result is bit-identical.
 It saves one gather launch per MoE layer per step. The stacked copy replaces the
 separate tensors (the model drops them), so memory does not grow (footprint 19 GB
 either way; the load takes ~4 s longer). Not used with SSD expert offload, whose pool
 serves gate and up as separate roles, nor by the MTP block.

 Default OFF and **tested null**: Qwen3.6-35B-A3B affine, M3 Max, pipelined greedy
 decode, ABBA over four server runs: 12.0 / 11.8 ms/token off against 11.9 / 11.9 on,
 819 generated bytes identical in all four. Decode is GPU-bound, so a launch saved
 between matmuls that already queue behind each other buys nothing. Kept as the layout a
 fused gate+up+SwiGLU routed-expert kernel would read.

### `LUMEN_NATIVE_FUSE_SIGMOID_MUL`

*Optimization, default off.*

Compile-wrapped `sigmoid(gate) * other`, bit-identical to the
 two-op composition. Default OFF (anti-pattern #30 calibration).
 (Parse note: previously only `=1` enabled this; the uniform rule
 now accepts any non-`"0"`.)

### `LUMEN_NATIVE_FUSE_SWIGLU`

*Optimization, default on.*

Fuse `silu(gate) * up` into one compile dispatch. Default ON
 2026-05-11 — net WIN −0.671 ms (Welch t=−29σ) at n=10 STEPS=300 on
 Qwen3.6-35B-A3B-mxfp4; closes ~62% of the Native-vs-PyO3 decode gap.

### `LUMEN_NATIVE_KV_STEP_PREALLOC`

*Optimization, default on.*

Grow the full-attn KV buffer in `KV_CACHE_STEP` (256) token blocks
 and fill via `slice_update`, instead of a per-step
 `concatenate_axis`. The legacy concat path stays live as the `=0`
 emergency revert.

 Default ON since Phase G2: microbench long-prompt 35B-mxfp4 n=12
 interleaved A/B (PROMPT_LEN=2048, STEPS=1500) showed Δtps
 +4.67 tok/s (Welch t = +4.33σ), Δp50 −1.02 ms/step (t = −4.68σ),
 with a 32-step PyO3 golden bit-identical PASS for both paths.

 This flag is the ancestor of the whole `lumen_flags` design: it was
 the only one of the audited 370 env gates whose alternate path a
 test could reach, via a hand-rolled thread-local override that the
 macro now generates for every flag.

### `LUMEN_NATIVE_LINEAR_ATTN_SCALE_FUSE`

*Optimization, default on.*

Absorb `scale_q/k` into the rms_norm weight (bit-identical by
 linearity of rms_norm weight scaling). Default ON 2026-05-11:
 thermal-clean A/B (n=10, STEPS=100) Δ=−0.315 ms, Welch t=−3.33σ.
 Only active when `LUMEN_NATIVE_ALLOC_REUSE` is also on.

### `LUMEN_NATIVE_PIPELINE_DECODE`

*Optimization, default on.*

Build each greedy decode step while the one before it is still running.

 Blocking decode reads a token, then builds the next step's graph with
 the GPU idle; on a small-active-parameter MoE that build is ~2.5 ms of a
 ~13 ms step. With this on, a loop that feeds every token back as the
 next input (`decode_step_chained`) submits step N+1 -- built from step
 N's unread argmax -- before it reads step N, as mlx-lm's `generate_step`
 does. The same graphs run in the same order, so the tokens are
 bit-identical; the one extra step a loop leaves running when it stops is
 taken back before anything else touches the sequence.

 Does nothing (falls back to blocking decode) under SSD expert offload,
 MTP, `LUMEN_NATIVE_TIMING`, or a quantized / TurboQuant KV cache.

### `LUMEN_NATIVE_RMS_NORM_GATED_FUSED`

*Optimization, default off.*

Fused rms_norm + silu(gate) + multiply Metal kernel. Default OFF —
 reserved for super-kernel composition. (Parse note: previously only
 `=1` enabled this; the uniform rule now accepts any non-`"0"`.)

### `LUMEN_NATIVE_TIMING`

*Diagnostic, default off.*

Per-step decode timing capture (`take_native_decode_timing_log`).
 Costs an eval barrier per timed stage; default OFF. (Parse note:
 previously a truthy list `1|true|TRUE|yes`; the uniform rule now
 accepts any non-`"0"`.)

### `LUMEN_NATIVE_WIDEN_SCALES`

*Optimization, default on.*

Keep the affine projections' scales and biases in f32 so a matmul on the f32 residual
 stream stops re-casting them on every call.

 MLX gives `quantized_matmul` the common dtype of the activations and the scales and
 biases and casts whichever differs on every call. The trunk's activations are f32 and
 the checkpoint's scales bf16, so each of a decode step's ~500 matmuls re-widened its
 scales and biases before reading the weights. On stock MLX 0.32.3 (M3 Max, 17408x5120
 4-bit group 64) that is 187 GB/s against 276 GB/s with bf16 activations; pre-widened
 scales recover most of it (237-250 GB/s). The widening is exact (bf16 -> f32), so the
 output is bit-identical; the cost is the f32 copy minus the bf16 original the model
 drops (`NativeWeights::release_scales_and_biases`): +1.45 GB of GPU memory on the 27B
 (16.0 -> 17.5 GB footprint, `footprint`; `ps` RSS does not see GPU buffers).
 Affine projections only (mxfp4/nvfp4 scales are integer exponents). Applies to
 `resolve_linear`, i.e. the Qwen 3.5/3.6 trunk and lm_head. Measured on Qwen3.6-27B
 (4-bit group 64, M3 Max): decode 84.3 -> 67.1 ms/step, 192 greedy tokens identical.
 `=0` restores the per-call cast.

 A projection whose matvecs `LUMEN_NATIVE_DECODE_QMV` runs (4-bit group 64, inputs a
 multiple of 512, outputs a multiple of 8) keeps its bf16 scales instead: those
 kernels read them as they are, faster than MLX reads the f32 copy, and only a
 prefill-sized matmul still casts them. On the 4-bit Qwen checkpoints that is every
 trunk projection, so the copy, and its memory, is gone there.

### `LUMEN_PREFIX_EXACT_REUSE`

*Behavior, default on.*

Make a later round of a conversation on a cut-sensitive runner (Flash-Next, a mixture of
 experts whose numbers depend on where a feed is cut) come out exactly as that round's cold
 prefill would. Off, a round extends the previous round's full-prompt snapshot, so its feed
 is cut where the previous prompt ended, a place the round's own cold prefill (cut at the
 system/tools head, then every prefill chunk from there) does not cut, and the reply can
 differ from a cold run's. On, a round resumes only from the head or from a point a whole
 number of prefill chunks past it, kept instead of the full-prompt snapshot (one per
 conversation, so no more memory), and re-feeds the rest on the cold grid: less than one
 chunk more per round than the snapshot it replaces, and an identical prompt retried
 re-feeds that much too. Changes no cold prefill, and nothing on other runners. On by
 default: on three 6-round agent tasks the snapshot path changed the tool call in 4 of 15
 later rounds and was no faster on average; `0` restores it.

### `LUMEN_QWEN35_QUANT_KV`

*Behavior, default off.*

Store the full-attention KV cache affine-quantized (8-bit, group
 64): half the bytes of bf16. Decode reads it through the fused
 kernel in `native_quant_attention`; prefill dequantizes each chunk
 and takes the bf16 SDPA path. On an M3 Max at 64K keys (task 021)
 decode attention is ~7% faster than bf16 at GQA 4 (9B) and 15-20%
 slower at GQA 6-8 (27B, 35B-A3B), so this is a memory lever. Disk
 persistence of these caches is not supported yet. Default off.

### `LUMEN_QWEN35_REASONING_EFFORT`

*Behavior, default on.*

Honour a checkpoint's own `reasoning_effort` declaration (Qwen 3.8).

 **`Behavior`, not `Optimization`: this changes the prompt.** ON means
 "inject the effort sentence when the checkpoint's `chat_template.jinja`
 declares the block" — so it is inert for every 3.5/3.6 checkpoint, which
 declares no such thing. `=0` suppresses it even on 3.8, which is the
 A/B hatch. The equivalence matrix must never flip this expecting
 identical output: on a 3.8 checkpoint the two settings render different
 system blocks by design.

### `LUMEN_QWEN4_BUILD_AHEAD`

*Optimization, default off.*

Decode with the offloaded layers pipelined one segment ahead: submit the GPU work that
 ends in a layer's routing, then build the rest of that layer and the whole next layer's
 graph while the GPU runs it, and only then read the routing back. The graph is the same
 one the plain path builds (the gather reads its slot numbers from a persistent array the
 CPU fills in after the readback), so the output is bit-identical; what changes is that the
 host's graph building no longer sits between two GPU segments. Measured on the real
 checkpoint (M3 Max): identical text, but no end-to-end gain (15 GB pool 9.79 -> 9.81 tok/s,
 N=10; 6.7 GB +2%, noise); only ~5.5 ms of host build per token is hideable. Kept opt-in.

### `LUMEN_QWEN4_KV_BF16`

*Behavior, default on.*

Keep Flash-Next's attention KV cache (and the sparse indexer's keys) in bf16 rather than
 f32: 30 KB per token of context instead of 60, so the memory plan reserves half as much
 for the longest prompt (+0.74 GB of expert pool at a 24 GB-class limit) and every retained
 conversation state (a session, an agent's system-prompt head, a resume point, a disk
 record) is half the size. Attention then runs in bf16 (the cast lands after the norm and
 RoPE, as `LUMEN_MLX_KV_BF16` does for Qwen 3.5); the indexer's keys are pooled in f32.

 **`Behavior`: a numerics change.** Teacher-forced over 98K positions of Korean, English and
 code (16K-token contexts), against f32: the same top token at 95.9% of positions, KL 0.0093
 nats, perplexity -0.05%, flat over depth (for scale, the gs64 and gs32 4-bit builds differ
 by KL 0.20). A 16K-token prefill is ~6% faster; decode reads half the KV bytes (~1% of a
 token there). States saved under one setting are never restored under the other (the disk
 identity includes it). `=0` restores f32.

### `LUMEN_QWEN4_MEM_LOG`

*Diagnostic, default off.*

Log MLX's memory (active, cache, peak) and what each sequence holds: after every prefill,
 when a sequence is dropped, and every 32 decode steps with the speed of those steps. For
 finding out where a memory limit is being hit; it adds one allocator query per line.
 After each prefill it also prints the expert bank's account of that feed alone (layer
 calls, reads, layer calls that took the staging path), restarting the bank's counters.

### `LUMEN_QWEN4_STAGE_SYNC`

*Diagnostic, default off.*

Evaluate every decode step stage by stage (hyper-connection, mixer, router, route, experts,
 shared expert, write-back) and print each stage's build time, evaluation time and
 dispatch counts. Costs a sync per stage; for diagnosis only.

### `LUMEN_QWEN4_TOKEN_COUNTERS`

*Diagnostic, default off.*

Count the Metal command buffers, kernel dispatches and encoded primitives of every decode
 step, plus the CPU time spent encoding, and print them every 100 steps. Adds no sync.

### `LUMEN_QWEN4_WIDEN_SCALES`

*Optimization, default on.*

Keep the resident quantized projections' scales and biases in f32 so the activations'
 matmul stops re-casting them on every call. Bit-identical (bf16 -> f32 is exact).

### `LUMEN_SPARSE_DECODE`

*Behavior, default off.*

Page-sparse decode on the full-attention layers (Qwen 3.5/3.6,
 Gemma 4): once a cache holds `SPARSE_DECODE_MIN_KEYS` keys, a
 single-token step attends to the pages whose key bounds score
 highest for its query, plus the first and most recent pages,
 instead of every key (`sdpa_page_sparse`). Prefill, MTP verify and
 batched decode stay dense. Changes which keys are attended, so
 `Behavior`. Default off.

### `LUMEN_TOKENIZE_MEMO`

*Optimization, default on.*

Remember the encodes of long prompt pieces that recur across requests —
 the system+tools head and the history an agent resends every turn — so
 a turn encodes only what is new. Qwen prompts are cut right before each
 `<|im_start|>`, where HF splits them itself; other strings are
 remembered whole. Exact: the cut is used only for a tokenizer where it
 cannot change the ids, checked at load. Bounded at 64 MB. On by
 default since task 016's Gate 2: warm agentic turns at ~35K tokens
 reached their first token 38 ms sooner on Qwen3.5-9B (279 → 241 ms,
 Welch t 16.3) and 45 ms sooner on Gemma 4 (272 → 227 ms, t 16.2).
