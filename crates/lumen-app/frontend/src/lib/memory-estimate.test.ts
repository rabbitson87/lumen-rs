// Unit tests for the memory estimator and the Flash-Next card's settings.
//
//   node --test src/lib/memory-estimate.test.ts        (Node >= 22.18; no extra dependency)
//
// Not part of `npm run check`: the tsconfig leaves *.test.ts out, because `node:test` has no
// type declarations in this package. Nothing in the build imports this file.

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { registerHooks } from "node:module";
import { test } from "node:test";

// The app's modules import each other without extensions (bundler resolution). Give Node the
// same, then load them after the hook is in place (static imports would run first).
registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier.startsWith(".") && !/\.[cm]?[jt]s$/.test(specifier)) {
      try {
        return nextResolve(`${specifier}.ts`, context);
      } catch {
        // fall through to the plain specifier
      }
    }
    return nextResolve(specifier, context);
  },
});

const {
  FLASH_NEXT,
  FLASH_NEXT_DECODE_TPS,
  attnScoreBytes,
  envKeysForFamily,
  estimateAt,
  flashNextDecodeTps,
  FLASH_NEXT_LANE_PRESETS,
  FLASH_NEXT_LANES_DEFAULT,
  FLASH_NEXT_DEFAULTS,
  FLASH_NEXT_ENV_KEYS,
  FLASH_NEXT_POOL_PRESETS_GB,
  FLASH_NEXT_PROMPT_PRESETS,
  envChoice,
  envShadowsTypedKey,
  flashNextChipTps,
  flashNextDefaultContext,
  flashNextDiskBytes,
  flashNextEffective,
  flashNextEnv,
  flashNextIsDefault,
  flashNextPoolChip,
  flashNextPromptChip,
  formatGbValue,
  formatPromptShort,
  lanesValue,
  mtpInUse,
  mtpMode,
  packInUse,
  parseGbInput,
  parsePromptInput,
  flashNextEstimate,
  flashNextFits,
  flashNextLaunchContext,
  flashNextMaxContext,
  flashNextMaxPoolGb,
  flashNextMinRamGb,
  flashNextPlan,
  flashNextPlanFits,
  flashNextPlannedContext,
  flashNextSettingsEnabled,
  flashNextSettingsEnv,
  flashNextShadowedKeys,
  flashNextUsableBytes,
  geometryForModel,
  isFlashNextModelId,
  kvBytes,
  maxContextTokens,
} = await import("./memory-estimate.ts");

const GIB = 1024 ** 3;
const near = (a: number, b: number, eps = 1e-6) => assert.ok(Math.abs(a - b) <= eps, `${a} vs ${b}`);

// ── The families that existed before Flash-Next keep their numbers ──────────

test("the existing families still resolve to the same geometry", () => {
  assert.equal(geometryForModel("mlx-community/Qwen3.6-35B-A3B-mxfp4")?.family, "qwen35");
  assert.equal(geometryForModel("Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed")?.fullAttnLayers, 16);
  assert.equal(geometryForModel("mlx-community/gemma-4-26b-a4b-4bit")?.family, "gemma4");
  assert.equal(envKeysForFamily("qwen35").prefillChunk, "LUMEN_QWEN35_PREFILL_CHUNK");
  assert.equal(envKeysForFamily("gemma4").prefillChunk, "LUMEN_GEMMA4_PREFILL_CHUNK");
});

test("the existing estimator formulas are untouched", () => {
  const g = geometryForModel("Qwen3.6-35B")!;
  // 10 full-attn layers x K,V x 2 KV heads x 256 dims x 2 B per element.
  assert.equal(kvBytes(g, "bf16", 1000), 10 * 2 * 2 * 256 * 2 * 1000);
  assert.equal(attnScoreBytes(g, 46_638, 2048), 16 * 2048 * 46_638 * 4);
  const at = estimateAt(
    { geometry: g, budgetBytes: 31 * GIB, kvMode: "bf16", chunkTokens: 2048, prefixCache: false },
    46_638,
  );
  assert.equal(at.peakBytes, g.weightsBytes + kvBytes(g, "bf16", 46_638) + at.attnScoreBytes + at.overheadBytes);
  assert.ok(maxContextTokens({ geometry: g, budgetBytes: 31 * GIB, kvMode: "bf16", chunkTokens: 2048, prefixCache: false }) > 0);
});

test("Flash-Next is not mistaken for a resident-weights family", () => {
  assert.equal(geometryForModel("pipenetwork/Qwen3.8-Flash-Next-MLX-4bit"), null);
  assert.ok(isFlashNextModelId("pipenetwork/Qwen3.8-Flash-Next-MLX-4bit"));
  assert.ok(isFlashNextModelId("/Users/x/models/pipenetwork--Qwen3.8-Flash-Next-MLX-4bit"));
  assert.ok(isFlashNextModelId("qwen4_exp-local"));
  assert.ok(!isFlashNextModelId("mlx-community/Qwen3.6-35B-A3B-mxfp4"));
  assert.ok(!isFlashNextModelId(null));
  assert.ok(!isFlashNextModelId(undefined));
});

// ── Flash-Next: the memory plan the server makes (runner.rs `plan_memory`) ────

test("KV is 59,904 bytes per token: 13 layers x (K,V of 2 heads x 256 dims + a 128-dim indexer key), f32", () => {
  assert.equal(FLASH_NEXT.kvBytesPerToken, 13 * (2 * 2 * 256 * 4 + 128 * 4));
});

test("the pool floor is 2 x 512 slots and the second staging region needs 3 x 512", () => {
  assert.equal(FLASH_NEXT.poolFloorBytes, 2 * 512 * FLASH_NEXT.expertBytes);
  assert.equal(FLASH_NEXT.poolPrefetchBytes, 3 * 512 * FLASH_NEXT.expertBytes);
});

test("usable memory is 2/3 of RAM up to 32 GiB and 3/4 above, times 0.88", () => {
  near(flashNextUsableBytes(16), 16 * GIB * (2 / 3) * 0.88, 1);
  near(flashNextUsableBytes(32), 32 * GIB * (2 / 3) * 0.88, 1);
  near(flashNextUsableBytes(36), 36 * GIB * 0.75 * 0.88, 1);
  // A memory limit (GiB) only ever lowers it.
  assert.equal(flashNextUsableBytes(36, 9), 9 * GIB);
  near(flashNextUsableBytes(16, 64), flashNextUsableBytes(16), 1);
  assert.equal(flashNextUsableBytes(36, null), flashNextUsableBytes(36));
});

test("the reserve adds buffers, chunk activations, KV of the longest prompt and half the scores", () => {
  const usable = flashNextUsableBytes(36);
  const p = flashNextPlan(usable);
  assert.deepEqual([p.chunk, p.ctxTokens, p.tight], [1024, 16_384, false]);
  const scores = 24 * 1024 * 16_384 * 4; // 1.61 GB, under the 2 GB budget
  assert.equal(p.buffersBytes, 600_000_000 + 1024 * 1_100_000);
  assert.equal(p.kvBytes, 16_384 * 59_904);
  assert.equal(p.scoresBytes, scores / 2);
  assert.equal(p.reserveBytes, p.buffersBytes + p.kvBytes + p.scoresBytes);
  near(p.poolBytes, usable - 3.6e9 - p.reserveBytes, 1);
  // Past the 2 GB budget the scores stop growing.
  assert.equal(flashNextPlan(usable, null, 65_536).scoresBytes, 1_000_000_000);
});

test("a tight machine takes the smaller chunk, context and scores budget", () => {
  const p = flashNextPlan(flashNextUsableBytes(16));
  assert.deepEqual([p.chunk, p.ctxTokens, p.tight], [512, 8192, true]);
  assert.equal(p.reserveBytes, 1_855_260_160);
  near(p.poolBytes, 4_623_596_427, 2);
  // 24 GB is not tight: 1,024-token chunks, 16K prompts, an ~8 GB pool.
  const q = flashNextPlan(flashNextUsableBytes(24));
  assert.deepEqual([q.chunk, q.ctxTokens, q.tight], [1024, 16_384, false]);
  near(q.poolBytes, 8_005_111_377, 2);
});

test("an override replaces the default it names and nothing else", () => {
  const usable = flashNextUsableBytes(16);
  const c = flashNextPlan(usable, 2048);
  assert.deepEqual([c.chunk, c.ctxTokens], [2048, 8192]);
  const x = flashNextPlan(usable, null, 32_768);
  assert.deepEqual([x.chunk, x.ctxTokens], [512, 32_768]);
});

test("the longest prompt the user allows comes out of the pool", () => {
  const usable = flashNextUsableBytes(36);
  const small = flashNextPlan(usable, null, 16_384);
  const big = flashNextPlan(usable, null, 81_920);
  assert.ok(big.poolBytes < small.poolBytes);
  near(big.poolBytes, 14_278_370_058, 2);
  // The app's default Max context (81,920) leaves a 16 GB Mac no room at all.
  const tight = flashNextPlan(flashNextUsableBytes(16), null, 81_920);
  assert.ok(!flashNextPlanFits(tight));
});

test("the planned prompt is the prompt cap, lowered to Max context; Max context alone never raises it", () => {
  // Same cases as `the_prompt_cap_sets_the_planned_context...` in flash_next.rs.
  assert.equal(flashNextPlannedContext(16_384, 81_920), 16_384);
  assert.equal(flashNextPlannedContext(100_000, 81_920), 81_920);
  assert.equal(flashNextPlannedContext(5_000, null), 5_000);
  assert.equal(flashNextPlannedContext(null, 81_920), FLASH_NEXT.defaultContext);
  assert.equal(flashNextPlannedContext(null, 4_096), 4_096);
  assert.equal(flashNextPlannedContext(null, null), null);
});

test("a default Flash-Next launch plans the machine's own prompt, not the Prefill card's", () => {
  const u36 = flashNextUsableBytes(36);
  const u16 = flashNextUsableBytes(16);
  // The app sends the machine's default: 16,384, or 8,192 on a tight Mac.
  assert.equal(flashNextDefaultContext(u36), 16_384);
  assert.equal(flashNextDefaultContext(u16), 8_192);
  const dflt = flashNextLaunchContext({}, 81_920, u36);
  assert.deepEqual(
    [dflt.planned, dflt.promptCap, dflt.maxCtx, dflt.explicit],
    [16_384, 16_384, 81_920, false],
  );
  assert.equal(flashNextLaunchContext({}, 81_920, u16).planned, 8_192);
  // The default follows the memory limit a launch carries (a tight machine to the server).
  assert.equal(flashNextDefaultContext(flashNextUsableBytes(36, 8)), 8_192);
});

test("a LUMEN_MAX_PROMPT_TOKENS the user set wins, and Max context only lowers it", () => {
  const u36 = flashNextUsableBytes(36);
  const set = flashNextLaunchContext({ LUMEN_MAX_PROMPT_TOKENS: "32768" }, 81_920, u36);
  assert.deepEqual([set.planned, set.explicit], [32_768, true]);
  // A smaller Max context (card or LUMEN_MAX_CTX override) lowers it; a larger one does not raise it.
  assert.equal(flashNextLaunchContext({ LUMEN_MAX_PROMPT_TOKENS: "32768" }, 8_000, u36).planned, 8_000);
  assert.equal(
    flashNextLaunchContext({ LUMEN_MAX_PROMPT_TOKENS: "32768", LUMEN_MAX_CTX: "4096" }, 81_920, u36).planned,
    4_096,
  );
  assert.equal(flashNextLaunchContext({ LUMEN_MAX_PROMPT_TOKENS: "32768", LUMEN_MAX_CTX: "200000" }, 81_920, u36).planned, 32_768);
  assert.equal(flashNextLaunchContext({}, 4_096, u36).planned, 4_096);
  // The Prefill card (LUMEN_PREFILL_CHUNK) is the other families' setting and is not read.
  assert.equal(flashNextLaunchContext({ LUMEN_PREFILL_CHUNK: "40960" }, 81_920, u36).planned, 16_384);
  // Values the server's parser would refuse are ignored.
  const junk = flashNextLaunchContext({ LUMEN_MAX_PROMPT_TOKENS: "abc", LUMEN_MAX_CTX: "-5" }, 81_920, u36);
  assert.deepEqual([junk.planned, junk.explicit], [16_384, false]);
  assert.equal(flashNextLaunchContext({ LUMEN_MAX_PROMPT_TOKENS: "0" }, 81_920, u36).planned, 16_384);
});

test("the longest prompt that fits is found by search", () => {
  const usable = flashNextUsableBytes(16);
  assert.equal(flashNextMaxContext(usable), 34_879);
  assert.ok(flashNextPlanFits(flashNextPlan(usable, null, 34_879)));
  assert.ok(!flashNextPlanFits(flashNextPlan(usable, null, 34_880)));
  assert.equal(flashNextMaxContext(flashNextUsableBytes(36)), FLASH_NEXT.maxContext);
  assert.equal(flashNextMaxContext(flashNextUsableBytes(8)), 0);
  // A bigger pool leaves less.
  const u36 = flashNextUsableBytes(36);
  assert.ok(flashNextMaxContext(u36, null, 20e9) < flashNextMaxContext(u36, null, 10e9));
});

test("a 14 GB Mac is the smallest that runs the model", () => {
  assert.ok(!flashNextFits(13));
  assert.ok(flashNextFits(14));
  assert.ok(flashNextFits(16));
  assert.ok(!flashNextFits(8));
  assert.equal(flashNextMinRamGb(), 14);
  // A memory limit can take a big Mac below what it needs.
  assert.ok(flashNextFits(36, 8)); // 8 GiB makes it a tight machine, which still fits
  assert.ok(!flashNextFits(36, 6));
});

test("the estimate sums trunk, pool and reserve; auto spends exactly the usable memory", () => {
  const usable = flashNextUsableBytes(36);
  const auto = flashNextEstimate({ usableBytes: usable, poolBytes: null, chunk: 1024, ctxTokens: 16_384 });
  assert.ok(auto.fits);
  near(auto.residentBytes, usable, 1);
  near(auto.utilization, 1, 1e-9);
  const manual = flashNextEstimate({ usableBytes: usable, poolBytes: 10e9, chunk: 1024, ctxTokens: 16_384 });
  near(manual.residentBytes, 3.6e9 + 10e9 + manual.plan.reserveBytes, 1);
  assert.ok(manual.fits && manual.utilization < 1);
  const over = flashNextEstimate({ usableBytes: usable, poolBytes: 30e9, chunk: 1024, ctxTokens: 16_384 });
  assert.ok(!over.fits && over.utilization > 1);
});

test("a pool under the floor is raised to it, and one under 4.2 GB lends a single staging region", () => {
  const usable = flashNextUsableBytes(36);
  const e = flashNextEstimate({ usableBytes: usable, poolBytes: 1e9, chunk: 1024, ctxTokens: 16_384 });
  assert.equal(e.poolBytes, FLASH_NEXT.poolFloorBytes);
  assert.ok(e.poolRaisedToFloor && e.oneStagingRegion);
  const three = flashNextEstimate({ usableBytes: usable, poolBytes: 3.5e9, chunk: 1024, ctxTokens: 16_384 });
  assert.ok(!three.poolRaisedToFloor && three.oneStagingRegion);
  const six = flashNextEstimate({ usableBytes: usable, poolBytes: 6e9, chunk: 1024, ctxTokens: 16_384 });
  assert.ok(!six.oneStagingRegion);
});

test("a hand-typed pool is bounded by RAM, never below the floor", () => {
  assert.equal(flashNextMaxPoolGb(36), 29);
  assert.equal(flashNextMaxPoolGb(4), 2.82624);
});

// ── Decode speed ────────────────────────────────────────────────────────────

test("the speed table is reproduced at its measured pools", () => {
  for (const p of FLASH_NEXT_DECODE_TPS) {
    if (p.shards != null) {
      const r = flashNextDecodeTps(p.poolGb * 1e9)!;
      near(r.tps, p.shards);
      assert.equal(r.basis, "measured");
    }
    if (p.pack != null) near(flashNextDecodeTps(p.poolGb * 1e9, true)!.tps, p.pack);
  }
});

test("the emulated 16 and 24 GB points are marked as such", () => {
  const emulated = FLASH_NEXT_DECODE_TPS.filter((p) => p.emulated).map((p) => p.poolGb);
  assert.deepEqual(emulated, [4.2, 6.85]);
});

test("between measured pools the speed is a straight line", () => {
  near(flashNextDecodeTps(12.5e9)!.tps, 8.63);
  near(flashNextDecodeTps(12.5e9, true)!.tps, 9.54);
  // Inside the pack's new low range: halfway between 4.2 (5.6) and 6.7 (6.83).
  near(flashNextDecodeTps(5.45e9, true)!.tps, 6.215);
});

test("outside the measured range the table is held above and extrapolated below", () => {
  const above = flashNextDecodeTps(60e9)!;
  assert.deepEqual([above.tps, above.basis], [13, "above"]);
  const below = flashNextDecodeTps(5e9)!;
  assert.equal(below.basis, "below");
  near(below.tps, 5.75 - 1.7 * ((7.53 - 5.75) / 3.3), 1e-9);
  assert.equal(flashNextDecodeTps(-50e9)!.tps, 0);
});

test("the pack speed is not claimed where it was not measured (under 4.2 or over 15 GB)", () => {
  assert.equal(flashNextDecodeTps(19.8e9, true), null);
  assert.equal(flashNextDecodeTps(4.0e9, true), null);
  assert.ok(flashNextDecodeTps(4.2e9, true) !== null);
  assert.ok(flashNextDecodeTps(15e9, true) !== null);
});

test("the disk the model needs adds the optional pack and draft head", () => {
  assert.equal(flashNextDiskBytes({ pack: false, mtp: false }), 97e9);
  assert.equal(flashNextDiskBytes({ pack: true, mtp: true }), 97e9 + 68e9 + 1.5e9);
});

// ── The constants mirror the Rust side (src/flash_next.rs) ──────────────────

test("the sizes, plan constants and speed tables match src/flash_next.rs", () => {
  const rs = readFileSync(new URL("../../../src/flash_next.rs", import.meta.url), "utf8");
  // Rust literals: 3_600_000_000, 2.0 / 3.0, 32 << 30.
  const value = (expr: string): number => {
    const e = expr.replaceAll("_", "").trim();
    const shift = e.match(/^([0-9.]+) << ([0-9.]+)$/);
    if (shift) return Number(shift[1]) * 2 ** Number(shift[2]);
    const div = e.match(/^([0-9.]+) \/ ([0-9.]+)$/);
    if (div) return Number(div[1]) / Number(div[2]);
    return Number(e);
  };
  const num = (name: string): number => {
    const m = rs.match(new RegExp(`const ${name}: [a-z0-9]+ = ([^;]+);`));
    assert.ok(m, `${name} not found in flash_next.rs`);
    return value(m[1]);
  };
  const F = FLASH_NEXT;
  assert.equal(num("TRUNK_BYTES"), F.trunkBytes);
  assert.equal(num("EXPERT_BYTES"), F.expertBytes);
  assert.equal(num("POOL_FLOOR_BYTES"), F.poolFloorBytes);
  assert.equal(num("POOL_PREFETCH_BYTES"), F.poolPrefetchBytes);
  assert.equal(num("DEFAULT_CHUNK"), F.defaultChunk);
  assert.equal(num("TIGHT_CHUNK"), F.tightChunk);
  assert.equal(num("DEFAULT_CONTEXT"), F.defaultContext);
  assert.equal(num("TIGHT_CONTEXT"), F.tightContext);
  assert.equal(num("MAX_CONTEXT"), F.maxContext);
  assert.equal(num("TIGHT_BELOW_BYTES"), F.tightBelowBytes);
  assert.equal(num("SCORES_BUDGET_BYTES"), F.scoresBudgetBytes);
  assert.equal(num("TIGHT_SCORES_BUDGET_BYTES"), F.tightScoresBudgetBytes);
  assert.equal(num("RESERVE_BASE_BYTES"), F.reserveBaseBytes);
  assert.equal(num("RESERVE_PER_TOKEN_BYTES"), F.reservePerTokenBytes);
  assert.equal(num("KV_BYTES_PER_TOKEN"), F.kvBytesPerToken);
  assert.equal(num("ATTENTION_HEADS"), F.attentionHeads);
  assert.equal(num("WORKING_SET_SHARE_SMALL"), F.workingSetShareSmall);
  assert.equal(num("WORKING_SET_SHARE_LARGE"), F.workingSetShareLarge);
  assert.equal(num("SMALL_RAM_BYTES"), F.smallRamGib * GIB);
  assert.equal(num("WORKING_SET_FILLABLE"), F.fillableShare);

  const table = (name: string): number[][] => {
    const m = rs.match(new RegExp(`${name}: \\[\\(f64, f64\\); \\d\\] =\\s*\\[([^;]+)\\];`));
    assert.ok(m, `${name} not found`);
    return [...m[1].matchAll(/\(([0-9.]+), ([0-9.]+)\)/g)].map((x) => [Number(x[1]), Number(x[2])]);
  };
  const column = (key: "shards" | "pack") =>
    FLASH_NEXT_DECODE_TPS.flatMap((p) => (p[key] == null ? [] : [[p.poolGb, p[key] as number]]));
  assert.deepEqual(table("DECODE_TPS_SHARDS"), column("shards"));
  assert.deepEqual(table("DECODE_TPS_PACK"), column("pack"));
});

// ── The settings the card stores (src/config.rs `FlashNextConfig`) ──────────

const settings = (patch: Partial<typeof FLASH_NEXT_DEFAULTS> = {}) => ({ ...FLASH_NEXT_DEFAULTS, ...patch });
const everyOptionChosen = settings({
  expert_cache_gb: 12.5,
  max_prompt_tokens: 32_768,
  prefill_chunk: 512,
  pack: "build",
  draft_head: "on",
  io_mode: "direct",
  io_lanes: 16,
});

test("the card is operable only while the Flash-Next model is the selected one", () => {
  assert.equal(flashNextSettingsEnabled("pipenetwork/Qwen3.8-Flash-Next-MLX-4bit"), true);
  assert.equal(flashNextSettingsEnabled("/Users/me/models/pipenetwork--Qwen3.8-Flash-Next-MLX-4bit"), true);
  assert.equal(flashNextSettingsEnabled("qwen4_exp-local"), true);
  for (const other of [
    "mlx-community/Qwen3.6-35B-A3B-mxfp4",
    "Youssofal/Qwen3.8-27B-MTPLX-Optimized-Speed",
    "mlx-community/gemma-4-26b-a4b-4bit",
  ]) {
    assert.equal(flashNextSettingsEnabled(other), false, other);
  }
  // No model selected, or nothing known yet: disabled, never enabled by default.
  assert.equal(flashNextSettingsEnabled(null), false);
  assert.equal(flashNextSettingsEnabled(undefined), false);
  assert.equal(flashNextSettingsEnabled(""), false);
});

test("switching models neither resets nor depends on the stored settings", () => {
  // The predicate looks at the model only; the settings object is the same either way.
  const stored = everyOptionChosen;
  assert.equal(flashNextSettingsEnabled("mlx-community/gemma-4-26b-a4b-4bit"), false);
  assert.equal(flashNextSettingsEnabled("pipenetwork/Qwen3.8-Flash-Next-MLX-4bit"), true);
  assert.deepEqual(stored, everyOptionChosen);
  assert.ok(!flashNextIsDefault(stored));
});

test("the defaults send nothing: an unset option is the server's own default", () => {
  assert.deepEqual(flashNextSettingsEnv(FLASH_NEXT_DEFAULTS), {});
  assert.ok(flashNextIsDefault(FLASH_NEXT_DEFAULTS));
  // Choosing the server's default (32 reader threads) is not choosing.
  assert.deepEqual(flashNextSettingsEnv(settings({ io_lanes: FLASH_NEXT_LANES_DEFAULT })), {});
});

test("every chosen option maps to its variable as a plain value (as FlashNextConfig::env does)", () => {
  // The same expectations as the Rust test `every_set_option_maps_to_its_variable_as_a_plain_value`.
  assert.deepEqual(flashNextSettingsEnv(everyOptionChosen), {
    LUMEN_MOE_EXPERT_CACHE_GB: "12.5",
    LUMEN_MAX_PROMPT_TOKENS: "32768",
    LUMEN_QWEN35_PREFILL_CHUNK: "512",
    LUMEN_MOE_PACK: "build",
    LUMEN_QWEN4_MTP: "on",
    LUMEN_MOE_IO_MODE: "direct",
    LUMEN_MOE_IO_LANES: "16",
  });
  assert.deepEqual(Object.keys(flashNextSettingsEnv(everyOptionChosen)), [...FLASH_NEXT_ENV_KEYS]);
  const env = flashNextSettingsEnv(settings({ expert_cache_gb: 8, pack: "off", draft_head: "off", io_mode: "buffered" }));
  assert.deepEqual(env, {
    LUMEN_MOE_EXPERT_CACHE_GB: "8",
    LUMEN_MOE_PACK: "off",
    LUMEN_QWEN4_MTP: "off",
    LUMEN_MOE_IO_MODE: "buffered",
  });
  assert.equal(flashNextSettingsEnv(settings({ expert_cache_gb: 6.349 })).LUMEN_MOE_EXPERT_CACHE_GB, "6.35");
});

test("a value that could not be one counts as unset, like the Rust side", () => {
  for (const bad of [0, -3, Number.NaN, Number.POSITIVE_INFINITY]) {
    assert.deepEqual(flashNextSettingsEnv(settings({ expert_cache_gb: bad })), {}, String(bad));
  }
  for (const lanes of [0, 65, 1000, 2.5]) {
    assert.deepEqual(flashNextSettingsEnv(settings({ io_lanes: lanes })), {}, String(lanes));
  }
  assert.deepEqual(flashNextSettingsEnv(settings({ max_prompt_tokens: 0, prefill_chunk: 0 })), {});
});

test("an env override replaces the setting of the same name and nothing else", () => {
  const env = flashNextEnv(everyOptionChosen, { LUMEN_MOE_EXPERT_CACHE_GB: "20", LUMEN_MOE_PACK: "off", OTHER: "x" });
  assert.equal(env.LUMEN_MOE_EXPERT_CACHE_GB, "20");
  assert.equal(env.LUMEN_MOE_PACK, "off");
  assert.equal(env.LUMEN_MOE_IO_LANES, "16");
  assert.equal(env.OTHER, "x");
  assert.deepEqual(flashNextShadowedKeys({ LUMEN_MOE_PACK: "off", OTHER: "x", LUMEN_MOE_IO_LANES: "8" }), [
    "LUMEN_MOE_PACK",
    "LUMEN_MOE_IO_LANES",
  ]);
  assert.deepEqual(flashNextShadowedKeys({}), []);
});

test("the Flash-Next names shadow a typed setting only while Flash-Next is selected", () => {
  const typed = new Set(["LUMEN_MAX_PROMPT_TOKENS", "LUMEN_QWEN35_PREFILL_CHUNK", "LUMEN_MOE_PACK", "LUMEN_KV_DISK"]);
  // The prompt cap and the prefill chunk are rows of the env panel for every family.
  assert.equal(envShadowsTypedKey("LUMEN_MAX_PROMPT_TOKENS", typed, false), false);
  assert.equal(envShadowsTypedKey("LUMEN_MAX_PROMPT_TOKENS", typed, true), true);
  assert.equal(envShadowsTypedKey("LUMEN_QWEN35_PREFILL_CHUNK", typed, false), false);
  assert.equal(envShadowsTypedKey("LUMEN_MOE_PACK", typed, true), true);
  // Anything else typed shadows regardless; an untyped name never does.
  assert.equal(envShadowsTypedKey("LUMEN_KV_DISK", typed, false), true);
  assert.equal(envShadowsTypedKey("LUMEN_TEMPERATURE", typed, true), false);
});

test("the settings, their variables and the defaults match the Rust side", () => {
  const cfg = readFileSync(new URL("../../../src/config.rs", import.meta.url), "utf8");
  const srv = readFileSync(new URL("../../../src/server.rs", import.meta.url), "utf8");
  // The field names of `FlashNextConfig` are the keys of the settings object.
  const body = cfg.match(/pub struct FlashNextConfig \{([^}]+)\}/)?.[1];
  assert.ok(body, "FlashNextConfig not found in config.rs");
  const fields = [...body.matchAll(/pub (\w+):/g)].map((m) => m[1]);
  assert.deepEqual(fields, Object.keys(FLASH_NEXT_DEFAULTS));
  // The variables the Rust `env()` writes are the ones listed here, in the same order.
  const envFn = cfg.slice(cfg.indexOf("pub fn env(&self)"));
  const written = [...envFn.slice(0, envFn.indexOf("\n    }\n")).matchAll(/"(LUMEN_[A-Z0-9_]+)"/g)].map((m) => m[1]);
  assert.deepEqual([...new Set(written)], [...FLASH_NEXT_ENV_KEYS]);
  // Each is declared typed, so an override of it shows the shadowed warning.
  const typedBlock = srv.slice(srv.indexOf("pub const TYPED_ENV_KEYS"));
  const typed = new Set([...typedBlock.slice(0, typedBlock.indexOf("];")).matchAll(/"(LUMEN_[A-Z0-9_]+)"/g)].map((m) => m[1]));
  for (const k of FLASH_NEXT_ENV_KEYS) assert.ok(typed.has(k), `${k} is not in TYPED_ENV_KEYS`);
  // The choices and the server's own number of reader threads.
  const variants = (name: string) =>
    [...(cfg.match(new RegExp(`pub enum ${name} \\{([^}]+)\\}`))?.[1] ?? "").matchAll(/^\s{4}([A-Z]\w*),/gm)].map((m) => m[1].toLowerCase());
  assert.deepEqual(variants("FlashNextPack"), ["auto", "off", "build"]);
  assert.deepEqual(variants("FlashNextDraftHead"), ["auto", "on", "off"]);
  assert.deepEqual(variants("FlashNextIoMode"), ["auto", "direct", "buffered"]);
  assert.equal(Number(cfg.match(/FLASH_NEXT_IO_LANES_DEFAULT: u32 = (\d+);/)?.[1]), FLASH_NEXT_LANES_DEFAULT);
  assert.equal(Number(cfg.match(/FLASH_NEXT_IO_LANES_MAX: u32 = (\d+);/)?.[1]), 64);
});

// ── The option pickers ──────────────────────────────────────────────────────

test("a pool size is read as 8, 8g, 8GB or 12.5g, and written as a plain number", () => {
  for (const [text, gb] of [["8", 8], ["8g", 8], ["8G", 8], ["8GB", 8], ["8 gb", 8], [" 12.5g ", 12.5], ["12.5", 12.5], ["0.5g", 0.5]] as const) {
    assert.equal(parseGbInput(text), gb, text);
  }
  for (const bad of ["", "0", "0g", "-3", "abc", "8 gigs", ".5g", "8..5", "g", "8gbb"]) {
    assert.equal(parseGbInput(bad), null, bad);
  }
  assert.equal(formatGbValue(8), "8");
  assert.equal(formatGbValue(12.5), "12.5");
  assert.equal(formatGbValue(8.333), "8.33");
  assert.equal(formatGbValue(12.504), "12.5");
});

test("a prompt length is read as tokens or K of 1,024, and shown as K when it is whole", () => {
  assert.equal(parsePromptInput("32768"), 32_768);
  assert.equal(parsePromptInput("32k"), 32_768);
  assert.equal(parsePromptInput("32K"), 32_768);
  assert.equal(parsePromptInput("1.5k"), 1_536);
  assert.equal(parsePromptInput(" 48 k "), 49_152);
  for (const bad of ["", "0", "0k", "k", "abc", "-5", "1e3"]) assert.equal(parsePromptInput(bad), null, bad);
  assert.equal(formatPromptShort(8_192), "8K");
  assert.equal(formatPromptShort(131_072), "128K");
  assert.equal(formatPromptShort(40_000), "40,000");
});

test("the presets are the ones the pickers promise, none under the floor", () => {
  assert.deepEqual([...FLASH_NEXT_POOL_PRESETS_GB], [4, 6, 8, 12, 16, 20]);
  assert.deepEqual([...FLASH_NEXT_PROMPT_PRESETS], [8192, 16384, 32768, 65536, 131072]);
  assert.deepEqual([...FLASH_NEXT_LANE_PRESETS], [8, 16, 32, 64]);
  assert.equal(FLASH_NEXT_LANES_DEFAULT, 32);
  assert.ok(FLASH_NEXT_POOL_PRESETS_GB.every((g) => g * 1e9 > FLASH_NEXT.poolFloorBytes));
  assert.ok(FLASH_NEXT_PROMPT_PRESETS.includes(FLASH_NEXT.defaultContext as never));
  assert.ok(FLASH_NEXT_PROMPT_PRESETS.includes(FLASH_NEXT.tightContext as never));
});

test("a pool chip is disabled when the pool does not fit beside the current longest prompt", () => {
  const usable = flashNextUsableBytes(36);
  const at = (gb: number | null, ctxTokens: number) =>
    flashNextPoolChip({ usableBytes: usable, chunk: 1024, ctxTokens, gb, pack: false });
  // At 16K a 36 GB Mac leaves an ~18.4 GB pool: 16 GB fits, 20 GB does not.
  assert.ok(at(16, 16_384).fits);
  assert.ok(!at(20, 16_384).fits);
  // The largest pool that fits is told to 0.1 GB, rounded down.
  assert.equal(at(20, 16_384).maxFitGb, 18.3);
  // At 128K the reserve takes ~10.6 GB: only ~11.3 GB is left, so 12 GB goes and 8 GB stays.
  assert.ok(at(8, 131_072).fits);
  assert.ok(!at(12, 131_072).fits);
  assert.ok(at(null, 131_072).fits);
  // Auto is what the plan leaves; the chips share the table's speeds.
  near(at(null, 16_384).poolBytes, flashNextPlan(usable, 1024, 16_384).poolBytes, 1);
  // Nothing fits when the model does not run at this prompt at all.
  const none = flashNextPoolChip({ usableBytes: flashNextUsableBytes(16), chunk: 512, ctxTokens: 131_072, gb: null, pack: false });
  assert.ok(!none.fits);
});

test("a pool chip's speed follows the pack's table when the pack is in use, and says where it is shaky", () => {
  const eight = flashNextChipTps(8e9, true)!;
  assert.ok(eight.withPack);
  // Between the emulated 6.85 GB point and 10 GB: interpolated, resting on an emulated point.
  near(eight.tps, 6.7 + ((8 - 6.85) / (10 - 6.85)) * (8.22 - 6.7), 1e-9);
  assert.ok(eight.emulated);
  // 10 GB is a real measurement; the segment above it is real too.
  assert.ok(!flashNextChipTps(12e9, true)!.emulated);
  // Without the pack, the shipped weights' table; no emulated points there.
  const plain = flashNextChipTps(8e9, false)!;
  assert.ok(!plain.withPack && !plain.emulated);
  near(plain.tps, 5.75 + ((8 - 6.7) / (10 - 6.7)) * (7.53 - 5.75), 1e-9);
  // Under the pack table's range the shard table answers, extrapolated.
  const low = flashNextChipTps(3.5e9, true)!;
  assert.ok(!low.withPack && low.basis === "below");
  // Past 15 GB the pack was not measured: the shard table's last value is held.
  assert.deepEqual([flashNextChipTps(19e9, true)!.withPack, flashNextChipTps(19e9, true)!.basis], [false, "measured"]);
});

test("the emulated flag follows the points the estimate rests on", () => {
  assert.ok(flashNextDecodeTps(4.2e9, true)!.emulated);
  assert.ok(flashNextDecodeTps(5e9, true)!.emulated);
  assert.ok(flashNextDecodeTps(6.85e9, true)!.emulated);
  assert.ok(!flashNextDecodeTps(10e9, true)!.emulated);
  assert.ok(!flashNextDecodeTps(10e9)!.emulated);
});

test("a prompt chip shows the pool auto would leave, and is disabled under the floor", () => {
  const u36 = flashNextUsableBytes(36);
  const u16 = flashNextUsableBytes(16);
  const chip = (usableBytes: number, chunk: number, tokens: number) =>
    flashNextPromptChip({ usableBytes, chunk, tokens });
  near(chip(u36, 1024, 16_384).poolBytes, flashNextPlan(u36, 1024, 16_384).poolBytes, 1);
  assert.ok(chip(u36, 1024, 131_072).fits);
  // A 16 GB Mac: 8K and 16K fit, 128K does not.
  assert.ok(chip(u16, 512, 8_192).fits && chip(u16, 512, 16_384).fits);
  assert.ok(!chip(u16, 512, 131_072).fits);
  // The pool shrinks as the prompt grows.
  assert.ok(chip(u36, 1024, 65_536).poolBytes < chip(u36, 1024, 32_768).poolBytes);
});

test("choices are read like the server reads them", () => {
  assert.equal(envChoice(undefined, ["auto", "off", "build"] as const, "auto"), "auto");
  assert.equal(envChoice(" Build ", ["auto", "off", "build"] as const, "auto"), "build");
  assert.equal(envChoice("nope", ["auto", "off", "build"] as const, "auto"), "auto");
  assert.equal(mtpMode("off"), "off");
  assert.equal(mtpMode("0"), "off");
  assert.equal(mtpMode("TRUE"), "on");
  assert.equal(mtpMode(undefined), "auto");
  assert.equal(lanesValue("16"), 16);
  assert.equal(lanesValue(undefined), 32);
  assert.equal(lanesValue("0"), 32);
  assert.equal(lanesValue("65"), 32);
  assert.equal(lanesValue("abc"), 32);
});

test("the pack is in use when it is complete (auto), asked for (build) or not switched off", () => {
  const complete = { installed: true, pack: "complete", mtp_present: false };
  const absent = { installed: true, pack: "absent", mtp_present: true };
  assert.ok(packInUse("auto", complete));
  assert.ok(!packInUse("auto", absent));
  assert.ok(!packInUse("auto", null));
  assert.ok(packInUse("build", absent));
  assert.ok(!packInUse("off", complete));
  assert.ok(mtpInUse("auto", absent));
  assert.ok(mtpInUse("on", absent));
  assert.ok(!mtpInUse("off", absent));
  assert.ok(!mtpInUse("auto", complete));
  assert.ok(!mtpInUse("auto", null));
});

test("the summary follows the saved settings", () => {
  const usable = flashNextUsableBytes(36);
  const status = { installed: true, pack: "absent", mtp_present: true };
  const dflt = flashNextEffective({ env: {}, contextMax: 81_920, usableBytes: usable, status });
  assert.deepEqual(
    [dflt.poolExplicit, dflt.promptTokens, dflt.promptExplicit, dflt.pack, dflt.head],
    [false, 16_384, false, false, true],
  );
  near(dflt.estimate.poolBytes, flashNextPlan(usable, 1024, 16_384).poolBytes, 1);
  const set = flashNextEffective({
    env: {
      LUMEN_MOE_EXPERT_CACHE_GB: "8",
      LUMEN_MAX_PROMPT_TOKENS: "32768",
      LUMEN_MOE_PACK: "build",
      LUMEN_QWEN4_MTP: "off",
    },
    contextMax: 81_920,
    usableBytes: usable,
    status,
  });
  assert.deepEqual(
    [set.poolExplicit, set.promptTokens, set.promptExplicit, set.pack, set.head],
    [true, 32_768, true, true, false],
  );
  assert.equal(set.estimate.poolBytes, 8e9);
  assert.ok(set.tps!.withPack);
  // The server reads 8g as 8 GB too.
  assert.equal(
    flashNextEffective({ env: { LUMEN_MOE_EXPERT_CACHE_GB: "8g" }, usableBytes: usable, status }).estimate.poolBytes,
    8e9,
  );
  // `auto` written by hand is not an explicit pool.
  assert.ok(!flashNextEffective({ env: { LUMEN_MOE_EXPERT_CACHE_GB: "auto" }, usableBytes: usable, status }).poolExplicit);
});

test("reset puts every option back to the default, and the defaults send nothing", () => {
  assert.ok(!flashNextIsDefault(everyOptionChosen));
  assert.ok(flashNextIsDefault({ ...FLASH_NEXT_DEFAULTS }));
  // Each option on its own is enough to make the card resettable.
  for (const [k, v] of Object.entries(everyOptionChosen)) {
    assert.ok(!flashNextIsDefault(settings({ [k]: v })), k);
  }
});
