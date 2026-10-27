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
  FLASH_NEXT_SPEED_MODEL,
  attnScoreBytes,
  envKeysForFamily,
  estimateAt,
  flashNextDecodeTps,
  flashNextHardware,
  flashNextHitRate,
  flashNextSlowSsd,
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
  flashNextLiveHeadroom,
  flashNextUsableNow,
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

test("KV is 29,952 bytes per token: 13 layers x (K,V of 2 heads x 256 dims + a 128-dim indexer key), bf16", () => {
  assert.equal(FLASH_NEXT.kvBytesPerToken, 13 * (2 * 2 * 256 * 2 + 128 * 2));
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

// Same cases as `a_server_started_now_plans_within_what_other_apps_leave` in flash_next.rs.
test("a server started now plans within what other apps leave, less 2 GB or 5% of RAM", () => {
  const ram = 36 * GIB;
  assert.equal(flashNextLiveHeadroom(ram), 2e9);
  assert.equal(flashNextLiveHeadroom(64 * GIB), 3_435_973_836);
  assert.equal(flashNextLiveHeadroom(16 * GIB), 2e9);
  assert.equal(flashNextLiveHeadroom(24 * GIB), 2e9);
  const usable = flashNextUsableBytes(36);
  // Idle: more free than the GPU may use, so the plan is the usual one.
  assert.equal(flashNextUsableNow(usable, ram, 30e9, 0), usable);
  // An editor, a browser and chat apps open: 17.7 GB reclaimable, 15.7 GB to plan within.
  assert.equal(flashNextUsableNow(usable, ram, 17_707_089_920, 0), 15_707_089_920);
  // A running server's memory is a restarted one's too.
  assert.equal(flashNextUsableNow(usable, ram, 7e9, 10e9), 15e9);
  // Less than the headroom free is nothing.
  assert.equal(flashNextUsableNow(usable, 16 * GIB, 1e9, 0), 0);
});

test("the reserve adds buffers, chunk activations, KV of the longest prompt and half the scores", () => {
  const usable = flashNextUsableBytes(36);
  const p = flashNextPlan(usable);
  assert.deepEqual([p.chunk, p.ctxTokens, p.tight], [1024, 16_384, false]);
  const scores = 24 * 1024 * 16_384 * 4; // 1.61 GB, under the 2 GB budget
  assert.equal(p.buffersBytes, 600_000_000 + 1024 * 1_100_000);
  assert.equal(p.kvBytes, 16_384 * 29_952);
  assert.equal(p.scoresBytes, scores / 2);
  assert.equal(p.reserveBytes, p.buffersBytes + p.kvBytes + p.scoresBytes);
  near(p.poolBytes, usable - 3.6e9 - p.reserveBytes, 1);
  // Past the 2 GB budget the scores stop growing.
  assert.equal(flashNextPlan(usable, null, 65_536).scoresBytes, 1_000_000_000);
});

test("a tight machine takes the smaller chunk, context and scores budget", () => {
  const p = flashNextPlan(flashNextUsableBytes(16));
  assert.deepEqual([p.chunk, p.ctxTokens, p.tight], [512, 8192, true]);
  assert.equal(p.reserveBytes, 1_609_893_376);
  near(p.poolBytes, 4_868_963_211, 2);
  // 24 GB is not tight: 1,024-token chunks, 16K prompts, an ~8 GB pool.
  const q = flashNextPlan(flashNextUsableBytes(24));
  assert.deepEqual([q.chunk, q.ctxTokens, q.tight], [1024, 16_384, false]);
  near(q.poolBytes, 8_495_844_945, 2);
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
  near(big.poolBytes, 16_732_037_898, 2);
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
  assert.equal(flashNextMaxContext(usable), 69_758);
  assert.ok(flashNextPlanFits(flashNextPlan(usable, null, 69_758)));
  assert.ok(!flashNextPlanFits(flashNextPlan(usable, null, 69_759)));
  assert.equal(flashNextMaxContext(flashNextUsableBytes(36)), FLASH_NEXT.maxContext);
  assert.equal(flashNextMaxContext(flashNextUsableBytes(8)), 0);
  // A bigger pool leaves less.
  const u36 = flashNextUsableBytes(36);
  assert.ok(flashNextMaxContext(u36, null, 20e9) < flashNextMaxContext(u36, null, 10e9));
});

test("a 13 GB Mac is the smallest that runs the model", () => {
  assert.ok(!flashNextFits(12));
  assert.ok(flashNextFits(13));
  assert.ok(flashNextFits(16));
  assert.ok(!flashNextFits(8));
  assert.equal(flashNextMinRamGb(), 13);
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

// ── Decode speed: the bandwidth/SSD model ───────────────────────────────────

const M = FLASH_NEXT_SPEED_MODEL;
const hitPoints = M.hitRatePoints as readonly (readonly [number, number])[];
const REF = { memoryGbps: M.referenceMemoryGbps, ssdRandomGbps: M.referenceSsdGbps };

/** The model written out step by step from a set of constants, the way `by_hand` in
 *  `flash_next.rs` does: what the model is, so changing the constants moves both sides together. */
function byHand(
  c: { streamGb: number; otherMs: number; expertGbPerToken: number; points: readonly (readonly [number, number])[]; effShards: number; effPack: number },
  pool: number,
  mem: number,
  ssd: number,
  pack: boolean,
): number {
  // The engine runs no pool under its floor.
  pool = Math.max(pool, FLASH_NEXT.poolFloorBytes / 1e9);
  const p = c.points;
  let hit: number;
  if (pool >= p[p.length - 1][0]) hit = p[p.length - 1][1];
  else {
    let i = 1;
    while (i < p.length - 1 && pool >= p[i][0]) i++;
    const [a, b] = [p[i - 1], p[i]];
    hit = Math.max(0, a[1] + ((pool - a[0]) * (b[1] - a[1])) / (b[0] - a[0]));
  }
  const ms =
    (c.streamGb / mem) * 1000 + c.otherMs + (((1 - hit) * c.expertGbPerToken) / (ssd * (pack ? c.effPack : c.effShards))) * 1000;
  return 1000 / ms;
}

const tsConstants = {
  streamGb: M.streamGb,
  otherMs: M.otherMs,
  expertGbPerToken: M.expertGbPerToken,
  points: hitPoints,
  effShards: M.ssdEfficiencyShards,
  effPack: M.ssdEfficiencyPack,
};

test("the model is the formula over its constants", () => {
  for (const pool of [0.5, 2.3, 4.2, 6.7, 8, 12.5, 19.8, 40]) {
    for (const mem of [100, 281, 400]) {
      for (const ssd of [0.5, 3.1, 6.57, 14]) {
        for (const pack of [false, true]) {
          near(flashNextDecodeTps(pool, mem, ssd, pack)!, byHand(tsConstants, pool, mem, ssd, pack), 1e-9);
        }
      }
    }
  }
});

test("the model's constants, hit-rate points and reference table match src/flash_next.rs", () => {
  const rs = readFileSync(new URL("../../../src/flash_next.rs", import.meta.url), "utf8");
  const num = (name: string): number => {
    const m = rs.match(new RegExp(`const ${name}: f64 = ([^;]+);`));
    assert.ok(m, `${name} not found in flash_next.rs`);
    return Number(m[1].replaceAll("_", "").trim());
  };
  assert.equal(num("STREAM_GB"), M.streamGb);
  assert.equal(num("OTHER_MS"), M.otherMs);
  assert.equal(num("EXPERT_GB_PER_TOKEN"), M.expertGbPerToken);
  assert.equal(num("SSD_EFFICIENCY_SHARDS"), M.ssdEfficiencyShards);
  assert.equal(num("SSD_EFFICIENCY_PACK"), M.ssdEfficiencyPack);
  assert.equal(num("REFERENCE_MEMORY_GBPS"), M.referenceMemoryGbps);
  assert.equal(num("REFERENCE_READ_GBPS"), M.referenceSsdGbps);
  assert.equal(num("SLOW_READ_GBPS"), M.slowSsdGbps);
  const m = rs.match(/HIT_RATE_POINTS: \[\(f64, f64\); \d\] =\s*\[([^;]+)\];/);
  assert.ok(m, "HIT_RATE_POINTS not found");
  const rsPoints = [...m[1].matchAll(/\(([0-9.]+), ([0-9.]+)\)/g)].map((x) => [Number(x[1]), Number(x[2])]);
  assert.deepEqual(rsPoints, hitPoints.map((p) => [...p]));

  // The same numbers out of both sides: the Rust constants run through the same formula.
  const rsConstants = {
    streamGb: num("STREAM_GB"),
    otherMs: num("OTHER_MS"),
    expertGbPerToken: num("EXPERT_GB_PER_TOKEN"),
    points: rsPoints as [number, number][],
    effShards: num("SSD_EFFICIENCY_SHARDS"),
    effPack: num("SSD_EFFICIENCY_PACK"),
  };
  for (const pool of [2.9, 4.2, 8, 15, 19.8, 30]) {
    for (const [mem, ssd] of [[281, 6.57], [150, 2], [500, 12]]) {
      for (const pack of [false, true]) {
        near(flashNextDecodeTps(pool, mem, ssd, pack)!, byHand(rsConstants, pool, mem, ssd, pack), 1e-9);
      }
    }
  }
  // The constants describe what a token streams, as the Rust test checks too.
  near(M.expertGbPerToken, (48 * 10 * FLASH_NEXT.expertBytes) / 1e9, 1e-3);
  near(M.streamGb, M.expertGbPerToken + FLASH_NEXT.trunkBytes / 1e9, 1e-9);
});

test("at the reference hardware the model reproduces every measured point (12 cells) within 4%", () => {
  for (const p of FLASH_NEXT_DECODE_TPS) {
    for (const [col, pack] of [["shards", false], ["pack", true]] as const) {
      const measured = p[col];
      if (measured == null) continue;
      const got = flashNextDecodeTps(p.poolGb, REF.memoryGbps, REF.ssdRandomGbps, pack)!;
      assert.ok(Math.abs(got - measured) / measured <= 0.04, `${p.poolGb} GB ${col}: model ${got.toFixed(2)} vs measured ${measured}`);
    }
  }
});

test("the hit rate follows its points, holds past the last, and continues the first slope below (never under 0)", () => {
  for (const [pool, hit] of hitPoints) near(flashNextHitRate(pool), hit, 1e-12);
  const [a, b] = [hitPoints[0], hitPoints[1]];
  near(flashNextHitRate((a[0] + b[0]) / 2), (a[1] + b[1]) / 2, 1e-12);
  const last = hitPoints[hitPoints.length - 1];
  assert.equal(flashNextHitRate(last[0] + 50), last[1]);
  near(flashNextHitRate(a[0] - 1), a[1] - (b[1] - a[1]) / (b[0] - a[0]), 1e-12);
  assert.ok(flashNextHitRate(0) >= 0 && flashNextHitRate(-100) >= 0);
});

test("a bigger pool, faster memory or a faster disk is never slower, and the pack beats the shards", () => {
  const tps = (pool: number, mem: number, ssd: number, pack: boolean) => flashNextDecodeTps(pool, mem, ssd, pack)!;
  for (const pack of [false, true]) {
    const pools = [3, 4.2, 6.7, 8, 10, 15, 19.8];
    for (let i = 1; i < pools.length; i++) assert.ok(tps(pools[i], 281, 6.57, pack) > tps(pools[i - 1], 281, 6.57, pack), `pool ${pools[i]}`);
    // Past the last hit-rate point the pool buys nothing more.
    assert.equal(tps(30, 281, 6.57, pack), tps(19.8, 281, 6.57, pack));
    const mems = [100, 200, 281, 400, 800];
    for (let i = 1; i < mems.length; i++) assert.ok(tps(8, mems[i], 6.57, pack) > tps(8, mems[i - 1], 6.57, pack), `memory ${mems[i]}`);
    const ssds = [0.5, 1, 3.1, 6.57, 12, 25];
    for (let i = 1; i < ssds.length; i++) assert.ok(tps(8, 281, ssds[i], pack) > tps(8, 281, ssds[i - 1], pack), `ssd ${ssds[i]}`);
  }
  for (const pool of [2.9, 4.2, 8, 15]) {
    for (const [mem, ssd] of [[281, 6.57], [150, 2], [500, 12]]) {
      assert.ok(tps(pool, mem, ssd, true) > tps(pool, mem, ssd, false), `${pool} ${mem} ${ssd}`);
    }
  }
});

test("a faster machine than the reference one is promised more, not clamped", () => {
  const at = flashNextDecodeTps(10, REF.memoryGbps, REF.ssdRandomGbps, false)!;
  assert.ok(flashNextDecodeTps(10, 2 * REF.memoryGbps, REF.ssdRandomGbps, false)! > at);
  assert.ok(flashNextDecodeTps(10, REF.memoryGbps, 3 * REF.ssdRandomGbps, false)! > at);
  assert.ok(flashNextDecodeTps(10, REF.memoryGbps, 0.5, false)! < 1);
});

test("a missing or non-positive input gives no estimate, never zero or infinity", () => {
  for (const bad of [0, -1, Number.NaN, Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY, null, undefined]) {
    assert.equal(flashNextDecodeTps(bad as number, 281, 6.57, false), null, `pool ${bad}`);
    assert.equal(flashNextDecodeTps(8, bad, 6.57, false), null, `memory ${bad}`);
    assert.equal(flashNextDecodeTps(8, 281, bad, true), null, `ssd ${bad}`);
  }
  const crawl = flashNextDecodeTps(2.9, 1e-3, 1e-3, false)!;
  assert.ok(Number.isFinite(crawl) && crawl > 0);
});

test("the reference table is the measured one, with no emulated points", () => {
  assert.deepEqual(
    FLASH_NEXT_DECODE_TPS.map((p) => [p.poolGb, p.shards, p.pack]),
    [
      [2.83, 4.72, 5.44],
      [4.2, 5.58, 6.14],
      [6.7, 6.69, 7.15],
      [10, 8.06, 8.96],
      [15, 10.39, 11.15],
      [19.8, 11.96, 12.95],
    ],
  );
  assert.ok(FLASH_NEXT_DECODE_TPS.every((p) => !("emulated" in p)));
  // The reference Mac is the one the model was calibrated on.
  assert.deepEqual([REF.memoryGbps, REF.ssdRandomGbps], [281, 6.57]);
});

test("a pool under the engine's floor is estimated at the floor", () => {
  const floorGb = FLASH_NEXT.poolFloorBytes / 1e9;
  for (const pack of [false, true]) {
    const at = flashNextDecodeTps(floorGb, REF.memoryGbps, REF.ssdRandomGbps, pack)!;
    for (const pool of [0.1, 1, 2.5, floorGb - 1e-6]) {
      assert.equal(flashNextDecodeTps(pool, REF.memoryGbps, REF.ssdRandomGbps, pack), at, `${pool} GB`);
    }
    assert.ok(flashNextDecodeTps(floorGb + 0.5, REF.memoryGbps, REF.ssdRandomGbps, pack)! > at);
  }
});

// ── What is measured on this Mac ────────────────────────────────────────────

const hwView = (over: Partial<NonNullable<Parameters<typeof flashNextHardware>[0]>> = {}) => ({
  memory_gbps: 231 as number | null,
  memory_measured_at: 1_000 as number | null,
  ssd_random_gbps: 3.1 as number | null,
  ssd_sequential_gbps: 5 as number | null,
  ssd_measured_at: 2_000 as number | null,
  ssd_stale: false,
  ...over,
});

test("an estimate needs both numbers, and says which are missing", () => {
  const both = flashNextHardware(hwView());
  assert.deepEqual(both.hw, { memoryGbps: 231, ssdRandomGbps: 3.1 });
  assert.deepEqual(both.missing, []);
  // The older of the two times is what the card dates the estimate by.
  assert.equal(both.measuredAt, 1_000);
  const none = flashNextHardware(hwView({ memory_gbps: null, memory_measured_at: null, ssd_random_gbps: null, ssd_measured_at: null }));
  assert.deepEqual([none.hw, none.missing, none.measuredAt], [null, ["memory", "ssd"], null]);
  const memoryOnly = flashNextHardware(hwView({ ssd_random_gbps: null, ssd_measured_at: null }));
  assert.deepEqual([memoryOnly.hw, memoryOnly.missing], [null, ["ssd"]]);
  const ssdOnly = flashNextHardware(hwView({ memory_gbps: null, memory_measured_at: null }));
  assert.deepEqual([ssdOnly.hw, ssdOnly.missing], [null, ["memory"]]);
  // Nothing loaded yet.
  assert.deepEqual(flashNextHardware(undefined).missing, ["memory", "ssd"]);
  assert.equal(flashNextHardware(null).hw, null);
});

test("an SSD number from another model folder counts as not measured", () => {
  const stale = flashNextHardware(hwView({ ssd_random_gbps: null, ssd_measured_at: null, ssd_stale: true }));
  assert.equal(stale.hw, null);
  assert.deepEqual(stale.missing, ["ssd"]);
  assert.ok(stale.ssdStale);
  // Even if a number came along with the flag, the stale one is never used.
  assert.equal(flashNextHardware(hwView({ ssd_stale: true })).hw, null);
});

test("a number that is not a positive speed does not count", () => {
  for (const bad of [0, -3, Number.NaN]) {
    assert.deepEqual(flashNextHardware(hwView({ memory_gbps: bad })).missing, ["memory"], String(bad));
    assert.deepEqual(flashNextHardware(hwView({ ssd_random_gbps: bad })).missing, ["ssd"], String(bad));
  }
});

test("the slow-disk warning follows the measured random speed", () => {
  assert.ok(flashNextSlowSsd({ memoryGbps: 281, ssdRandomGbps: 1.99 }));
  assert.ok(!flashNextSlowSsd({ memoryGbps: 281, ssdRandomGbps: 2 }));
  assert.ok(!flashNextSlowSsd(null), "no warning without a measurement");
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
  assert.equal(num("LIVE_HEADROOM_BYTES"), F.liveHeadroomBytes);
  assert.equal(num("LIVE_HEADROOM_RAM_SHARE"), F.liveHeadroomRamShare);

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

test("the draft head is off by default and sent only when chosen (auto / on)", () => {
  assert.equal(FLASH_NEXT_DEFAULTS.draft_head, "off");
  assert.equal(flashNextSettingsEnv(settings({ draft_head: "off" })).LUMEN_QWEN4_MTP, undefined);
  assert.equal(flashNextSettingsEnv(settings({ draft_head: "auto" })).LUMEN_QWEN4_MTP, "auto");
  assert.equal(flashNextSettingsEnv(settings({ draft_head: "on" })).LUMEN_QWEN4_MTP, "on");
  assert.ok(flashNextIsDefault(settings({ draft_head: "off" })));
  assert.ok(!flashNextIsDefault(settings({ draft_head: "auto" })));
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
  const env = flashNextSettingsEnv(settings({ expert_cache_gb: 8, pack: "off", draft_head: "auto", io_mode: "buffered" }));
  assert.deepEqual(env, {
    LUMEN_MOE_EXPERT_CACHE_GB: "8",
    LUMEN_MOE_PACK: "off",
    LUMEN_QWEN4_MTP: "auto",
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
  assert.deepEqual(variants("FlashNextDraftHead"), ["off", "auto", "on"]);
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
    flashNextPoolChip({ usableBytes: usable, chunk: 1024, ctxTokens, gb, pack: false, hw: null });
  // At 16K a 36 GB Mac leaves an ~18.9 GB pool: 16 GB fits, 20 GB does not.
  assert.ok(at(16, 16_384).fits);
  assert.ok(!at(20, 16_384).fits);
  // The largest pool that fits is told to 0.1 GB, rounded down.
  assert.equal(at(20, 16_384).maxFitGb, 18.8);
  // At 128K the reserve takes ~6.7 GB: ~15.3 GB is left, so 16 GB goes and 12 GB stays.
  assert.ok(at(12, 131_072).fits);
  assert.ok(!at(16, 131_072).fits);
  assert.ok(at(null, 131_072).fits);
  // Auto is what the plan leaves.
  near(at(null, 16_384).poolBytes, flashNextPlan(usable, 1024, 16_384).poolBytes, 1);
  // Nothing fits when the model does not run at this prompt at all.
  const none = flashNextPoolChip({ usableBytes: flashNextUsableBytes(16), chunk: 512, ctxTokens: 131_072, gb: null, pack: false, hw: null });
  assert.ok(!none.fits);
});

test("a pool chip shows no speed until this Mac is measured, then the model's", () => {
  const usable = flashNextUsableBytes(36);
  const chip = (gb: number | null, pack: boolean, hw: typeof REF | null) =>
    flashNextPoolChip({ usableBytes: usable, chunk: 1024, ctxTokens: 16_384, gb, pack, hw });
  // Not measured: no speed on any chip, and none from the reference table.
  assert.equal(chip(8, false, null).tps, null);
  assert.equal(chip(null, true, null).tps, null);
  assert.equal(flashNextChipTps(8e9, true, null), null);
  // Measured: the model at this Mac's numbers, the pack's efficiency when the pack is in use.
  const mine = { memoryGbps: 231, ssdRandomGbps: 3.1 };
  near(chip(8, false, mine).tps!.tps, flashNextDecodeTps(8, 231, 3.1, false)!, 1e-12);
  near(chip(8, true, mine).tps!.tps, flashNextDecodeTps(8, 231, 3.1, true)!, 1e-12);
  assert.deepEqual([chip(8, true, mine).tps!.withPack, chip(8, false, mine).tps!.withPack], [true, false]);
  // Auto is the pool the plan leaves.
  near(chip(null, false, mine).tps!.tps, flashNextDecodeTps(chip(null, false, mine).poolBytes / 1e9, 231, 3.1, false)!, 1e-12);
  // A bigger pool on the same Mac is faster.
  assert.ok(chip(12, false, mine).tps!.tps > chip(6, false, mine).tps!.tps);
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
  assert.equal(mtpMode("on"), "on");
  assert.equal(mtpMode(" Auto "), "auto");
  // Unset is off: the server's own default.
  assert.equal(mtpMode(undefined), "off");
  assert.equal(mtpMode("nope"), "off");
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
  // The head is used only when chosen (auto / on) and installed; off is never used.
  assert.ok(mtpInUse("auto", absent));
  assert.ok(mtpInUse("on", absent));
  assert.ok(!mtpInUse("off", absent));
  assert.ok(!mtpInUse("auto", complete), "no head installed");
  assert.ok(!mtpInUse("auto", null));
});

test("the summary follows the saved settings", () => {
  const usable = flashNextUsableBytes(36);
  const status = { installed: true, pack: "absent", mtp_present: true };
  const dflt = flashNextEffective({ env: {}, contextMax: 81_920, usableBytes: usable, status });
  assert.deepEqual(
    [dflt.poolExplicit, dflt.promptTokens, dflt.promptExplicit, dflt.pack, dflt.head],
    [false, 16_384, false, false, false],
  );
  // The head is off by default even with the file installed; choosing auto uses it.
  assert.ok(
    flashNextEffective({ env: { LUMEN_QWEN4_MTP: "auto" }, contextMax: 81_920, usableBytes: usable, status }).head,
  );
  near(dflt.estimate.poolBytes, flashNextPlan(usable, 1024, 16_384).poolBytes, 1);
  const set = flashNextEffective({
    env: {
      LUMEN_MOE_EXPERT_CACHE_GB: "8",
      LUMEN_MAX_PROMPT_TOKENS: "32768",
      LUMEN_MOE_PACK: "build",
      LUMEN_QWEN4_MTP: "on",
    },
    contextMax: 81_920,
    usableBytes: usable,
    status,
  });
  assert.deepEqual(
    [set.poolExplicit, set.promptTokens, set.promptExplicit, set.pack, set.head],
    [true, 32_768, true, true, true],
  );
  assert.equal(set.estimate.poolBytes, 8e9);
  // No measurement, no speed: the summary never quotes a reference machine.
  assert.equal(set.tps, null);
  assert.equal(dflt.tps, null);
  const measured = flashNextEffective({
    env: { LUMEN_MOE_EXPERT_CACHE_GB: "8", LUMEN_MOE_PACK: "build" },
    usableBytes: usable,
    status,
    hw: { memoryGbps: 231, ssdRandomGbps: 3.1 },
  });
  assert.ok(measured.tps!.withPack);
  near(measured.tps!.tps, flashNextDecodeTps(8, 231, 3.1, true)!, 1e-12);
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
