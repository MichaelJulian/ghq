/** @jest-environment node */

import { beforeAll, describe, expect, it, jest } from "@jest/globals";
import type { GameEngine } from "@/game/engine-v2";
import { VALUE_FEATURE_NAMES } from "@/game/value-model/features";
import { runSelfPlayGeneration } from "@/game/self-play/run-generation";
import { loadV2Engine } from "@/server/engine";
import { analyzeFen } from "@/server/fen-analysis";

jest.setTimeout(30_000);

let engine: GameEngine;

beforeAll(async () => {
  engine = await loadV2Engine();
});

describe("runSelfPlayGeneration", () => {
  it("runs color-swapped seeded games and emits model-ready records", async () => {
    const shared = {
      timeMs: 100,
      maxDepth: 1,
      beamWidth: 2,
      explorationTemperature: 0.6,
    } as const;
    const generation = await runSelfPlayGeneration({
      generationId: "test-generation",
      engine,
      analyze: analyzeFen,
      population: [
        { id: "balanced-g0", personality: "balanced", ...shared },
        { id: "raider-g0", personality: "mobile_raider", ...shared },
      ],
      games: 2,
      maxTurns: 6,
      concurrency: 1,
      seed: 4242,
    });

    expect(generation.games).toHaveLength(2);
    expect(
      generation.games.map((game) => [
        game.result.redAgentId,
        game.result.blueAgentId,
      ])
    ).toEqual(
      expect.arrayContaining([
        ["balanced-g0", "raider-g0"],
        ["raider-g0", "balanced-g0"],
      ])
    );
    expect(generation.standings).toHaveLength(2);
    expect(generation.metrics.uniqueContinuationCount).toBeGreaterThan(0);
    const records = generation.games.flatMap((game) => game.trainingRecords);
    expect(records.length).toBeGreaterThan(0);
    expect(records[0].features).toHaveLength(VALUE_FEATURE_NAMES.length);
    expect(records[0].candidateTurns).toBeDefined();
  });
});
