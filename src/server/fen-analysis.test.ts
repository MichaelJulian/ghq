/** @jest-environment node */

import { describe, expect, it, jest } from "@jest/globals";
import { GHQ_STARTING_FEN } from "@/game/analysis/types";
import { analyzeFen } from "./fen-analysis";

jest.setTimeout(30_000);

describe("production FEN analysis", () => {
  it("returns a legal complete turn plus model and search outputs", async () => {
    const result = await analyzeFen({
      fen: GHQ_STARTING_FEN,
      personality: "battery_commander",
      turnNumber: 1,
      timeMs: 100,
      maxDepth: 1,
      beamWidth: 2,
    });

    expect(result.sideToMove).toBe("RED");
    expect(result.search.best_turn.actions).toHaveLength(3);
    expect([
      "best move",
      "best found",
      "safe fallback",
      "greedy fallback",
    ]).toContain(result.search.recommendation_label);
    if (result.search.search.completed_depth_in_turns === 0) {
      expect(result.search.search.fallback_used).not.toBe("none");
    }
    expect(result.resultingFen).not.toBe(result.fen);
    expect(result.serializedState.length).toBeGreaterThan(20);
    expect(result.model.before.redWinProbability).toBeGreaterThan(0);
    expect(result.model.before.redWinProbability).toBeLessThan(1);
    expect(result.model.before.personality.personality).toBe(
      "battery_commander"
    );
  });

  it("continues from serialized production state", async () => {
    const first = await analyzeFen({
      fen: GHQ_STARTING_FEN,
      timeMs: 100,
      maxDepth: 1,
      beamWidth: 2,
    });
    const second = await analyzeFen({
      serializedState: first.serializedState,
      turnNumber: 2,
      personality: "mobile_raider",
      timeMs: 100,
      maxDepth: 1,
      beamWidth: 2,
    });

    expect(second.fen).toBe(first.resultingFen);
    expect(second.sideToMove).toBe("BLUE");
    expect(second.personality).toBe("mobile_raider");
  });

  it("uses the trained value model during a completed leaf search", async () => {
    const result = await analyzeFen({
      fen: GHQ_STARTING_FEN,
      turnNumber: 1,
      timeMs: 1000,
      maxDepth: 1,
      beamWidth: 2,
    });

    expect(result.search.search.completed_depth_in_turns).toBe(1);
    expect(result.search.search.value_model_evaluations).toBeGreaterThan(0);
    expect(result.search.search.fallback_used).toBe("none");
  });
});
