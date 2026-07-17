/** @jest-environment node */

import { describe, expect, it, jest } from "@jest/globals";
import { GHQ_STARTING_FEN } from "@/game/analysis/types";
import type { GhqCandidateTurn, GhqSearchResult } from "@/game/analysis/types";
import { analyzeFen, applyHistoryAvoidance } from "./fen-analysis";

jest.setTimeout(30_000);

describe("production FEN analysis", () => {
  it("chooses a near-best novel turn over a multi-move undo cycle", () => {
    const candidate = (
      rank: number,
      actions: string[],
      resulting_fen: string,
      score: number
    ) =>
      ({
        rank,
        actions,
        all_moves: actions,
        automatic_captures: [],
        resulting_fen,
        score,
        action_purposes: [],
        purpose: {},
      } as unknown as GhqCandidateTurn);
    const repeating = candidate(1, ["a2a1", "b2b1", "c1c2"], "old", 4);
    const novel = candidate(2, ["a2a3", "b2c3", "c1d2"], "new", 3.4);
    const result = {
      recommendation_label: "best found",
      best_turn: repeating,
      principal_variation: repeating.all_moves,
      candidate_turns: [repeating, novel],
      score: { current_player: 4, red: 4 },
      search: { opening_book_used: false },
      exploration: {
        temperature: 0,
        seed: 1,
        selectedRank: 1,
        candidateCount: 2,
      },
    } as unknown as GhqSearchResult;

    applyHistoryAvoidance(result, "RED", ["old"], ["a1a2", "b1b2"]);

    expect(result.best_turn.resulting_fen).toBe("new");
    expect(result.recommendation_label).toBe("history avoidance");
    expect(result.score.current_player).toBe(3.4);
    expect(result.exploration?.selectedRank).toBe(2);
  });

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
      "complete-turn seed",
      "opening book",
    ]).toContain(result.search.recommendation_label);
    if (
      result.search.search.completed_depth_in_turns === 0 &&
      !result.search.search.opening_book_used
    ) {
      expect(result.search.search.fallback_used).not.toBe("none");
    }
    expect(result.search.best_turn.purpose.total_penalty).toBe(0);
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
      turnNumber: 5,
      // Cohesion and phase-frontier scoring deliberately inspect complete
      // structures, so leave enough time to finish a real leaf search.
      timeMs: 5000,
      maxDepth: 1,
      beamWidth: 2,
    });

    expect(result.search.search.completed_depth_in_turns).toBe(1);
    expect(result.search.search.value_model_evaluations).toBeGreaterThan(0);
    expect(result.search.search.fallback_used).toBe("none");
    expect(result.search.candidate_turns.length).toBeGreaterThan(1);
  });

  it("samples safe near-best turns reproducibly", async () => {
    const request = {
      fen: GHQ_STARTING_FEN,
      turnNumber: 5,
      timeMs: 5000,
      maxDepth: 1,
      beamWidth: 6,
      explorationTemperature: 0.8,
      explorationSeed: 991,
    } as const;
    const first = await analyzeFen(request);
    const second = await analyzeFen(request);

    expect(first.search.exploration).toEqual(second.search.exploration);
    expect(first.search.best_turn.all_moves).toEqual(
      second.search.best_turn.all_moves
    );
    expect(first.resultingFen).toBe(second.resultingFen);
    expect(first.search.exploration?.candidateCount).toBeGreaterThan(1);
  });
});
