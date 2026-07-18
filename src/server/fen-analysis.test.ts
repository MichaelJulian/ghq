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

    applyHistoryAvoidance(result, "RED", ["old"], [["a1a2", "b1b2"]], 0);

    expect(result.best_turn.resulting_fen).toBe("new");
    expect(result.recommendation_label).toBe("history avoidance");
    expect(result.score.current_player).toBe(3.4);
    expect(result.exploration?.selectedRank).toBe(2);
  });

  it("never replaces a reply-verified HQ survival turn to avoid history", () => {
    const safe = {
      rank: 1,
      actions: ["f2g2", "h5h6", "skip"],
      all_moves: ["f2g2", "h5h6", "skip"],
      automatic_captures: [],
      resulting_fen: "safe-repeated-position",
      score: -1_000_005.7049,
      action_purposes: [],
      purpose: { stagnation_progress: 0 },
    } as unknown as GhqCandidateTurn;
    const immediateLoss = {
      ...safe,
      rank: 2,
      actions: ["f2f3", "h5g4", "skip"],
      all_moves: ["f2f3", "h5g4", "skip"],
      resulting_fen: "novel-but-losing-position",
      score: -1_000_002.649,
      purpose: { stagnation_progress: 4.83 },
    } as unknown as GhqCandidateTurn;
    const result = {
      recommendation_label: "safe fallback",
      best_turn: safe,
      principal_variation: safe.all_moves,
      candidate_turns: [safe, immediateLoss],
      score: { current_player: safe.score, red: safe.score },
      search: {
        opening_book_used: false,
        fallback_used: "safe",
        hq_survival_override_used: true,
        hq_survival_reply_verified: true,
      },
      exploration: {
        temperature: 0,
        seed: 1,
        selectedRank: 1,
        candidateCount: 2,
      },
    } as unknown as GhqSearchResult;

    applyHistoryAvoidance(
      result,
      "RED",
      ["safe-repeated-position"],
      [],
      24
    );

    expect(result.best_turn.resulting_fen).toBe("safe-repeated-position");
    expect(result.recommendation_label).toBe("safe fallback");
    expect(result.exploration?.selectedRank).toBe(1);
  });

  it("widens the quality window late in a quiet multi-turn cycle", () => {
    const stalled = {
      rank: 1,
      actions: ["a2a1", "b2b1", "c2c1"],
      all_moves: ["a2a1", "b2b1", "c2c1"],
      automatic_captures: [],
      resulting_fen: "stalled",
      score: 10,
      action_purposes: [],
      purpose: {
        backfills: 2,
        reversals: 0,
        forcing_gain: 0,
        purposeful_actions: 0,
      },
    } as unknown as GhqCandidateTurn;
    const breaker = {
      ...stalled,
      rank: 2,
      actions: ["a2a3", "b2c3", "c2d3"],
      all_moves: ["a2a3", "b2c3", "c2d3"],
      resulting_fen: "breaker",
      score: 5.5,
      purpose: {
        backfills: 0,
        reversals: 0,
        forcing_gain: 2,
        purposeful_actions: 3,
      },
    } as unknown as GhqCandidateTurn;
    const result = {
      recommendation_label: "best found",
      best_turn: stalled,
      principal_variation: stalled.all_moves,
      candidate_turns: [stalled, breaker],
      score: { current_player: 10, red: 10 },
      search: { opening_book_used: false },
      exploration: {
        temperature: 0,
        seed: 1,
        selectedRank: 1,
        candidateCount: 2,
      },
    } as unknown as GhqSearchResult;

    applyHistoryAvoidance(
      result,
      "RED",
      [],
      [
        ["a1a2", "b1b2", "c1c2"],
        ["d1d2", "e1e2", "f1f2"],
      ],
      24
    );

    expect(result.best_turn.resulting_fen).toBe("breaker");
    expect(result.score.current_player).toBe(5.5);
  });

  it("widens early when an entire turn reverses the previous turn", () => {
    const cycling = {
      rank: 1,
      actions: ["d6d7", "a6a7", "f7f6"],
      all_moves: ["d6d7", "a6a7", "f7f6"],
      automatic_captures: [],
      resulting_fen: "cycled-position",
      score: 18.3915,
      action_purposes: [],
      purpose: {
        backfills: 0,
        reversals: 0,
        forcing_gain: 0.79,
        purposeful_actions: 3,
        stagnation_progress: 0,
      },
    } as unknown as GhqCandidateTurn;
    const breaker = {
      ...cycling,
      rank: 2,
      actions: ["d6e5", "c7d8↓", "skip"],
      all_moves: ["d6e5", "c7d8↓", "skip"],
      resulting_fen: "cycle-breaker",
      score: 16.4362,
      purpose: {
        backfills: 0,
        reversals: 0,
        forcing_gain: 2.0375,
        purposeful_actions: 2,
        stagnation_progress: 1,
      },
    } as unknown as GhqCandidateTurn;
    const result = {
      recommendation_label: "best found",
      best_turn: cycling,
      principal_variation: cycling.all_moves,
      candidate_turns: [cycling, breaker],
      score: { current_player: cycling.score, red: -cycling.score },
      search: { opening_book_used: false, fallback_used: "none" },
      exploration: {
        temperature: 0,
        seed: 1,
        selectedRank: 1,
        candidateCount: 2,
      },
    } as unknown as GhqSearchResult;

    applyHistoryAvoidance(
      result,
      "BLUE",
      [],
      [["d7d6", "a7a6", "f6f7"]],
      5
    );

    expect(result.best_turn.resulting_fen).toBe("cycle-breaker");
    expect(result.recommendation_label).toBe("history avoidance");
  });

  it("rejects cosmetically purposeful shuffles with no durable progress", () => {
    const shuffle = {
      rank: 1,
      actions: ["a2b2", "b3a3", "c2c3"],
      all_moves: ["a2b2", "b3a3", "c2c3"],
      automatic_captures: [],
      resulting_fen: "new-shuffle-square",
      score: 8,
      action_purposes: [],
      purpose: {
        backfills: 0,
        reversals: 0,
        forcing_gain: 2,
        purposeful_actions: 3,
        stagnation_progress: 0,
      },
    } as unknown as GhqCandidateTurn;
    const contactBreak = {
      ...shuffle,
      rank: 2,
      actions: ["a2a4", "b3b4", "c2d3"],
      all_moves: ["a2a4", "b3b4", "c2d3"],
      resulting_fen: "contact-break",
      score: 5.5,
      purpose: {
        ...shuffle.purpose,
        stagnation_progress: 3,
      },
    } as unknown as GhqCandidateTurn;
    const result = {
      recommendation_label: "best found",
      best_turn: shuffle,
      principal_variation: shuffle.all_moves,
      candidate_turns: [shuffle, contactBreak],
      score: { current_player: 8, red: 8 },
      search: { opening_book_used: false },
      exploration: {
        temperature: 0,
        seed: 1,
        selectedRank: 1,
        candidateCount: 2,
      },
    } as unknown as GhqSearchResult;

    applyHistoryAvoidance(result, "RED", [], [], 24);

    expect(result.best_turn.resulting_fen).toBe("contact-break");
    expect(result.recommendation_label).toBe("history avoidance");
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
