/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";
import { summarizeValueModelArena } from "./arena-results";

function game(
  index: number,
  challengerWins: boolean
): DurableSelfPlayGameResult {
  const challengerRed = index % 2 === 0;
  return {
    generationId: "arena",
    gameId: `arena-${String(index + 1).padStart(4, "0")}`,
    seed: Math.floor(index / 2),
    redAgentId: challengerRed
      ? "balanced-challenger-a3"
      : "balanced-incumbent-a3",
    blueAgentId: challengerRed
      ? "balanced-incumbent-a3"
      : "balanced-challenger-a3",
    redMaxActions: 3,
    blueMaxActions: 3,
    redValueModel: challengerRed ? "challenger" : "incumbent",
    blueValueModel: challengerRed ? "incumbent" : "challenger",
    initialFen: "initial",
    finalFen: "final",
    decisions: [],
    outcome: {
      winner: challengerWins
        ? challengerRed
          ? "RED"
          : "BLUE"
        : challengerRed
        ? "BLUE"
        : "RED",
      termination: "hq-capture",
    },
    completed: true,
    trainingPositions: 0,
    quality: {
      decisions: 0,
      eligibleDecisions: 0,
      completedSearches: 0,
      fallbackDecisions: 0,
      verifiedFallbackDecisions: 0,
      unverifiedFallbackDecisions: 0,
      timedOutDecisions: 0,
      decisive: true,
      trainingEligible: false,
      trainingRejectionReasons: [],
    },
    storage: {
      status: "saved",
      gamePathname: "game",
      trainingPathname: "training",
      trainingSamples: 0,
    },
  };
}

describe("value-model arena promotion gate", () => {
  it("passes a color-balanced, statistically decisive challenger", () => {
    const games = Array.from({ length: 100 }, (_, index) =>
      game(index, index < 70)
    );
    const summary = summarizeValueModelArena(games, 1_000)!;
    expect(summary.challenger.scoreRate).toBe(0.7);
    expect(summary.challenger.byColor.RED.games).toBe(50);
    expect(summary.challenger.byColor.BLUE.games).toBe(50);
    expect(summary.promotionGate).toEqual({ passed: true, reasons: [] });
  });

  it("rejects an incomplete or statistically uncertain arena", () => {
    const games = Array.from({ length: 40 }, (_, index) =>
      game(index, index < 22)
    );
    const summary = summarizeValueModelArena(games, 500)!;
    expect(summary.promotionGate.passed).toBe(false);
    expect(summary.promotionGate.reasons).toContain("fewer-than-100-games");
    expect(summary.promotionGate.reasons).toContain(
      "paired-ci-does-not-clear-50-percent"
    );
  });

  it("does not manufacture pairs from nonadjacent completed games", () => {
    const summary = summarizeValueModelArena(
      [game(0, true), game(2, true), game(3, false)],
      100
    )!;
    expect(summary.games).toBe(3);
    expect(summary.pairs).toBe(1);
    expect(summary.promotionGate.reasons).toContain("incomplete-color-pair");
  });
});
