/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import type { DurableSelfPlayDecision } from "@/workflows/self-play-game";
import { summarizeSearchRuntime } from "./search-runtime-summary";

function decision(
  completedDepth: number,
  backend?: DurableSelfPlayDecision["searchBackend"]
): DurableSelfPlayDecision {
  return {
    turnNumber: 1,
    player: "RED",
    fen: "before",
    resultingFen: "after",
    personality: "balanced",
    agentId: "balanced-a3",
    opponentId: "fortress-a3",
    selectedMoves: ["skip"],
    selectedRank: 1,
    candidateTurns: [],
    currentPlayerScore: 0,
    winProbability: 0.5,
    completedDepth,
    searchTelemetry: {
      nodes: completedDepth * 10,
      elapsedMs: 100,
      ruleFilteredActions: 0,
      beamPrunedActions: 0,
      partialTurnsPruned: 0,
      completeTurnsGenerated: completedDepth * 20,
      completeTurnsDeduplicated: 0,
      completeTurnsPruned: 0,
      tacticallyUnsafeTurns: 0,
      rotationQuotaPruned: 0,
      purposeFilteredTurns: 0,
      valueModelEvaluations: 0,
      turnCacheHits: 0,
      transpositionHits: 0,
    },
    timedOut: false,
    fallback: "none",
    searchBackend: backend,
    searchValueModelBackend:
      backend === "native-python" ? "native-gbdt" : undefined,
    explorationSeed: 1,
    explorationTemperature: 0,
    features: [],
    completedTurn: true,
  };
}

describe("self-play search runtime summary", () => {
  it("exposes backend provenance, depth coverage, and search work", () => {
    const guarded = decision(2, "native-python");
    guarded.searchTelemetry!.hqExactReturnProbeUsed = true;
    guarded.searchTelemetry!.tacticalReturnGuardUsed = true;
    guarded.searchTelemetry!.safeFallbackReplyVerified = true;
    guarded.searchTelemetry!.seedReplyVerified = true;
    guarded.searchTelemetry!.seedSafetyRetryUsed = true;
    guarded.searchTelemetry!.seedSafetyRetryVerified = true;
    const summary = summarizeSearchRuntime([
      {
        decisions: [
          guarded,
          decision(1, "native-python"),
          decision(0),
        ],
      },
    ]);

    expect(summary.backendCounts).toEqual({
      "native-python": 2,
      unknown: 1,
    });
    expect(summary.valueModelBackendCounts).toEqual({
      "native-gbdt": 2,
      unknown: 1,
    });
    expect(summary.completedDepthCounts).toEqual({ "0": 1, "1": 1, "2": 1 });
    expect(summary.averageCompletedDepth).toBe(1);
    expect(summary.depthAtLeastTwoRate).toBeCloseTo(1 / 3);
    expect(summary.zeroDepthRate).toBeCloseTo(1 / 3);
    expect(summary.averageNodes).toBe(10);
    expect(summary.averageCompleteTurnsGenerated).toBe(20);
    expect(summary.hqExactReturnProbeDecisions).toBe(1);
    expect(summary.tacticalReturnGuardDecisions).toBe(1);
    expect(summary.tacticalReturnGuardRate).toBeCloseTo(1 / 3);
    expect(summary.safeFallbackReplyVerifiedDecisions).toBe(1);
    expect(summary.safeFallbackReplyVerifiedRate).toBeCloseTo(1 / 3);
    expect(summary.seedReplyVerifiedDecisions).toBe(1);
    expect(summary.seedReplyVerifiedRate).toBeCloseTo(1 / 3);
    expect(summary.seedSafetyRetryDecisions).toBe(1);
    expect(summary.seedSafetyRetryVerifiedDecisions).toBe(1);
    expect(summary.seedSafetyRetryVerificationRate).toBe(1);
  });

  it("returns finite zero rates for an empty generation", () => {
    expect(summarizeSearchRuntime([])).toEqual({
      decisions: 0,
      backendCounts: {},
      valueModelBackendCounts: {},
      completedDepthCounts: {},
      averageCompletedDepth: 0,
      depthAtLeastTwoRate: 0,
      zeroDepthRate: 0,
      averageNodes: undefined,
      averageCompleteTurnsGenerated: undefined,
      hqExactReturnProbeDecisions: 0,
      tacticalReturnGuardDecisions: 0,
      tacticalReturnGuardRate: 0,
      safeFallbackReplyVerifiedDecisions: 0,
      safeFallbackReplyVerifiedRate: 0,
      seedReplyVerifiedDecisions: 0,
      seedReplyVerifiedRate: 0,
      seedSafetyRetryDecisions: 0,
      seedSafetyRetryVerifiedDecisions: 0,
      seedSafetyRetryVerificationRate: 0,
    });
  });
});
