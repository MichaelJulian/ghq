/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import {
  actionMadeProgress,
  durableSearchSlotAt,
  durableSelfPlayProgressSnapshot,
  durableTrainingSample,
  durableGameTrainingRejectionReasons,
  isDurableTrainingDecisionEligible,
  resolveDurableInitialState,
  type DurableSelfPlayDecision,
} from "./self-play-game";

function decision(
  overrides: Partial<DurableSelfPlayDecision> = {}
): DurableSelfPlayDecision {
  return {
    turnNumber: 1,
    player: "RED",
    fen: "before",
    resultingFen: "after",
    personality: "balanced",
    agentId: "balanced-a3",
    opponentId: "fortress-a3",
    selectedMoves: ["a1a2", "b1b2", "c1c2"],
    selectedRank: 1,
    candidateTurns: [],
    selectedPurpose: {
      paratrooper_mission_penalty: 0,
    } as NonNullable<DurableSelfPlayDecision["selectedPurpose"]>,
    currentPlayerScore: 0,
    winProbability: 0.5,
    completedDepth: 2,
    timedOut: false,
    fallback: "none",
    searchBackend: "native-python",
    searchValueModelBackend: "native-gbdt",
    searchCodeVersion: "test-code-version",
    explorationSeed: 1,
    explorationTemperature: 0,
    features: [],
    completedTurn: true,
    selfActionLimit: 3,
    opponentActionLimit: 3,
    ...overrides,
  };
}

describe("durable self-play training quality", () => {
  it("preserves behavior-quality telemetry in compact training artifacts", () => {
    const sample = durableTrainingSample(
      decision({
        player: "BLUE",
        fallback: "safe",
        timedOut: true,
        completedDepth: 2,
        selectedMoves: ["a8a7", "b8b7", "c8c7"],
      }),
      {
        generationId: "generation-1",
        gameId: "game-1",
        red: {
          id: "red",
          personality: "balanced",
          timeMs: 20_000,
          maxDepth: 2,
          beamWidth: 6,
          explorationTemperature: 0,
          valueModelCheckpoint: "red-checkpoint",
        },
        blue: {
          id: "blue",
          personality: "fortress",
          timeMs: 20_000,
          maxDepth: 2,
          beamWidth: 6,
          explorationTemperature: 0,
          valueModelCheckpoint: "blue-checkpoint",
        },
      },
      "RED"
    );

    expect(sample).toMatchObject({
      generationId: "generation-1",
      gameId: "game-1",
      player: "BLUE",
      outcomeValue: 0,
      selectedMoves: ["a8a7", "b8b7", "c8c7"],
      completedDepth: 2,
      fallback: "safe",
      timedOut: true,
      valueModelCheckpoint: "blue-checkpoint",
    });
  });

  it("places concurrent games into stable absolute search lanes", () => {
    const schedule = {
      epochMs: 1_000,
      lane: 2,
      laneCount: 3,
      slotMs: 50_000,
    };

    expect(durableSearchSlotAt(schedule, 1)?.getTime()).toBe(101_000);
    expect(durableSearchSlotAt(schedule, 2)?.getTime()).toBe(251_000);
    expect(durableSearchSlotAt(undefined, 2)).toBeUndefined();
  });

  it("derives a counterfactual start's side and absolute turn from its FEN", () => {
    const state = resolveDurableInitialState({
      initialFen: "qr↓6/iii5/8/8/8/8/5III/6R↑Q IIIIIFFFPRRTH iiiiifffprrth b",
      initialTurnNumber: 27,
      dataRole: "counterfactual",
    });

    expect(state).toMatchObject({
      initialTurnNumber: 27,
      initialPlayer: "BLUE",
      dataRole: "counterfactual",
    });
  });

  it("rejects invalid counterfactual turn numbers", () => {
    expect(() => resolveDurableInitialState({ initialTurnNumber: 0 })).toThrow(
      "initialTurnNumber must be a positive integer"
    );
  });

  it("summarizes durable mid-game progress without storing candidate trees", () => {
    const snapshot = durableSelfPlayProgressSnapshot({
      config: {
        generationId: "generation-1",
        gameId: "game-1",
        seed: 0xffff_ffff + 2,
        codeVersion: "revision-1",
        red: {
          id: "balanced-challenger-a3",
          personality: "balanced",
          timeMs: 20_000,
          maxDepth: 2,
          beamWidth: 6,
          explorationTemperature: 0,
          valueModelCheckpoint: "challenger-1",
        },
        blue: {
          id: "balanced-incumbent-a3",
          personality: "balanced",
          timeMs: 20_000,
          maxDepth: 2,
          beamWidth: 6,
          explorationTemperature: 0,
          valueModelCheckpoint: "incumbent-1",
        },
      },
      decisions: [
        decision({ timedOut: true }),
        decision({
          turnNumber: 2,
          player: "BLUE",
          fallback: "safe",
          completedDepth: 2,
        }),
        decision({
          turnNumber: 3,
          fallback: "safe",
          completedDepth: 0,
          fen: "unsafe-before",
          selectedMoves: ["d1d2", "skip"],
          searchTelemetry: {
            nodes: 10,
            elapsedMs: 20_000,
            ruleFilteredActions: 0,
            beamPrunedActions: 0,
            partialTurnsPruned: 0,
            completeTurnsGenerated: 1,
            completeTurnsDeduplicated: 0,
            completeTurnsPruned: 0,
            tacticallyUnsafeTurns: 1,
            rotationQuotaPruned: 0,
            purposeFilteredTurns: 0,
            valueModelEvaluations: 0,
            turnCacheHits: 0,
            transpositionHits: 0,
            seedReplyVerified: true,
            seedSafetyRetryUsed: true,
            seedSafetyRetryVerified: false,
            tacticalReturnGuardUsed: true,
          },
        }),
      ],
      completedTurns: 3,
      currentPlayer: "BLUE",
      currentFen: "position-3",
      status: "running",
    });

    expect(snapshot).toMatchObject({
      format: "ghq-self-play-progress-v1",
      seed: 1,
      decisions: 3,
      completedTurns: 3,
      depthAtLeastTwoDecisions: 2,
      fallbackDecisions: 2,
      unverifiedFallbackDecisions: 1,
      latestUnverifiedFallback: {
        turnNumber: 3,
        player: "RED",
        fen: "unsafe-before",
        selectedMoves: ["d1d2", "skip"],
        completedDepth: 0,
        fallback: "safe",
        timedOut: false,
        seedReplyVerified: true,
        seedSafetyRetryUsed: true,
        seedSafetyRetryVerified: false,
        safeFallbackReplyVerified: false,
        tacticalReturnGuardUsed: true,
      },
      timedOutDecisions: 1,
      redValueModelCheckpoint: "challenger-1",
      blueValueModelCheckpoint: "incumbent-1",
      status: "running",
    });
    expect(snapshot).not.toHaveProperty("candidateTurns");
  });

  it("does not let skip reset the no-progress clock", () => {
    expect(actionMadeProgress("skip")).toBe(false);
    expect(actionMadeProgress("sbe5")).toBe(true);
    expect(actionMadeProgress("ria1")).toBe(true);
  });

  it("requires a complete opponent reply for an individual training label", () => {
    const outcome = { winner: "RED" as const, termination: "hq-capture" };
    expect(isDurableTrainingDecisionEligible(decision(), outcome)).toBe(true);
    expect(
      isDurableTrainingDecisionEligible(
        decision({ completedDepth: 1 }),
        outcome
      )
    ).toBe(false);
    expect(
      isDurableTrainingDecisionEligible(
        decision({ fallback: "safe", completedDepth: 2 }),
        outcome
      )
    ).toBe(true);
    expect(
      isDurableTrainingDecisionEligible(
        decision({ fallback: "seeded", completedDepth: 2 }),
        outcome
      )
    ).toBe(false);
  });

  it("rejects labels and games containing a paratrooper policy violation", () => {
    const outcome = { winner: "RED" as const, termination: "hq-capture" };
    const violating = decision({
      selectedPurpose: {
        paratrooper_mission_penalty: 9,
      } as NonNullable<DurableSelfPlayDecision["selectedPurpose"]>,
    });

    expect(isDurableTrainingDecisionEligible(violating, outcome)).toBe(false);
    expect(durableGameTrainingRejectionReasons([violating], outcome)).toContain(
      "paratrooper-policy-violation"
    );
  });

  it("rejects labels and games with missing paratrooper policy telemetry", () => {
    const outcome = { winner: "RED" as const, termination: "hq-capture" };
    const missingTelemetry = decision({ selectedPurpose: undefined });

    expect(isDurableTrainingDecisionEligible(missingTelemetry, outcome)).toBe(
      false
    );
    expect(
      durableGameTrainingRejectionReasons([missingTelemetry], outcome)
    ).toContain("missing-paratrooper-policy-telemetry");
  });

  it("accepts a clean three-action HQ-capture game", () => {
    expect(
      durableGameTrainingRejectionReasons(
        Array.from({ length: 20 }, (_, index) =>
          decision({ turnNumber: index + 1 })
        ),
        { winner: "RED", termination: "hq-capture" }
      )
    ).toEqual([]);
  });

  it("distinguishes reply-verified safe fallbacks from unverified ones", () => {
    const decisions = Array.from({ length: 20 }, (_, index) =>
      decision({
        turnNumber: index + 1,
        fallback:
          index === 0 ? "seeded" : index === 1 || index === 2 ? "safe" : "none",
        completedDepth: index === 1 ? 2 : index <= 2 ? 0 : 2,
      })
    );
    expect(
      durableGameTrainingRejectionReasons(decisions, {
        winner: "BLUE",
        termination: "hq-capture",
      })
    ).toEqual([
      "unverified-complete-turn-seed",
      "unverified-fallback-decision",
    ]);
  });

  it("quarantines a whole game after one shallow safe fallback", () => {
    const decisions = Array.from({ length: 100 }, (_, index) =>
      decision({
        turnNumber: index + 1,
        fallback: index === 50 ? "safe" : "none",
        completedDepth: index === 50 ? 0 : 2,
      })
    );
    expect(
      durableGameTrainingRejectionReasons(decisions, {
        winner: "RED",
        termination: "hq-capture",
      })
    ).toContain("unverified-fallback-decision");
  });

  it("keeps reply-verified safe fallbacks training-eligible", () => {
    const decisions = Array.from({ length: 20 }, (_, index) =>
      decision({
        turnNumber: index + 1,
        fallback: index === 10 ? "safe" : "none",
        completedDepth: 2,
      })
    );
    expect(
      durableGameTrainingRejectionReasons(decisions, {
        winner: "BLUE",
        termination: "hq-capture",
      })
    ).toEqual([]);
  });

  it("rejects draws and experimental action limits", () => {
    expect(
      durableGameTrainingRejectionReasons([decision({ selfActionLimit: 2 })], {
        termination: "repetition",
      })
    ).toEqual(["not-hq-capture", "nonstandard-action-limit"]);
  });

  it("requires one exact search runtime throughout a training game", () => {
    expect(
      durableGameTrainingRejectionReasons(
        [decision({ searchBackend: undefined })],
        { winner: "RED", termination: "hq-capture" }
      )
    ).toContain("missing-search-runtime-provenance");
    expect(
      durableGameTrainingRejectionReasons(
        [
          decision(),
          decision({
            turnNumber: 2,
            searchBackend: "pyodide",
            searchValueModelBackend: "typescript-callback",
          }),
        ],
        { winner: "RED", termination: "hq-capture" }
      )
    ).toContain("mixed-search-runtime-provenance");
    expect(
      durableGameTrainingRejectionReasons(
        [
          decision(),
          decision({ turnNumber: 2, searchCodeVersion: "different-version" }),
        ],
        { winner: "RED", termination: "hq-capture" },
        "test-code-version"
      )
    ).toEqual(
      expect.arrayContaining([
        "mixed-search-code-version",
        "mismatched-search-code-version",
      ])
    );
  });
});
