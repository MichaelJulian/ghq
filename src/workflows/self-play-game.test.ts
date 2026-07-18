/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import {
  actionMadeProgress,
  durableSelfPlayProgressSnapshot,
  durableGameTrainingRejectionReasons,
  isDurableTrainingDecisionEligible,
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
      "excessive-unverified-fallback-rate",
    ]);
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
