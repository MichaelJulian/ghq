import { describe, expect, it } from "@jest/globals";
import type { GhqCandidateTurn } from "@/game/analysis/types";
import type {
  DurableSelfPlayDecision,
  DurableSelfPlayGameResult,
} from "@/workflows/self-play-game";
import {
  counterfactualReplicateSeed,
  counterfactualReplicateEvidence,
  counterfactualRootSeed,
  selectCounterfactualRoots,
} from "./counterfactual";

function candidate(rank: number, score: number): GhqCandidateTurn {
  return {
    rank,
    automatic_captures: [],
    actions: [`a${rank}a${rank + 1}`],
    all_moves: [`a${rank}a${rank + 1}`],
    resulting_fen: `8/8/8/8/8/8/8/8 - - ${rank % 2 ? "b" : "r"}`,
    score,
    action_purposes: [],
    purpose: {} as GhqCandidateTurn["purpose"],
  };
}

function decision(
  turnNumber: number,
  scores: number[],
  player: "RED" | "BLUE" = "RED"
): DurableSelfPlayDecision {
  return {
    turnNumber,
    player,
    fen: "before",
    resultingFen: "after",
    personality: player === "RED" ? "fortress" : "mobile_raider",
    agentId: `${player}-agent`,
    opponentId: "opponent",
    selectedMoves: [],
    selectedRank: 1,
    candidateTurns: scores.map((score, index) => candidate(index + 1, score)),
    currentPlayerScore: scores[0],
    winProbability: 0.5,
    completedDepth: 2,
    timedOut: false,
    fallback: "none",
    explorationSeed: 1,
    explorationTemperature: 0,
    features: [],
    completedTurn: true,
  };
}

function game(gameId: string, decisions: DurableSelfPlayDecision[]) {
  return {
    gameId,
    decisions,
  } as DurableSelfPlayGameResult;
}

describe("counterfactual rollout selection", () => {
  it("selects close, reply-verified candidates and preserves branch metadata", () => {
    const roots = selectCounterfactualRoots(
      [
        game("game-a", [
          decision(4, [1, 1.1]),
          decision(12, [2, 2.4]),
          decision(20, [4, 6]),
          decision(13, [1_000_000, 999_999]),
        ]),
      ],
      { maxRoots: 4, maxScoreMargin: 1 }
    );

    expect(roots).toHaveLength(1);
    expect(roots[0]).toMatchObject({
      rootId: "game-a:t12",
      rootPlayer: "RED",
    });
    expect(roots[0].scoreMargin).toBeCloseTo(0.4);
    expect(roots[0].branches).toEqual([
      expect.objectContaining({
        candidateRank: 1,
        initialTurnNumber: 13,
        redPersonality: "fortress",
      }),
      expect.objectContaining({
        candidateRank: 2,
        initialTurnNumber: 13,
        redPersonality: "fortress",
      }),
    ]);
  });

  it("uses one identical deterministic seed for every branch of a root", () => {
    expect(counterfactualRootSeed(42, "game:t9")).toBe(
      counterfactualRootSeed(42, "game:t9")
    );
    expect(counterfactualRootSeed(42, "game:t9")).not.toBe(
      counterfactualRootSeed(42, "game:t10")
    );
  });

  it("uses matched seeds within a stochastic replicate", () => {
    expect(counterfactualReplicateSeed(42, "game:t9", 0)).toBe(
      counterfactualReplicateSeed(42, "game:t9", 0)
    );
    expect(counterfactualReplicateSeed(42, "game:t9", 0)).not.toBe(
      counterfactualReplicateSeed(42, "game:t9", 1)
    );
    expect(() => counterfactualReplicateSeed(42, "game:t9", -1)).toThrow(
      "non-negative integer"
    );
  });

  it("requires consistent matched-replicate evidence", () => {
    expect(
      counterfactualReplicateEvidence(
        [
          { replicate: 0, rolloutValue: 1 },
          { replicate: 1, rolloutValue: 0.8 },
        ],
        [
          { replicate: 0, rolloutValue: 0 },
          { replicate: 1, rolloutValue: 0.4 },
        ],
        2,
        0.02
      )
    ).toMatchObject({
      supportingReplicates: 2,
      conflictingReplicates: 0,
      requiredReplicateSupport: 2,
      replicateReliable: true,
    });
    expect(
      counterfactualReplicateEvidence(
        [
          { replicate: 0, rolloutValue: 1 },
          { replicate: 1, rolloutValue: 0 },
        ],
        [
          { replicate: 0, rolloutValue: 0 },
          { replicate: 1, rolloutValue: 1 },
        ],
        2,
        0.02
      )
    ).toMatchObject({
      supportingReplicates: 1,
      conflictingReplicates: 1,
      replicateReliable: false,
    });
  });

  it("skips duplicate resulting states and balances roots across phases", () => {
    const duplicate = decision(10, [1, 1.01, 1.02]);
    duplicate.candidateTurns[1].resulting_fen =
      duplicate.candidateTurns[0].resulting_fen;
    duplicate.candidateTurns[2].resulting_fen = "8/8/8/8/8/8/7Q/q7 - - b";
    const roots = selectCounterfactualRoots(
      [
        game("early", [duplicate]),
        game("middle", [decision(40, [2, 2.1])]),
        game("late", [decision(70, [3, 3.2])]),
      ],
      { maxRoots: 3, maxRootsPerGame: 1 }
    );

    expect(roots.map((root) => root.sourceGameId)).toEqual([
      "early",
      "middle",
      "late",
    ]);
    expect(roots[0].branches.map((branch) => branch.candidateRank)).toEqual([
      1, 3,
    ]);
  });

  it("interleaves both root players within each phase", () => {
    const roots = selectCounterfactualRoots(
      [
        game("early-red", [decision(10, [1, 1.1], "RED")]),
        game("early-blue", [decision(11, [1, 1.1], "BLUE")]),
        game("middle-red", [decision(40, [1, 1.1], "RED")]),
        game("middle-blue", [decision(41, [1, 1.1], "BLUE")]),
        game("late-red", [decision(70, [1, 1.1], "RED")]),
        game("late-blue", [decision(71, [1, 1.1], "BLUE")]),
      ],
      { maxRoots: 6, maxRootsPerGame: 1 }
    );

    expect(roots.map((root) => root.rootPlayer)).toEqual([
      "RED",
      "BLUE",
      "RED",
      "BLUE",
      "RED",
      "BLUE",
    ]);
  });

  it("pages through the balanced root order without repeating roots", () => {
    const games = Array.from({ length: 8 }, (_, index) =>
      game(`game-${index}`, [
        decision(10 + index, [1, 1.1], index % 2 ? "BLUE" : "RED"),
      ])
    );
    const first = selectCounterfactualRoots(games, {
      maxRoots: 4,
      maxRootsPerGame: 1,
    });
    const second = selectCounterfactualRoots(games, {
      maxRoots: 4,
      skipRoots: 4,
      maxRootsPerGame: 1,
    });

    expect(second).toHaveLength(4);
    expect(new Set([...first, ...second].map((root) => root.rootId)).size).toBe(
      8
    );
  });

  it("excludes roots already used for training or evaluation", () => {
    const games = Array.from({ length: 4 }, (_, index) =>
      game(`freshness-${index}`, [decision(10 + index, [1, 1.1])])
    );
    const first = selectCounterfactualRoots(games, { maxRoots: 2 });
    const excluded = new Set(first.map((root) => root.rootId));
    const fresh = selectCounterfactualRoots(games, {
      maxRoots: 2,
      excludeRootIds: excluded,
    });

    expect(fresh).toHaveLength(2);
    expect(fresh.every((root) => !excluded.has(root.rootId))).toBe(true);
  });
});
