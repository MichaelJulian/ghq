import { describe, expect, it } from "@jest/globals";
import type { GhqCandidateTurn } from "@/game/analysis/types";
import type {
  DurableSelfPlayDecision,
  DurableSelfPlayGameResult,
} from "@/workflows/self-play-game";
import {
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
});
