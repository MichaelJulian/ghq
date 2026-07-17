/** @jest-environment node */

import { beforeAll, describe, expect, it, jest } from "@jest/globals";
import { loadV2Engine } from "@/server/engine";
import type { GameEngine } from "@/game/engine-v2";
import { createRandomAgent, playOneGame, SelfPlayAgent } from "./play-one-game";

jest.setTimeout(30_000);

let engine: GameEngine;

beforeAll(async () => {
  engine = await loadV2Engine();
});

const skipAgent = (id: string): SelfPlayAgent => ({
  id,
  selectMove: ({ legalMoves }) => {
    const skip = legalMoves.find((move) => move.name === "Skip");
    if (!skip) throw new Error("Expected Skip to be legal");
    return skip;
  },
});

describe("playOneGame", () => {
  it("uses production double-skip termination and records complete turns", async () => {
    const result = await playOneGame({
      engine,
      red: skipAgent("red-skip"),
      blue: skipAgent("blue-skip"),
      maxTurns: 4,
      seed: 7,
    });

    expect(result.completed).toBe(true);
    expect(result.outcome).toEqual({ termination: "double-skip" });
    expect(result.turns).toHaveLength(2);
    expect(result.turns.map((turn) => turn.player)).toEqual(["RED", "BLUE"]);
    expect(result.turns.map((turn) => turn.moves[0].name)).toEqual([
      "Skip",
      "Skip",
    ]);
    expect(result.actionCount).toBe(2);
    expect(result.automaticCaptureCount).toBe(0);
  });

  it("lets the production engine enforce the three-action turn limit", async () => {
    const firstNonSkip: SelfPlayAgent = {
      id: "first-action",
      selectMove: ({ legalMoves }) =>
        legalMoves.find(
          (move) => move.name !== "Skip" && move.name !== "AutoCapture"
        ) ?? legalMoves[0],
    };

    const result = await playOneGame({
      engine,
      red: firstNonSkip,
      blue: firstNonSkip,
      maxTurns: 1,
      seed: 11,
    });

    expect(result.completed).toBe(false);
    expect(result.outcome.termination).toBe("max-turns");
    expect(result.turns).toHaveLength(1);
    expect(result.turns[0].player).toBe("RED");
    expect(
      result.turns[0].moves.filter((move) => !move.automaticCapture)
    ).toHaveLength(3);
    expect(result.turns[0].moves.map((move) => move.actionNumber)).toEqual([
      1, 2, 3,
    ]);
  });

  it("terminates a game after the configured no-progress window", async () => {
    const firstNonSkip: SelfPlayAgent = {
      id: "quiet-action",
      selectMove: ({ legalMoves }) =>
        legalMoves.find((move) =>
          (move.name === "Move" || move.name === "MoveAndOrient") &&
          !move.uci().includes("x")
        ) ?? legalMoves.find((move) => move.name === "Skip") ?? legalMoves[0],
    };
    const result = await playOneGame({
      engine,
      red: firstNonSkip,
      blue: firstNonSkip,
      maxTurns: 10,
      repetitionLimit: 10,
      noProgressTurns: 1,
      seed: 111,
    });

    expect(result.completed).toBe(true);
    expect(result.outcome.termination).toBe("no-progress");
    // Red's first quiet turn reaches a new frontier and therefore counts as
    // strategic progress. Blue's following retreat does not, so the one-turn
    // quiet window expires there.
    expect(result.turns).toHaveLength(2);
  });

  it("resolves production automatic captures without spending an action", async () => {
    const forcedThenSkip: SelfPlayAgent = {
      id: "forced-then-skip",
      selectMove: ({ legalMoves, forcedAutomaticCapture }) =>
        forcedAutomaticCapture
          ? legalMoves[0]
          : legalMoves.find((move) => move.name === "Skip") ?? legalMoves[0],
    };
    const result = await playOneGame({
      engine,
      red: forcedThenSkip,
      blue: skipAgent("blue-skip"),
      fen: "7q/8/8/8/8/8/6i1/6R↑Q - - r",
      maxTurns: 1,
      seed: 12,
    });

    expect(result.outcome.termination).toBe("max-turns");
    expect(result.turns[0].moves.map((move) => move.name)).toEqual([
      "AutoCapture",
      "Skip",
    ]);
    expect(result.turns[0].moves[0].uci).toBe("sbg2");
    expect(result.turns[0].moves[0].actionNumber).toBeUndefined();
    expect(result.turns[0].moves[1].actionNumber).toBe(1);
    expect(result.actionCount).toBe(1);
    expect(result.automaticCaptureCount).toBe(1);
  });

  it("rejects moves that are not legal in the production position", async () => {
    const illegal: SelfPlayAgent = {
      id: "broken-agent",
      selectMove: () => "a1a8",
    };

    await expect(
      playOneGame({
        engine,
        red: illegal,
        blue: illegal,
        maxTurns: 1,
        seed: 13,
      })
    ).rejects.toThrow(/broken-agent selected illegal move/);
  });

  it("replays random games exactly when given the same seed", async () => {
    const random = createRandomAgent();
    const config = {
      engine,
      red: random,
      blue: random,
      maxTurns: 8,
      seed: 101,
    };

    const first = await playOneGame(config);
    const second = await playOneGame(config);

    expect(second.finalFen).toBe(first.finalFen);
    expect(second.turns).toEqual(first.turns);
  });
});
