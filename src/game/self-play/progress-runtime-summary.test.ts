import type { SelfPlayProgressSnapshot } from "@/server/self-play-storage";
import { summarizeActiveProgressRuntime } from "./progress-runtime-summary";

function snapshot(
  overrides: Partial<SelfPlayProgressSnapshot>
): SelfPlayProgressSnapshot {
  return {
    format: "ghq-self-play-progress-v1",
    generationId: "generation",
    gameId: "game",
    seed: 1,
    codeVersion: "checkpoint",
    redAgentId: "red",
    blueAgentId: "blue",
    redValueModelCheckpoint: "model",
    blueValueModelCheckpoint: "model",
    completedTurns: 10,
    currentPlayer: "RED",
    currentFen: "fen",
    decisions: 10,
    depthAtLeastTwoDecisions: 8,
    fallbackDecisions: 2,
    unverifiedFallbackDecisions: 1,
    timedOutDecisions: 6,
    status: "running",
    ...overrides,
  };
}

describe("active self-play progress telemetry", () => {
  it("aggregates live snapshot counts and rates", () => {
    expect(
      summarizeActiveProgressRuntime([
        snapshot({ gameId: "one" }),
        snapshot({
          gameId: "two",
          decisions: 20,
          depthAtLeastTwoDecisions: 14,
          fallbackDecisions: 3,
          unverifiedFallbackDecisions: 2,
          timedOutDecisions: 15,
        }),
      ])
    ).toEqual({
      games: 2,
      decisions: 30,
      depthAtLeastTwoDecisions: 22,
      fallbackDecisions: 5,
      unverifiedFallbackDecisions: 3,
      timedOutDecisions: 21,
      depthAtLeastTwoRate: 22 / 30,
      fallbackRate: 5 / 30,
      unverifiedFallbackRate: 3 / 30,
      timedOutRate: 21 / 30,
    });
  });

  it("returns zero rates before the first persisted decision", () => {
    expect(summarizeActiveProgressRuntime([])).toEqual({
      games: 0,
      decisions: 0,
      depthAtLeastTwoDecisions: 0,
      fallbackDecisions: 0,
      unverifiedFallbackDecisions: 0,
      timedOutDecisions: 0,
      depthAtLeastTwoRate: 0,
      fallbackRate: 0,
      unverifiedFallbackRate: 0,
      timedOutRate: 0,
    });
  });
});
