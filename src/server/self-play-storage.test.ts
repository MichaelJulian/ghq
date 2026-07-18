/** @jest-environment node */

import { afterEach, describe, expect, it } from "@jest/globals";
import {
  persistSelfPlayProgress,
  SELF_PLAY_PROGRESS_PUT_OPTIONS,
  selfPlayProgressPathname,
  type SelfPlayProgressSnapshot,
} from "./self-play-storage";

function snapshot(
  overrides: Partial<SelfPlayProgressSnapshot> = {}
): SelfPlayProgressSnapshot {
  return {
    format: "ghq-self-play-progress-v1",
    generationId: "generation-1",
    gameId: "game-1",
    seed: 1,
    codeVersion: "revision-1",
    redAgentId: "balanced-challenger-a3",
    blueAgentId: "balanced-incumbent-a3",
    redValueModelCheckpoint: "challenger-1",
    blueValueModelCheckpoint: "incumbent-1",
    completedTurns: 10,
    currentPlayer: "RED",
    currentFen: "position-10",
    decisions: 10,
    depthAtLeastTwoDecisions: 9,
    fallbackDecisions: 1,
    unverifiedFallbackDecisions: 0,
    timedOutDecisions: 4,
    status: "running",
    ...overrides,
  };
}

describe("self-play progress storage", () => {
  const previousToken = process.env.BLOB_READ_WRITE_TOKEN;
  const previousStoreId = process.env.BLOB_STORE_ID;

  afterEach(() => {
    if (previousToken === undefined) delete process.env.BLOB_READ_WRITE_TOKEN;
    else process.env.BLOB_READ_WRITE_TOKEN = previousToken;
    if (previousStoreId === undefined) delete process.env.BLOB_STORE_ID;
    else process.env.BLOB_STORE_ID = previousStoreId;
  });

  it("uses one stable overwrite-enabled private object per game", () => {
    expect(selfPlayProgressPathname("generation-1", "game-1")).toBe(
      "self-play/generations/generation-1/progress/game-1.json"
    );
    expect(SELF_PLAY_PROGRESS_PUT_OPTIONS).toEqual({
      access: "private",
      addRandomSuffix: false,
      allowOverwrite: true,
      contentType: "application/json",
    });
  });

  it("does not attempt a write when Blob storage is unavailable", async () => {
    delete process.env.BLOB_READ_WRITE_TOKEN;
    delete process.env.BLOB_STORE_ID;

    await expect(persistSelfPlayProgress(snapshot())).resolves.toBe(
      "not-configured"
    );
  });
});
