/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import {
  colorSwapPairIntegrityRejectionReasons,
  partitionColorSwapPairs,
} from "./color-pairs";

function game(generationId: string, number: number) {
  return {
    generationId,
    gameId: `${generationId}-${String(number).padStart(4, "0")}`,
  };
}

describe("color-swapped pair partitioning", () => {
  it("keeps only actual adjacent members of one generation", () => {
    const one = game("one", 1);
    const two = game("one", 2);
    const orphan = game("one", 5);
    const otherGeneration = game("two", 6);
    const result = partitionColorSwapPairs([otherGeneration, two, orphan, one]);
    expect(result.pairs).toEqual([[one, two]]);
    expect(result.orphans).toEqual([orphan, otherGeneration]);
  });

  it("does not manufacture a pair from duplicate or malformed ids", () => {
    const duplicate = game("one", 1);
    const malformed = { generationId: "one", gameId: "missing-suffix" };
    const result = partitionColorSwapPairs([
      duplicate,
      { ...duplicate },
      malformed,
    ]);
    expect(result.pairs).toEqual([]);
    expect(result.orphans).toHaveLength(3);
  });

  it("verifies the experimental configuration was actually swapped", () => {
    const first = {
      ...game("one", 1),
      seed: 42,
      redAgentId: "balanced-incumbent-a3",
      blueAgentId: "fortress-incumbent-a3",
      redMaxActions: 3,
      blueMaxActions: 3,
      redValueModelCheckpoint: "incumbent",
      blueValueModelCheckpoint: "incumbent",
      initialFen: "start",
      initialTurnNumber: 1,
      dataRole: "standard",
    };
    const second = {
      ...game("one", 2),
      ...first,
      gameId: "one-0002",
      redAgentId: first.blueAgentId,
      blueAgentId: first.redAgentId,
    };

    expect(colorSwapPairIntegrityRejectionReasons(first, second)).toEqual([]);
    expect(
      colorSwapPairIntegrityRejectionReasons(first, {
        ...second,
        seed: 43,
        redAgentId: "other",
        initialFen: "different",
      })
    ).toEqual([
      "mismatched-seed",
      "agents-not-swapped",
      "initial-state-mismatch",
    ]);
  });
});
