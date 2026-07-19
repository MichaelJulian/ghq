import { describe, expect, it } from "@jest/globals";

import {
  MINIMUM_AUDITED_TRAINING_PAIRS,
  summarizeSelfPlayTrainingReadiness,
} from "./training-readiness";

describe("summarizeSelfPlayTrainingReadiness", () => {
  it("counts only complete adjacent odd/even color swaps", () => {
    expect(
      summarizeSelfPlayTrainingReadiness([
        "generation-a-0001",
        "generation-a-0002",
        "generation-a-0003",
        "generation-a-0005",
        "generation-a-0006",
      ])
    ).toEqual({
      qualityEligibleGames: 5,
      preAuditCompletePairs: 2,
      minimumAuditedPairs: MINIMUM_AUDITED_TRAINING_PAIRS,
      preAuditPairDeficit: MINIMUM_AUDITED_TRAINING_PAIRS - 2,
    });
  });

  it("does not combine games across generations or count duplicates", () => {
    const readiness = summarizeSelfPlayTrainingReadiness([
      "generation-a-0001",
      "generation-a-0001",
      "generation-b-0002",
      "invalid",
    ]);

    expect(readiness.qualityEligibleGames).toBe(3);
    expect(readiness.preAuditCompletePairs).toBe(0);
  });

  it("floors the deficit at zero", () => {
    const gameIds = Array.from(
      { length: MINIMUM_AUDITED_TRAINING_PAIRS * 2 },
      (_, index) => `generation-a-${String(index + 1).padStart(4, "0")}`
    );

    expect(summarizeSelfPlayTrainingReadiness(gameIds)).toMatchObject({
      preAuditCompletePairs: MINIMUM_AUDITED_TRAINING_PAIRS,
      preAuditPairDeficit: 0,
    });
  });
});
