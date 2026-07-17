/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import type { DurableSelfPlayCompetitor } from "@/workflows/self-play-game";
import { scheduleDurableCompetitors } from "./durable-schedule";

const competitors: DurableSelfPlayCompetitor[] = ["balanced", "fortress"].map(
  (personality) => ({
    id: personality,
    personality: personality as "balanced" | "fortress",
    timeMs: 20_000,
    maxDepth: 2,
    beamWidth: 6,
    explorationTemperature: 0.2,
  })
);

describe("durable self-play scheduling", () => {
  it("isolates value checkpoint and color within an arena pair", () => {
    const first = scheduleDurableCompetitors({
      index: 0,
      competitors,
      redMaxActions: 3,
      blueMaxActions: 3,
      valueModelArena: true,
    });
    const second = scheduleDurableCompetitors({
      index: 1,
      competitors,
      redMaxActions: 3,
      blueMaxActions: 3,
      valueModelArena: true,
    });

    expect(first.red.personality).toBe(first.blue.personality);
    expect(second.red.personality).toBe(first.red.personality);
    expect(first.red.valueModel).toBe("challenger");
    expect(first.blue.valueModel).toBe("incumbent");
    expect(second.red.valueModel).toBe("incumbent");
    expect(second.blue.valueModel).toBe("challenger");
    expect(first.red.valueModelCheckpoint).toContain(
      "three-actions:challenger:"
    );
    expect(first.blue.valueModelCheckpoint).toContain(
      "three-actions:incumbent:"
    );
    expect(second.red.valueModelCheckpoint).toBe(
      first.blue.valueModelCheckpoint
    );
    expect(second.blue.valueModelCheckpoint).toBe(
      first.red.valueModelCheckpoint
    );
  });

  it("keeps the normal personality matchup on the incumbent", () => {
    const scheduled = scheduleDurableCompetitors({
      index: 0,
      competitors,
      redMaxActions: 3,
      blueMaxActions: 3,
      valueModelArena: false,
    });
    expect(scheduled.red.personality).not.toBe(scheduled.blue.personality);
    expect(scheduled.red.valueModel).toBe("incumbent");
    expect(scheduled.blue.valueModel).toBe("incumbent");
  });
});
