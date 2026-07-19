/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import type { DurableSelfPlayCompetitor } from "@/workflows/self-play-game";
import {
  scheduleDurableCompetitors,
  scheduleDurableSearch,
} from "./durable-schedule";

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
  it("caps a large batch at four concurrent search lanes", () => {
    const schedules = Array.from({ length: 12 }, (_, index) =>
      scheduleDurableSearch(index, 12, 1_000)
    );

    expect(schedules.slice(0, 4).map((schedule) => schedule?.lane)).toEqual([
      0, 0, 0, 0,
    ]);
    expect(schedules.slice(4, 8).map((schedule) => schedule?.lane)).toEqual([
      1, 1, 1, 1,
    ]);
    expect(schedules.slice(8).map((schedule) => schedule?.lane)).toEqual([
      2, 2, 2, 2,
    ]);
    expect(schedules.every((schedule) => schedule?.laneCount === 3)).toBe(true);
    expect(scheduleDurableSearch(3, 4, 1_000)).toBeUndefined();
  });

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
