import {
  extendsStrategicBest,
  mergeStrategicBest,
  strategicProgress,
} from "./strategic-progress";

describe("strategic self-play progress", () => {
  const quietApproachBefore = "q1i3i1/3i3i/2i3i1/8/8/8/1IIII1I1/1F3I1Q - - r";
  const quietApproachAfter = "q1i3i1/3i3i/2i3i1/8/8/2I5/1I1II1I1/F4I1Q - - b";

  it("recognizes a new quiet frontier instead of declaring no progress", () => {
    const before = strategicProgress(quietApproachBefore, "RED");
    const after = strategicProgress(quietApproachAfter, "RED");
    expect(extendsStrategicBest(before, after)).toBe(true);
    expect(after.frontierRank).toBeGreaterThan(before.frontierRank);
  });

  it("does not let a retreat erase the historical frontier", () => {
    const before = strategicProgress(quietApproachBefore, "RED");
    const after = strategicProgress(quietApproachAfter, "RED");
    const best = mergeStrategicBest(before, after);
    expect(extendsStrategicBest(best, before)).toBe(false);
    expect(mergeStrategicBest(best, before)).toEqual(best);
  });

  it("counts additional infantry closing an HQ net after nearest distance saturates", () => {
    const before = strategicProgress(
      "8/8/5Iq1/8/2I5/1F2I2I/F4I2/7Q - - r",
      "RED"
    );
    const after = strategicProgress(
      "8/8/5Iq1/8/2I2II1/1F4I1/F7/7Q - - b",
      "RED"
    );

    expect(after.enemyHqDistance).toBe(before.enemyHqDistance);
    expect(after.enemyHqPressure).toBeGreaterThan(before.enemyHqPressure);
    expect(extendsStrategicBest(before, after)).toBe(true);
  });
});
