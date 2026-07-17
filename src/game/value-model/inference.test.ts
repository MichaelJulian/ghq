import { endgame } from "@/game/variants";
import { extractValueFeatures, VALUE_FEATURE_NAMES } from "./features";
import {
  assertValueModelCompatible,
  CHALLENGER_VALUE_MODEL_METADATA,
  predictFromFeatures,
  predictWinProbability,
  predictZeroSumWinProbability,
  TWO_ACTION_VALUE_MODEL_METADATA,
  valueModelCheckpointId,
} from "./inference";

const position = {
  board: endgame.board,
  redReserve: endgame.redReserve,
  blueReserve: endgame.blueReserve,
  currentPlayer: "RED" as const,
  turnNumber: 40,
};

describe("gradient-boosted value model", () => {
  it("uses the same feature schema as the generated artifact", () => {
    expect(() => assertValueModelCompatible()).not.toThrow();
    expect(extractValueFeatures(position, "RED")).toHaveLength(
      VALUE_FEATURE_NAMES.length
    );
  });

  it("returns finite calibrated probabilities for either perspective", () => {
    const red = predictWinProbability(position, "RED");
    const blue = predictWinProbability(position, "BLUE");
    expect(red).toBeGreaterThan(0);
    expect(red).toBeLessThan(1);
    expect(blue).toBeGreaterThan(0);
    expect(blue).toBeLessThan(1);
  });

  it("normalizes independent side predictions for zero-sum search", () => {
    const red = predictZeroSumWinProbability(position, "RED");
    const blue = predictZeroSumWinProbability(position, "BLUE");
    expect(red + blue).toBeCloseTo(1, 12);
  });

  it("selects the dedicated two-action checkpoint", () => {
    const standard = predictWinProbability(position, "RED", "three-actions");
    const twoAction = predictWinProbability(position, "RED", "two-actions");
    expect(Number.isFinite(twoAction)).toBe(true);
    expect(twoAction).not.toBe(standard);
    expect(TWO_ACTION_VALUE_MODEL_METADATA).toBeDefined();
  });

  it("can evaluate the staged three-action challenger", () => {
    const incumbent = predictZeroSumWinProbability(
      position,
      "RED",
      "three-actions",
      "incumbent"
    );
    const challenger = predictZeroSumWinProbability(
      position,
      "RED",
      "three-actions",
      "challenger"
    );
    expect(challenger).toBeGreaterThan(0);
    expect(challenger).toBeLessThan(1);
    expect(challenger).not.toBe(incumbent);
    expect(CHALLENGER_VALUE_MODEL_METADATA).toBeDefined();
  });

  it("exposes distinct persistent checkpoint fingerprints", () => {
    const incumbent = valueModelCheckpointId("three-actions", "incumbent");
    const challenger = valueModelCheckpointId("three-actions", "challenger");
    expect(incumbent).toMatch(/^three-actions:incumbent:[0-9a-f]{16}:/);
    expect(challenger).toMatch(/^three-actions:challenger:[0-9a-f]{16}:/);
    expect(challenger).not.toBe(incumbent);
    const calibrationDataset =
      CHALLENGER_VALUE_MODEL_METADATA.calibration_dataset_sha256;
    expect(typeof calibrationDataset).toBe("string");
    expect(challenger).toContain(
      `:${(calibrationDataset as string).slice(0, 16)}:`
    );
  });

  it("evaluates an exported tree vector directly", () => {
    const probability = predictFromFeatures(
      extractValueFeatures(position, "RED")
    );
    expect(Number.isFinite(probability)).toBe(true);
  });

  it("overrides the model for a captured HQ", () => {
    const terminal = JSON.parse(JSON.stringify(position)) as typeof position;
    terminal.board[0][0] = null;
    expect(predictWinProbability(terminal, "RED")).toBe(1);
    expect(predictWinProbability(terminal, "BLUE")).toBe(0);
  });
});
