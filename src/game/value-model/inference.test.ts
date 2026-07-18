import { endgame } from "@/game/variants";
import {
  extractValueFeatures,
  extractValueFeaturesV2,
  extractValueFeaturesV3,
  VALUE_FEATURE_NAMES,
  VALUE_FEATURE_NAMES_V2,
  VALUE_FEATURE_NAMES_V3,
} from "./features";
import {
  assertValueModelCompatible,
  CHALLENGER_VALUE_MODEL_METADATA,
  predictFromFeatures,
  predictWinProbability,
  predictZeroSumWinProbability,
  TWO_ACTION_VALUE_MODEL_METADATA,
  valueModelCheckpointId,
  type ValueModelArtifact,
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
    const checkpointDataset =
      CHALLENGER_VALUE_MODEL_METADATA.correction_dataset_sha256 ??
      CHALLENGER_VALUE_MODEL_METADATA.calibration_dataset_sha256 ??
      CHALLENGER_VALUE_MODEL_METADATA.dataset_sha256;
    expect(typeof checkpointDataset).toBe("string");
    expect(challenger).toContain(
      `:${(checkpointDataset as string).slice(0, 16)}:`
    );
  });

  it("evaluates an exported tree vector directly", () => {
    const probability = predictFromFeatures(
      extractValueFeatures(position, "RED")
    );
    expect(Number.isFinite(probability)).toBe(true);
  });

  it("accepts the append-only v2 schema without changing incumbent indices", () => {
    expect(VALUE_FEATURE_NAMES_V2.slice(0, VALUE_FEATURE_NAMES.length)).toEqual(
      [...VALUE_FEATURE_NAMES]
    );
    const artifact: ValueModelArtifact = {
      format: "ghq-gradient-boosted-value-v1",
      feature_names: [...VALUE_FEATURE_NAMES_V2],
      base_raw_score: 0,
      learning_rate: 0.1,
      calibration: { kind: "platt", scale: 1, intercept: 0 },
      trees: [],
      metadata: {},
    };
    expect(() => assertValueModelCompatible(artifact)).not.toThrow();
    expect(
      predictFromFeatures(extractValueFeaturesV2(position, "RED"), artifact)
    ).toBeCloseTo(0.5, 12);
  });

  it("accepts the append-only tactical v3 schema without changing v2 indices", () => {
    expect(
      VALUE_FEATURE_NAMES_V3.slice(0, VALUE_FEATURE_NAMES_V2.length)
    ).toEqual([...VALUE_FEATURE_NAMES_V2]);
    const artifact: ValueModelArtifact = {
      format: "ghq-gradient-boosted-value-v1",
      feature_names: [...VALUE_FEATURE_NAMES_V3],
      base_raw_score: 0,
      learning_rate: 0.1,
      calibration: { kind: "platt", scale: 1, intercept: 0 },
      trees: [],
      metadata: {},
    };
    expect(() => assertValueModelCompatible(artifact)).not.toThrow();
    expect(
      predictFromFeatures(extractValueFeaturesV3(position, "RED"), artifact)
    ).toBeCloseTo(0.5, 12);
  });

  it("applies a validated sparse correction in append-only feature space", () => {
    const base: ValueModelArtifact = {
      format: "ghq-gradient-boosted-value-v1",
      feature_names: [...VALUE_FEATURE_NAMES_V2],
      base_raw_score: 0,
      learning_rate: 0.1,
      calibration: { kind: "platt", scale: 1, intercept: 0 },
      trees: [],
      metadata: {},
    };
    const corrected: ValueModelArtifact = {
      ...base,
      linear_correction: {
        feature_indices: [VALUE_FEATURE_NAMES.length],
        coefficients: [0.5],
      },
    };
    const features = extractValueFeaturesV2(position, "RED");
    const expected =
      1 / (1 + Math.exp(-0.5 * features[VALUE_FEATURE_NAMES.length]));
    expect(predictFromFeatures(features, corrected)).toBeCloseTo(expected, 12);
  });

  it("rejects malformed sparse corrections", () => {
    const artifact: ValueModelArtifact = {
      format: "ghq-gradient-boosted-value-v1",
      feature_names: [...VALUE_FEATURE_NAMES_V2],
      base_raw_score: 0,
      learning_rate: 0.1,
      calibration: { kind: "platt", scale: 1, intercept: 0 },
      trees: [],
      metadata: {},
      linear_correction: {
        feature_indices: [VALUE_FEATURE_NAMES_V2.length],
        coefficients: [1],
      },
    };
    expect(() => assertValueModelCompatible(artifact)).toThrow(
      "linear correction is invalid"
    );
  });

  it("applies a validated pairwise tree correction after calibration", () => {
    const artifact: ValueModelArtifact = {
      format: "ghq-gradient-boosted-value-v1",
      feature_names: [...VALUE_FEATURE_NAMES_V2],
      base_raw_score: 0,
      learning_rate: 0.1,
      calibration: { kind: "platt", scale: 1, intercept: 0 },
      trees: [],
      metadata: {},
      tree_correction: {
        learning_rate: 0.25,
        trees: [
          {
            children_left: [1, -1, -1],
            children_right: [2, -1, -1],
            feature: [VALUE_FEATURE_NAMES.length, -2, -2],
            threshold: [0.5, -2, -2],
            value: [0, -1, 1],
          },
        ],
      },
    };
    const features = extractValueFeaturesV2(position, "RED");
    const leaf = features[VALUE_FEATURE_NAMES.length] <= 0.5 ? -1 : 1;
    const expected = 1 / (1 + Math.exp(-0.25 * leaf));
    expect(() => assertValueModelCompatible(artifact)).not.toThrow();
    expect(predictFromFeatures(features, artifact)).toBeCloseTo(expected, 12);
  });

  it("rejects malformed pairwise tree corrections", () => {
    const artifact: ValueModelArtifact = {
      format: "ghq-gradient-boosted-value-v1",
      feature_names: [...VALUE_FEATURE_NAMES_V2],
      base_raw_score: 0,
      learning_rate: 0.1,
      calibration: { kind: "platt", scale: 1, intercept: 0 },
      trees: [],
      metadata: {},
      tree_correction: {
        learning_rate: 0.1,
        trees: [
          {
            children_left: [-1],
            children_right: [-1],
            feature: [VALUE_FEATURE_NAMES_V2.length],
            threshold: [-2],
            value: [0],
          },
        ],
      },
    };
    expect(() => assertValueModelCompatible(artifact)).toThrow(
      "tree correction is invalid"
    );
  });

  it("overrides the model for a captured HQ", () => {
    const terminal = JSON.parse(JSON.stringify(position)) as typeof position;
    terminal.board[0][0] = null;
    expect(predictWinProbability(terminal, "RED")).toBe(1);
    expect(predictWinProbability(terminal, "BLUE")).toBe(0);
  });
});
