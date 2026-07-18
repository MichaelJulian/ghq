import type { Player } from "@/game/engine";
import {
  extractValueFeatures,
  extractValueFeaturesV2,
  extractValueFeaturesV3,
  VALUE_FEATURE_NAMES,
  VALUE_FEATURE_NAMES_V2,
  VALUE_FEATURE_NAMES_V3,
  ValuePosition,
} from "@/game/value-model/features";
import generatedModel from "@/game/value-model/model.generated.json";
import challengerGeneratedModel from "@/game/value-model/model.challenger.generated.json";
import twoActionGeneratedModel from "@/game/value-model/model.two-action.generated.json";

interface ExportedTree {
  children_left: number[];
  children_right: number[];
  feature: number[];
  threshold: number[];
  value: number[];
}

export interface ValueModelArtifact {
  format: "ghq-gradient-boosted-value-v1";
  generated_at?: string;
  feature_names: string[];
  base_raw_score: number;
  learning_rate: number;
  calibration: {
    kind: "platt";
    scale: number;
    intercept: number;
  };
  /**
   * Optional sparse additive correction in the artifact's feature space.
   * This lets a challenger preserve every incumbent tree while learning how
   * append-only strategic features should adjust its calibrated log-odds.
   */
  linear_correction?: {
    feature_indices: number[];
    coefficients: number[];
  };
  /** Move-ranking logit applied to completed-turn transitions, never leaves. */
  policy_correction?: {
    feature_indices: number[];
    coefficients: number[];
  };
  /** Shallow post-calibration trees trained under a pairwise ranking loss. */
  tree_correction?: {
    learning_rate: number;
    trees: ExportedTree[];
  };
  trees: ExportedTree[];
  metadata: Record<string, unknown>;
}

const model = generatedModel as unknown as ValueModelArtifact;
const challengerModel =
  challengerGeneratedModel as unknown as ValueModelArtifact;
const twoActionModel = twoActionGeneratedModel as unknown as ValueModelArtifact;

export type ValueModelRuleset = "three-actions" | "two-actions";
export type ValueModelVersion = "incumbent" | "challenger";

function modelForRuleset(
  ruleset: ValueModelRuleset,
  version: ValueModelVersion
): ValueModelArtifact {
  if (ruleset === "two-actions") return twoActionModel;
  return version === "challenger" ? challengerModel : model;
}

/** Stable, human-readable provenance for persisted arena results. */
export function valueModelCheckpointId(
  ruleset: ValueModelRuleset = "three-actions",
  version: ValueModelVersion = "incumbent"
): string {
  const artifact = modelForRuleset(ruleset, version);
  // Calibration-only challengers intentionally retain the incumbent trees,
  // so `dataset_sha256` still names the incumbent's original human dataset.
  // Prefer the calibration dataset hash when present; otherwise distinct
  // recalibrations can look like the same checkpoint apart from a timestamp.
  const datasetHash =
    artifact.metadata.correction_dataset_sha256 ??
    artifact.metadata.calibration_dataset_sha256 ??
    artifact.metadata.dataset_sha256;
  const fingerprint =
    typeof datasetHash === "string" ? datasetHash.slice(0, 16) : "unknown";
  const generated = artifact.generated_at ?? "unknown";
  return `${ruleset}:${version}:${fingerprint}:${generated}`;
}

function sigmoid(value: number): number {
  if (value >= 0) return 1 / (1 + Math.exp(-value));
  const exponential = Math.exp(value);
  return exponential / (1 + exponential);
}

function evaluateTree(tree: ExportedTree, features: number[]): number {
  let node = 0;
  while (tree.children_left[node] !== -1) {
    const feature = tree.feature[node];
    node =
      features[feature] <= tree.threshold[node]
        ? tree.children_left[node]
        : tree.children_right[node];
  }
  return tree.value[node];
}

export function assertValueModelCompatible(
  artifact: ValueModelArtifact = model
): void {
  if (artifact.format !== "ghq-gradient-boosted-value-v1") {
    throw new Error(`Unsupported GHQ value model format: ${artifact.format}`);
  }
  const matches = (featureNames: readonly string[]) =>
    artifact.feature_names.length === featureNames.length &&
    artifact.feature_names.every(
      (feature, index) => feature === featureNames[index]
    );
  if (
    !matches(VALUE_FEATURE_NAMES) &&
    !matches(VALUE_FEATURE_NAMES_V2) &&
    !matches(VALUE_FEATURE_NAMES_V3)
  ) {
    throw new Error(
      "GHQ value model feature schema does not match the runtime"
    );
  }
  const correction = artifact.linear_correction;
  if (correction) {
    if (
      correction.feature_indices.length !== correction.coefficients.length ||
      correction.feature_indices.some(
        (index) =>
          !Number.isInteger(index) ||
          index < 0 ||
          index >= artifact.feature_names.length
      ) ||
      correction.coefficients.some(
        (coefficient) => !Number.isFinite(coefficient)
      )
    ) {
      throw new Error("GHQ value model linear correction is invalid");
    }
  }
  const policyCorrection = artifact.policy_correction;
  if (policyCorrection) {
    if (
      policyCorrection.feature_indices.length !==
        policyCorrection.coefficients.length ||
      policyCorrection.feature_indices.some(
        (index) =>
          !Number.isInteger(index) ||
          index < 0 ||
          index >= artifact.feature_names.length
      ) ||
      policyCorrection.coefficients.some(
        (coefficient) => !Number.isFinite(coefficient)
      )
    ) {
      throw new Error("GHQ value model policy correction is invalid");
    }
  }
  const treeCorrection = artifact.tree_correction;
  if (correction && treeCorrection) {
    throw new Error("GHQ value model cannot combine correction kinds");
  }
  if (treeCorrection) {
    const validTree = (tree: ExportedTree) => {
      const length = tree.children_left.length;
      return (
        length > 0 &&
        tree.children_right.length === length &&
        tree.feature.length === length &&
        tree.threshold.length === length &&
        tree.value.length === length &&
        tree.threshold.every(Number.isFinite) &&
        tree.value.every(Number.isFinite) &&
        tree.feature.every(
          (feature, node) =>
            (tree.children_left[node] === -1 && feature === -2) ||
            (Number.isInteger(feature) &&
              feature >= 0 &&
              feature < artifact.feature_names.length)
        ) &&
        tree.children_left.every(
          (child, node) =>
            (child === -1 && tree.children_right[node] === -1) ||
            (Number.isInteger(child) &&
              child >= 0 &&
              child < length &&
              Number.isInteger(tree.children_right[node]) &&
              tree.children_right[node] >= 0 &&
              tree.children_right[node] < length)
        )
      );
    };
    if (
      !Number.isFinite(treeCorrection.learning_rate) ||
      treeCorrection.learning_rate <= 0 ||
      treeCorrection.trees.length === 0 ||
      !treeCorrection.trees.every(validTree)
    ) {
      throw new Error("GHQ value model tree correction is invalid");
    }
  }
}

function featuresForArtifact(
  position: ValuePosition,
  perspective: Player,
  artifact: ValueModelArtifact
): number[] {
  assertValueModelCompatible(artifact);
  if (artifact.feature_names.length === VALUE_FEATURE_NAMES_V3.length) {
    return extractValueFeaturesV3(position, perspective);
  }
  if (artifact.feature_names.length === VALUE_FEATURE_NAMES_V2.length) {
    return extractValueFeaturesV2(position, perspective);
  }
  return extractValueFeatures(position, perspective);
}

export function predictFromFeatures(
  features: number[],
  artifact: ValueModelArtifact = model
): number {
  assertValueModelCompatible(artifact);
  if (features.length !== artifact.feature_names.length) {
    throw new Error(
      `Expected ${artifact.feature_names.length} value features, received ${features.length}`
    );
  }
  let raw = artifact.base_raw_score;
  for (const tree of artifact.trees) {
    raw += artifact.learning_rate * evaluateTree(tree, features);
  }
  let calibrated =
    artifact.calibration.scale * raw + artifact.calibration.intercept;
  const correction = artifact.linear_correction;
  if (correction) {
    for (let index = 0; index < correction.feature_indices.length; index += 1) {
      calibrated +=
        correction.coefficients[index] *
        features[correction.feature_indices[index]];
    }
  }
  const treeCorrection = artifact.tree_correction;
  if (treeCorrection) {
    for (const tree of treeCorrection.trees) {
      calibrated +=
        treeCorrection.learning_rate * evaluateTree(tree, features);
    }
  }
  return sigmoid(calibrated);
}

export function policyAdjustmentFromFeatures(
  features: number[],
  artifact: ValueModelArtifact = model
): number {
  assertValueModelCompatible(artifact);
  if (features.length !== artifact.feature_names.length) {
    throw new Error(
      `Expected ${artifact.feature_names.length} value features, received ${features.length}`
    );
  }
  const correction = artifact.policy_correction;
  if (!correction) return 0;
  return correction.feature_indices.reduce(
    (score, featureIndex, index) =>
      score + correction.coefficients[index] * features[featureIndex],
    0
  );
}

export function predictPolicyAdjustment(
  position: ValuePosition,
  perspective: Player,
  ruleset: ValueModelRuleset = "three-actions",
  version: ValueModelVersion = "incumbent"
): number {
  const artifact = modelForRuleset(ruleset, version);
  return policyAdjustmentFromFeatures(
    featuresForArtifact(position, perspective, artifact),
    artifact
  );
}

export function predictWinProbability(
  position: ValuePosition,
  perspective: Player,
  ruleset: ValueModelRuleset = "three-actions",
  version: ValueModelVersion = "incumbent"
): number {
  const ownHq = position.board
    .flat()
    .some((piece) => piece?.type === "HQ" && piece.player === perspective);
  const opponent = perspective === "RED" ? "BLUE" : "RED";
  const opponentHq = position.board
    .flat()
    .some((piece) => piece?.type === "HQ" && piece.player === opponent);
  if (!ownHq) return 0;
  if (!opponentHq) return 1;
  const artifact = modelForRuleset(ruleset, version);
  return predictFromFeatures(
    featuresForArtifact(position, perspective, artifact),
    artifact
  );
}

/**
 * Convert the independently calibrated perspective predictions into the
 * zero-sum probability required by minimax. Without this normalization both
 * sides can be above 50%, which injects a fixed Red offset when Python search
 * consumes only the Red prediction.
 */
export function predictZeroSumWinProbability(
  position: ValuePosition,
  perspective: Player,
  ruleset: ValueModelRuleset = "three-actions",
  version: ValueModelVersion = "incumbent"
): number {
  const opponent = perspective === "RED" ? "BLUE" : "RED";
  const own = predictWinProbability(position, perspective, ruleset, version);
  const other = predictWinProbability(position, opponent, ruleset, version);
  const total = own + other;
  return total > 0 ? own / total : 0.5;
}

export const VALUE_MODEL_METADATA = model.metadata;
export const CHALLENGER_VALUE_MODEL_METADATA = challengerModel.metadata;
export const TWO_ACTION_VALUE_MODEL_METADATA = twoActionModel.metadata;
