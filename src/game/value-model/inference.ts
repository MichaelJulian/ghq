import type { Player } from "@/game/engine";
import {
  extractValueFeatures,
  extractValueFeaturesV2,
  VALUE_FEATURE_NAMES,
  VALUE_FEATURE_NAMES_V2,
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
  if (!matches(VALUE_FEATURE_NAMES) && !matches(VALUE_FEATURE_NAMES_V2)) {
    throw new Error(
      "GHQ value model feature schema does not match the runtime"
    );
  }
}

function featuresForArtifact(
  position: ValuePosition,
  perspective: Player,
  artifact: ValueModelArtifact
): number[] {
  assertValueModelCompatible(artifact);
  return artifact.feature_names.length === VALUE_FEATURE_NAMES_V2.length
    ? extractValueFeaturesV2(position, perspective)
    : extractValueFeatures(position, perspective);
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
  return sigmoid(
    artifact.calibration.scale * raw + artifact.calibration.intercept
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
