import type { PersonalityId } from "@/game/analysis/types";
import type { Player } from "@/game/engine";
import { FENtoBoardState } from "@/game/notation";
import {
  extractValueFeaturesV3,
  VALUE_FEATURE_NAMES_V3,
} from "@/game/value-model/features";
import type {
  DurableSelfPlayDecision,
  DurableSelfPlayGameResult,
} from "@/workflows/self-play-game";

const MATE_SCORE = 1_000_000;

export interface CounterfactualBranch {
  rootId: string;
  sourceGameId: string;
  sourceTurnNumber: number;
  rootPlayer: Player;
  candidateRank: number;
  candidateScore: number;
  candidateMoves: string[];
  initialFen: string;
  initialTurnNumber: number;
  redPersonality: PersonalityId;
  bluePersonality: PersonalityId;
}

export interface CounterfactualRoot {
  rootId: string;
  rootFingerprint: string;
  sourceGameId: string;
  sourceTurnNumber: number;
  rootPlayer: Player;
  scoreMargin: number;
  strategicDivergence: number;
  branches: CounterfactualBranch[];
}

export interface CounterfactualSelectionOptions {
  maxRoots?: number;
  skipRoots?: number;
  maxRootsPerGame?: number;
  candidatesPerRoot?: number;
  maxScoreMargin?: number;
  minTurnNumber?: number;
  minStrategicDivergence?: number;
  excludeRootIds?: ReadonlySet<string>;
  excludeRootFingerprints?: ReadonlySet<string>;
  excludeSourceGameIds?: ReadonlySet<string>;
}

/** Identify the same policy question even when it came from another game. */
export function counterfactualRootFingerprint(
  rootPlayer: Player,
  candidateFens: readonly string[]
): string {
  return `${rootPlayer}:${[...new Set(candidateFens)].sort().join("||")}`;
}

const DIVERGENCE_FEATURE_SCALES: Readonly<Record<string, number>> = {
  diff_material_total: 6,
  diff_material_board: 6,
  diff_infantry_board: 2,
  diff_artillery_board: 2,
  diff_unsupported_value: 4,
  diff_overextended_value: 4,
  diff_bombarded_enemy_value: 4,
  diff_paratrooper_deployed: 1,
  diff_hq_bombarded: 1,
  diff_hq_escape_squares: 2,
  diff_pseudo_mobility: 1,
  diff_home_rank_immobile_count: 2,
  diff_infantry_diagonal_adjacent_pairs: 2,
  diff_infantry_vertical_adjacent_pairs: 2,
  diff_hq_attack_pressure: 3,
};

const DIVERGENCE_FEATURES = Object.entries(DIVERGENCE_FEATURE_SCALES).map(
  ([name, scale]) => {
    const index = VALUE_FEATURE_NAMES_V3.indexOf(name);
    if (index < 0) throw new Error(`Missing counterfactual feature ${name}`);
    return { index, scale };
  }
);

function valuePosition(fen: string, turnNumber: number) {
  const state = FENtoBoardState(fen);
  return {
    board: state.board,
    redReserve: state.redReserve,
    blueReserve: state.blueReserve,
    currentPlayer: state.currentPlayerTurn ?? "RED",
    turnNumber,
  };
}

export function candidateStrategicDivergence(
  candidates: ReturnType<typeof distinctCandidateStates>,
  player: Player,
  turnNumber: number
): number {
  const vectors = candidates.map((candidate) =>
    extractValueFeaturesV3(
      valuePosition(candidate.resulting_fen, turnNumber + 1),
      player
    )
  );
  const baseline = vectors[0];
  return Math.max(
    0,
    ...vectors.slice(1).map((vector) =>
      DIVERGENCE_FEATURES.reduce(
        (total, feature) =>
          total +
          Math.min(
            3,
            Math.abs(vector[feature.index] - baseline[feature.index]) /
              feature.scale
          ),
        0
      )
    )
  );
}

function personalityFor(
  game: DurableSelfPlayGameResult,
  player: Player
): PersonalityId {
  return (
    game.decisions.find((decision) => decision.player === player)
      ?.personality ?? "balanced"
  );
}

function eligibleDecision(
  decision: DurableSelfPlayDecision,
  options: Required<CounterfactualSelectionOptions>
): boolean {
  if (
    decision.turnNumber < options.minTurnNumber ||
    decision.completedDepth < 2 ||
    decision.fallback === "seeded"
  ) {
    return false;
  }
  const candidates = distinctCandidateStates(decision).slice(
    0,
    options.candidatesPerRoot
  );
  if (candidates.length < options.candidatesPerRoot) return false;
  return candidates.every(
    (candidate) =>
      Number.isFinite(candidate.score) && Math.abs(candidate.score) < MATE_SCORE
  );
}

function distinctCandidateStates(decision: DurableSelfPlayDecision) {
  const seen = new Set<string>();
  return [...decision.candidateTurns]
    .sort((left, right) => left.rank - right.rank)
    .filter((candidate) => {
      if (seen.has(candidate.resulting_fen)) return false;
      seen.add(candidate.resulting_fen);
      return true;
    });
}

/**
 * Find search decisions where the incumbent could not clearly separate its
 * leading candidates. Those are the positions where continuation rollouts can
 * provide policy-relevant supervision that a final game label cannot.
 */
export function selectCounterfactualRoots(
  games: DurableSelfPlayGameResult[],
  rawOptions: CounterfactualSelectionOptions = {}
): CounterfactualRoot[] {
  const options: Required<CounterfactualSelectionOptions> = {
    maxRoots: rawOptions.maxRoots ?? 8,
    skipRoots: rawOptions.skipRoots ?? 0,
    maxRootsPerGame: rawOptions.maxRootsPerGame ?? 2,
    candidatesPerRoot: rawOptions.candidatesPerRoot ?? 2,
    maxScoreMargin: rawOptions.maxScoreMargin ?? 1,
    minTurnNumber: rawOptions.minTurnNumber ?? 5,
    minStrategicDivergence: rawOptions.minStrategicDivergence ?? 0,
    excludeRootIds: rawOptions.excludeRootIds ?? new Set<string>(),
    excludeRootFingerprints:
      rawOptions.excludeRootFingerprints ?? new Set<string>(),
    excludeSourceGameIds:
      rawOptions.excludeSourceGameIds ?? new Set<string>(),
  };
  for (const [name, value] of Object.entries(options).filter(
    ([name]) =>
      name !== "excludeRootIds" &&
      name !== "excludeRootFingerprints" &&
      name !== "excludeSourceGameIds"
  )) {
    if (typeof value !== "number") continue;
    const mayBeZero =
      name === "skipRoots" || name === "minStrategicDivergence";
    if (
      !Number.isFinite(value) ||
      value < 0 ||
      (!mayBeZero && value === 0)
    ) {
      throw new RangeError(
        `${name} must be ${mayBeZero ? "non-negative" : "positive"}`
      );
    }
  }
  if (!Number.isSafeInteger(options.maxRoots)) {
    throw new RangeError("maxRoots must be an integer");
  }
  if (!Number.isSafeInteger(options.skipRoots)) {
    throw new RangeError("skipRoots must be an integer");
  }
  if (!Number.isSafeInteger(options.maxRootsPerGame)) {
    throw new RangeError("maxRootsPerGame must be an integer");
  }
  if (
    !Number.isSafeInteger(options.candidatesPerRoot) ||
    options.candidatesPerRoot < 2 ||
    options.candidatesPerRoot > 4
  ) {
    throw new RangeError("candidatesPerRoot must be an integer from 2 to 4");
  }
  if (!Number.isSafeInteger(options.minTurnNumber)) {
    throw new RangeError("minTurnNumber must be an integer");
  }

  const possible = games.flatMap((game) => {
    if (options.excludeSourceGameIds.has(game.gameId)) return [];
    return game.decisions.flatMap((decision) => {
      if (!eligibleDecision(decision, options)) return [];
      const candidates = distinctCandidateStates(decision).slice(
        0,
        options.candidatesPerRoot
      );
      const scoreMargin = Math.max(
        ...candidates.map((candidate) =>
          Math.abs(candidate.score - candidates[0].score)
        )
      );
      if (scoreMargin > options.maxScoreMargin) return [];
      const strategicDivergence = candidateStrategicDivergence(
        candidates,
        decision.player,
        decision.turnNumber
      );
      if (strategicDivergence < options.minStrategicDivergence) return [];
      const rootId = `${game.gameId}:t${decision.turnNumber}`;
      if (options.excludeRootIds.has(rootId)) return [];
      const rootFingerprint = counterfactualRootFingerprint(
        decision.player,
        candidates.map((candidate) => candidate.resulting_fen)
      );
      if (options.excludeRootFingerprints.has(rootFingerprint)) return [];
      const redPersonality = personalityFor(game, "RED");
      const bluePersonality = personalityFor(game, "BLUE");
      return [
        {
          rootId,
          rootFingerprint,
          sourceGameId: game.gameId,
          sourceTurnNumber: decision.turnNumber,
          rootPlayer: decision.player,
          scoreMargin,
          strategicDivergence,
          branches: candidates.map((candidate) => ({
            rootId,
            sourceGameId: game.gameId,
            sourceTurnNumber: decision.turnNumber,
            rootPlayer: decision.player,
            candidateRank: candidate.rank,
            candidateScore: candidate.score,
            candidateMoves: candidate.all_moves,
            initialFen: candidate.resulting_fen,
            initialTurnNumber: decision.turnNumber + 1,
            redPersonality,
            bluePersonality,
          })),
        } satisfies CounterfactualRoot,
      ];
    });
  });
  possible.sort(
    (left, right) =>
      right.strategicDivergence - left.strategicDivergence ||
      left.scoreMargin - right.scoreMargin ||
      left.sourceTurnNumber - right.sourceTurnNumber ||
      left.rootId.localeCompare(right.rootId)
  );
  const seenFingerprints = new Set<string>();
  const uniquePossible = possible.filter((root) => {
    if (seenFingerprints.has(root.rootFingerprint)) return false;
    seenFingerprints.add(root.rootFingerprint);
    return true;
  });

  // Interleave opening/early, middle, and late roots. A pure smallest-margin
  // sort over-samples late sparse positions, even though the search also
  // needs policy supervision for development and formation decisions.
  const phasePredicates = [
    (root: CounterfactualRoot) => root.sourceTurnNumber <= 24,
    (root: CounterfactualRoot) =>
      root.sourceTurnNumber > 24 && root.sourceTurnNumber <= 59,
    (root: CounterfactualRoot) => root.sourceTurnNumber > 59,
  ];
  const phaseBuckets = phasePredicates.flatMap((inPhase) =>
    (["RED", "BLUE"] as const).map((player) =>
      uniquePossible.filter(
        (root) => inPhase(root) && root.rootPlayer === player
      )
    )
  );
  const selected: CounterfactualRoot[] = [];
  const selectionLimit = options.skipRoots + options.maxRoots;
  const perGame = new Map<string, number>();
  while (selected.length < selectionLimit) {
    let added = false;
    for (const bucket of phaseBuckets) {
      let root: CounterfactualRoot | undefined;
      while (bucket.length && !root) {
        const candidate = bucket.shift()!;
        if (
          (perGame.get(candidate.sourceGameId) ?? 0) < options.maxRootsPerGame
        ) {
          root = candidate;
        }
      }
      if (!root) continue;
      selected.push(root);
      perGame.set(root.sourceGameId, (perGame.get(root.sourceGameId) ?? 0) + 1);
      added = true;
      if (selected.length >= selectionLimit) break;
    }
    if (!added) break;
  }
  return selected.slice(options.skipRoots, selectionLimit);
}

export function counterfactualRootSeed(
  batchSeed: number,
  rootId: string
): number {
  let hash = batchSeed >>> 0;
  for (let index = 0; index < rootId.length; index += 1) {
    hash ^= rootId.charCodeAt(index);
    hash = Math.imul(hash, 0x01000193) >>> 0;
  }
  return hash;
}

/** Matched candidates share a seed within each replicate, while replicates vary. */
export function counterfactualReplicateSeed(
  batchSeed: number,
  rootId: string,
  replicate: number
): number {
  if (!Number.isSafeInteger(replicate) || replicate < 0) {
    throw new RangeError("counterfactual replicate must be a non-negative integer");
  }
  return counterfactualRootSeed(
    counterfactualRootSeed(batchSeed, rootId),
    `replicate:${replicate}`
  );
}

interface ReplicateValue {
  replicate: number;
  rolloutValue: number;
  /** A replicate with shallow/seeded fallback cannot supply label evidence. */
  unverifiedFallbackDecisions?: number;
}

/** Require repeated matched continuations to support the same policy label. */
export function counterfactualReplicateEvidence(
  preferred: ReplicateValue[],
  runnerUp: ReplicateValue[],
  expectedReplicates: number,
  minimumDelta: number
) {
  const runnerUpByReplicate = new Map(
    runnerUp.map((replicate) => [replicate.replicate, replicate])
  );
  const replicateDeltas = preferred.flatMap((replicate) => {
    const other = runnerUpByReplicate.get(replicate.replicate);
    return other === undefined
      ? []
      : [
          {
            replicate: replicate.replicate,
            preferredMinusRunnerUp:
              replicate.rolloutValue - other.rolloutValue,
            verified:
              (replicate.unverifiedFallbackDecisions ?? 0) === 0 &&
              (other.unverifiedFallbackDecisions ?? 0) === 0,
          },
        ];
  });
  const supportingReplicates = replicateDeltas.filter(
    (replicate) => replicate.preferredMinusRunnerUp >= minimumDelta
  ).length;
  const conflictingReplicates = replicateDeltas.filter(
    (replicate) => replicate.preferredMinusRunnerUp <= -minimumDelta
  ).length;
  const cleanSupportingReplicates = replicateDeltas.filter(
    (replicate) =>
      replicate.verified &&
      replicate.preferredMinusRunnerUp >= minimumDelta
  ).length;
  const cleanConflictingReplicates = replicateDeltas.filter(
    (replicate) =>
      replicate.verified &&
      replicate.preferredMinusRunnerUp <= -minimumDelta
  ).length;
  const unverifiedReplicates = replicateDeltas.filter(
    (replicate) => !replicate.verified
  ).length;
  const requiredReplicateSupport = Math.min(2, expectedReplicates);
  return {
    replicateDeltas,
    supportingReplicates,
    conflictingReplicates,
    cleanSupportingReplicates,
    cleanConflictingReplicates,
    unverifiedReplicates,
    requiredReplicateSupport,
    replicateReliable:
      cleanSupportingReplicates >= requiredReplicateSupport &&
      cleanConflictingReplicates === 0,
  };
}
