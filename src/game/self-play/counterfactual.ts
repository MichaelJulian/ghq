import type { PersonalityId } from "@/game/analysis/types";
import type { Player } from "@/game/engine";
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
  sourceGameId: string;
  sourceTurnNumber: number;
  rootPlayer: Player;
  scoreMargin: number;
  branches: CounterfactualBranch[];
}

export interface CounterfactualSelectionOptions {
  maxRoots?: number;
  maxRootsPerGame?: number;
  candidatesPerRoot?: number;
  maxScoreMargin?: number;
  minTurnNumber?: number;
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
    maxRootsPerGame: rawOptions.maxRootsPerGame ?? 2,
    candidatesPerRoot: rawOptions.candidatesPerRoot ?? 2,
    maxScoreMargin: rawOptions.maxScoreMargin ?? 1,
    minTurnNumber: rawOptions.minTurnNumber ?? 5,
  };
  for (const [name, value] of Object.entries(options)) {
    if (!Number.isFinite(value) || value <= 0) {
      throw new RangeError(`${name} must be positive`);
    }
  }
  if (!Number.isSafeInteger(options.maxRoots)) {
    throw new RangeError("maxRoots must be an integer");
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

  const possible = games.flatMap((game) =>
    game.decisions.flatMap((decision) => {
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
      const rootId = `${game.gameId}:t${decision.turnNumber}`;
      const redPersonality = personalityFor(game, "RED");
      const bluePersonality = personalityFor(game, "BLUE");
      return [
        {
          rootId,
          sourceGameId: game.gameId,
          sourceTurnNumber: decision.turnNumber,
          rootPlayer: decision.player,
          scoreMargin,
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
    })
  );
  possible.sort(
    (left, right) =>
      left.scoreMargin - right.scoreMargin ||
      left.sourceTurnNumber - right.sourceTurnNumber ||
      left.rootId.localeCompare(right.rootId)
  );

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
      possible.filter((root) => inPhase(root) && root.rootPlayer === player)
    )
  );
  const selected: CounterfactualRoot[] = [];
  const perGame = new Map<string, number>();
  while (selected.length < options.maxRoots) {
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
      if (selected.length >= options.maxRoots) break;
    }
    if (!added) break;
  }
  return selected;
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
