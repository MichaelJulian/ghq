#!/usr/bin/env tsx
/** Score paired counterfactual continuations and emit policy-training rows. */

import { writeFile } from "node:fs/promises";
import { config } from "dotenv";

import type { Player } from "../src/game/engine";
import { FENtoBoardState } from "../src/game/notation";
import {
  extractValueFeaturesV3,
  VALUE_FEATURE_NAMES_V3,
} from "../src/game/value-model/features";
import { predictZeroSumWinProbability } from "../src/game/value-model/inference";
import {
  readPersistedSelfPlayGames,
  readSelfPlayGenerationManifest,
} from "../src/server/self-play-storage";
import type { DurableSelfPlayGameResult } from "../src/workflows/self-play-game";

config({ path: ".env.local" });

function argument(name: string): string | undefined {
  const index = process.argv.lastIndexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function position(fen: string, turnNumber: number) {
  const state = FENtoBoardState(fen);
  return {
    board: state.board,
    redReserve: state.redReserve,
    blueReserve: state.blueReserve,
    currentPlayer: state.currentPlayerTurn ?? "RED",
    turnNumber,
  };
}

function rolloutValue(
  game: DurableSelfPlayGameResult,
  rootPlayer: Player
): {
  value: number;
  source: "terminal" | "leaf-model";
  leafTurnNumber: number;
} {
  const leafTurnNumber =
    (game.decisions.at(-1)?.turnNumber ?? game.initialTurnNumber ?? 1) + 1;
  if (game.outcome.winner) {
    return {
      value: game.outcome.winner === rootPlayer ? 1 : 0,
      source: "terminal",
      leafTurnNumber,
    };
  }
  return {
    value: predictZeroSumWinProbability(
      position(game.finalFen, leafTurnNumber),
      rootPlayer,
      "three-actions",
      "incumbent"
    ),
    source: "leaf-model",
    leafTurnNumber,
  };
}

async function main() {
  const generationId = argument("--generation");
  if (!generationId) {
    throw new Error("Pass --generation <counterfactual-generation-id>");
  }
  const [manifest, games] = await Promise.all([
    readSelfPlayGenerationManifest(generationId),
    readPersistedSelfPlayGames<DurableSelfPlayGameResult>(generationId),
  ]);
  if (!manifest?.counterfactual) {
    throw new Error(`${generationId} has no counterfactual manifest metadata`);
  }
  const gameById = new Map(games.map((game) => [game.gameId, game]));
  const rollouts = manifest.counterfactual.branches.map((branch) => {
    const game = gameById.get(branch.gameId);
    if (!game) return { ...branch, status: "missing" as const };
    const scored = rolloutValue(game, branch.rootPlayer);
    const start = position(branch.initialFen, branch.initialTurnNumber);
    return {
      ...branch,
      status: "completed" as const,
      rolloutValue: scored.value,
      valueSource: scored.source,
      leafTurnNumber: scored.leafTurnNumber,
      finalFen: game.finalFen,
      outcome: game.outcome,
      decisions: game.decisions.length,
      fallbackDecisions: game.quality.fallbackDecisions,
      unverifiedFallbackDecisions: game.quality.unverifiedFallbackDecisions,
      featuresV3: extractValueFeaturesV3(start, branch.rootPlayer),
    };
  });
  const completedRollouts = rollouts.filter(
    (
      branch
    ): branch is Extract<(typeof rollouts)[number], { status: "completed" }> =>
      branch.status === "completed"
  );
  const rolloutGroups = new Map<string, typeof rollouts>();
  for (const branch of rollouts) {
    const key = `${branch.rootId}:${branch.candidateRank}`;
    const replicates = rolloutGroups.get(key) ?? [];
    replicates.push(branch);
    rolloutGroups.set(key, replicates);
  }
  const branches = [...rolloutGroups.values()].map((replicates) => {
    const first = replicates[0];
    const completed = replicates.filter(
      (
        branch
      ): branch is Extract<(typeof rollouts)[number], { status: "completed" }> =>
        branch.status === "completed"
    );
    if (completed.length !== replicates.length) {
      return {
        rootId: first.rootId,
        sourceGameId: first.sourceGameId,
        sourceTurnNumber: first.sourceTurnNumber,
        rootPlayer: first.rootPlayer,
        candidateRank: first.candidateRank,
        status: "missing" as const,
        expectedReplicates: replicates.length,
        completedReplicates: completed.length,
      };
    }
    const representative = completed[0];
    return {
      ...representative,
      gameId: representative.gameId,
      status: "completed" as const,
      rolloutValue:
        completed.reduce((sum, branch) => sum + branch.rolloutValue, 0) /
        completed.length,
      valueSource: completed.every(
        (branch) => branch.valueSource === "terminal"
      )
        ? ("terminal" as const)
        : ("leaf-model" as const),
      leafTurnNumber: Math.max(
        ...completed.map((branch) => branch.leafTurnNumber)
      ),
      decisions: completed.reduce((sum, branch) => sum + branch.decisions, 0),
      fallbackDecisions: completed.reduce(
        (sum, branch) => sum + branch.fallbackDecisions,
        0
      ),
      unverifiedFallbackDecisions: completed.reduce(
        (sum, branch) => sum + branch.unverifiedFallbackDecisions,
        0
      ),
      expectedReplicates: replicates.length,
      completedReplicates: completed.length,
      replicateValues: completed.map((branch) => ({
        replicate: branch.replicate ?? 0,
        gameId: branch.gameId,
        rolloutValue: branch.rolloutValue,
        valueSource: branch.valueSource,
        outcome: branch.outcome,
        finalFen: branch.finalFen,
      })),
    };
  });
  const completed = branches.filter(
    (
      branch
    ): branch is Extract<(typeof branches)[number], { status: "completed" }> =>
      branch.status === "completed"
  );
  const grouped = new Map<string, typeof completed>();
  for (const branch of completed) {
    const siblings = grouped.get(branch.rootId) ?? [];
    siblings.push(branch);
    grouped.set(branch.rootId, siblings);
  }
  const minimumDeltaRaw = Number(argument("--minimum-delta") ?? 0.02);
  if (
    !Number.isFinite(minimumDeltaRaw) ||
    minimumDeltaRaw < 0 ||
    minimumDeltaRaw > 1
  ) {
    throw new RangeError("--minimum-delta must be between zero and one");
  }
  const pairs = [...grouped.entries()].flatMap(([rootId, siblings]) => {
    if (siblings.length < 2) return [];
    const ranked = [...siblings].sort(
      (left, right) =>
        right.rolloutValue - left.rolloutValue ||
        left.candidateRank - right.candidateRank
    );
    const best = ranked[0];
    const runnerUp = ranked[1];
    const delta = best.rolloutValue - runnerUp.rolloutValue;
    const confident = delta >= minimumDeltaRaw;
    const hasUnverifiedFallback = siblings.some(
      (branch) => branch.unverifiedFallbackDecisions > 0
    );
    return [
      {
        rootId,
        sourceGameId: best.sourceGameId,
        sourceTurnNumber: best.sourceTurnNumber,
        rootPlayer: best.rootPlayer,
        preferredCandidateRank: best.candidateRank,
        searchPreferredCandidateRank: Math.min(
          ...siblings.map((branch) => branch.candidateRank)
        ),
        rolloutDelta: delta,
        confident,
        trainingEligible: confident && !hasUnverifiedFallback,
        trainingExclusion: !confident
          ? "insufficient-rollout-separation"
          : hasUnverifiedFallback
            ? "unverified-fallback"
            : undefined,
        branches: [...siblings].sort(
          (left, right) => left.candidateRank - right.candidateRank
        ),
      },
    ];
  });
  const confident = pairs.filter((pair) => pair.confident);
  const trainingEligible = pairs.filter((pair) => pair.trainingEligible);
  const report = {
    format: "ghq-counterfactual-rollout-report-v1",
    generationId,
    sourceGenerationId: manifest.counterfactual.sourceGenerationId,
    expectedBranches: branches.length,
    completedBranches: completed.length,
    missingBranches: branches.length - completed.length,
    expectedRollouts: rollouts.length,
    completedRollouts: completedRollouts.length,
    missingRollouts: rollouts.length - completedRollouts.length,
    rootsWithCompletePairs: pairs.length,
    confidentPairs: confident.length,
    trainingEligiblePairs: trainingEligible.length,
    trainingExcludedForUnverifiedFallback: confident.filter(
      (pair) => pair.trainingExclusion === "unverified-fallback"
    ).length,
    minimumDelta: minimumDeltaRaw,
    searchTopCandidatePreferred: pairs.filter(
      (pair) =>
        pair.preferredCandidateRank === pair.searchPreferredCandidateRank
    ).length,
    searchTopCandidateRejected: pairs.filter(
      (pair) =>
        pair.confident &&
        pair.preferredCandidateRank !== pair.searchPreferredCandidateRank
    ).length,
    searchTopCandidateRejectedTrainingEligible: trainingEligible.filter(
      (pair) =>
        pair.preferredCandidateRank !== pair.searchPreferredCandidateRank
    ).length,
    terminalBranches: completed.filter((branch) =>
      branch.replicateValues.every(
        (replicate) => replicate.valueSource === "terminal"
      )
    ).length,
    leafModelBranches: completed.filter((branch) =>
      branch.replicateValues.some(
        (replicate) => replicate.valueSource === "leaf-model"
      )
    ).length,
    terminalRollouts: completedRollouts.filter(
      (branch) => branch.valueSource === "terminal"
    ).length,
    leafModelRollouts: completedRollouts.filter(
      (branch) => branch.valueSource === "leaf-model"
    ).length,
    fallbackDecisions: completed.reduce(
      (sum, branch) => sum + branch.fallbackDecisions,
      0
    ),
    unverifiedFallbackDecisions: completed.reduce(
      (sum, branch) => sum + branch.unverifiedFallbackDecisions,
      0
    ),
    featureSchema: VALUE_FEATURE_NAMES_V3,
    pairs,
  };
  const rendered = `${JSON.stringify(report, null, 2)}\n`;
  const outputPath = argument("--output");
  if (outputPath) await writeFile(outputPath, rendered, "utf8");
  process.stdout.write(rendered);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
