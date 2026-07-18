#!/usr/bin/env tsx
/** Summarize selected persisted Vercel self-play generations. */

import "dotenv/config";
import { writeFile } from "node:fs/promises";
import { get, list, type ListBlobResultBlob } from "@vercel/blob";

import {
  actionMadeProgress,
  type DurableSelfPlayGameResult,
} from "../src/workflows/self-play-game";
import { summarizeValueModelArena } from "../src/game/self-play/arena-results";
import { partitionColorSwapPairs } from "../src/game/self-play/color-pairs";
import {
  extendsStrategicBest,
  mergeStrategicBest,
  strategicProgress,
} from "../src/game/self-play/strategic-progress";

function argumentsFor(name: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < process.argv.length; index++) {
    if (process.argv[index] === name && process.argv[index + 1]) {
      values.push(process.argv[index + 1]);
    }
  }
  return values;
}

async function gameBlobs(generationId: string) {
  const blobs: ListBlobResultBlob[] = [];
  let cursor: string | undefined;
  do {
    const page = await list({
      prefix: `self-play/generations/${generationId}/games/`,
      cursor,
      limit: 1000,
    });
    blobs.push(...page.blobs.filter((blob) => blob.pathname.endsWith(".json")));
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs;
}

async function readGame(blob: ListBlobResultBlob) {
  const result = await get(blob.pathname, {
    access: "private",
    useCache: false,
  });
  if (!result?.stream || result.statusCode !== 200) {
    throw new Error(`Unable to read ${blob.pathname}`);
  }
  return (await new Response(
    result.stream
  ).json()) as DurableSelfPlayGameResult;
}

function personality(agentId: string): string {
  return agentId
    .replace(/-workflow.*$/, "")
    .replace(/-(?:incumbent|challenger)-a[23]$/, "");
}

function valueModel(agentId: string, recorded?: string): string {
  if (recorded === "challenger" || recorded === "incumbent") return recorded;
  return agentId.includes("-challenger-") ? "challenger" : "incumbent";
}

function increment(counts: Record<string, number>, key: string, amount = 1) {
  counts[key] = (counts[key] ?? 0) + amount;
}

function sameMoves(left: string[], right: string[]): boolean {
  return (
    left.length === right.length &&
    left.every((move, index) => move === right[index])
  );
}

function percentile(values: number[], fraction: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.round((sorted.length - 1) * fraction)];
}

function distribution(values: number[]) {
  if (!values.length) return { count: 0 };
  return {
    count: values.length,
    min: Math.min(...values),
    median: percentile(values, 0.5),
    p90: percentile(values, 0.9),
    p99: percentile(values, 0.99),
    max: Math.max(...values),
    mean: Number(
      (values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(1)
    ),
  };
}

async function main() {
  const generationIds = argumentsFor("--generation");
  if (!generationIds.length) {
    throw new Error("Pass at least one --generation <generation-id>");
  }

  const blobs = (await Promise.all(generationIds.map(gameBlobs))).flat();
  const games: DurableSelfPlayGameResult[] = [];
  for (let index = 0; index < blobs.length; index += 12) {
    games.push(
      ...(await Promise.all(blobs.slice(index, index + 12).map(readGame)))
    );
  }

  const outcomes: Record<string, number> = {};
  const terminations: Record<string, number> = {};
  const actionCounts: Record<string, number> = {};
  const countedActionCounts: Record<string, number> = {};
  const depths: Record<string, number> = {};
  const fallbackTypes: Record<string, number> = {};
  const fallbackByColor: Record<string, number> = {};
  const fallbackByPhase: Record<string, number> = {};
  const fallbackByTurn: Record<string, number> = {};
  const fallbackExamples: Array<{
    gameId: string;
    turn: number;
    player: string;
    personality: string;
    turnsWithoutProgress: number;
    fen: string;
    moves: string[];
    fallback: string;
    depth: number;
    candidates: number;
    score: number;
    nodes?: number;
    elapsedMs?: number;
    completeTurnsGenerated?: number;
    beamPrunedActions?: number;
    hqSurvivalProbeNodes?: number;
    hqSurvivalReplyNodes?: number;
    recommendation?: string;
  }> = [];
  const historyAvoidanceByTurn: Record<string, number> = {};
  const purposeTelemetry = {
    decisions: 0,
    decisionsWithNoNewEffect: 0,
    noNewEffectActions: 0,
    decisionsWithSetup: 0,
    setupActions: 0,
    unpurposedActions: 0,
    reversals: 0,
    pureRotations: 0,
    paratrooperMissionPenaltyDecisions: 0,
    totalParatrooperMissionPenalty: 0,
    totalNetPurposePenalty: 0,
    decisionsWithGeneratedEarlyStops: 0,
    generatedEarlyStopCandidates: 0,
    selectedPurposefulEarlyStops: 0,
  };
  const purposeExamples: Array<{
    gameId: string;
    turn: number;
    player: string;
    fen: string;
    moves: string[];
    actionPurposes: Array<{ move: string; roles: string[] }>;
    setupActions: number;
    unpurposedActions: number;
    reversals: number;
    pureRotations: number;
    paratrooperMissionPenalty: number;
    netPurposePenalty: number;
  }> = [];
  const depthByColor: Record<string, number> = {};
  const trainingRejectionReasons: Record<string, number> = {};
  const personalities: Record<
    string,
    { games: number; wins: number; losses: number; draws: number }
  > = {};
  const valueModels: Record<
    string,
    {
      games: number;
      wins: number;
      losses: number;
      draws: number;
      points: number;
    }
  > = {};
  const rejected: Array<{
    gameId: string;
    termination: string;
    winner?: string;
    decisions: number;
    fallbackDecisions: number;
  }> = [];
  const stalledGameTails: Array<{
    gameId: string;
    termination: string;
    tail: Array<{
      turn: number;
      player: string;
      personality: string;
      agentId: string;
      fen: string;
      moves: string[];
      score: number;
      depth: number;
      fallback: string;
      recommendation?: string;
      selectedRank: number;
    }>;
  }> = [];
  const overLimitExamples: Array<{
    gameId: string;
    turnNumber: number;
    player: string;
    moves: string[];
    fallback: string;
    depth: number;
  }> = [];
  const immediateHqLosses: Array<{
    gameId: string;
    winner: string;
    potentialHorizonBlunder: boolean;
    potentialValueModelTacticalContradiction: boolean;
    highConfidenceValueModelTacticalContradiction: boolean;
    loserDecision: {
      turn: number;
      player: string;
      personality: string;
      agentId: string;
      fen: string;
      moves: string[];
      score: number;
      winProbability: number;
      depth: number;
      fallback: string;
    };
    winningDecision: {
      turn: number;
      player: string;
      moves: string[];
      score: number;
    };
  }> = [];
  const lengths: number[] = [];
  const searchElapsedMs: number[] = [];
  const searchNodes: number[] = [];
  const completeTurnsGenerated: number[] = [];
  const hqSurvivalProbeNodes: number[] = [];
  const hqSurvivalReplyNodes: number[] = [];
  const fallbackElapsedMs: number[] = [];
  let decisions = 0;
  let trainingPositions = 0;
  let fallbackDecisions = 0;
  let verifiedFallbackDecisions = 0;
  let unverifiedFallbackDecisions = 0;
  let timedOutDecisions = 0;
  let incompleteTurnDecisions = 0;
  let persistentCacheHits = 0;
  let qualityEligibleGames = 0;

  for (const game of games) {
    let turnsWithoutProgress = 0;
    const strategicBest = {
      RED: strategicProgress(game.initialFen, "RED"),
      BLUE: strategicProgress(game.initialFen, "BLUE"),
    };
    increment(outcomes, game.outcome.winner ?? "DRAW");
    increment(terminations, game.outcome.termination);
    lengths.push(
      Math.max(0, ...game.decisions.map((decision) => decision.turnNumber))
    );
    decisions += game.decisions.length;
    trainingPositions += game.trainingPositions;
    fallbackDecisions += game.quality.fallbackDecisions;
    verifiedFallbackDecisions +=
      game.quality.verifiedFallbackDecisions ??
      game.decisions.filter(
        (decision) =>
          decision.fallback === "safe" && decision.completedDepth >= 2
      ).length;
    unverifiedFallbackDecisions +=
      game.quality.unverifiedFallbackDecisions ??
      game.decisions.filter(
        (decision) =>
          decision.fallback === "seeded" ||
          (decision.fallback !== "none" && decision.completedDepth < 2)
      ).length;
    timedOutDecisions += game.quality.timedOutDecisions;
    if (game.quality.trainingEligible) qualityEligibleGames++;
    for (const reason of game.quality.trainingRejectionReasons ?? []) {
      increment(trainingRejectionReasons, reason);
    }
    for (const decision of game.decisions) {
      increment(actionCounts, String(decision.selectedMoves.length));
      increment(
        countedActionCounts,
        String(
          decision.selectedMoves.filter((move) => !move.startsWith("s")).length
        )
      );
      increment(depths, String(decision.completedDepth));
      if (decision.searchTelemetry) {
        searchElapsedMs.push(decision.searchTelemetry.elapsedMs);
        searchNodes.push(decision.searchTelemetry.nodes);
        completeTurnsGenerated.push(
          decision.searchTelemetry.completeTurnsGenerated
        );
        hqSurvivalProbeNodes.push(
          decision.searchTelemetry.hqSurvivalProbeNodes ?? 0
        );
        hqSurvivalReplyNodes.push(
          decision.searchTelemetry.hqSurvivalReplyNodes ?? 0
        );
        if (decision.fallback !== "none") {
          fallbackElapsedMs.push(decision.searchTelemetry.elapsedMs);
        }
      }
      if (decision.persistentCacheHit) persistentCacheHits++;
      increment(
        depthByColor,
        `${decision.player}:depth-${decision.completedDepth}`
      );
      increment(fallbackTypes, decision.fallback);
      if (decision.fallback !== "none") {
        increment(fallbackByColor, decision.player);
        const phase =
          decision.turnNumber <= 12
            ? "early"
            : decision.turnNumber <= 60
            ? "mid"
            : "late";
        increment(fallbackByPhase, phase);
        increment(fallbackByTurn, String(decision.turnNumber));
        if (fallbackExamples.length < 40) {
          fallbackExamples.push({
            gameId: game.gameId,
            turn: decision.turnNumber,
            player: decision.player,
            personality: decision.personality,
            turnsWithoutProgress,
            fen: decision.fen,
            moves: decision.selectedMoves,
            fallback: decision.fallback,
            depth: decision.completedDepth,
            candidates: decision.candidateTurns?.length ?? 0,
            score: decision.currentPlayerScore,
            nodes: decision.searchTelemetry?.nodes,
            elapsedMs: decision.searchTelemetry?.elapsedMs,
            completeTurnsGenerated:
              decision.searchTelemetry?.completeTurnsGenerated,
            beamPrunedActions: decision.searchTelemetry?.beamPrunedActions,
            hqSurvivalProbeNodes:
              decision.searchTelemetry?.hqSurvivalProbeNodes,
            hqSurvivalReplyNodes:
              decision.searchTelemetry?.hqSurvivalReplyNodes,
            recommendation: decision.recommendationLabel,
          });
        }
      }
      if (decision.recommendationLabel === "history avoidance") {
        increment(historyAvoidanceByTurn, String(decision.turnNumber));
      }
      const selectedCandidate = decision.candidateTurns?.find((candidate) =>
        sameMoves(candidate.all_moves, decision.selectedMoves)
      );
      const selectedPurpose =
        decision.selectedPurpose ?? selectedCandidate?.purpose;
      const selectedActionPurposes =
        decision.selectedActionPurposes ?? selectedCandidate?.action_purposes;
      if (selectedPurpose && selectedActionPurposes) {
        const purpose = selectedPurpose;
        const noNewEffectActions = selectedActionPurposes.filter((action) =>
          action.roles.includes("no_new_effect")
        ).length;
        const setupActions = selectedActionPurposes.filter((action) =>
          action.roles.includes("setup")
        ).length;
        purposeTelemetry.decisions++;
        purposeTelemetry.noNewEffectActions += noNewEffectActions;
        purposeTelemetry.setupActions += setupActions;
        purposeTelemetry.unpurposedActions += purpose.unpurposed_actions;
        purposeTelemetry.reversals += purpose.reversals;
        purposeTelemetry.pureRotations += purpose.pure_rotations;
        purposeTelemetry.totalParatrooperMissionPenalty +=
          purpose.paratrooper_mission_penalty;
        purposeTelemetry.totalNetPurposePenalty += purpose.net_purpose_penalty;
        const generatedEarlyStops = decision.purposefulEarlyStopsGenerated ?? 0;
        purposeTelemetry.generatedEarlyStopCandidates += generatedEarlyStops;
        if (generatedEarlyStops > 0) {
          purposeTelemetry.decisionsWithGeneratedEarlyStops++;
        }
        const countedActions = decision.selectedMoves.filter(
          (move) => move !== "skip" && !move.startsWith("s")
        ).length;
        if (
          countedActions === 2 &&
          decision.selectedMoves.includes("skip") &&
          noNewEffectActions === 0
        ) {
          purposeTelemetry.selectedPurposefulEarlyStops++;
        }
        if (noNewEffectActions) {
          purposeTelemetry.decisionsWithNoNewEffect++;
        }
        if (setupActions) {
          purposeTelemetry.decisionsWithSetup++;
        }
        if (purpose.paratrooper_mission_penalty > 0) {
          purposeTelemetry.paratrooperMissionPenaltyDecisions++;
        }
        if (
          purposeExamples.length < 40 &&
          (noNewEffectActions > 0 ||
            setupActions > 0 ||
            purpose.unpurposed_actions > 0 ||
            purpose.reversals > 0 ||
            purpose.pure_rotations > 0 ||
            purpose.paratrooper_mission_penalty > 0)
        ) {
          purposeExamples.push({
            gameId: game.gameId,
            turn: decision.turnNumber,
            player: decision.player,
            fen: decision.fen,
            moves: decision.selectedMoves,
            actionPurposes: selectedActionPurposes,
            setupActions,
            unpurposedActions: purpose.unpurposed_actions,
            reversals: purpose.reversals,
            pureRotations: purpose.pure_rotations,
            paratrooperMissionPenalty: purpose.paratrooper_mission_penalty,
            netPurposePenalty: purpose.net_purpose_penalty,
          });
        }
      }
      if (!decision.completedTurn) incompleteTurnDecisions++;

      if (decision.completedTurn) {
        const currentProgress = strategicProgress(
          decision.resultingFen,
          decision.player
        );
        const madeStrategicProgress = extendsStrategicBest(
          strategicBest[decision.player],
          currentProgress
        );
        strategicBest[decision.player] = mergeStrategicBest(
          strategicBest[decision.player],
          currentProgress
        );
        turnsWithoutProgress =
          madeStrategicProgress ||
          decision.selectedMoves.some(actionMadeProgress)
            ? 0
            : turnsWithoutProgress + 1;
      }
      if (
        decision.selectedMoves.filter((move) => !move.startsWith("s")).length >
          3 &&
        overLimitExamples.length < 12
      ) {
        overLimitExamples.push({
          gameId: game.gameId,
          turnNumber: decision.turnNumber,
          player: decision.player,
          moves: decision.selectedMoves,
          fallback: decision.fallback,
          depth: decision.completedDepth,
        });
      }
    }
    if (!game.trainingPositions) {
      rejected.push({
        gameId: game.gameId,
        termination: game.outcome.termination,
        winner: game.outcome.winner,
        decisions: game.decisions.length,
        fallbackDecisions: game.quality.fallbackDecisions,
      });
    }
    if (
      game.outcome.termination === "no-progress" ||
      game.outcome.termination === "repetition" ||
      game.outcome.termination === "max-turns"
    ) {
      stalledGameTails.push({
        gameId: game.gameId,
        termination: game.outcome.termination,
        tail: game.decisions.slice(-24).map((decision) => ({
          turn: decision.turnNumber,
          player: decision.player,
          personality: decision.personality,
          agentId: decision.agentId,
          fen: decision.fen,
          moves: decision.selectedMoves,
          score: decision.currentPlayerScore,
          depth: decision.completedDepth,
          fallback: decision.fallback,
          recommendation: decision.recommendationLabel,
          selectedRank: decision.selectedRank,
        })),
      });
    }
    if (
      game.outcome.termination === "hq-capture" &&
      game.outcome.winner &&
      game.decisions.length >= 2
    ) {
      const loserDecision = game.decisions.at(-2)!;
      const winningDecision = game.decisions.at(-1)!;
      if (
        winningDecision.player === game.outcome.winner &&
        loserDecision.player !== game.outcome.winner &&
        winningDecision.turnNumber === loserDecision.turnNumber + 1
      ) {
        immediateHqLosses.push({
          gameId: game.gameId,
          winner: game.outcome.winner,
          // A complete opponent reply should see an available next-turn HQ
          // capture. Scores below this threshold already encode forced mate;
          // ordinary scores indicate a reply-generation or horizon miss.
          potentialHorizonBlunder:
            loserDecision.completedDepth >= 2 &&
            loserDecision.currentPlayerScore > -50_000,
          // Search and the raw value model serve different purposes. When the
          // completed search already encodes a forced loss but the static
          // model still favors the defender, retain the position as an
          // explicit hard-negative candidate for the next value checkpoint.
          // Exact HQ auditing remains the authority on whether the loss was
          // truly forced.
          potentialValueModelTacticalContradiction:
            loserDecision.currentPlayerScore <= -50_000 &&
            loserDecision.winProbability > 0.5,
          highConfidenceValueModelTacticalContradiction:
            loserDecision.currentPlayerScore <= -50_000 &&
            loserDecision.winProbability >= 0.8,
          loserDecision: {
            turn: loserDecision.turnNumber,
            player: loserDecision.player,
            personality: loserDecision.personality,
            agentId: loserDecision.agentId,
            fen: loserDecision.fen,
            moves: loserDecision.selectedMoves,
            score: loserDecision.currentPlayerScore,
            winProbability: loserDecision.winProbability,
            depth: loserDecision.completedDepth,
            fallback: loserDecision.fallback,
          },
          winningDecision: {
            turn: winningDecision.turnNumber,
            player: winningDecision.player,
            moves: winningDecision.selectedMoves,
            score: winningDecision.currentPlayerScore,
          },
        });
      }
    }

    for (const [color, agentId] of [
      ["RED", game.redAgentId],
      ["BLUE", game.blueAgentId],
    ] as const) {
      const id = personality(agentId);
      const record = (personalities[id] ??= {
        games: 0,
        wins: 0,
        losses: 0,
        draws: 0,
      });
      record.games++;
      if (!game.outcome.winner) record.draws++;
      else if (game.outcome.winner === color) record.wins++;
      else record.losses++;
    }

    for (const [color, agentId, recordedModel] of [
      ["RED", game.redAgentId, game.redValueModel],
      ["BLUE", game.blueAgentId, game.blueValueModel],
    ] as const) {
      const id = valueModel(agentId, recordedModel);
      const record = (valueModels[id] ??= {
        games: 0,
        wins: 0,
        losses: 0,
        draws: 0,
        points: 0,
      });
      record.games++;
      if (!game.outcome.winner) {
        record.draws++;
        record.points += 0.5;
      } else if (game.outcome.winner === color) {
        record.wins++;
        record.points++;
      } else {
        record.losses++;
      }
    }
  }

  const pairedOutcomes: Record<string, number> = {};
  const colorPairs = partitionColorSwapPairs(games);
  for (const [firstGame, secondGame] of colorPairs.pairs) {
    const first = firstGame.outcome.winner ?? "DRAW";
    const second = secondGame.outcome.winner ?? "DRAW";
    increment(pairedOutcomes, `${first}/${second}`);
  }

  const report = {
    generationIds,
    games: games.length,
    threeActionGames: games.filter(
      (game) => game.redMaxActions === 3 && game.blueMaxActions === 3
    ).length,
    outcomes,
    terminations,
    gameLengthTurns: {
      min: Math.min(...lengths),
      median: percentile(lengths, 0.5),
      p90: percentile(lengths, 0.9),
      max: Math.max(...lengths),
      mean: Number(
        (
          lengths.reduce((sum, length) => sum + length, 0) / lengths.length
        ).toFixed(1)
      ),
    },
    decisions,
    actionCounts,
    countedActionCounts,
    completedDepths: depths,
    depthByColor,
    searchTelemetry: {
      coverageRate: Number((searchElapsedMs.length / decisions).toFixed(4)),
      elapsedMs: distribution(searchElapsedMs),
      nodes: distribution(searchNodes),
      completeTurnsGenerated: distribution(completeTurnsGenerated),
      hqSurvivalProbeNodes: distribution(hqSurvivalProbeNodes),
      hqSurvivalReplyNodes: distribution(hqSurvivalReplyNodes),
      fallbackElapsedMs: distribution(fallbackElapsedMs),
    },
    fallbackTypes,
    fallbackByColor,
    fallbackByPhase,
    fallbackByTurn,
    fallbackExamples,
    historyAvoidanceByTurn,
    purposeTelemetry: {
      ...purposeTelemetry,
      coverageRate: Number((purposeTelemetry.decisions / decisions).toFixed(4)),
      decisionsWithNoNewEffectRate: Number(
        (
          purposeTelemetry.decisionsWithNoNewEffect /
          Math.max(1, purposeTelemetry.decisions)
        ).toFixed(4)
      ),
      noNewEffectActionsPerDecision: Number(
        (
          purposeTelemetry.noNewEffectActions /
          Math.max(1, purposeTelemetry.decisions)
        ).toFixed(4)
      ),
      decisionsWithSetupRate: Number(
        (
          purposeTelemetry.decisionsWithSetup /
          Math.max(1, purposeTelemetry.decisions)
        ).toFixed(4)
      ),
      setupActionsPerDecision: Number(
        (
          purposeTelemetry.setupActions /
          Math.max(1, purposeTelemetry.decisions)
        ).toFixed(4)
      ),
      unpurposedActionsPerDecision: Number(
        (
          purposeTelemetry.unpurposedActions /
          Math.max(1, purposeTelemetry.decisions)
        ).toFixed(4)
      ),
      selectedPurposefulEarlyStopRate: Number(
        (
          purposeTelemetry.selectedPurposefulEarlyStops /
          Math.max(1, purposeTelemetry.decisions)
        ).toFixed(4)
      ),
    },
    purposeExamples,
    fallbackDecisions,
    fallbackRate: Number((fallbackDecisions / decisions).toFixed(4)),
    verifiedFallbackDecisions,
    unverifiedFallbackDecisions,
    unverifiedFallbackRate: Number(
      (unverifiedFallbackDecisions / decisions).toFixed(4)
    ),
    timedOutDecisions,
    timedOutRate: Number((timedOutDecisions / decisions).toFixed(4)),
    incompleteTurnDecisions,
    persistentCacheHits,
    persistentCacheHitRate: Number(
      (persistentCacheHits / decisions).toFixed(4)
    ),
    trainingGames: games.filter((game) => game.trainingPositions > 0).length,
    qualityEligibleGames,
    trainingRejectionReasons,
    trainingPositions,
    personalities,
    valueModels: Object.fromEntries(
      Object.entries(valueModels).map(([id, record]) => [
        id,
        {
          ...record,
          scoreRate: Number((record.points / record.games).toFixed(4)),
        },
      ])
    ),
    pairIntegrity: {
      completePairs: colorPairs.pairs.length,
      orphanGames: colorPairs.orphans.length,
      orphanGameIds: colorPairs.orphans.map((game) => game.gameId),
    },
    pairedOutcomes,
    valueModelTerminalCalibration: {
      positions: immediateHqLosses.length,
      potentialContradictions: immediateHqLosses.filter(
        (loss) => loss.potentialValueModelTacticalContradiction
      ).length,
      highConfidencePotentialContradictions: immediateHqLosses.filter(
        (loss) => loss.highConfidenceValueModelTacticalContradiction
      ).length,
      meanLosingWinProbability: immediateHqLosses.length
        ? Number(
            (
              immediateHqLosses.reduce(
                (sum, loss) => sum + loss.loserDecision.winProbability,
                0
              ) / immediateHqLosses.length
            ).toFixed(4)
          )
        : null,
      maxLosingWinProbability: immediateHqLosses.length
        ? Number(
            Math.max(
              ...immediateHqLosses.map(
                (loss) => loss.loserDecision.winProbability
              )
            ).toFixed(4)
          )
        : null,
    },
    immediateHqLosses: immediateHqLosses.sort(
      (left, right) =>
        Number(right.highConfidenceValueModelTacticalContradiction) -
          Number(left.highConfidenceValueModelTacticalContradiction) ||
        Number(right.potentialValueModelTacticalContradiction) -
          Number(left.potentialValueModelTacticalContradiction) ||
        Number(right.potentialHorizonBlunder) -
          Number(left.potentialHorizonBlunder) ||
        right.loserDecision.score - left.loserDecision.score
    ),
    valueModelArena: summarizeValueModelArena(games),
    rejected,
    stalledGameTails,
    overLimitExamples,
  };
  const rendered = `${JSON.stringify(report, null, 2)}\n`;
  const outputPath = argumentsFor("--output").at(-1);
  if (outputPath) await writeFile(outputPath, rendered, "utf8");
  process.stdout.write(rendered);
}

void main();
