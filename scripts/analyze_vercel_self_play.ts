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
import { auditParatrooperTrainingPolicy } from "../src/game/self-play/training-policy";
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

const MOVE_UCI = /^([a-h])([1-8])([a-h])([1-8])/;
const DIAGONAL_ORIENTATIONS = new Set(["↗", "↘", "↙", "↖"]);
const ARTILLERY_ORIENTATIONS = new Set([
  "↑",
  "↗",
  "→",
  "↘",
  "↓",
  "↙",
  "←",
  "↖",
]);

function isCaptureMove(move: string): boolean {
  return move.includes("x") || (move.startsWith("s") && move !== "skip");
}

function isVoluntaryBoardMove(move: string): boolean {
  return move !== "skip" && !move.startsWith("r") && !move.startsWith("s");
}

function isLongRangeParadrop(move: string, player: "RED" | "BLUE"): boolean {
  const parsed = MOVE_UCI.exec(move);
  if (!parsed) return false;
  const [, fromFile, fromRank, toFile, toRank] = parsed;
  if (fromRank !== (player === "RED" ? "1" : "8")) return false;
  const fileDistance = Math.abs(fromFile.charCodeAt(0) - toFile.charCodeAt(0));
  const rankDistance = Math.abs(Number(fromRank) - Number(toRank));
  // No ordinary GHQ unit can relocate more than two squares. This is a
  // conservative classifier: short airborne commitments are deliberately
  // omitted rather than misclassifying an armored unit.
  return Math.max(fileDistance, rankDistance) > 2;
}

function isDiagonalArtilleryAction(move: string): boolean {
  return DIAGONAL_ORIENTATIONS.has(move.at(-1) ?? "");
}

function isArtilleryAction(move: string): boolean {
  return ARTILLERY_ORIENTATIONS.has(move.at(-1) ?? "");
}

function isPureRotation(move: string): boolean {
  const parsed = MOVE_UCI.exec(move);
  return Boolean(parsed && parsed[1] === parsed[3] && parsed[2] === parsed[4]);
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

function summarizeTacticalPatterns<
  T extends {
    decisions: number;
    voluntaryBoardActions: number;
    artilleryActions: number;
    multiCaptureTurns: number;
    captureSetupMultiCaptureTurns: number;
    paradropActions: number;
    paradropSameTurnFollowupCaptures: number;
    paradropHqCombinations: number;
    paradropConcreteMissions: number;
    paradropSingleCaptureOnly: number;
    diagonalArtilleryActions: number;
    diagonalPureRotations: number;
  }
>(patterns: T) {
  return {
    ...patterns,
    multiCaptureTurnRate: Number(
      (patterns.multiCaptureTurns / Math.max(1, patterns.decisions)).toFixed(4)
    ),
    captureSetupShareOfMultiCaptureTurns: Number(
      (
        patterns.captureSetupMultiCaptureTurns /
        Math.max(1, patterns.multiCaptureTurns)
      ).toFixed(4)
    ),
    paradropActionRate: Number(
      (
        patterns.paradropActions / Math.max(1, patterns.voluntaryBoardActions)
      ).toFixed(4)
    ),
    concreteParadropMissionRate: Number(
      (
        patterns.paradropConcreteMissions /
        Math.max(1, patterns.paradropActions)
      ).toFixed(4)
    ),
    singleCaptureOnlyParadropRate: Number(
      (
        patterns.paradropSingleCaptureOnly /
        Math.max(1, patterns.paradropActions)
      ).toFixed(4)
    ),
    diagonalArtilleryActionRate: Number(
      (
        patterns.diagonalArtilleryActions /
        Math.max(1, patterns.artilleryActions)
      ).toFixed(4)
    ),
    diagonalPureRotationRate: Number(
      (
        patterns.diagonalPureRotations /
        Math.max(1, patterns.diagonalArtilleryActions)
      ).toFixed(4)
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
  const tacticalPatternTelemetry = {
    decisions: 0,
    voluntaryBoardActions: 0,
    artilleryActions: 0,
    multiCaptureTurns: 0,
    captureSetupMultiCaptureTurns: 0,
    paradropActions: 0,
    paradropDirectCaptures: 0,
    paradropSameTurnFollowupCaptures: 0,
    paradropHqCombinations: 0,
    paradropConcreteMissions: 0,
    paradropSingleCaptureOnly: 0,
    diagonalArtilleryActions: 0,
    diagonalArtilleryRelocations: 0,
    diagonalPureRotations: 0,
  };
  const tacticalPatternsByPersonality: Record<
    string,
    typeof tacticalPatternTelemetry
  > = {};
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
  let hqSurvivalOverrideDecisions = 0;
  let hqSurvivalReplyVerifiedDecisions = 0;
  let hqExactReturnProbeDecisions = 0;
  let tacticalReturnGuardDecisions = 0;
  let safeFallbackReplyVerifiedDecisions = 0;
  let policyReturnGuardDecisions = 0;
  let seedReplyVerifiedDecisions = 0;
  let seedReplyRetryDecisions = 0;
  let timedOutDecisions = 0;
  let incompleteTurnDecisions = 0;
  let persistentCacheHits = 0;
  let qualityEligibleGames = 0;
  let policyQuarantinedGames = 0;
  let policyViolationDecisions = 0;
  let policyMissingTelemetryGames = 0;
  let policyMissingTelemetryDecisions = 0;
  let policyQuarantinedPersistedTrainingPositions = 0;
  let policyCleanTrainingGames = 0;
  let policyCleanTrainingPositions = 0;
  let policyUnverifiedFallbackGames = 0;

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
    const policyAudit = auditParatrooperTrainingPolicy(game.decisions);
    const gameUnverifiedFallbackDecisions =
      game.quality.unverifiedFallbackDecisions ??
      game.decisions.filter(
        (decision) =>
          decision.fallback === "seeded" ||
          (decision.fallback !== "none" && decision.completedDepth < 2)
      ).length;
    const strictTrainingEligible =
      policyAudit.eligible &&
      game.quality.trainingEligible &&
      gameUnverifiedFallbackDecisions === 0;
    if (!strictTrainingEligible) {
      policyQuarantinedGames++;
      policyViolationDecisions += policyAudit.violatingDecisions;
      policyMissingTelemetryDecisions += policyAudit.missingTelemetryDecisions;
      if (!policyAudit.telemetryComplete) policyMissingTelemetryGames++;
      if (gameUnverifiedFallbackDecisions > 0) {
        policyUnverifiedFallbackGames++;
      }
      policyQuarantinedPersistedTrainingPositions += game.trainingPositions;
    } else {
      if (game.trainingPositions > 0) policyCleanTrainingGames++;
      policyCleanTrainingPositions += game.trainingPositions;
    }
    fallbackDecisions += game.quality.fallbackDecisions;
    verifiedFallbackDecisions +=
      game.quality.verifiedFallbackDecisions ??
      game.decisions.filter(
        (decision) =>
          decision.fallback === "safe" && decision.completedDepth >= 2
      ).length;
    unverifiedFallbackDecisions += gameUnverifiedFallbackDecisions;
    timedOutDecisions += game.quality.timedOutDecisions;
    if (strictTrainingEligible) qualityEligibleGames++;
    for (const reason of game.quality.trainingRejectionReasons ?? []) {
      increment(trainingRejectionReasons, reason);
    }
    for (const decision of game.decisions) {
      const personalityPatterns = (tacticalPatternsByPersonality[
        decision.personality
      ] ??= {
        decisions: 0,
        voluntaryBoardActions: 0,
        artilleryActions: 0,
        multiCaptureTurns: 0,
        captureSetupMultiCaptureTurns: 0,
        paradropActions: 0,
        paradropDirectCaptures: 0,
        paradropSameTurnFollowupCaptures: 0,
        paradropHqCombinations: 0,
        paradropConcreteMissions: 0,
        paradropSingleCaptureOnly: 0,
        diagonalArtilleryActions: 0,
        diagonalArtilleryRelocations: 0,
        diagonalPureRotations: 0,
      });
      const patternRecords = [tacticalPatternTelemetry, personalityPatterns];
      for (const record of patternRecords) record.decisions++;
      const captureIndexes = decision.selectedMoves
        .map((move, index) => (isCaptureMove(move) ? index : -1))
        .filter((index) => index >= 0);
      if (captureIndexes.length >= 2) {
        for (const record of patternRecords) record.multiCaptureTurns++;
        const firstCapture = captureIndexes[0];
        if (
          decision.selectedMoves
            .slice(0, firstCapture)
            .some(isVoluntaryBoardMove)
        ) {
          for (const record of patternRecords) {
            record.captureSetupMultiCaptureTurns++;
          }
        }
      }
      for (
        let moveIndex = 0;
        moveIndex < decision.selectedMoves.length;
        moveIndex++
      ) {
        const move = decision.selectedMoves[moveIndex];
        if (isVoluntaryBoardMove(move)) {
          for (const record of patternRecords) record.voluntaryBoardActions++;
        }
        if (isArtilleryAction(move)) {
          for (const record of patternRecords) record.artilleryActions++;
        }
        if (isDiagonalArtilleryAction(move)) {
          for (const record of patternRecords) {
            record.diagonalArtilleryActions++;
            if (isPureRotation(move)) record.diagonalPureRotations++;
            else record.diagonalArtilleryRelocations++;
          }
        }
        if (!isLongRangeParadrop(move, decision.player)) continue;
        const directCapture = isCaptureMove(move);
        const followupCapture = decision.selectedMoves
          .slice(moveIndex + 1)
          .some(isCaptureMove);
        const roles =
          (decision.selectedActionPurposes ?? []).find(
            (item) => item.move === move
          )?.roles ?? [];
        const hqCombination =
          roles.some((role) =>
            ["hq_capture_unlock", "capture"].includes(role)
          ) && decision.currentPlayerScore >= 50_000;
        for (const record of patternRecords) {
          record.paradropActions++;
          if (directCapture) record.paradropDirectCaptures++;
          if (followupCapture) record.paradropSameTurnFollowupCaptures++;
          if (hqCombination) record.paradropHqCombinations++;
          if (followupCapture || hqCombination) {
            record.paradropConcreteMissions++;
          }
          if (directCapture && !followupCapture && !hqCombination) {
            record.paradropSingleCaptureOnly++;
          }
        }
      }
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
        if (decision.searchTelemetry.hqSurvivalOverrideUsed) {
          hqSurvivalOverrideDecisions++;
        }
        if (decision.searchTelemetry.hqSurvivalReplyVerified) {
          hqSurvivalReplyVerifiedDecisions++;
        }
        if (decision.searchTelemetry.hqExactReturnProbeUsed) {
          hqExactReturnProbeDecisions++;
        }
        if (decision.searchTelemetry.tacticalReturnGuardUsed) {
          tacticalReturnGuardDecisions++;
        }
        if (decision.searchTelemetry.safeFallbackReplyVerified) {
          safeFallbackReplyVerifiedDecisions++;
        }
        if (decision.searchTelemetry.policyReturnGuardUsed) {
          policyReturnGuardDecisions++;
        }
        if (decision.searchTelemetry.seedReplyVerified) {
          seedReplyVerifiedDecisions++;
        }
        if (decision.searchTelemetry.seedReplyRetryUsed) {
          seedReplyRetryDecisions++;
        }
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
      hqSurvivalOverrideDecisions,
      hqSurvivalReplyVerifiedDecisions,
      hqExactReturnProbeDecisions,
      tacticalReturnGuardDecisions,
      tacticalReturnGuardRate: Number(
        (tacticalReturnGuardDecisions / Math.max(1, decisions)).toFixed(4)
      ),
      safeFallbackReplyVerifiedDecisions,
      safeFallbackReplyVerifiedRate: Number(
        (
          safeFallbackReplyVerifiedDecisions / Math.max(1, decisions)
        ).toFixed(4)
      ),
      policyReturnGuardDecisions,
      seedReplyVerifiedDecisions,
      seedReplyRetryDecisions,
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
    tacticalPatternTelemetry: summarizeTacticalPatterns(
      tacticalPatternTelemetry
    ),
    tacticalPatternsByPersonality: Object.fromEntries(
      Object.entries(tacticalPatternsByPersonality).map(([id, patterns]) => [
        id,
        summarizeTacticalPatterns(patterns),
      ])
    ),
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
    policyTrainingQuarantine: {
      games: policyQuarantinedGames,
      violatingDecisions: policyViolationDecisions,
      missingTelemetryGames: policyMissingTelemetryGames,
      missingTelemetryDecisions: policyMissingTelemetryDecisions,
      unverifiedFallbackGames: policyUnverifiedFallbackGames,
      persistedTrainingPositions: policyQuarantinedPersistedTrainingPositions,
      effectiveTrainingGames: policyCleanTrainingGames,
      effectiveTrainingPositions: policyCleanTrainingPositions,
    },
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
