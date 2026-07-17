import { GHQ_STARTING_FEN } from "@/game/analysis/types";
import type { GhqCandidateTurn, PersonalityId } from "@/game/analysis/types";
import type { Player } from "@/game/engine-v2";
import { FENtoBoardState } from "@/game/notation";
import { extractValueFeatures } from "@/game/value-model/features";
import type { ValueModelVersion } from "@/game/value-model/inference";
import {
  extendsStrategicBest,
  mergeStrategicBest,
  strategicProgress,
  type StrategicProgress,
} from "@/game/self-play/strategic-progress";
import { analyzeFen } from "@/server/fen-analysis";
import {
  persistSelfPlayArtifacts,
  type PersistedSelfPlayArtifacts,
} from "@/server/self-play-storage";

export interface DurableSelfPlayCompetitor {
  id: string;
  personality: PersonalityId;
  timeMs: number;
  maxDepth: number;
  beamWidth: number;
  explorationTemperature: number;
  maxActions?: 2 | 3;
  valueModel?: ValueModelVersion;
  valueModelCheckpoint?: string;
}

export interface DurableSelfPlayGameConfig {
  generationId: string;
  gameId: string;
  seed: number;
  red: DurableSelfPlayCompetitor;
  blue: DurableSelfPlayCompetitor;
  initialFen?: string;
  maxTurns?: number;
  repetitionLimit?: number;
  noProgressTurns?: number;
  codeVersion?: string;
}

export interface DurableSelfPlayDecision {
  turnNumber: number;
  player: Player;
  fen: string;
  resultingFen: string;
  personality: PersonalityId;
  agentId: string;
  opponentId: string;
  selectedMoves: string[];
  selectedRank: number;
  candidateTurns: GhqCandidateTurn[];
  currentPlayerScore: number;
  winProbability: number;
  completedDepth: number;
  persistentCacheHit?: boolean;
  timedOut: boolean;
  fallback: "none" | "safe" | "seeded";
  /** Missing on historical records created before recommendation telemetry. */
  recommendationLabel?: string;
  explorationSeed: number;
  explorationTemperature: number;
  features: number[];
  completedTurn: boolean;
  /** Missing on historical records, which used the standard three-action turn. */
  selfActionLimit?: number;
  /** Missing on historical records, which used the standard three-action turn. */
  opponentActionLimit?: number;
  /** Missing on historical records, which used the incumbent checkpoint. */
  valueModel?: ValueModelVersion;
}

interface DurableTurnStepInput {
  fen?: string;
  serializedState?: string;
  turnNumber: number;
  player: Player;
  competitor: DurableSelfPlayCompetitor;
  opponentId: string;
  opponentMaxActions: number;
  explorationSeed: number;
  recentFens: string[];
  previousOwnTurnMoves: string[];
  previousOwnTurns: string[][];
  turnsWithoutProgress: number;
}

interface DurableTurnStepResult {
  decision: DurableSelfPlayDecision;
  serializedState: string;
  outcome?: { winner?: Player; termination: string };
}

export interface DurableSelfPlayGameResult {
  generationId: string;
  gameId: string;
  seed: number;
  redAgentId: string;
  blueAgentId: string;
  redMaxActions: number;
  blueMaxActions: number;
  redValueModel: ValueModelVersion;
  blueValueModel: ValueModelVersion;
  redValueModelCheckpoint: string;
  blueValueModelCheckpoint: string;
  codeVersion: string;
  initialFen: string;
  finalFen: string;
  decisions: DurableSelfPlayDecision[];
  outcome: { winner?: Player; termination: string };
  completed: boolean;
  trainingPositions: number;
  quality: DurableSelfPlayQuality;
  storage: PersistedSelfPlayArtifacts;
}

export interface DurableSelfPlayQuality {
  decisions: number;
  eligibleDecisions: number;
  completedSearches: number;
  fallbackDecisions: number;
  /** Safe fallbacks that still completed a full opponent reply. */
  verifiedFallbackDecisions: number;
  /** Seeded or shallow fallbacks without a complete opponent reply. */
  unverifiedFallbackDecisions: number;
  timedOutDecisions: number;
  decisive: boolean;
  trainingEligible: boolean;
  trainingRejectionReasons: string[];
}

interface DurableTrainingSample {
  generationId: string;
  gameId: string;
  turnNumber: number;
  player: Player;
  agentId: string;
  opponentId: string;
  personality: PersonalityId;
  fen: string;
  features: number[];
  outcomeValue: number;
  winner: Player;
  selectedMoves: string[];
  completedDepth: number;
  valueModel: ValueModelVersion;
  valueModelCheckpoint: string;
  codeVersion: string;
}

async function playDurableTurn(
  input: DurableTurnStepInput
): Promise<DurableTurnStepResult> {
  "use step";

  const analysis = await analyzeFen({
    fen: input.fen,
    serializedState: input.serializedState,
    turnNumber: input.turnNumber,
    personality: input.competitor.personality,
    timeMs: input.competitor.timeMs,
    maxDepth: input.competitor.maxDepth,
    beamWidth: input.competitor.beamWidth,
    maxActions: input.competitor.maxActions ?? 3,
    valueModel: input.competitor.valueModel ?? "incumbent",
    explorationTemperature: input.competitor.explorationTemperature,
    explorationSeed: input.explorationSeed,
    recentFens: input.recentFens,
    previousOwnTurnMoves: input.previousOwnTurnMoves,
    previousOwnTurns: input.previousOwnTurns,
    turnsWithoutProgress: input.turnsWithoutProgress,
  });
  const state = FENtoBoardState(analysis.fen);
  const resultingState = FENtoBoardState(analysis.resultingFen);
  const decision: DurableSelfPlayDecision = {
    turnNumber: input.turnNumber,
    player: input.player,
    fen: analysis.fen,
    resultingFen: analysis.resultingFen,
    personality: input.competitor.personality,
    agentId: input.competitor.id,
    opponentId: input.opponentId,
    selectedMoves: [...analysis.search.best_turn.all_moves],
    selectedRank: analysis.search.exploration?.selectedRank ?? 1,
    candidateTurns: analysis.search.candidate_turns ?? [],
    currentPlayerScore: analysis.search.score.current_player,
    winProbability:
      input.player === "RED"
        ? analysis.model.before.redWinProbability
        : analysis.model.before.blueWinProbability,
    completedDepth: analysis.search.search.completed_depth_in_turns,
    persistentCacheHit: analysis.search.search.persistent_cache_hit === true,
    timedOut: analysis.search.search.timed_out,
    fallback: analysis.search.search.fallback_used,
    recommendationLabel: analysis.search.recommendation_label,
    explorationSeed: input.explorationSeed,
    explorationTemperature: input.competitor.explorationTemperature,
    features: extractValueFeatures(
      {
        board: state.board,
        redReserve: state.redReserve,
        blueReserve: state.blueReserve,
        currentPlayer: state.currentPlayerTurn ?? input.player,
        turnNumber: input.turnNumber,
      },
      input.player
    ),
    completedTurn: Boolean(
      analysis.outcome || resultingState.currentPlayerTurn !== input.player
    ),
    selfActionLimit: input.competitor.maxActions ?? 3,
    opponentActionLimit: input.opponentMaxActions,
    valueModel: input.competitor.valueModel ?? "incumbent",
  };
  return {
    decision,
    serializedState: analysis.serializedState,
    outcome: analysis.outcome,
  };
}

export function actionMadeProgress(uci: string): boolean {
  return (
    uci.startsWith("r") ||
    (uci.startsWith("s") && uci !== "skip") ||
    uci.includes("x")
  );
}

function turnSeed(seed: number, turnNumber: number): number {
  return (seed + Math.imul(turnNumber, 0x9e3779b1)) >>> 0;
}

export function isDurableTrainingDecisionEligible(
  decision: DurableSelfPlayDecision,
  outcome: DurableSelfPlayGameResult["outcome"]
): outcome is { winner: Player; termination: string } {
  return Boolean(
    outcome.winner &&
      outcome.termination === "hq-capture" &&
      decision.selectedMoves.length > 0 &&
      decision.completedTurn &&
      (decision.selfActionLimit ?? 3) === 3 &&
      (decision.opponentActionLimit ?? 3) === 3 &&
      decision.fallback !== "seeded" &&
      // Depth one evaluates only our resulting position. Depth two includes a
      // complete opponent reply and is the minimum tactically verified label
      // admitted to the value-model dataset.
      decision.completedDepth >= 2
  );
}

export function durableGameTrainingRejectionReasons(
  decisions: DurableSelfPlayDecision[],
  outcome: DurableSelfPlayGameResult["outcome"]
): string[] {
  const reasons: string[] = [];
  if (!outcome.winner || outcome.termination !== "hq-capture") {
    reasons.push("not-hq-capture");
  }
  if (!decisions.length) reasons.push("no-decisions");
  if (decisions.some((decision) => !decision.completedTurn)) {
    reasons.push("incomplete-turn");
  }
  if (
    decisions.some(
      (decision) =>
        (decision.selfActionLimit ?? 3) !== 3 ||
        (decision.opponentActionLimit ?? 3) !== 3
    )
  ) {
    reasons.push("nonstandard-action-limit");
  }
  if (decisions.some((decision) => decision.fallback === "seeded")) {
    reasons.push("unverified-complete-turn-seed");
  }
  const unverifiedFallbackDecisions = decisions.filter(
    (decision) =>
      decision.fallback === "seeded" ||
      (decision.fallback !== "none" && decision.completedDepth < 2)
  ).length;
  if (
    decisions.length &&
    unverifiedFallbackDecisions / decisions.length > 0.05
  ) {
    reasons.push("excessive-unverified-fallback-rate");
  }
  return reasons;
}

async function persistDurableGame(input: {
  result: Omit<DurableSelfPlayGameResult, "storage">;
  trainingSamples: DurableTrainingSample[];
}): Promise<PersistedSelfPlayArtifacts> {
  "use step";

  return persistSelfPlayArtifacts({
    generationId: input.result.generationId,
    gameId: input.result.gameId,
    game: input.result,
    trainingSamples: input.trainingSamples,
  });
}

export async function playDurableSelfPlayGame(
  config: DurableSelfPlayGameConfig
): Promise<DurableSelfPlayGameResult> {
  "use workflow";

  const initialFen = config.initialFen ?? GHQ_STARTING_FEN;
  const maxTurns = config.maxTurns ?? 160;
  const repetitionLimit = config.repetitionLimit ?? 3;
  const noProgressTurns = config.noProgressTurns ?? 24;
  const decisions: DurableSelfPlayDecision[] = [];
  const positionOccurrences: Record<string, number> = { [initialFen]: 1 };
  const positionHistory: string[] = [initialFen];
  const lastTurnMoves: Record<Player, string[]> = { RED: [], BLUE: [] };
  const ownTurnHistory: Record<Player, string[][]> = { RED: [], BLUE: [] };
  let fen: string | undefined = initialFen;
  let serializedState: string | undefined;
  let player: Player = "RED";
  let turnsWithoutProgress = 0;
  const strategicBest: Record<Player, StrategicProgress> = {
    RED: strategicProgress(initialFen, "RED"),
    BLUE: strategicProgress(initialFen, "BLUE"),
  };
  let partialTurnAttempts = 0;
  let pendingTurnMoves: string[] = [];
  let outcome: DurableSelfPlayGameResult["outcome"] | undefined;

  let turnNumber = 1;
  while (turnNumber <= maxTurns && !outcome) {
    const competitor = player === "RED" ? config.red : config.blue;
    const opponent = player === "RED" ? config.blue : config.red;
    const step = await playDurableTurn({
      fen,
      serializedState,
      turnNumber,
      player,
      competitor,
      opponentId: opponent.id,
      opponentMaxActions: opponent.maxActions ?? 3,
      explorationSeed: turnSeed(config.seed, turnNumber),
      recentFens: positionHistory.slice(-32),
      previousOwnTurnMoves: lastTurnMoves[player],
      previousOwnTurns: ownTurnHistory[player].slice(-4),
      turnsWithoutProgress,
    });
    decisions.push(step.decision);
    fen = step.decision.resultingFen;
    serializedState = step.serializedState;
    outcome = step.outcome;
    pendingTurnMoves.push(...step.decision.selectedMoves);

    const resultingPlayer = FENtoBoardState(fen).currentPlayerTurn;
    const nextPlayer: Player =
      resultingPlayer ?? (player === "RED" ? "BLUE" : "RED");
    if (!outcome && !step.decision.completedTurn) {
      partialTurnAttempts++;
      player = nextPlayer;
      if (partialTurnAttempts >= 3) {
        outcome = { termination: "incomplete-turn" };
      }
      continue;
    }

    const currentStrategicProgress = strategicProgress(fen, player);
    const madeStrategicProgress = extendsStrategicBest(
      strategicBest[player],
      currentStrategicProgress
    );
    strategicBest[player] = mergeStrategicBest(
      strategicBest[player],
      currentStrategicProgress
    );
    const madeProgress =
      pendingTurnMoves.some(actionMadeProgress) || madeStrategicProgress;
    turnsWithoutProgress = madeProgress ? 0 : turnsWithoutProgress + 1;
    const occurrences = (positionOccurrences[fen] ?? 0) + 1;
    positionOccurrences[fen] = occurrences;
    positionHistory.push(fen);
    lastTurnMoves[player] = [...pendingTurnMoves];
    ownTurnHistory[player].push([...pendingTurnMoves]);
    if (!outcome && occurrences >= repetitionLimit) {
      outcome = { termination: "repetition" };
    } else if (!outcome && turnsWithoutProgress >= noProgressTurns) {
      outcome = { termination: "no-progress" };
    }
    player = nextPlayer;
    pendingTurnMoves = [];
    partialTurnAttempts = 0;
    turnNumber++;
  }

  if (!outcome) outcome = { termination: "max-turns" };
  const trainingRejectionReasons = durableGameTrainingRejectionReasons(
    decisions,
    outcome
  );
  const trainingEligible = trainingRejectionReasons.length === 0;
  const eligibleDecisions = trainingEligible
    ? decisions.filter((decision) =>
        isDurableTrainingDecisionEligible(decision, outcome)
      )
    : [];
  const trainingSamples: DurableTrainingSample[] = outcome.winner
    ? eligibleDecisions.map((decision) => ({
        generationId: config.generationId,
        gameId: config.gameId,
        turnNumber: decision.turnNumber,
        player: decision.player,
        agentId: decision.agentId,
        opponentId: decision.opponentId,
        personality: decision.personality,
        fen: decision.fen,
        features: decision.features,
        outcomeValue: decision.player === outcome.winner ? 1 : 0,
        winner: outcome.winner!,
        selectedMoves: decision.selectedMoves,
        completedDepth: decision.completedDepth,
        valueModel: decision.valueModel ?? "incumbent",
        valueModelCheckpoint:
          decision.player === "RED"
            ? config.red.valueModelCheckpoint ?? "unknown"
            : config.blue.valueModelCheckpoint ?? "unknown",
        codeVersion: config.codeVersion ?? "unknown",
      }))
    : [];
  const result: Omit<DurableSelfPlayGameResult, "storage"> = {
    generationId: config.generationId,
    gameId: config.gameId,
    seed: config.seed >>> 0,
    redAgentId: config.red.id,
    blueAgentId: config.blue.id,
    redMaxActions: config.red.maxActions ?? 3,
    blueMaxActions: config.blue.maxActions ?? 3,
    redValueModel: config.red.valueModel ?? "incumbent",
    blueValueModel: config.blue.valueModel ?? "incumbent",
    redValueModelCheckpoint: config.red.valueModelCheckpoint ?? "unknown",
    blueValueModelCheckpoint: config.blue.valueModelCheckpoint ?? "unknown",
    codeVersion: config.codeVersion ?? "unknown",
    initialFen,
    finalFen: fen ?? initialFen,
    decisions,
    outcome,
    completed: outcome.termination !== "max-turns",
    trainingPositions: trainingSamples.length,
    quality: {
      decisions: decisions.length,
      eligibleDecisions: trainingSamples.length,
      completedSearches: decisions.filter(
        (decision) =>
          decision.completedDepth >= 1 && decision.fallback === "none"
      ).length,
      fallbackDecisions: decisions.filter(
        (decision) => decision.fallback !== "none"
      ).length,
      verifiedFallbackDecisions: decisions.filter(
        (decision) =>
          decision.fallback === "safe" && decision.completedDepth >= 2
      ).length,
      unverifiedFallbackDecisions: decisions.filter(
        (decision) =>
          decision.fallback === "seeded" ||
          (decision.fallback !== "none" && decision.completedDepth < 2)
      ).length,
      timedOutDecisions: decisions.filter((decision) => decision.timedOut)
        .length,
      decisive: outcome.winner !== undefined,
      trainingEligible,
      trainingRejectionReasons,
    },
  };
  const storage = await persistDurableGame({ result, trainingSamples });
  return { ...result, storage };
}
