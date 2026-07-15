import { GHQ_STARTING_FEN } from "@/game/analysis/types";
import type { GhqCandidateTurn, PersonalityId } from "@/game/analysis/types";
import type { Player } from "@/game/engine-v2";
import { FENtoBoardState } from "@/game/notation";
import { extractValueFeatures } from "@/game/value-model/features";
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
  timedOut: boolean;
  fallback: "none" | "safe" | "greedy";
  explorationSeed: number;
  explorationTemperature: number;
  features: number[];
}

interface DurableTurnStepInput {
  fen?: string;
  serializedState?: string;
  turnNumber: number;
  player: Player;
  competitor: DurableSelfPlayCompetitor;
  opponentId: string;
  explorationSeed: number;
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
  timedOutDecisions: number;
  decisive: boolean;
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
    explorationTemperature: input.competitor.explorationTemperature,
    explorationSeed: input.explorationSeed,
  });
  const state = FENtoBoardState(analysis.fen);
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
    timedOut: analysis.search.search.timed_out,
    fallback: analysis.search.search.fallback_used,
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
  };
  return {
    decision,
    serializedState: analysis.serializedState,
    outcome: analysis.outcome,
  };
}

function actionMadeProgress(uci: string): boolean {
  return uci.startsWith("r") || uci.startsWith("s") || uci.includes("x");
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
      decision.fallback === "none" &&
      decision.completedDepth >= 1
  );
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
  let fen: string | undefined = initialFen;
  let serializedState: string | undefined;
  let player: Player = "RED";
  let turnsWithoutProgress = 0;
  let outcome: DurableSelfPlayGameResult["outcome"] | undefined;

  for (let turnNumber = 1; turnNumber <= maxTurns && !outcome; turnNumber++) {
    const competitor = player === "RED" ? config.red : config.blue;
    const opponent = player === "RED" ? config.blue : config.red;
    const step = await playDurableTurn({
      fen,
      serializedState,
      turnNumber,
      player,
      competitor,
      opponentId: opponent.id,
      explorationSeed: turnSeed(config.seed, turnNumber),
    });
    decisions.push(step.decision);
    fen = step.decision.resultingFen;
    serializedState = step.serializedState;
    outcome = step.outcome;

    const madeProgress = step.decision.selectedMoves.some(actionMadeProgress);
    turnsWithoutProgress = madeProgress ? 0 : turnsWithoutProgress + 1;
    const occurrences = (positionOccurrences[fen] ?? 0) + 1;
    positionOccurrences[fen] = occurrences;
    if (!outcome && occurrences >= repetitionLimit) {
      outcome = { termination: "repetition" };
    } else if (!outcome && turnsWithoutProgress >= noProgressTurns) {
      outcome = { termination: "no-progress" };
    }
    player = player === "RED" ? "BLUE" : "RED";
  }

  if (!outcome) outcome = { termination: "max-turns" };
  const eligibleDecisions = decisions.filter((decision) =>
    isDurableTrainingDecisionEligible(decision, outcome)
  );
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
      }))
    : [];
  const result: Omit<DurableSelfPlayGameResult, "storage"> = {
    generationId: config.generationId,
    gameId: config.gameId,
    seed: config.seed >>> 0,
    redAgentId: config.red.id,
    blueAgentId: config.blue.id,
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
      timedOutDecisions: decisions.filter((decision) => decision.timedOut)
        .length,
      decisive: outcome.winner !== undefined,
    },
  };
  const storage = await persistDurableGame({ result, trainingSamples });
  return { ...result, storage };
}
