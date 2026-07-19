import { GHQ_STARTING_FEN } from "@/game/analysis/types";
import type {
  GhqCandidateTurn,
  GhqTurnPurpose,
  PersonalityId,
} from "@/game/analysis/types";
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
import { auditParatrooperTrainingPolicy } from "@/game/self-play/training-policy";
import { analyzeFen } from "@/server/fen-analysis";
import { sleep } from "workflow";
import {
  persistSelfPlayArtifacts,
  persistSelfPlayProgress,
  type PersistedSelfPlayArtifacts,
  type SelfPlayProgressSnapshot,
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
  /** Absolute turn number represented by `initialFen`; defaults to turn one. */
  initialTurnNumber?: number;
  /** Counterfactual rollouts are persisted, but never enter terminal training. */
  dataRole?: "standard" | "counterfactual";
  maxTurns?: number;
  repetitionLimit?: number;
  noProgressTurns?: number;
  codeVersion?: string;
  /** Absolute slots that cap concurrent CPU-heavy searches within a batch. */
  searchSchedule?: {
    epochMs: number;
    lane: number;
    laneCount: number;
    slotMs: number;
  };
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
  /** Exact purpose labels for the selected turn, including personality reranks. */
  selectedActionPurposes?: Array<{ move: string; roles: string[] }>;
  /** Exact aggregate purpose telemetry for the selected turn. */
  selectedPurpose?: GhqTurnPurpose;
  /** Number of clean two-action alternatives synthesized during this search. */
  purposefulEarlyStopsGenerated?: number;
  selectedRank: number;
  candidateTurns: GhqCandidateTurn[];
  currentPlayerScore: number;
  winProbability: number;
  completedDepth: number;
  /** Search-shape telemetry for diagnosing budget and branching failures. */
  searchTelemetry?: {
    nodes: number;
    elapsedMs: number;
    ruleFilteredActions: number;
    beamPrunedActions: number;
    partialTurnsPruned: number;
    completeTurnsGenerated: number;
    completeTurnsDeduplicated: number;
    completeTurnsPruned: number;
    tacticallyUnsafeTurns: number;
    rotationQuotaPruned: number;
    purposeFilteredTurns: number;
    valueModelEvaluations: number;
    turnCacheHits: number;
    transpositionHits: number;
    hqSurvivalProbeNodes?: number;
    hqSurvivalReplyNodes?: number;
    hqSurvivalOverrideUsed?: boolean;
    hqSurvivalReplyVerified?: boolean;
    hqExactReturnProbeUsed?: boolean;
    tacticalReturnGuardUsed?: boolean;
    safeFallbackReplyVerified?: boolean;
    safeFallbackReplyNodes?: number;
    policyReturnGuardUsed?: boolean;
    seedReplyVerified?: boolean;
    seedReplyRetryUsed?: boolean;
    seedSafetyRetryUsed?: boolean;
    seedSafetyRetryVerified?: boolean;
  };
  persistentCacheHit?: boolean;
  timedOut: boolean;
  fallback: "none" | "safe" | "seeded";
  /** Missing on historical records created before recommendation telemetry. */
  recommendationLabel?: string;
  /** Exact runtime used for the behavior search. */
  searchBackend?: "pyodide" | "native-python";
  /** Exact value-inference bridge used inside that search. */
  searchValueModelBackend?: "typescript-callback" | "native-gbdt";
  /** Exact deployed source revision actually executed by the search backend. */
  searchCodeVersion?: string;
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
  initialTurnNumber: number;
  dataRole: "standard" | "counterfactual";
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

export interface DurableTrainingSample {
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
  fallback: DurableSelfPlayDecision["fallback"];
  timedOut: boolean;
  valueModel: ValueModelVersion;
  valueModelCheckpoint: string;
  codeVersion: string;
  searchBackend: NonNullable<DurableSelfPlayDecision["searchBackend"]>;
  searchValueModelBackend: NonNullable<
    DurableSelfPlayDecision["searchValueModelBackend"]
  >;
  searchCodeVersion: string;
}

export function durableTrainingSample(
  decision: DurableSelfPlayDecision,
  config: Pick<
    DurableSelfPlayGameConfig,
    "generationId" | "gameId" | "red" | "blue"
  >,
  winner: Player
): DurableTrainingSample {
  return {
    generationId: config.generationId,
    gameId: config.gameId,
    turnNumber: decision.turnNumber,
    player: decision.player,
    agentId: decision.agentId,
    opponentId: decision.opponentId,
    personality: decision.personality,
    fen: decision.fen,
    features: decision.features,
    outcomeValue: decision.player === winner ? 1 : 0,
    winner,
    selectedMoves: decision.selectedMoves,
    completedDepth: decision.completedDepth,
    fallback: decision.fallback,
    timedOut: decision.timedOut,
    valueModel: decision.valueModel ?? "incumbent",
    valueModelCheckpoint:
      decision.player === "RED"
        ? config.red.valueModelCheckpoint ?? "unknown"
        : config.blue.valueModelCheckpoint ?? "unknown",
    codeVersion: decision.searchCodeVersion!,
    searchBackend: decision.searchBackend!,
    searchValueModelBackend: decision.searchValueModelBackend!,
    searchCodeVersion: decision.searchCodeVersion!,
  };
}

async function playDurableTurn(
  input: DurableTurnStepInput
): Promise<DurableTurnStepResult> {
  "use step";

  let analysis: Awaited<ReturnType<typeof analyzeFen>>;
  try {
    analysis = await analyzeFen({
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
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(
      `Self-play ${input.player} turn ${input.turnNumber} failed from ${
        input.fen ?? "serialized state"
      }: ${message}`
    );
  }
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
    selectedActionPurposes: analysis.search.best_turn.action_purposes,
    selectedPurpose: analysis.search.best_turn.purpose,
    purposefulEarlyStopsGenerated:
      analysis.search.search.purposeful_early_stops_generated,
    selectedRank: analysis.search.exploration?.selectedRank ?? 1,
    candidateTurns: analysis.search.candidate_turns ?? [],
    currentPlayerScore: analysis.search.score.current_player,
    winProbability:
      input.player === "RED"
        ? analysis.model.before.redWinProbability
        : analysis.model.before.blueWinProbability,
    completedDepth: analysis.search.search.completed_depth_in_turns,
    searchTelemetry: {
      nodes: analysis.search.search.nodes,
      elapsedMs: analysis.search.search.elapsed_ms,
      ruleFilteredActions: analysis.search.search.rule_filtered_actions,
      beamPrunedActions: analysis.search.search.beam_pruned_actions,
      partialTurnsPruned: analysis.search.search.partial_turns_pruned,
      completeTurnsGenerated: analysis.search.search.complete_turns_generated,
      completeTurnsDeduplicated:
        analysis.search.search.complete_turns_deduplicated,
      completeTurnsPruned: analysis.search.search.complete_turns_pruned,
      tacticallyUnsafeTurns: analysis.search.search.tactically_unsafe_turns,
      rotationQuotaPruned: analysis.search.search.rotation_quota_pruned,
      purposeFilteredTurns: analysis.search.search.purpose_filtered_turns,
      valueModelEvaluations: analysis.search.search.value_model_evaluations,
      turnCacheHits: analysis.search.search.turn_cache_hits,
      transpositionHits: analysis.search.search.transposition_hits,
      hqSurvivalProbeNodes: analysis.search.search.hq_survival_probe_nodes ?? 0,
      hqSurvivalReplyNodes: analysis.search.search.hq_survival_reply_nodes ?? 0,
      hqSurvivalOverrideUsed:
        analysis.search.search.hq_survival_override_used === true,
      hqSurvivalReplyVerified:
        analysis.search.search.hq_survival_reply_verified === true,
      hqExactReturnProbeUsed:
        analysis.search.search.hq_exact_return_probe_used === true,
      tacticalReturnGuardUsed:
        analysis.search.search.tactical_return_guard_used === true,
      safeFallbackReplyVerified:
        analysis.search.search.safe_fallback_reply_verified === true,
      safeFallbackReplyNodes:
        analysis.search.search.safe_fallback_reply_nodes ?? 0,
      policyReturnGuardUsed:
        analysis.search.search.policy_return_guard_used === true,
      seedReplyVerified: analysis.search.search.seed_reply_verified === true,
      seedReplyRetryUsed: analysis.search.search.seed_reply_retry_used === true,
      seedSafetyRetryUsed:
        analysis.search.search.seed_safety_retry_used === true,
      seedSafetyRetryVerified:
        analysis.search.search.seed_safety_retry_verified === true,
    },
    persistentCacheHit: analysis.search.search.persistent_cache_hit === true,
    timedOut: analysis.search.search.timed_out,
    fallback: analysis.search.search.fallback_used,
    recommendationLabel: analysis.search.recommendation_label,
    searchBackend: analysis.search.search.backend,
    searchValueModelBackend: analysis.search.search.value_model_backend,
    searchCodeVersion: analysis.search.search.code_version,
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

export function durableSearchSlotAt(
  schedule: DurableSelfPlayGameConfig["searchSchedule"],
  turnNumber: number,
  initialTurnNumber = 1
): Date | undefined {
  if (!schedule) return undefined;
  const { epochMs, lane, laneCount, slotMs } = schedule;
  if (
    !Number.isSafeInteger(epochMs) ||
    epochMs < 0 ||
    !Number.isSafeInteger(lane) ||
    lane < 0 ||
    !Number.isSafeInteger(laneCount) ||
    laneCount < 1 ||
    lane >= laneCount ||
    !Number.isSafeInteger(slotMs) ||
    slotMs < 1 ||
    !Number.isSafeInteger(turnNumber) ||
    turnNumber < initialTurnNumber
  ) {
    throw new RangeError("Invalid durable self-play search schedule");
  }
  const turnOffset = turnNumber - initialTurnNumber;
  return new Date(epochMs + (lane + turnOffset * laneCount) * slotMs);
}

export function isDurableTrainingDecisionEligible(
  decision: DurableSelfPlayDecision,
  outcome: DurableSelfPlayGameResult["outcome"]
): outcome is { winner: Player; termination: string } {
  const paratrooperPolicy = auditParatrooperTrainingPolicy([decision]);
  return Boolean(
    outcome.winner &&
      outcome.termination === "hq-capture" &&
      decision.selectedMoves.length > 0 &&
      decision.completedTurn &&
      (decision.selfActionLimit ?? 3) === 3 &&
      (decision.opponentActionLimit ?? 3) === 3 &&
      decision.fallback !== "seeded" &&
      decision.searchBackend !== undefined &&
      decision.searchValueModelBackend !== undefined &&
      decision.searchCodeVersion !== undefined &&
      paratrooperPolicy.eligible &&
      // Depth one evaluates only our resulting position. Depth two includes a
      // complete opponent reply and is the minimum tactically verified label
      // admitted to the value-model dataset.
      decision.completedDepth >= 2
  );
}

export function durableGameTrainingRejectionReasons(
  decisions: DurableSelfPlayDecision[],
  outcome: DurableSelfPlayGameResult["outcome"],
  expectedCodeVersion?: string
): string[] {
  const reasons: string[] = [];
  const paratrooperPolicy = auditParatrooperTrainingPolicy(decisions);
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
  if (!paratrooperPolicy.telemetryComplete) {
    reasons.push("missing-paratrooper-policy-telemetry");
  }
  if (paratrooperPolicy.violatingDecisions > 0) {
    reasons.push("paratrooper-policy-violation");
  }
  if (
    decisions.some(
      (decision) =>
        decision.searchBackend === undefined ||
        decision.searchValueModelBackend === undefined ||
        decision.searchCodeVersion === undefined
    )
  ) {
    reasons.push("missing-search-runtime-provenance");
  } else {
    const runtimePairs = new Set(
      decisions.map(
        (decision) =>
          `${decision.searchBackend}:${decision.searchValueModelBackend}`
      )
    );
    if (runtimePairs.size > 1) reasons.push("mixed-search-runtime-provenance");
  }
  const searchCodeVersions = new Set(
    decisions
      .map((decision) => decision.searchCodeVersion)
      .filter((value): value is string => value !== undefined)
  );
  if (searchCodeVersions.size > 1) reasons.push("mixed-search-code-version");
  if (
    expectedCodeVersion !== undefined &&
    decisions.some(
      (decision) => decision.searchCodeVersion !== expectedCodeVersion
    )
  ) {
    reasons.push("mismatched-search-code-version");
  }
  const unverifiedFallbackDecisions = decisions.filter(
    (decision) =>
      decision.fallback === "seeded" ||
      (decision.fallback !== "none" && decision.completedDepth < 2)
  ).length;
  // One blind turn changes the position, the eventual result, and therefore
  // every later value label in the game.  A low aggregate rate cannot make
  // those downstream labels clean.  Reply-verified safe fallbacks remain
  // eligible because they have completed the same tactical floor as an
  // ordinary depth-two decision.
  if (unverifiedFallbackDecisions > 0) {
    reasons.push("unverified-fallback-decision");
  }
  return reasons;
}

export function resolveDurableInitialState(
  config: Pick<
    DurableSelfPlayGameConfig,
    "initialFen" | "initialTurnNumber" | "dataRole"
  >
): {
  initialFen: string;
  initialTurnNumber: number;
  initialPlayer: Player;
  dataRole: "standard" | "counterfactual";
} {
  const initialFen = config.initialFen ?? GHQ_STARTING_FEN;
  const initialTurnNumber = config.initialTurnNumber ?? 1;
  if (!Number.isSafeInteger(initialTurnNumber) || initialTurnNumber < 1) {
    throw new RangeError("initialTurnNumber must be a positive integer");
  }
  return {
    initialFen,
    initialTurnNumber,
    initialPlayer: FENtoBoardState(initialFen).currentPlayerTurn ?? "RED",
    dataRole: config.dataRole ?? "standard",
  };
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

export function durableSelfPlayProgressSnapshot(input: {
  config: DurableSelfPlayGameConfig;
  decisions: DurableSelfPlayDecision[];
  completedTurns: number;
  currentPlayer: Player;
  currentFen: string;
  status: SelfPlayProgressSnapshot["status"];
  outcome?: DurableSelfPlayGameResult["outcome"];
}): SelfPlayProgressSnapshot {
  const latestUnverifiedFallback = [...input.decisions]
    .reverse()
    .find(
      (decision) =>
        decision.fallback === "seeded" ||
        (decision.fallback !== "none" && decision.completedDepth < 2)
    );
  return {
    format: "ghq-self-play-progress-v1",
    generationId: input.config.generationId,
    gameId: input.config.gameId,
    seed: input.config.seed >>> 0,
    codeVersion: input.config.codeVersion ?? "unknown",
    redAgentId: input.config.red.id,
    blueAgentId: input.config.blue.id,
    redValueModelCheckpoint: input.config.red.valueModelCheckpoint ?? "unknown",
    blueValueModelCheckpoint:
      input.config.blue.valueModelCheckpoint ?? "unknown",
    completedTurns: input.completedTurns,
    currentPlayer: input.currentPlayer,
    currentFen: input.currentFen,
    decisions: input.decisions.length,
    depthAtLeastTwoDecisions: input.decisions.filter(
      (decision) => decision.completedDepth >= 2
    ).length,
    fallbackDecisions: input.decisions.filter(
      (decision) => decision.fallback !== "none"
    ).length,
    unverifiedFallbackDecisions: input.decisions.filter(
      (decision) =>
        decision.fallback === "seeded" ||
        (decision.fallback !== "none" && decision.completedDepth < 2)
    ).length,
    latestUnverifiedFallback: latestUnverifiedFallback
      ? {
          turnNumber: latestUnverifiedFallback.turnNumber,
          player: latestUnverifiedFallback.player,
          fen: latestUnverifiedFallback.fen,
          selectedMoves: [...latestUnverifiedFallback.selectedMoves],
          completedDepth: latestUnverifiedFallback.completedDepth,
          fallback: latestUnverifiedFallback.fallback as "safe" | "seeded",
          timedOut: latestUnverifiedFallback.timedOut,
          seedReplyVerified:
            latestUnverifiedFallback.searchTelemetry?.seedReplyVerified ===
            true,
          seedSafetyRetryUsed:
            latestUnverifiedFallback.searchTelemetry?.seedSafetyRetryUsed ===
            true,
          seedSafetyRetryVerified:
            latestUnverifiedFallback.searchTelemetry
              ?.seedSafetyRetryVerified === true,
          safeFallbackReplyVerified:
            latestUnverifiedFallback.searchTelemetry
              ?.safeFallbackReplyVerified === true,
          tacticalReturnGuardUsed:
            latestUnverifiedFallback.searchTelemetry
              ?.tacticalReturnGuardUsed === true,
        }
      : undefined,
    timedOutDecisions: input.decisions.filter((decision) => decision.timedOut)
      .length,
    status: input.status,
    outcome: input.outcome,
  };
}

async function persistDurableProgress(
  snapshot: SelfPlayProgressSnapshot
): Promise<"saved" | "not-configured" | "failed"> {
  "use step";

  try {
    return await persistSelfPlayProgress(snapshot);
  } catch (error) {
    console.error("Unable to persist self-play progress", error);
    return "failed";
  }
}

export async function playDurableSelfPlayGame(
  config: DurableSelfPlayGameConfig
): Promise<DurableSelfPlayGameResult> {
  "use workflow";

  const { initialFen, initialTurnNumber, initialPlayer, dataRole } =
    resolveDurableInitialState(config);
  const maxTurns = config.maxTurns ?? 160;
  if (!Number.isSafeInteger(maxTurns) || maxTurns < initialTurnNumber) {
    throw new RangeError(
      "maxTurns must be an integer at least as large as initialTurnNumber"
    );
  }
  const repetitionLimit = config.repetitionLimit ?? 3;
  const noProgressTurns = config.noProgressTurns ?? 36;
  const decisions: DurableSelfPlayDecision[] = [];
  const positionOccurrences: Record<string, number> = { [initialFen]: 1 };
  const positionHistory: string[] = [initialFen];
  const lastTurnMoves: Record<Player, string[]> = { RED: [], BLUE: [] };
  const ownTurnHistory: Record<Player, string[][]> = { RED: [], BLUE: [] };
  let fen: string | undefined = initialFen;
  let serializedState: string | undefined;
  let player: Player = initialPlayer;
  let turnsWithoutProgress = 0;
  const strategicBest: Record<Player, StrategicProgress> = {
    RED: strategicProgress(initialFen, "RED"),
    BLUE: strategicProgress(initialFen, "BLUE"),
  };
  let partialTurnAttempts = 0;
  let pendingTurnMoves: string[] = [];
  let outcome: DurableSelfPlayGameResult["outcome"] | undefined;

  let turnNumber = initialTurnNumber;
  while (turnNumber <= maxTurns && !outcome) {
    const competitor = player === "RED" ? config.red : config.blue;
    const opponent = player === "RED" ? config.blue : config.red;
    const searchSlot = durableSearchSlotAt(
      config.searchSchedule,
      turnNumber,
      initialTurnNumber
    );
    if (searchSlot) await sleep(searchSlot);
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
    if (!outcome && (turnNumber - 1) % 10 === 0) {
      await persistDurableProgress(
        durableSelfPlayProgressSnapshot({
          config,
          decisions,
          completedTurns: turnNumber - 1,
          currentPlayer: player,
          currentFen: fen,
          status: "running",
        })
      );
    }
  }

  if (!outcome) outcome = { termination: "max-turns" };
  const trainingRejectionReasons = [
    ...(dataRole === "counterfactual" ? ["counterfactual-rollout"] : []),
    ...durableGameTrainingRejectionReasons(
      decisions,
      outcome,
      config.codeVersion
    ),
  ];
  const trainingEligible = trainingRejectionReasons.length === 0;
  const eligibleDecisions = trainingEligible
    ? decisions.filter((decision) =>
        isDurableTrainingDecisionEligible(decision, outcome)
      )
    : [];
  const trainingSamples: DurableTrainingSample[] = outcome.winner
    ? eligibleDecisions.map((decision) =>
        durableTrainingSample(decision, config, outcome.winner!)
      )
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
    initialTurnNumber,
    dataRole,
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
  await persistDurableProgress(
    durableSelfPlayProgressSnapshot({
      config,
      decisions,
      completedTurns: Math.min(turnNumber - 1, maxTurns),
      currentPlayer: player,
      currentFen: fen ?? initialFen,
      status: "completed",
      outcome,
    })
  );
  const storage = await persistDurableGame({ result, trainingSamples });
  return { ...result, storage };
}
