import { readFile } from "fs/promises";
import path from "path";
import type { GameEngine, PythonBoard } from "@/game/engine-v2";
import type {
  FenAnalysisRequest,
  FenAnalysisResponse,
  GhqSearchResult,
  ModelPositionOutput,
  PersonalityId,
} from "@/game/analysis/types";
import { FENtoBoardState } from "@/game/notation";
import type { Player } from "@/game/engine";
import {
  predictZeroSumWinProbability,
  type ValueModelVersion,
} from "@/game/value-model/inference";
import { evaluatePersonalityPosition } from "@/game/value-model/styled-evaluation";
import { PERSONALITIES } from "@/game/value-model/personalities";
import { loadServerPyodide } from "@/server/pyodide";
import {
  persistSearch,
  readPersistedSearch,
  type SearchCacheKey,
} from "@/server/search-cache";

interface PythonProxy {
  destroy?: () => void;
  toJs?: (options?: unknown) => unknown;
}

interface SearchModule extends PythonProxy {
  search: (
    board: PythonBoard,
    personality: PersonalityId,
    timeMs: number,
    maxDepth: number,
    beamWidth: number,
    turnNumber: number,
    valueFunction: (fen: string, turnNumber: number) => number,
    openingSeed: number,
    maxActions: number,
    stagnationTurns: number
  ) => PythonProxy;
  evaluation_breakdown: (
    board: PythonBoard,
    personality: PersonalityId,
    turnNumber: number
  ) => PythonProxy;
}

interface LoadedAnalysisEngine {
  engine: GameEngine & PythonProxy;
  search: SearchModule;
}

let loadedEngine: Promise<LoadedAnalysisEngine> | undefined;

function destroyProxy(value: unknown): void {
  const destroy = (value as PythonProxy | undefined)?.destroy;
  if (typeof destroy === "function") destroy.call(value);
}

async function loadAnalysisEngine(): Promise<LoadedAnalysisEngine> {
  if (loadedEngine) return loadedEngine;
  loadedEngine = (async () => {
    const pyodide = await loadServerPyodide();
    const [engineCode, searchCode] = await Promise.all([
      readFile(path.join(process.cwd(), "public/engine.py"), "utf8"),
      readFile(path.join(process.cwd(), "scripts/ghq_ai.py"), "utf8"),
    ]);
    pyodide.FS.writeFile("engine.py", new TextEncoder().encode(engineCode));
    pyodide.FS.writeFile("ghq_ai.py", new TextEncoder().encode(searchCode));
    const engine = pyodide.pyimport("engine") as GameEngine & PythonProxy;
    const search = pyodide.pyimport("ghq_ai") as SearchModule;
    return { engine, search };
  })().catch((error) => {
    loadedEngine = undefined;
    throw error;
  });
  return loadedEngine;
}

function integerInRange(
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
  name: string
): number {
  const resolved = value ?? fallback;
  if (
    !Number.isSafeInteger(resolved) ||
    resolved < minimum ||
    resolved > maximum
  ) {
    throw new AnalysisInputError(
      `${name} must be an integer from ${minimum} through ${maximum}`
    );
  }
  return resolved;
}

function numberInRange(
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
  name: string
): number {
  const resolved = value ?? fallback;
  if (!Number.isFinite(resolved) || resolved < minimum || resolved > maximum) {
    throw new AnalysisInputError(
      `${name} must be a number from ${minimum} through ${maximum}`
    );
  }
  return resolved;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

export function applyExploration(
  result: GhqSearchResult,
  sideToMove: Player,
  temperature: number,
  seed: number
): GhqSearchResult {
  const candidates = result.candidate_turns ?? [];
  let selected = candidates[0];
  if (
    temperature > 0 &&
    candidates.length > 1 &&
    !result.search.opening_book_used
  ) {
    const bestScore = candidates[0].score;
    const qualityWindow = Math.max(0.35, Math.min(2.5, temperature * 3));
    const eligible = candidates.filter(
      (candidate) => candidate.score >= bestScore - qualityWindow
    );
    const scale = Math.max(0.05, temperature);
    const weights = eligible.map((candidate) =>
      Math.exp((candidate.score - bestScore) / scale)
    );
    let draw =
      seededRandom(seed)() * weights.reduce((sum, value) => sum + value, 0);
    selected = eligible[eligible.length - 1];
    for (let index = 0; index < eligible.length; index++) {
      draw -= weights[index];
      if (draw <= 0) {
        selected = eligible[index];
        break;
      }
    }
  }

  if (selected && selected.rank > 1) {
    result.best_turn = {
      automatic_captures: selected.automatic_captures,
      actions: selected.actions,
      all_moves: selected.all_moves,
      resulting_fen: selected.resulting_fen,
      action_purposes: selected.action_purposes,
      purpose: selected.purpose,
    };
    result.principal_variation = selected.all_moves;
    result.score.current_player = selected.score;
    result.score.red = sideToMove === "RED" ? selected.score : -selected.score;
    result.recommendation_label = "exploratory";
  }
  result.exploration = {
    temperature,
    seed,
    selectedRank: selected?.rank ?? 1,
    candidateCount: candidates.length || 1,
  };
  return result;
}

function ordinaryMoveEndpoints(uci: string): [string, string] | undefined {
  const match = /^([a-h][1-8])([a-h][1-8])/.exec(uci);
  return match ? [match[1], match[2]] : undefined;
}

function reversalCount(
  actions: string[],
  previousOwnTurnMoves: string[]
): number {
  const reversedPrevious = new Set(
    previousOwnTurnMoves.flatMap((move) => {
      const endpoints = ordinaryMoveEndpoints(move);
      return endpoints ? [`${endpoints[1]}${endpoints[0]}`] : [];
    })
  );
  return actions.reduce((count, move) => {
    const endpoints = ordinaryMoveEndpoints(move);
    return (
      count + (endpoints && reversedPrevious.has(endpoints.join("")) ? 1 : 0)
    );
  }, 0);
}

/** Prefer a near-best novel turn when the nominal choice repeats or undoes play. */
export function applyHistoryAvoidance(
  result: GhqSearchResult,
  sideToMove: Player,
  recentFens: string[],
  previousOwnTurns: string[][],
  turnsWithoutProgress: number
): GhqSearchResult {
  const candidates = result.candidate_turns ?? [];
  if (candidates.length < 2 || result.search.opening_book_used) return result;

  const recentCounts = new Map<string, number>();
  for (const fen of recentFens.slice(-32)) {
    recentCounts.set(fen, (recentCounts.get(fen) ?? 0) + 1);
  }
  const stagnation = Math.max(0, Math.min(1, (turnsWithoutProgress - 4) / 20));
  const priorTurns = previousOwnTurns.slice(-4);
  const penalty = (candidate: (typeof candidates)[number]): number => {
    const repeats = recentCounts.get(candidate.resulting_fen) ?? 0;
    const reversals = priorTurns.reduce(
      (total, moves, index) =>
        total +
        reversalCount(candidate.actions, moves) *
          (index === priorTurns.length - 1 ? 1 : 0.65),
      0
    );
    const skips = candidate.actions.filter((move) => move === "skip").length;
    const purpose = candidate.purpose;
    const conveyorMotion = (purpose.backfills ?? 0) + (purpose.reversals ?? 0);
    const lowPurpose = (purpose.forcing_gain ?? 0) < 0.25 ? 1 : 0;
    const emptyPurpose = (purpose.purposeful_actions ?? 0) < 1 ? 1 : 0;
    return (
      repeats * 100 +
      reversals * (1.5 + 3.5 * stagnation) +
      stagnation *
        (2.5 * conveyorMotion + 5 * skips + 5 * lowPurpose + 4 * emptyPurpose)
    );
  };

  const selectedRank = result.exploration?.selectedRank ?? 1;
  const selected =
    candidates.find((candidate) => candidate.rank === selectedRank) ??
    candidates.find(
      (candidate) => candidate.resulting_fen === result.best_turn.resulting_fen
    ) ??
    candidates[0];
  const selectedPenalty = penalty(selected);
  if (selectedPenalty <= 0) return result;

  const bestScore = candidates[0].score;
  // Early in a quiet spell, preserve normal search quality. As the draw clock
  // grows, permit a larger controlled score sacrifice to break a proven loop.
  const qualityWindow = 1.25 + 4.75 * stagnation;
  const replacement = candidates
    .filter((candidate) => candidate.score >= bestScore - qualityWindow)
    .sort(
      (left, right) =>
        right.score - penalty(right) - (left.score - penalty(left)) ||
        penalty(left) - penalty(right) ||
        right.score - left.score
    )[0];
  if (
    !replacement ||
    replacement.rank === selected.rank ||
    replacement.score - penalty(replacement) <=
      selected.score - selectedPenalty + 0.05
  ) {
    return result;
  }

  result.best_turn = {
    automatic_captures: replacement.automatic_captures,
    actions: replacement.actions,
    all_moves: replacement.all_moves,
    resulting_fen: replacement.resulting_fen,
    action_purposes: replacement.action_purposes,
    purpose: replacement.purpose,
  };
  result.principal_variation = replacement.all_moves;
  result.score.current_player = replacement.score;
  result.score.red =
    sideToMove === "RED" ? replacement.score : -replacement.score;
  result.recommendation_label = "history avoidance";
  result.exploration = {
    temperature: result.exploration?.temperature ?? 0,
    seed: result.exploration?.seed ?? 0,
    selectedRank: replacement.rank,
    candidateCount: candidates.length,
  };
  return result;
}

function playerToMove(board: PythonBoard): Player {
  return board.is_red_turn() ? "RED" : "BLUE";
}

function modelOutput(
  fen: string,
  turnNumber: number,
  perspective: Player,
  personality: PersonalityId,
  maxActions: number,
  valueModel: ValueModelVersion
): ModelPositionOutput {
  const state = FENtoBoardState(fen);
  const position = {
    board: state.board,
    redReserve: state.redReserve,
    blueReserve: state.blueReserve,
    currentPlayer: state.currentPlayerTurn ?? "RED",
    turnNumber,
  };
  return {
    redWinProbability: predictZeroSumWinProbability(
      position,
      "RED",
      maxActions === 2 ? "two-actions" : "three-actions",
      valueModel
    ),
    blueWinProbability: predictZeroSumWinProbability(
      position,
      "BLUE",
      maxActions === 2 ? "two-actions" : "three-actions",
      valueModel
    ),
    personality: evaluatePersonalityPosition(
      position,
      perspective,
      personality
    ),
  };
}

function redModelValue(
  fen: string,
  turnNumber: number,
  maxActions: number,
  valueModel: ValueModelVersion
): number {
  const state = FENtoBoardState(fen);
  return predictZeroSumWinProbability(
    {
      board: state.board,
      redReserve: state.redReserve,
      blueReserve: state.blueReserve,
      currentPlayer: state.currentPlayerTurn ?? "RED",
      turnNumber,
    },
    "RED",
    maxActions === 2 ? "two-actions" : "three-actions",
    valueModel
  );
}

function pythonOutcome(board: PythonBoard): FenAnalysisResponse["outcome"] {
  const outcome = board.outcome();
  if (!outcome) return undefined;
  try {
    return {
      winner:
        outcome.winner === false
          ? "RED"
          : outcome.winner === true
          ? "BLUE"
          : undefined,
      termination: outcome.termination,
    };
  } finally {
    destroyProxy(outcome);
  }
}

function toSearchResult(result: PythonProxy): GhqSearchResult {
  if (typeof result.toJs !== "function") {
    throw new Error("Python search did not return a convertible result");
  }
  return result.toJs({
    dict_converter: Object.fromEntries,
  }) as GhqSearchResult;
}

export class AnalysisInputError extends Error {}

export async function analyzeFen(
  request: FenAnalysisRequest
): Promise<FenAnalysisResponse> {
  if (!request.fen && !request.serializedState) {
    throw new AnalysisInputError("fen or serializedState is required");
  }
  const personality = request.personality ?? "balanced";
  if (!(personality in PERSONALITIES)) {
    throw new AnalysisInputError(`Unknown personality: ${personality}`);
  }
  const valueModel = request.valueModel ?? "incumbent";
  if (valueModel !== "incumbent" && valueModel !== "challenger") {
    throw new AnalysisInputError(`Unknown value model: ${valueModel}`);
  }
  const turnNumber = integerInRange(
    request.turnNumber,
    1,
    1,
    2_000,
    "turnNumber"
  );
  const timeMs = integerInRange(request.timeMs, 30_000, 50, 30_000, "timeMs");
  const maxDepth = integerInRange(request.maxDepth, 3, 1, 3, "maxDepth");
  const beamWidth = integerInRange(request.beamWidth, 8, 2, 16, "beamWidth");
  const maxActions = integerInRange(request.maxActions, 3, 2, 3, "maxActions");
  const explorationTemperature = numberInRange(
    request.explorationTemperature,
    0,
    0,
    2,
    "explorationTemperature"
  );
  const explorationSeed = integerInRange(
    request.explorationSeed,
    Math.floor(Math.random() * 0x1_0000_0000),
    0,
    0xffff_ffff,
    "explorationSeed"
  );
  const turnsWithoutProgress = integerInRange(
    request.turnsWithoutProgress,
    0,
    0,
    400,
    "turnsWithoutProgress"
  );
  const { engine, search } = await loadAnalysisEngine();
  let board: PythonBoard;
  try {
    board = request.serializedState
      ? engine.BaseBoard.deserialize(request.serializedState)
      : engine.BaseBoard(request.fen);
  } catch (error) {
    throw new AnalysisInputError(
      error instanceof Error ? error.message : "Invalid GHQ position"
    );
  }

  try {
    const fen = board.board_fen();
    const sideToMove = playerToMove(board);
    const searchCacheKey: SearchCacheKey = {
      serializedPosition: board.serialize(),
      personality,
      turnNumber,
      timeMs,
      maxDepth,
      beamWidth,
      maxActions,
      stagnationTurns: turnsWithoutProgress,
      valueModel,
    };
    let rawSearchResult = await readPersistedSearch(searchCacheKey);
    if (rawSearchResult) {
      rawSearchResult = structuredClone(rawSearchResult);
      rawSearchResult.search.persistent_cache_hit = true;
    } else {
      const resultProxy = search.search(
        board,
        personality,
        timeMs,
        maxDepth,
        beamWidth,
        turnNumber,
        (valueFen, valueTurnNumber) =>
          redModelValue(valueFen, valueTurnNumber, maxActions, valueModel),
        explorationSeed,
        maxActions,
        turnsWithoutProgress
      );
      try {
        rawSearchResult = toSearchResult(resultProxy);
      } finally {
        destroyProxy(resultProxy);
      }
      await persistSearch(searchCacheKey, rawSearchResult);
    }
    const searchResult = applyHistoryAvoidance(
      applyExploration(
        rawSearchResult,
        sideToMove,
        explorationTemperature,
        explorationSeed
      ),
      sideToMove,
      (request.recentFens ?? []).filter(
        (value): value is string => typeof value === "string"
      ),
      (request.previousOwnTurns ?? [request.previousOwnTurnMoves ?? []])
        .filter((turn): turn is string[] => Array.isArray(turn))
        .map((turn) =>
          turn.filter((value): value is string => typeof value === "string")
        ),
      turnsWithoutProgress
    );

    for (const uci of searchResult.best_turn.all_moves) {
      const move = engine.Move.from_uci(uci);
      try {
        if (!board.is_legal(move)) {
          throw new Error(`Search returned illegal production move: ${uci}`);
        }
        board.push(move);
      } finally {
        destroyProxy(move);
      }
    }

    const resultingFen = board.board_fen();
    const afterEvaluationProxy = search.evaluation_breakdown(
      board,
      personality,
      turnNumber + 1
    );
    try {
      if (typeof afterEvaluationProxy.toJs === "function") {
        searchResult.evaluation.after_best_turn = afterEvaluationProxy.toJs({
          dict_converter: Object.fromEntries,
        }) as GhqSearchResult["evaluation"]["after_best_turn"];
      }
    } finally {
      destroyProxy(afterEvaluationProxy);
    }
    return {
      fen,
      resultingFen,
      serializedState: board.serialize(),
      sideToMove,
      turnNumber,
      personality,
      effectiveConfig: {
        timeMs,
        maxDepth,
        beamWidth,
        explorationTemperature,
        explorationSeed,
        maxActions: maxActions as 2 | 3,
        valueModel,
      },
      outcome: pythonOutcome(board),
      model: {
        before: modelOutput(
          fen,
          turnNumber,
          sideToMove,
          personality,
          maxActions,
          valueModel
        ),
        after: modelOutput(
          resultingFen,
          turnNumber + 1,
          sideToMove,
          personality,
          maxActions,
          valueModel
        ),
      },
      search: searchResult,
    };
  } finally {
    destroyProxy(board);
  }
}
