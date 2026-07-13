import { readFile } from "fs/promises";
import path from "path";
import { loadPyodide } from "pyodide";
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
import { predictWinProbability } from "@/game/value-model/inference";
import { evaluatePersonalityPosition } from "@/game/value-model/styled-evaluation";
import { PERSONALITIES } from "@/game/value-model/personalities";

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
    beamWidth: number
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
    const pyodidePackage = path.dirname(
      require.resolve("pyodide/package.json")
    );
    const pyodide = await loadPyodide({
      indexURL: `${pyodidePackage}${path.sep}`,
    });
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

function playerToMove(board: PythonBoard): Player {
  return board.is_red_turn() ? "RED" : "BLUE";
}

function modelOutput(
  fen: string,
  turnNumber: number,
  perspective: Player,
  personality: PersonalityId
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
    redWinProbability: predictWinProbability(position, "RED"),
    blueWinProbability: predictWinProbability(position, "BLUE"),
    personality: evaluatePersonalityPosition(
      position,
      perspective,
      personality
    ),
  };
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
  const turnNumber = integerInRange(
    request.turnNumber,
    1,
    1,
    2_000,
    "turnNumber"
  );
  const timeMs = integerInRange(request.timeMs, 750, 50, 3_000, "timeMs");
  const maxDepth = integerInRange(request.maxDepth, 2, 1, 3, "maxDepth");
  const beamWidth = integerInRange(request.beamWidth, 8, 2, 16, "beamWidth");
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
    const resultProxy = search.search(
      board,
      personality,
      timeMs,
      maxDepth,
      beamWidth
    );
    let searchResult: GhqSearchResult;
    try {
      searchResult = toSearchResult(resultProxy);
    } finally {
      destroyProxy(resultProxy);
    }

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
    return {
      fen,
      resultingFen,
      serializedState: board.serialize(),
      sideToMove,
      turnNumber,
      personality,
      effectiveConfig: { timeMs, maxDepth, beamWidth },
      outcome: pythonOutcome(board),
      model: {
        before: modelOutput(fen, turnNumber, sideToMove, personality),
        after: modelOutput(
          resultingFen,
          turnNumber + 1,
          sideToMove,
          personality
        ),
      },
      search: searchResult,
    };
  } finally {
    destroyProxy(board);
  }
}
