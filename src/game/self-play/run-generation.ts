import type { FenAnalysisRequest, FenAnalysisResponse, PersonalityId } from "@/game/analysis/types";
import type { GameEngine, Player } from "@/game/engine-v2";
import { FENtoBoardState } from "@/game/notation";
import { extractValueFeatures } from "@/game/value-model/features";
import {
  createSeededRandom,
  playOneGame,
  type SelfPlayGameResult,
} from "@/game/self-play/play-one-game";
import {
  createSearchSelfPlayAgent,
  type AnalyzePosition,
  type SearchDecisionRecord,
} from "@/game/self-play/search-agent";

export interface SelfPlayCompetitor {
  id: string;
  personality: PersonalityId;
  timeMs: number;
  maxDepth: number;
  beamWidth: number;
  explorationTemperature: number;
}

export interface SelfPlayTrainingRecord extends SearchDecisionRecord {
  generationId: string;
  gameId: string;
  agentId: string;
  opponentId: string;
  outcomeValue: number;
  winner?: Player;
  termination: string;
  trainingEligible: boolean;
  features: number[];
}

export interface SelfPlayGenerationGame {
  generationId: string;
  gameId: string;
  index: number;
  result: SelfPlayGameResult;
  trainingRecords: SelfPlayTrainingRecord[];
}

export interface SelfPlayStanding {
  agentId: string;
  games: number;
  wins: number;
  losses: number;
  draws: number;
  rating: number;
}

export interface SelfPlayGenerationResult {
  generationId: string;
  seed: number;
  games: SelfPlayGenerationGame[];
  standings: SelfPlayStanding[];
  metrics: {
    completedGames: number;
    decisiveGames: number;
    trainingPositions: number;
    exploratorySelections: number;
    uniqueContinuationCount: number;
    uniqueContinuationRate: number;
  };
}

export interface RunSelfPlayGenerationConfig {
  generationId: string;
  engine: GameEngine;
  analyze: AnalyzePosition;
  population: SelfPlayCompetitor[];
  games: number;
  seed: number;
  concurrency?: number;
  maxTurns?: number;
  /** Stop scheduling new games after this wall-clock timestamp. */
  deadlineAt?: number;
  onGame?: (game: SelfPlayGenerationGame) => void | Promise<void>;
}

function checkedInteger(value: number, minimum: number, label: string): number {
  if (!Number.isSafeInteger(value) || value < minimum) {
    throw new RangeError(`${label} must be an integer of at least ${minimum}`);
  }
  return value;
}

function matchupSchedule(
  population: SelfPlayCompetitor[],
  games: number,
  random: () => number
): Array<[SelfPlayCompetitor, SelfPlayCompetitor]> {
  const cycle: Array<[SelfPlayCompetitor, SelfPlayCompetitor]> = [];
  for (let first = 0; first < population.length; first++) {
    for (let second = first + 1; second < population.length; second++) {
      cycle.push([population[first], population[second]]);
      cycle.push([population[second], population[first]]);
    }
  }
  for (let index = cycle.length - 1; index > 0; index--) {
    const swap = Math.floor(random() * (index + 1));
    [cycle[index], cycle[swap]] = [cycle[swap], cycle[index]];
  }
  return Array.from({ length: games }, (_, index) => cycle[index % cycle.length]);
}

function outcomeValue(winner: Player | undefined, perspective: Player): number {
  if (!winner) return 0.5;
  return winner === perspective ? 1 : 0;
}

function summarizeStandings(
  population: SelfPlayCompetitor[],
  games: SelfPlayGenerationGame[]
): SelfPlayStanding[] {
  const table = new Map(
    population.map((agent) => [
      agent.id,
      { agentId: agent.id, games: 0, wins: 0, losses: 0, draws: 0, rating: 1500 },
    ])
  );
  for (const game of [...games].sort((a, b) => a.index - b.index)) {
    const red = table.get(game.result.redAgentId)!;
    const blue = table.get(game.result.blueAgentId)!;
    red.games++;
    blue.games++;
    const redScore = outcomeValue(game.result.outcome.winner, "RED");
    if (redScore === 1) {
      red.wins++;
      blue.losses++;
    } else if (redScore === 0) {
      blue.wins++;
      red.losses++;
    } else {
      red.draws++;
      blue.draws++;
    }
    const expectedRed = 1 / (1 + 10 ** ((blue.rating - red.rating) / 400));
    const change = 20 * (redScore - expectedRed);
    red.rating += change;
    blue.rating -= change;
  }
  return [...table.values()]
    .map((entry) => ({ ...entry, rating: Math.round(entry.rating) }))
    .sort((a, b) => b.rating - a.rating || a.agentId.localeCompare(b.agentId));
}

async function playScheduledGame(
  config: RunSelfPlayGenerationConfig,
  index: number,
  redConfig: SelfPlayCompetitor,
  blueConfig: SelfPlayCompetitor
): Promise<SelfPlayGenerationGame> {
  const gameId = `${config.generationId}-${String(index + 1).padStart(6, "0")}`;
  const decisions: Array<{ agentId: string; opponentId: string; record: SearchDecisionRecord }> = [];
  const makeAgent = (
    agent: SelfPlayCompetitor,
    opponent: SelfPlayCompetitor
  ) =>
    createSearchSelfPlayAgent({
      ...agent,
      analyze: config.analyze,
      onDecision: (record) => {
        decisions.push({ agentId: agent.id, opponentId: opponent.id, record });
      },
    });
  const result = await playOneGame({
    engine: config.engine,
    red: makeAgent(redConfig, blueConfig),
    blue: makeAgent(blueConfig, redConfig),
    seed: (config.seed + Math.imul(index + 1, 0x9e3779b1)) >>> 0,
    maxTurns: config.maxTurns ?? 160,
  });
  const trainingRecords = decisions.map(({ agentId, opponentId, record }) => {
    const state = FENtoBoardState(record.fen);
    return {
      ...record,
      generationId: config.generationId,
      gameId,
      agentId,
      opponentId,
      outcomeValue: outcomeValue(result.outcome.winner, record.player),
      winner: result.outcome.winner,
      termination: result.outcome.termination,
      trainingEligible: result.completed && result.outcome.winner !== undefined,
      features: extractValueFeatures(
        {
          board: state.board,
          redReserve: state.redReserve,
          blueReserve: state.blueReserve,
          currentPlayer: state.currentPlayerTurn ?? "RED",
          turnNumber: record.turnNumber,
        },
        record.player
      ),
    };
  });
  return { generationId: config.generationId, gameId, index, result, trainingRecords };
}

export async function runSelfPlayGeneration(
  config: RunSelfPlayGenerationConfig
): Promise<SelfPlayGenerationResult> {
  const gameCount = checkedInteger(config.games, 1, "games");
  const concurrency = checkedInteger(config.concurrency ?? 1, 1, "concurrency");
  if (config.population.length < 2) {
    throw new RangeError("population must contain at least two competitors");
  }
  if (new Set(config.population.map((agent) => agent.id)).size !== config.population.length) {
    throw new RangeError("population competitor ids must be unique");
  }

  const random = createSeededRandom(config.seed >>> 0);
  const schedule = matchupSchedule(config.population, gameCount, random);
  const completed = new Array<SelfPlayGenerationGame>(gameCount);
  let nextIndex = 0;
  const worker = async () => {
    while (true) {
      if (config.deadlineAt !== undefined && Date.now() >= config.deadlineAt) {
        return;
      }
      const index = nextIndex++;
      if (index >= gameCount) return;
      const [red, blue] = schedule[index];
      const game = await playScheduledGame(config, index, red, blue);
      completed[index] = game;
      await config.onGame?.(game);
    }
  };
  await Promise.all(
    Array.from({ length: Math.min(concurrency, gameCount) }, () => worker())
  );

  const finished = completed.filter(
    (game): game is SelfPlayGenerationGame => game !== undefined
  );
  const continuations = finished.map((game) =>
    game.result.turns
      .slice(4, 8)
      .map((turn) => turn.moves.map((move) => move.uci).join(" "))
      .join(" | ")
  );
  const uniqueContinuations = new Set(continuations);
  const trainingRecords = finished.flatMap((game) => game.trainingRecords);
  return {
    generationId: config.generationId,
    seed: config.seed >>> 0,
    games: finished,
    standings: summarizeStandings(config.population, finished),
    metrics: {
      completedGames: finished.filter((game) => game.result.completed).length,
      decisiveGames: finished.filter((game) => game.result.outcome.winner).length,
      trainingPositions: trainingRecords.filter((record) => record.trainingEligible).length,
      exploratorySelections: trainingRecords.filter((record) => record.selectedRank > 1).length,
      uniqueContinuationCount: uniqueContinuations.size,
      uniqueContinuationRate:
        finished.length === 0 ? 0 : uniqueContinuations.size / finished.length,
    },
  };
}

export type { AnalyzePosition, FenAnalysisRequest, FenAnalysisResponse };
