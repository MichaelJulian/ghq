import { appendFile, mkdir, writeFile } from "fs/promises";
import path from "path";
import { PERSONALITIES } from "@/game/value-model/personalities";
import { VALUE_FEATURE_NAMES } from "@/game/value-model/features";
import { runSelfPlayGeneration, type SelfPlayCompetitor } from "@/game/self-play/run-generation";
import { loadV2Engine } from "@/server/engine";
import { analyzeFen } from "@/server/fen-analysis";

function argument(name: string, fallback: string): string {
  const index = process.argv.indexOf(`--${name}`);
  return index === -1 ? fallback : process.argv[index + 1] ?? fallback;
}

function integerArgument(name: string, fallback: number): number {
  const value = Number(argument(name, String(fallback)));
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new RangeError(`--${name} must be a positive integer`);
  }
  return value;
}

function numberArgument(name: string, fallback: number): number {
  const value = Number(argument(name, String(fallback)));
  if (!Number.isFinite(value) || value <= 0) {
    throw new RangeError(`--${name} must be a positive number`);
  }
  return value;
}

function hasArgument(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

function gameLines(
  game: Awaited<ReturnType<typeof runSelfPlayGeneration>>["games"][number],
  generationId: string,
  createdAt: string
): string[] {
  const gameLine = JSON.stringify({
    type: "self-play-game",
    generationId: game.generationId,
    gameId: game.gameId,
    index: game.index,
    result: game.result,
  });
  const samples = game.trainingRecords
    .filter((record) => record.trainingEligible)
    .map((record) =>
      JSON.stringify({
        type: "sample",
        game_id: record.gameId,
        created_at: createdAt,
        outcome_reason: record.termination,
        turn: record.turnNumber,
        perspective: record.player,
        label: record.outcomeValue,
        features: record.features,
        generation_id: generationId,
        agent_id: record.agentId,
        opponent_id: record.opponentId,
        personality: record.personality,
        selected_rank: record.selectedRank,
        selected_moves: record.selectedMoves,
        candidate_turns: record.candidateTurns,
        search_score: record.currentPlayerScore,
        search_depth: record.completedDepth,
        search_timed_out: record.timedOut,
        search_fallback: record.fallback,
        exploration_seed: record.explorationSeed,
        exploration_temperature: record.explorationTemperature,
      })
    );
  return [gameLine, ...samples];
}

async function main() {
  const durationHours = hasArgument("duration-hours")
    ? numberArgument("duration-hours", 6)
    : undefined;
  const games = integerArgument("games", durationHours ? 1_000_000 : 20);
  const seed = integerArgument("seed", 20260713) >>> 0;
  const timeMs = integerArgument("time-ms", 750);
  const maxTurns = integerArgument("max-turns", 160);
  const concurrency = integerArgument("concurrency", 1);
  const maxDepth = integerArgument("depth", 1);
  const beamWidth = integerArgument("beam", 6);
  const generationId = argument(
    "generation",
    `generation-${new Date().toISOString().replaceAll(/[:.]/g, "-")}`
  );
  const output = path.resolve(
    argument("output", `.data/self-play/${generationId}.jsonl`)
  );
  const population: SelfPlayCompetitor[] = Object.values(PERSONALITIES).map(
    (profile) => ({
      id: `${profile.id}-g0`,
      personality: profile.id,
      timeMs,
      maxDepth,
      beamWidth,
      // Preserve each character's identity while guaranteeing enough league
      // exploration to avoid deterministic mirror games.
      explorationTemperature: Math.max(0.18, profile.explorationTemperature),
    })
  );

  const engine = await loadV2Engine();
  const createdAt = new Date().toISOString();
  await mkdir(path.dirname(output), { recursive: true });
  await writeFile(
    output,
    `${JSON.stringify({
      type: "schema",
      format: "ghq-self-play-generation-v1",
      version: 1,
      generationId,
      seed,
      population,
      durationHours,
      feature_names: VALUE_FEATURE_NAMES,
    })}\n`,
    "utf8"
  );
  const generation = await runSelfPlayGeneration({
    generationId,
    engine,
    analyze: analyzeFen,
    population,
    games,
    seed,
    concurrency,
    maxTurns,
    deadlineAt:
      durationHours === undefined
        ? undefined
        : Date.now() + durationHours * 60 * 60 * 1000,
    onGame: async (game) => {
      await appendFile(
        output,
        `${gameLines(game, generationId, createdAt).join("\n")}\n`,
        "utf8"
      );
      console.log(
        JSON.stringify({
          checkpoint: game.gameId,
          completed: game.result.completed,
          winner: game.result.outcome.winner,
          termination: game.result.outcome.termination,
          turns: game.result.turns.length,
          trainingPositions: game.trainingRecords.filter(
            (record) => record.trainingEligible
          ).length,
        })
      );
    },
  });
  await appendFile(
    output,
    `${JSON.stringify({
      type: "self-play-summary",
      generationId,
      standings: generation.standings,
      metrics: generation.metrics,
    })}\n`,
    "utf8"
  );
  console.log(
    JSON.stringify(
      { output, standings: generation.standings, metrics: generation.metrics },
      null,
      2
    )
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
