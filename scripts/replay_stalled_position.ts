#!/usr/bin/env tsx
/** Re-run one persisted self-play decision with its preceding history. */

import "dotenv/config";
import { get } from "@vercel/blob";

import type { Player } from "../src/game/engine-v2";
import { analyzeFen } from "../src/server/fen-analysis";
import {
  actionMadeProgress,
  type DurableSelfPlayGameResult,
} from "../src/workflows/self-play-game";

function argument(name: string): string | undefined {
  const index = process.argv.indexOf(`--${name}`);
  return index < 0 ? undefined : process.argv[index + 1];
}

async function readGame(generationId: string, gameId: string) {
  const pathname = `self-play/generations/${generationId}/games/${gameId}.json`;
  const result = await get(pathname, { access: "private", useCache: false });
  if (!result?.stream || result.statusCode !== 200) {
    throw new Error(`Unable to read ${pathname}`);
  }
  return (await new Response(
    result.stream
  ).json()) as DurableSelfPlayGameResult;
}

async function main() {
  const generationId = argument("generation");
  const gameId = argument("game");
  if (!generationId || !gameId) {
    throw new Error("Pass --generation <id> and --game <id>");
  }
  const game = await readGame(generationId, gameId);
  const requestedTurn = Number(argument("turn") ?? 0);
  const target = requestedTurn
    ? game.decisions.find((decision) => decision.turnNumber === requestedTurn)
    : game.decisions.at(-1);
  if (!target) throw new Error("Requested decision was not found");

  const recentFens = [game.initialFen];
  const ownTurns: Record<Player, string[][]> = { RED: [], BLUE: [] };
  let turnsWithoutProgress = 0;
  for (const decision of game.decisions) {
    if (decision === target) break;
    const madeProgress = decision.selectedMoves.some(actionMadeProgress);
    turnsWithoutProgress = madeProgress ? 0 : turnsWithoutProgress + 1;
    recentFens.push(decision.resultingFen);
    ownTurns[decision.player].push(decision.selectedMoves);
  }

  if (process.argv.includes("--inspect-only")) {
    console.log(
      JSON.stringify(
        {
          gameId,
          turn: target.turnNumber,
          player: target.player,
          fen: target.fen,
          turnsWithoutProgress,
          previousOwnTurns: ownTurns[target.player].slice(-4),
          originalMoves: target.selectedMoves,
        },
        null,
        2
      )
    );
    return;
  }

  const replay = await analyzeFen({
    fen: target.fen,
    turnNumber: target.turnNumber,
    personality: target.personality,
    timeMs: Number(argument("time-ms") ?? 20_000),
    maxDepth: 2,
    beamWidth: 6,
    maxActions: 3,
    explorationTemperature: 0,
    explorationSeed: target.explorationSeed,
    recentFens: recentFens.slice(-32),
    previousOwnTurns: ownTurns[target.player].slice(-4),
    turnsWithoutProgress,
  });

  console.log(
    JSON.stringify(
      {
        gameId,
        turn: target.turnNumber,
        player: target.player,
        fen: target.fen,
        turnsWithoutProgress,
        original: {
          moves: target.selectedMoves,
          score: target.currentPlayerScore,
          recommendation: target.recommendationLabel,
        },
        replay: {
          moves: replay.search.best_turn.all_moves,
          score: replay.search.score.current_player,
          recommendation: replay.search.recommendation_label,
          depth: replay.search.search.completed_depth_in_turns,
          fallback: replay.search.search.fallback_used,
          timedOut: replay.search.search.timed_out,
          candidates: replay.search.candidate_turns.map((candidate) => ({
            rank: candidate.rank,
            moves: candidate.all_moves,
            score: candidate.score,
            forcingGain: candidate.purpose.forcing_gain,
            backfills: candidate.purpose.backfills,
            purposefulActions: candidate.purpose.purposeful_actions,
          })),
        },
      },
      null,
      2
    )
  );
}

void main();
