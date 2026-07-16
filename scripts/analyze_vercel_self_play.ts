#!/usr/bin/env tsx
/** Summarize selected persisted Vercel self-play generations. */

import "dotenv/config";
import { get, list, type ListBlobResultBlob } from "@vercel/blob";

import type { DurableSelfPlayGameResult } from "../src/workflows/self-play-game";

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
  return agentId.replace(/-workflow.*$/, "");
}

function increment(counts: Record<string, number>, key: string, amount = 1) {
  counts[key] = (counts[key] ?? 0) + amount;
}

function percentile(values: number[], fraction: number): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((left, right) => left - right);
  return sorted[Math.round((sorted.length - 1) * fraction)];
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
  const depthByColor: Record<string, number> = {};
  const trainingRejectionReasons: Record<string, number> = {};
  const personalities: Record<
    string,
    { games: number; wins: number; losses: number; draws: number }
  > = {};
  const rejected: Array<{
    gameId: string;
    termination: string;
    winner?: string;
    decisions: number;
    fallbackDecisions: number;
  }> = [];
  const overLimitExamples: Array<{
    gameId: string;
    turnNumber: number;
    player: string;
    moves: string[];
    fallback: string;
    depth: number;
  }> = [];
  const lengths: number[] = [];
  let decisions = 0;
  let trainingPositions = 0;
  let fallbackDecisions = 0;
  let timedOutDecisions = 0;
  let incompleteTurnDecisions = 0;
  let qualityEligibleGames = 0;

  for (const game of games) {
    increment(outcomes, game.outcome.winner ?? "DRAW");
    increment(terminations, game.outcome.termination);
    lengths.push(
      Math.max(0, ...game.decisions.map((decision) => decision.turnNumber))
    );
    decisions += game.decisions.length;
    trainingPositions += game.trainingPositions;
    fallbackDecisions += game.quality.fallbackDecisions;
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
      }
      if (!decision.completedTurn) incompleteTurnDecisions++;
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
  }

  const pairedOutcomes: Record<string, number> = {};
  const sortedGames = [...games].sort((left, right) =>
    left.gameId.localeCompare(right.gameId)
  );
  for (let index = 0; index + 1 < sortedGames.length; index += 2) {
    const first = sortedGames[index].outcome.winner ?? "DRAW";
    const second = sortedGames[index + 1].outcome.winner ?? "DRAW";
    increment(pairedOutcomes, `${first}/${second}`);
  }

  console.log(
    JSON.stringify(
      {
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
        fallbackTypes,
        fallbackByColor,
        fallbackByPhase,
        fallbackDecisions,
        fallbackRate: Number((fallbackDecisions / decisions).toFixed(4)),
        timedOutDecisions,
        timedOutRate: Number((timedOutDecisions / decisions).toFixed(4)),
        incompleteTurnDecisions,
        trainingGames: games.filter((game) => game.trainingPositions > 0)
          .length,
        qualityEligibleGames,
        trainingRejectionReasons,
        trainingPositions,
        personalities,
        pairedOutcomes,
        rejected,
        overLimitExamples,
      },
      null,
      2
    )
  );
}

void main();
