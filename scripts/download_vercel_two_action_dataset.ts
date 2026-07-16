#!/usr/bin/env tsx
/** Build a draw-aware value dataset directly from persisted two-action games. */

import { createWriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { get, list, type ListBlobResultBlob } from "@vercel/blob";

import { VALUE_FEATURE_NAMES } from "../src/game/value-model/features";
import type { DurableSelfPlayGameResult } from "../src/workflows/self-play-game";

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  if (index === -1 || !process.argv[index + 1]) {
    throw new Error(`Missing required argument ${name}`);
  }
  return process.argv[index + 1];
}

async function gameBlobs(generationPrefix: string) {
  const blobs: ListBlobResultBlob[] = [];
  let cursor: string | undefined;
  do {
    const page = await list({
      prefix: `self-play/generations/${generationPrefix}`,
      cursor,
      limit: 1000,
    });
    blobs.push(
      ...page.blobs.filter(
        (blob) =>
          blob.pathname.includes("/games/") && blob.pathname.endsWith(".json")
      )
    );
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs.sort((left, right) =>
    left.pathname.localeCompare(right.pathname)
  );
}

async function readGame(
  blob: ListBlobResultBlob
): Promise<DurableSelfPlayGameResult> {
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

function isTwoActionGame(game: DurableSelfPlayGameResult): boolean {
  return (
    (game.redMaxActions ?? game.decisions[0]?.selfActionLimit ?? 3) === 2 &&
    (game.blueMaxActions ?? game.decisions[0]?.opponentActionLimit ?? 3) === 2
  );
}

function outcomeLabel(
  game: DurableSelfPlayGameResult,
  player: "RED" | "BLUE"
): number {
  if (!game.outcome.winner) return 0.5;
  return game.outcome.winner === player ? 1 : 0;
}

async function main() {
  const generationPrefix = argument("--generation-prefix");
  const outputPath = argument("--output");
  const blobs = await gameBlobs(generationPrefix);
  if (!blobs.length) throw new Error("No matching game artifacts found");

  await mkdir(dirname(outputPath), { recursive: true });
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  output.write(
    `${JSON.stringify({
      type: "schema",
      format: "ghq-value-features-v1",
      feature_names: VALUE_FEATURE_NAMES,
      ruleset: "two-actions",
      target: "eventual score: win=1, repetition draw=0.5, loss=0",
      source: "vercel-self-play-games",
      generation_prefix: generationPrefix,
    })}\n`
  );

  let games = 0;
  let decisiveGames = 0;
  let drawGames = 0;
  let rejectedGames = 0;
  let samples = 0;
  const generations = new Set<string>();
  const terminations: Record<string, number> = {};
  for (let index = 0; index < blobs.length; index += 12) {
    const batch = blobs.slice(index, index + 12);
    const records = await Promise.all(batch.map(readGame));
    for (let offset = 0; offset < records.length; offset++) {
      const game = records[offset];
      if (!isTwoActionGame(game)) {
        rejectedGames++;
        continue;
      }
      const eligibleOutcome =
        game.outcome.termination === "hq-capture" ||
        game.outcome.termination === "repetition";
      if (!eligibleOutcome) {
        rejectedGames++;
        continue;
      }
      games++;
      generations.add(game.generationId);
      terminations[game.outcome.termination] =
        (terminations[game.outcome.termination] ?? 0) + 1;
      if (game.outcome.winner) decisiveGames++;
      else drawGames++;
      const createdAt = batch[offset].uploadedAt.toISOString();
      for (const decision of game.decisions) {
        const eligible =
          decision.selectedMoves.length > 0 &&
          decision.completedTurn &&
          (decision.selfActionLimit ?? game.redMaxActions ?? 3) === 2 &&
          (decision.opponentActionLimit ?? game.blueMaxActions ?? 3) === 2 &&
          decision.fallback === "none" &&
          decision.completedDepth >= 1 &&
          decision.features.length === VALUE_FEATURE_NAMES.length;
        if (!eligible) continue;
        output.write(
          `${JSON.stringify({
            type: "sample",
            game_id: game.gameId,
            generation_id: game.generationId,
            source: "vercel_self_play_2a",
            created_at: createdAt,
            outcome_reason: game.outcome.termination,
            turn: decision.turnNumber,
            perspective: decision.player,
            label: outcomeLabel(game, decision.player),
            features: decision.features,
          })}\n`
        );
        samples++;
      }
    }
  }
  await new Promise<void>((resolve, reject) => {
    output.end(resolve);
    output.on("error", reject);
  });
  process.stderr.write(
    `${JSON.stringify({
      generationPrefix,
      artifacts: blobs.length,
      generations: generations.size,
      games,
      decisiveGames,
      drawGames,
      rejectedGames,
      terminations,
      samples,
      outputPath,
    })}\n`
  );
}

void main();
