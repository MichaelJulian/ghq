#!/usr/bin/env tsx
/** Download quality-gated Vercel self-play samples into the trainer's JSONL format. */

import { createWriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { get, list, type ListBlobResultBlob } from "@vercel/blob";

import { VALUE_FEATURE_NAMES } from "../src/game/value-model/features";

interface DurableTrainingSample {
  generationId: string;
  gameId: string;
  turnNumber: number;
  player: "RED" | "BLUE";
  outcomeValue: number;
  features: number[];
  codeVersion: string;
  valueModel?: "incumbent" | "challenger";
  valueModelCheckpoint?: string;
}

function pairedGameNumber(gameId: string): number {
  const match = gameId.match(/-(\d+)$/);
  if (!match) {
    throw new Error(`Unable to derive color-swapped pair from ${gameId}`);
  }
  const gameNumber = Number.parseInt(match[1], 10);
  if (!Number.isSafeInteger(gameNumber) || gameNumber < 1) {
    throw new Error(`Invalid paired game number in ${gameId}`);
  }
  return gameNumber;
}

function pairedGameId(generationId: string, gameId: string): string {
  const gameNumber = pairedGameNumber(gameId);
  const pairNumber = Math.floor((gameNumber - 1) / 2) + 1;
  return `${generationId}-pair-${String(pairNumber).padStart(4, "0")}`;
}

function isCompleteColorSwap(gameIds: string[]): boolean {
  if (gameIds.length !== 2) return false;
  const numbers = gameIds.map(pairedGameNumber).sort((a, b) => a - b);
  return numbers[0] % 2 === 1 && numbers[1] === numbers[0] + 1;
}

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  if (index === -1 || !process.argv[index + 1]) {
    throw new Error(`Missing required argument ${name}`);
  }
  return process.argv[index + 1];
}

async function selectedBlobs(generationPrefix: string) {
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
          blob.pathname.includes("/training/") &&
          blob.pathname.endsWith(".jsonl")
      )
    );
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs.sort((a, b) => a.pathname.localeCompare(b.pathname));
}

async function readBlob(
  blob: ListBlobResultBlob
): Promise<DurableTrainingSample[]> {
  const result = await get(blob.pathname, {
    access: "private",
    useCache: false,
  });
  if (!result?.stream || result.statusCode !== 200) {
    throw new Error(`Unable to read ${blob.pathname}`);
  }
  const body = await new Response(result.stream).text();
  return body
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as DurableTrainingSample);
}

async function main() {
  const generationPrefix = argument("--generation-prefix");
  const codeVersion = argument("--code-version");
  const valueModelCheckpoint = argument("--value-model-checkpoint");
  const outputPath = argument("--output");
  const blobs = await selectedBlobs(generationPrefix);
  if (!blobs.length) throw new Error("No matching training artifacts found");
  await mkdir(dirname(outputPath), { recursive: true });
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  output.write(
    `${JSON.stringify({
      type: "schema",
      format: "ghq-value-features-v1",
      feature_names: VALUE_FEATURE_NAMES,
      ruleset: "three-actions",
      source: "vercel-self-play",
      generation_prefix: generationPrefix,
      code_version: codeVersion,
      behavior_value_model_checkpoint: valueModelCheckpoint,
      paired_complete_only: true,
    })}\n`
  );

  const recordsByGame = new Map<
    string,
    {
      createdAt: string;
      generationId: string;
      pairId: string;
      samples: DurableTrainingSample[];
    }
  >();
  for (let index = 0; index < blobs.length; index += 12) {
    const batch = blobs.slice(index, index + 12);
    const records = await Promise.all(batch.map(readBlob));
    for (let offset = 0; offset < batch.length; offset++) {
      const createdAt = batch[offset].uploadedAt.toISOString();
      for (const sample of records[offset]) {
        if (sample.codeVersion !== codeVersion) {
          throw new Error(
            `Search provenance mismatch in ${
              sample.gameId
            }: expected ${codeVersion}, received ${
              sample.codeVersion || "missing"
            }`
          );
        }
        if (sample.valueModelCheckpoint !== valueModelCheckpoint) {
          throw new Error(
            `Behavior checkpoint mismatch in ${
              sample.gameId
            }: expected ${valueModelCheckpoint}, received ${
              sample.valueModelCheckpoint || "missing"
            }`
          );
        }
        if (sample.features.length !== VALUE_FEATURE_NAMES.length) {
          throw new Error(`Feature mismatch in ${sample.gameId}`);
        }
        const pairId = pairedGameId(sample.generationId, sample.gameId);
        const record = recordsByGame.get(sample.gameId);
        if (record) {
          if (
            record.generationId !== sample.generationId ||
            record.pairId !== pairId
          ) {
            throw new Error(
              `Inconsistent game provenance for ${sample.gameId}`
            );
          }
          record.samples.push(sample);
        } else {
          recordsByGame.set(sample.gameId, {
            createdAt,
            generationId: sample.generationId,
            pairId,
            samples: [sample],
          });
        }
      }
    }
  }

  const gamesByPair = new Map<string, string[]>();
  for (const [gameId, record] of recordsByGame) {
    const games = gamesByPair.get(record.pairId) ?? [];
    games.push(gameId);
    gamesByPair.set(record.pairId, games);
  }
  const completePairs = new Set(
    [...gamesByPair]
      .filter(([, gameIds]) => isCompleteColorSwap(gameIds))
      .map(([pairId]) => pairId)
  );
  if (!completePairs.size) {
    throw new Error("No complete quality-eligible color-swapped pairs found");
  }

  let samples = 0;
  const games = new Set<string>();
  const generations = new Set<string>();
  for (const [gameId, record] of [...recordsByGame].sort(([left], [right]) =>
    left.localeCompare(right)
  )) {
    if (!completePairs.has(record.pairId)) continue;
    games.add(gameId);
    generations.add(record.generationId);
    for (const sample of record.samples) {
      output.write(
        `${JSON.stringify({
          type: "sample",
          game_id: sample.gameId,
          generation_id: sample.generationId,
          pair_id: record.pairId,
          source: "vercel_self_play",
          code_version: sample.codeVersion,
          behavior_value_model: sample.valueModel ?? "unknown",
          behavior_value_model_checkpoint:
            sample.valueModelCheckpoint ?? "unknown",
          created_at: record.createdAt,
          outcome_reason: "hq-capture",
          turn: sample.turnNumber,
          perspective: sample.player,
          label: sample.outcomeValue,
          features: sample.features,
        })}\n`
      );
      samples++;
    }
  }
  await new Promise<void>((resolve, reject) => {
    output.end(resolve);
    output.on("error", reject);
  });
  process.stderr.write(
    `${JSON.stringify({
      generationPrefix,
      codeVersion,
      valueModelCheckpoint,
      generations: generations.size,
      artifacts: blobs.length,
      games: games.size,
      pairs: completePairs.size,
      excludedGames: recordsByGame.size - games.size,
      excludedIncompletePairs: gamesByPair.size - completePairs.size,
      samples,
      outputPath,
    })}\n`
  );
}

void main();
