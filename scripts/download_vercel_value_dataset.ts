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

async function readBlob(blob: ListBlobResultBlob): Promise<DurableTrainingSample[]> {
  const result = await get(blob.pathname, { access: "private", useCache: false });
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
    })}\n`
  );

  let samples = 0;
  const games = new Set<string>();
  const generations = new Set<string>();
  for (let index = 0; index < blobs.length; index += 12) {
    const batch = blobs.slice(index, index + 12);
    const records = await Promise.all(batch.map(readBlob));
    for (let offset = 0; offset < batch.length; offset++) {
      const createdAt = batch[offset].uploadedAt.toISOString();
      for (const sample of records[offset]) {
        if (sample.features.length !== VALUE_FEATURE_NAMES.length) {
          throw new Error(`Feature mismatch in ${sample.gameId}`);
        }
        output.write(
          `${JSON.stringify({
            type: "sample",
            game_id: sample.gameId,
            generation_id: sample.generationId,
            source: "vercel_self_play",
            created_at: createdAt,
            outcome_reason: "hq-capture",
            turn: sample.turnNumber,
            perspective: sample.player,
            label: sample.outcomeValue,
            features: sample.features,
          })}\n`
        );
        samples++;
        games.add(sample.gameId);
        generations.add(sample.generationId);
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
      generations: generations.size,
      artifacts: blobs.length,
      games: games.size,
      samples,
      outputPath,
    })}\n`
  );
}

void main();
