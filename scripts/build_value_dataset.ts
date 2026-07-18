#!/usr/bin/env tsx
/** Convert reconstructed GHQ positions into the exact TypeScript features used in production. */

import { createReadStream, createWriteStream } from "node:fs";
import { mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import { createInterface } from "node:readline";

import type { Board, Player, ReserveFleet } from "../src/game/engine";
import {
  extractValueFeatures,
  extractValueFeaturesV2,
  VALUE_FEATURE_NAMES,
  VALUE_FEATURE_NAMES_V2,
} from "../src/game/value-model/features";

type RawPosition = {
  type: "position";
  game_id: string;
  created_at: string;
  outcome_reason: string;
  winner: Player;
  turn: number;
  current_player: Player;
  board: Board;
  red_reserve: ReserveFleet;
  blue_reserve: ReserveFleet;
};

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  if (index === -1 || !process.argv[index + 1]) {
    throw new Error(`Missing required argument ${name}`);
  }
  return process.argv[index + 1];
}

function optionalArgument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

async function main() {
  const positionsPath = argument("--positions");
  const outputPath = argument("--output");
  const featureSchema = optionalArgument("--feature-schema") ?? "v1";
  if (featureSchema !== "v1" && featureSchema !== "v2") {
    throw new Error("--feature-schema must be v1 or v2");
  }
  const featureNames =
    featureSchema === "v2" ? VALUE_FEATURE_NAMES_V2 : VALUE_FEATURE_NAMES;
  const extractFeatures =
    featureSchema === "v2" ? extractValueFeaturesV2 : extractValueFeatures;
  await mkdir(dirname(outputPath), { recursive: true });
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  output.write(
    `${JSON.stringify({
      type: "schema",
      format: "ghq-value-features-v1",
      feature_names: featureNames,
      feature_schema: featureSchema,
    })}\n`
  );

  const reader = createInterface({
    input: createReadStream(positionsPath, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  let positions = 0;
  let samples = 0;
  const games = new Set<string>();
  for await (const line of reader) {
    if (!line.trim()) continue;
    const raw = JSON.parse(line) as RawPosition;
    const position = {
      board: raw.board,
      redReserve: raw.red_reserve,
      blueReserve: raw.blue_reserve,
      currentPlayer: raw.current_player,
      turnNumber: raw.turn,
    };
    for (const perspective of ["RED", "BLUE"] as const) {
      const features = extractFeatures(position, perspective);
      output.write(
        `${JSON.stringify({
          type: "sample",
          game_id: raw.game_id,
          created_at: raw.created_at,
          outcome_reason: raw.outcome_reason,
          turn: raw.turn,
          perspective,
          label: perspective === raw.winner ? 1 : 0,
          features,
        })}\n`
      );
      samples++;
    }
    positions++;
    games.add(raw.game_id);
  }
  await new Promise<void>((resolve, reject) => {
    output.end(resolve);
    output.on("error", reject);
  });
  process.stderr.write(
    `${JSON.stringify({
      games: games.size,
      positions,
      samples,
      features: featureNames.length,
      featureSchema,
    })}\n`
  );
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
