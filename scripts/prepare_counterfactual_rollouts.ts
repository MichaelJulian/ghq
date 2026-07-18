#!/usr/bin/env tsx
/** Mine close production-search candidates into a paired rollout request. */

import { writeFile } from "node:fs/promises";
import { config } from "dotenv";

import { selectCounterfactualRoots } from "../src/game/self-play/counterfactual";
import { readPersistedSelfPlayGames } from "../src/server/self-play-storage";
import type { DurableSelfPlayGameResult } from "../src/workflows/self-play-game";

config({ path: ".env.local" });

function argument(name: string): string | undefined {
  const index = process.argv.lastIndexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function integerArgument(
  name: string,
  fallback: number,
  minimum: number,
  maximum: number
): number {
  const raw = argument(name);
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isSafeInteger(value) || value < minimum || value > maximum) {
    throw new RangeError(
      `${name} must be an integer from ${minimum} to ${maximum}`
    );
  }
  return value;
}

function numberArgument(
  name: string,
  fallback: number,
  minimum: number,
  maximum: number
): number {
  const raw = argument(name);
  const value = raw === undefined ? fallback : Number(raw);
  if (!Number.isFinite(value) || value < minimum || value > maximum) {
    throw new RangeError(
      `${name} must be a number from ${minimum} to ${maximum}`
    );
  }
  return value;
}

async function main() {
  const generationId = argument("--generation");
  if (!generationId) {
    throw new Error("Pass --generation <source-generation-id>");
  }
  const games = await readPersistedSelfPlayGames<DurableSelfPlayGameResult>(
    generationId
  );
  if (!games.length) {
    throw new Error(`No completed games found for ${generationId}`);
  }

  const roots = selectCounterfactualRoots(games, {
    maxRoots: integerArgument("--max-roots", 8, 1, 16),
    maxRootsPerGame: integerArgument("--max-per-game", 2, 1, 8),
    candidatesPerRoot: integerArgument("--candidates", 2, 2, 4),
    maxScoreMargin: numberArgument("--max-margin", 1, 0.000_001, 100),
    minTurnNumber: integerArgument("--min-turn", 5, 1, 399),
  });
  if (!roots.length) {
    throw new Error("No eligible near-tied candidate roots found");
  }

  const request = {
    sourceGenerationId: generationId,
    seed: integerArgument("--seed", Date.now() >>> 0, 0, 0xffff_ffff),
    timeMs: integerArgument("--time-ms", 20_000, 50, 30_000),
    maxDepth: integerArgument("--max-depth", 2, 1, 3),
    beamWidth: integerArgument("--beam", 6, 2, 16),
    rolloutTurns: integerArgument("--rollout-turns", 24, 2, 120),
    repetitionLimit: integerArgument("--repetition", 3, 2, 10),
    noProgressTurns: integerArgument("--no-progress", 24, 4, 100),
    branches: roots.flatMap((root) => root.branches),
  };
  const rendered = `${JSON.stringify(request, null, 2)}\n`;
  const outputPath = argument("--output");
  if (outputPath) await writeFile(outputPath, rendered, "utf8");
  process.stderr.write(
    `Selected ${roots.length} roots / ${request.branches.length} branches from ${games.length} games\n`
  );
  process.stdout.write(rendered);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
