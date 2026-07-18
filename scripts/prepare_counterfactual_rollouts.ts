#!/usr/bin/env tsx
/** Mine close production-search candidates into a paired rollout request. */

import { readFile, writeFile } from "node:fs/promises";
import { config } from "dotenv";

import {
  counterfactualRootFingerprint,
  selectCounterfactualRoots,
} from "../src/game/self-play/counterfactual";
import { readPersistedSelfPlayGames } from "../src/server/self-play-storage";
import type { DurableSelfPlayGameResult } from "../src/workflows/self-play-game";

config({ path: ".env.local" });

function argument(name: string): string | undefined {
  const index = process.argv.lastIndexOf(name);
  return index >= 0 ? process.argv[index + 1] : undefined;
}

function argumentsFor(name: string): string[] {
  return process.argv.flatMap((value, index) =>
    value === name && process.argv[index + 1] ? [process.argv[index + 1]] : []
  );
}

async function excludedRootIds(): Promise<Set<string>> {
  const excluded = new Set<string>(argumentsFor("--exclude-root"));
  for (const path of argumentsFor("--exclude-report")) {
    const report = JSON.parse(await readFile(path, "utf8")) as {
      pairs?: Array<{ rootId?: unknown }>;
    };
    if (!Array.isArray(report.pairs)) {
      throw new Error(`${path} is not a counterfactual rollout report`);
    }
    for (const pair of report.pairs) {
      if (typeof pair.rootId === "string") excluded.add(pair.rootId);
    }
  }
  return excluded;
}

async function excludedSourceGameIds(): Promise<Set<string>> {
  const excluded = new Set<string>(argumentsFor("--exclude-source-game"));
  for (const path of argumentsFor("--exclude-source-games-report")) {
    const report = JSON.parse(await readFile(path, "utf8")) as {
      pairs?: Array<{
        sourceGameId?: unknown;
        confident?: unknown;
        trainingEligible?: unknown;
      }>;
    };
    if (!Array.isArray(report.pairs)) {
      throw new Error(`${path} is not a counterfactual rollout report`);
    }
    for (const pair of report.pairs) {
      const eligible =
        pair.trainingEligible === true ||
        (pair.trainingEligible === undefined && pair.confident === true);
      if (eligible && typeof pair.sourceGameId === "string") {
        excluded.add(pair.sourceGameId);
      }
    }
  }
  return excluded;
}

async function excludedRootFingerprints(): Promise<Set<string>> {
  const excluded = new Set<string>();
  for (const path of argumentsFor("--exclude-fingerprint-report")) {
    const report = JSON.parse(await readFile(path, "utf8")) as {
      pairs?: Array<{
        rootPlayer?: unknown;
        branches?: Array<{ initialFen?: unknown }>;
      }>;
    };
    if (!Array.isArray(report.pairs)) {
      throw new Error(`${path} is not a counterfactual rollout report`);
    }
    for (const pair of report.pairs) {
      if (
        (pair.rootPlayer !== "RED" && pair.rootPlayer !== "BLUE") ||
        !Array.isArray(pair.branches)
      ) {
        continue;
      }
      const fens = pair.branches.flatMap((branch) =>
        typeof branch.initialFen === "string" ? [branch.initialFen] : []
      );
      if (fens.length >= 2) {
        excluded.add(counterfactualRootFingerprint(pair.rootPlayer, fens));
      }
    }
  }
  return excluded;
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

  const [excluded, excludedFingerprints, excludedSourceGames] = await Promise.all([
    excludedRootIds(),
    excludedRootFingerprints(),
    excludedSourceGameIds(),
  ]);
  const roots = selectCounterfactualRoots(games, {
    maxRoots: integerArgument("--max-roots", 8, 1, 16),
    skipRoots: integerArgument("--skip-roots", 0, 0, 10_000),
    maxRootsPerGame: integerArgument("--max-per-game", 2, 1, 8),
    candidatesPerRoot: integerArgument("--candidates", 2, 2, 4),
    maxScoreMargin: numberArgument("--max-margin", 1, 0.000_001, 100),
    minTurnNumber: integerArgument("--min-turn", 5, 1, 399),
    minStrategicDivergence: numberArgument(
      "--min-divergence",
      0,
      0,
      100
    ),
    excludeRootIds: excluded,
    excludeRootFingerprints: excludedFingerprints,
    excludeSourceGameIds: excludedSourceGames,
  });
  if (!roots.length) {
    throw new Error("No eligible near-tied candidate roots found");
  }

  const request = {
    sourceGenerationId: generationId,
    selection: roots.map((root) => ({
      rootId: root.rootId,
      rootFingerprint: root.rootFingerprint,
      sourceGameId: root.sourceGameId,
      sourceTurnNumber: root.sourceTurnNumber,
      rootPlayer: root.rootPlayer,
      scoreMargin: root.scoreMargin,
      strategicDivergence: root.strategicDivergence,
    })),
    seed: integerArgument("--seed", Date.now() >>> 0, 0, 0xffff_ffff),
    timeMs: integerArgument("--time-ms", 20_000, 50, 30_000),
    maxDepth: integerArgument("--max-depth", 2, 1, 3),
    beamWidth: integerArgument("--beam", 6, 2, 16),
    rolloutTurns: integerArgument("--rollout-turns", 24, 2, 120),
    // A single deterministic continuation frequently gives both candidate
    // branches the same binary winner and therefore no policy information.
    // Two matched stochastic continuations are the cheapest setting that can
    // distinguish a repeatable advantage from one lucky trajectory.
    replicates: integerArgument("--replicates", 2, 1, 4),
    explorationTemperature: numberArgument(
      "--exploration-temperature",
      0.12,
      0,
      0.5
    ),
    repetitionLimit: integerArgument("--repetition", 3, 2, 10),
    noProgressTurns: integerArgument("--no-progress", 24, 4, 100),
    branches: roots.flatMap((root) => root.branches),
  };
  if (request.branches.length * request.replicates > 32) {
    throw new RangeError(
      `selected ${request.branches.length} branches x ${request.replicates} replicates; Vercel permits at most 32 runs`
    );
  }
  const rendered = `${JSON.stringify(request, null, 2)}\n`;
  const outputPath = argument("--output");
  if (outputPath) await writeFile(outputPath, rendered, "utf8");
  process.stderr.write(
    `Selected ${roots.length} roots / ${request.branches.length} branches from ${games.length} games; excluded ${excluded.size} prior roots, ${excludedFingerprints.size} semantic duplicates, and ${excludedSourceGames.size} training source games\n`
  );
  process.stdout.write(rendered);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
