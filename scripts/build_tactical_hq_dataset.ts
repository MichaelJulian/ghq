#!/usr/bin/env tsx
/** Build direct HQ-survival supervision from persisted Vercel games. */

import { execFile } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import { get, list, type ListBlobResultBlob } from "@vercel/blob";
import { config } from "dotenv";

import { FENtoBoardState } from "../src/game/notation";
import {
  extractValueFeaturesV3,
  VALUE_FEATURE_NAMES_V3,
} from "../src/game/value-model/features";
import type {
  DurableSelfPlayDecision,
  DurableSelfPlayGameResult,
} from "../src/workflows/self-play-game";

const execFileAsync = promisify(execFile);
config({ path: ".env.local" });

interface ExactHqAudit {
  fen: string;
  defender: "RED" | "BLUE";
  complete_turn_states: number;
  safe_turns: number;
  forced_hq_loss: boolean;
  inconclusive: boolean;
  exhaustive: boolean;
  nodes_visited: number;
  max_nodes: number;
  safe_examples: Array<{ moves: string[]; reason: string }>;
  terminal_states: number;
}

interface SelectedDecision {
  game: DurableSelfPlayGameResult;
  decision: DurableSelfPlayDecision;
  decisionIndex: number;
  ownTurnsFromEnd: number;
}

function argumentsFor(name: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < process.argv.length - 1; index++) {
    if (process.argv[index] === name) values.push(process.argv[index + 1]);
  }
  return values;
}

function positiveInteger(name: string, fallback: number): number {
  const raw = argumentsFor(name).at(-1);
  const value = raw === undefined ? fallback : Number.parseInt(raw, 10);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
}

function optionalPositiveInteger(name: string): number | undefined {
  const raw = argumentsFor(name).at(-1);
  if (raw === undefined) return undefined;
  const value = Number.parseInt(raw, 10);
  if (!Number.isSafeInteger(value) || value < 1) {
    throw new Error(`${name} must be a positive integer`);
  }
  return value;
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
  return blobs.sort((left, right) =>
    left.pathname.localeCompare(right.pathname)
  );
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

function selectedDecisions(
  game: DurableSelfPlayGameResult,
  lookback: number
): SelectedDecision[] {
  if (
    !game.completed ||
    game.outcome.termination !== "hq-capture" ||
    !game.outcome.winner
  ) {
    return [];
  }
  return (["RED", "BLUE"] as const).flatMap((player) => {
    const decisions = game.decisions
      .map((decision, decisionIndex) => ({ decision, decisionIndex }))
      .filter(({ decision }) => decision.player === player)
      .slice(-lookback);
    return decisions.map(({ decision, decisionIndex }, index) => ({
      game,
      decision,
      decisionIndex,
      ownTurnsFromEnd: decisions.length - index - 1,
    }));
  });
}

async function exactAudit(
  fen: string,
  maxNodes: number
): Promise<ExactHqAudit> {
  const python = process.env.GHQ_PYTHON ?? ".venv/bin/python";
  const { stdout } = await execFileAsync(
    python,
    ["scripts/audit_hq_loss.py", fen, "--max-nodes", String(maxNodes)],
    { maxBuffer: 1024 * 1024 }
  );
  return JSON.parse(stdout) as ExactHqAudit;
}

async function mapConcurrent<T, R>(
  values: T[],
  concurrency: number,
  mapper: (value: T, index: number) => Promise<R>
): Promise<R[]> {
  const results = new Array<R>(values.length);
  let next = 0;
  async function worker() {
    for (;;) {
      const index = next++;
      if (index >= values.length) return;
      results[index] = await mapper(values[index], index);
    }
  }
  await Promise.all(
    Array.from({ length: Math.min(concurrency, values.length) }, worker)
  );
  return results;
}

async function main() {
  const generationIds = argumentsFor("--generation");
  const outputPath = argumentsFor("--output").at(-1);
  if (!generationIds.length) {
    throw new Error("Pass at least one --generation <id>");
  }
  if (!outputPath) throw new Error("Pass --output <path>");
  const lookback = positiveInteger("--lookback", 4);
  const maxNodes = positiveInteger("--max-nodes", 2_000_000);
  const retryMaxNodes = optionalPositiveInteger("--retry-max-nodes");
  if (retryMaxNodes !== undefined && retryMaxNodes <= maxNodes) {
    throw new Error("--retry-max-nodes must be greater than --max-nodes");
  }
  const concurrency = positiveInteger("--concurrency", 4);

  const blobs = (await Promise.all(generationIds.map(gameBlobs))).flat();
  const games = await mapConcurrent(blobs, 12, (blob) => readGame(blob));
  const selected = games.flatMap((game) => selectedDecisions(game, lookback));
  let samples = await mapConcurrent(
    selected,
    concurrency,
    async ({ game, decision, decisionIndex, ownTurnsFromEnd }, index) => {
      process.stderr.write(
        `audit ${index + 1}/${selected.length} ${game.gameId} turn ${
          decision.turnNumber
        } ${decision.player}\n`
      );
      const exact = await exactAudit(decision.fen, maxNodes);
      if (exact.defender !== decision.player) {
        throw new Error(
          `Audit defender mismatch in ${game.gameId} turn ${decision.turnNumber}`
        );
      }
      const state = FENtoBoardState(decision.fen);
      return {
        ...exact,
        generationId: game.generationId,
        gameId: game.gameId,
        codeVersion: game.codeVersion,
        valueModelCheckpoint:
          decision.player === "RED"
            ? game.redValueModelCheckpoint
            : game.blueValueModelCheckpoint,
        winner: game.outcome.winner,
        player: decision.player,
        playerEventuallyLost: decision.player !== game.outcome.winner,
        turnNumber: decision.turnNumber,
        decisionIndex,
        ownTurnsFromEnd,
        fen: decision.fen,
        selectedMoves: decision.selectedMoves,
        searchScore: decision.currentPlayerScore,
        valueModelWinProbability: decision.winProbability,
        completedDepth: decision.completedDepth,
        features: extractValueFeaturesV3(
          {
            board: state.board,
            redReserve: state.redReserve,
            blueReserve: state.blueReserve,
            currentPlayer: state.currentPlayerTurn ?? decision.player,
            turnNumber: decision.turnNumber,
          },
          decision.player
        ),
        label: exact.forced_hq_loss ? 1 : exact.inconclusive ? null : 0,
      };
    }
  );

  if (retryMaxNodes !== undefined) {
    const inconclusiveIndices = samples
      .map((sample, index) => (sample.label === null ? index : -1))
      .filter((index) => index >= 0);
    const retried = await mapConcurrent(
      inconclusiveIndices,
      concurrency,
      async (sampleIndex, retryIndex) => {
        const sample = samples[sampleIndex];
        process.stderr.write(
          `retry ${retryIndex + 1}/${inconclusiveIndices.length} ${
            sample.gameId
          } turn ${sample.turnNumber} ${sample.player}\n`
        );
        const exact = await exactAudit(sample.fen, retryMaxNodes);
        if (exact.defender !== sample.player) {
          throw new Error(
            `Retry defender mismatch in ${sample.gameId} turn ${sample.turnNumber}`
          );
        }
        return {
          ...sample,
          ...exact,
          label: exact.forced_hq_loss ? 1 : exact.inconclusive ? null : 0,
        };
      }
    );
    samples = [...samples];
    inconclusiveIndices.forEach((sampleIndex, index) => {
      samples[sampleIndex] = retried[index];
    });
  }

  const eligible = samples.filter((sample) => sample.label !== null);
  const report = {
    format: "ghq-tactical-hq-dataset-v1",
    generatedAt: new Date().toISOString(),
    generationIds,
    games: games.length,
    hqCaptureGames: games.filter(
      (game) => game.completed && game.outcome.termination === "hq-capture"
    ).length,
    lookbackOwnTurnsPerPlayer: lookback,
    maxNodesPerAudit: maxNodes,
    retryMaxNodesPerInconclusiveAudit: retryMaxNodes ?? null,
    featureSchema: "v3",
    featureNames: [...VALUE_FEATURE_NAMES_V3],
    samples: samples.length,
    eligibleSamples: eligible.length,
    forcedHqLosses: eligible.filter((sample) => sample.label === 1).length,
    safePositions: eligible.filter((sample) => sample.label === 0).length,
    inconclusive: samples.filter((sample) => sample.label === null).length,
    exhaustive: samples.filter((sample) => sample.exhaustive).length,
    records: samples,
  };
  await writeFile(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf8");
  process.stdout.write(
    `${JSON.stringify({ ...report, records: undefined }, null, 2)}\n`
  );
}

void main();
