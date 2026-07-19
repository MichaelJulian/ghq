#!/usr/bin/env tsx
/** Exhaustively classify immediate HQ losses in persisted Vercel self-play. */

import { execFile } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { promisify } from "node:util";
import { get, list, type ListBlobResultBlob } from "@vercel/blob";
import { config } from "dotenv";

import type { DurableSelfPlayGameResult } from "../src/workflows/self-play-game";

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

function argumentsFor(name: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < process.argv.length - 1; index++) {
    if (process.argv[index] === name) values.push(process.argv[index + 1]);
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

async function readDownloadedGames(
  path: string
): Promise<DurableSelfPlayGameResult[]> {
  const body = await readFile(path, "utf8");
  return body
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as DurableSelfPlayGameResult);
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

async function main() {
  const requestedGenerationIds = argumentsFor("--generation");
  const inputPaths = argumentsFor("--input");
  if (!requestedGenerationIds.length && !inputPaths.length) {
    throw new Error(
      "Pass --generation <id> or --input <downloaded-games.jsonl>"
    );
  }
  const rawMaxNodes = argumentsFor("--max-nodes").at(-1);
  // Production counterfactual positions routinely need several hundred
  // thousand states to prove that every legal defense still loses. The old
  // 100k default discarded three of the first seven otherwise clean labels;
  // the observed high-water mark was 1.65m states, so keep a measured margin.
  const maxNodes = rawMaxNodes ? Number.parseInt(rawMaxNodes, 10) : 2_000_000;
  if (!Number.isSafeInteger(maxNodes) || maxNodes < 1) {
    throw new Error("--max-nodes must be a positive integer");
  }
  const blobs = (
    await Promise.all(requestedGenerationIds.map(gameBlobs))
  ).flat();
  const games: DurableSelfPlayGameResult[] = (
    await Promise.all(inputPaths.map(readDownloadedGames))
  ).flat();
  for (let index = 0; index < blobs.length; index += 12) {
    games.push(
      ...(await Promise.all(blobs.slice(index, index + 12).map(readGame)))
    );
  }
  const generationIds = [
    ...new Set([
      ...requestedGenerationIds,
      ...games.map((game) => game.generationId),
    ]),
  ].sort();

  const losses = games.flatMap((game) => {
    if (
      game.outcome.termination !== "hq-capture" ||
      !game.outcome.winner ||
      game.decisions.length < 2
    ) {
      return [];
    }
    const losingDecision = game.decisions.at(-2)!;
    const winningDecision = game.decisions.at(-1)!;
    if (
      losingDecision.player === game.outcome.winner ||
      winningDecision.player !== game.outcome.winner ||
      winningDecision.turnNumber !== losingDecision.turnNumber + 1
    ) {
      return [];
    }
    return [{ game, losingDecision, winningDecision }];
  });

  const audits = [];
  for (const { game, losingDecision, winningDecision } of losses) {
    const exact = await exactAudit(losingDecision.fen, maxNodes);
    audits.push({
      gameId: game.gameId,
      winner: game.outcome.winner,
      losingTurn: losingDecision.turnNumber,
      losingPlayer: losingDecision.player,
      selectedMoves: losingDecision.selectedMoves,
      winningMoves: winningDecision.selectedMoves,
      searchScore: losingDecision.currentPlayerScore,
      losingWinProbability: losingDecision.winProbability,
      completedDepth: losingDecision.completedDepth,
      ...exact,
    });
  }

  const avoidable = audits.filter((audit) => audit.safe_turns > 0);
  const forced = audits.filter((audit) => audit.forced_hq_loss);
  const inconclusive = audits.filter((audit) => audit.inconclusive);
  const report = {
    format: "ghq-exact-hq-audit-v1",
    generationIds,
    codeVersions: [
      ...new Set(games.map((game) => game.codeVersion ?? "unknown")),
    ].sort(),
    games: games.length,
    immediateHqLosses: audits.length,
    forcedHqLosses: forced.length,
    avoidableHqLosses: avoidable.length,
    inconclusiveHqLosses: inconclusive.length,
    maxNodesPerAudit: maxNodes,
    approvedTrainingGameIds: forced.map((audit) => audit.gameId).sort(),
    rejectedAvoidableGameIds: avoidable.map((audit) => audit.gameId).sort(),
    rejectedInconclusiveGameIds: inconclusive
      .map((audit) => audit.gameId)
      .sort(),
    audits,
  };
  const rendered = `${JSON.stringify(report, null, 2)}\n`;
  const outputPath = argumentsFor("--output").at(-1);
  if (outputPath) await writeFile(outputPath, rendered, "utf8");
  process.stdout.write(rendered);
  if (process.argv.includes("--fail-on-avoidable") && avoidable.length) {
    process.exitCode = 2;
  }
  if (process.argv.includes("--fail-on-inconclusive") && inconclusive.length) {
    process.exitCode = 3;
  }
}

void main();
