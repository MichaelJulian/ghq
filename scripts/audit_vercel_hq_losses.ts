#!/usr/bin/env tsx
/** Exhaustively classify immediate HQ losses in persisted Vercel self-play. */

import { execFile } from "node:child_process";
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

async function exactAudit(fen: string): Promise<ExactHqAudit> {
  const python = process.env.GHQ_PYTHON ?? ".venv/bin/python";
  const { stdout } = await execFileAsync(
    python,
    ["scripts/audit_hq_loss.py", fen],
    { maxBuffer: 1024 * 1024 }
  );
  return JSON.parse(stdout) as ExactHqAudit;
}

async function main() {
  const generationIds = argumentsFor("--generation");
  if (!generationIds.length) {
    throw new Error("Pass at least one --generation <id>");
  }
  const blobs = (await Promise.all(generationIds.map(gameBlobs))).flat();
  const games: DurableSelfPlayGameResult[] = [];
  for (let index = 0; index < blobs.length; index += 12) {
    games.push(
      ...(await Promise.all(blobs.slice(index, index + 12).map(readGame)))
    );
  }

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
    const exact = await exactAudit(losingDecision.fen);
    audits.push({
      gameId: game.gameId,
      winner: game.outcome.winner,
      losingTurn: losingDecision.turnNumber,
      losingPlayer: losingDecision.player,
      selectedMoves: losingDecision.selectedMoves,
      winningMoves: winningDecision.selectedMoves,
      searchScore: losingDecision.currentPlayerScore,
      completedDepth: losingDecision.completedDepth,
      ...exact,
    });
  }

  const avoidable = audits.filter((audit) => !audit.forced_hq_loss);
  console.log(
    JSON.stringify(
      {
        generationIds,
        games: games.length,
        immediateHqLosses: audits.length,
        forcedHqLosses: audits.length - avoidable.length,
        avoidableHqLosses: avoidable.length,
        audits,
      },
      null,
      2
    )
  );
  if (process.argv.includes("--fail-on-avoidable") && avoidable.length) {
    process.exitCode = 2;
  }
}

void main();
