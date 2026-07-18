#!/usr/bin/env tsx
/** Re-run a decrypted durable-turn input read from stdin with the current engine. */

import type { FenAnalysisRequest } from "../src/game/analysis/types";
import type { PersonalityId } from "../src/game/value-model/personalities";
import { analyzeFen } from "../src/server/fen-analysis";

interface DurableTurnInput {
  fen: string;
  serializedState?: string;
  turnNumber: number;
  competitor: {
    personality: PersonalityId;
    timeMs: number;
    maxDepth: number;
    beamWidth: number;
    maxActions?: 2 | 3;
    valueModel?: "incumbent" | "challenger";
    explorationTemperature?: number;
  };
  explorationSeed?: number;
  recentFens?: string[];
  previousOwnTurnMoves?: string[];
  previousOwnTurns?: string[][];
  turnsWithoutProgress?: number;
}

async function readStdin(): Promise<string> {
  const chunks: Buffer[] = [];
  for await (const chunk of process.stdin) {
    chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
  }
  return Buffer.concat(chunks).toString("utf8");
}

function analysisRequest(input: DurableTurnInput): FenAnalysisRequest {
  return {
    fen: input.fen,
    serializedState: input.serializedState,
    turnNumber: input.turnNumber,
    personality: input.competitor.personality,
    timeMs: input.competitor.timeMs,
    maxDepth: input.competitor.maxDepth,
    beamWidth: input.competitor.beamWidth,
    maxActions: input.competitor.maxActions ?? 3,
    valueModel: input.competitor.valueModel ?? "incumbent",
    explorationTemperature: input.competitor.explorationTemperature ?? 0,
    explorationSeed: input.explorationSeed,
    recentFens: input.recentFens,
    previousOwnTurnMoves: input.previousOwnTurnMoves,
    previousOwnTurns: input.previousOwnTurns,
    turnsWithoutProgress: input.turnsWithoutProgress,
  };
}

async function main() {
  const raw = await readStdin();
  if (!raw.trim())
    throw new Error("Expected a durable-turn input JSON object on stdin");
  const input = JSON.parse(raw) as DurableTurnInput;
  const replay = await analyzeFen(analysisRequest(input));
  console.log(
    JSON.stringify(
      {
        turn: input.turnNumber,
        player: replay.sideToMove,
        personality: replay.personality,
        fen: replay.fen,
        moves: replay.search.best_turn.all_moves,
        principalVariation: replay.search.principal_variation,
        recommendation: replay.search.recommendation_label,
        depth: replay.search.search.completed_depth_in_turns,
        fallback: replay.search.search.fallback_used,
        timedOut: replay.search.search.timed_out,
        elapsedMs: replay.search.search.elapsed_ms,
        nodes: replay.search.search.nodes,
        backend: replay.search.search.backend,
        valueModelBackend: replay.search.search.value_model_backend,
        tacticalReturnGuardUsed:
          replay.search.search.tactical_return_guard_used ?? false,
        safeFallbackReplyVerified:
          replay.search.search.safe_fallback_reply_verified ?? false,
        safeFallbackReplyNodes:
          replay.search.search.safe_fallback_reply_nodes ?? 0,
        seedReplyVerified: replay.search.search.seed_reply_verified ?? false,
        hqSurvivalReplyVerified:
          replay.search.search.hq_survival_reply_verified ?? false,
        codeVersion: replay.search.search.code_version,
      },
      null,
      2
    )
  );
}

void main();
