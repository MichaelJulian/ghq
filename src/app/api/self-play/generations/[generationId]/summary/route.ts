import { NextResponse } from "next/server";
import { getRun } from "workflow/api";
import { summarizeValueModelArena } from "@/game/self-play/arena-results";
import {
  readPersistedSelfPlayGames,
  readSelfPlayGenerationManifest,
} from "@/server/self-play-storage";
import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

function increment(counts: Record<string, number>, key: string) {
  counts[key] = (counts[key] ?? 0) + 1;
}

async function unresolvedWorkflowRunStatuses(
  runIds: string[]
): Promise<Record<string, number>> {
  const statuses: Record<string, number> = {};
  for (let index = 0; index < runIds.length; index += 20) {
    const chunk = await Promise.all(
      runIds.slice(index, index + 20).map(async (runId) => {
        try {
          const run = getRun(runId);
          if (!(await run.exists)) return "missing";
          return await run.status;
        } catch {
          return "unavailable";
        }
      })
    );
    for (const status of chunk) increment(statuses, status);
  }
  return statuses;
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ generationId: string }> }
) {
  try {
    const { generationId } = await params;
    const [games, manifest] = await Promise.all([
      readPersistedSelfPlayGames<DurableSelfPlayGameResult>(generationId),
      readSelfPlayGenerationManifest(generationId),
    ]);
    const persistedGameIds = new Set(games.map((game) => game.gameId));
    const unresolvedRunIds =
      manifest?.runs
        .filter((run) => !persistedGameIds.has(run.gameId))
        .map((run) => run.runId) ?? [];
    const workflowRunStatuses = manifest
      ? await unresolvedWorkflowRunStatuses(unresolvedRunIds)
      : undefined;
    if (workflowRunStatuses && games.length > 0) {
      workflowRunStatuses.completed =
        (workflowRunStatuses.completed ?? 0) + games.length;
    }
    const outcomes: Record<string, number> = {};
    const terminations: Record<string, number> = {};
    let decisions = 0;
    let fallbackDecisions = 0;
    let unverifiedFallbackDecisions = 0;
    let timedOutDecisions = 0;
    let persistentCacheHits = 0;
    const codeVersions = new Set<string>();
    const valueModelCheckpoints = new Set<string>();
    for (const game of games) {
      increment(outcomes, game.outcome.winner ?? "DRAW");
      increment(terminations, game.outcome.termination);
      decisions += game.decisions.length;
      fallbackDecisions += game.quality.fallbackDecisions;
      timedOutDecisions += game.quality.timedOutDecisions;
      persistentCacheHits += game.decisions.filter(
        (decision) => decision.persistentCacheHit
      ).length;
      unverifiedFallbackDecisions +=
        game.quality.unverifiedFallbackDecisions ??
        game.decisions.filter(
          (decision) =>
            decision.fallback === "seeded" ||
            (decision.fallback !== "none" && decision.completedDepth < 2)
        ).length;
      if (game.codeVersion && game.codeVersion !== "unknown") {
        codeVersions.add(game.codeVersion);
      }
      for (const checkpoint of [
        game.redValueModelCheckpoint,
        game.blueValueModelCheckpoint,
      ]) {
        if (checkpoint && checkpoint !== "unknown") {
          valueModelCheckpoints.add(checkpoint);
        }
      }
    }
    return NextResponse.json({
      generationId,
      games: games.length,
      expectedGames: manifest?.expectedGames,
      remainingGames:
        manifest === undefined
          ? undefined
          : Math.max(0, manifest.expectedGames - games.length),
      createdAt: manifest?.createdAt,
      manifestStorage: manifest ? "saved" : "historical-or-missing",
      workflowRuns: manifest
        ? {
            total: manifest.runs.length,
            statuses: workflowRunStatuses,
          }
        : undefined,
      outcomes,
      terminations,
      decisions,
      fallbackRate: decisions ? fallbackDecisions / decisions : 0,
      unverifiedFallbackRate: decisions
        ? unverifiedFallbackDecisions / decisions
        : 0,
      timedOutRate: decisions ? timedOutDecisions / decisions : 0,
      persistentCacheHitRate: decisions ? persistentCacheHits / decisions : 0,
      provenance: {
        codeVersions: [...codeVersions].sort(),
        valueModelCheckpoints: [...valueModelCheckpoints].sort(),
      },
      valueModelArena: summarizeValueModelArena(
        games,
        5_000,
        manifest?.valueModelArena
          ? {
              generationId: manifest.generationId,
              codeVersion: manifest.codeVersion,
              incumbentCheckpoints:
                manifest.expectedProvenance.incumbentCheckpoints,
              challengerCheckpoints:
                manifest.expectedProvenance.challengerCheckpoints,
            }
          : undefined
      ),
    });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to summarize generation",
      },
      { status: 500 }
    );
  }
}
