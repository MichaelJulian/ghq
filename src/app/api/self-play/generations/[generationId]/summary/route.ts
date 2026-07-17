import { NextResponse } from "next/server";
import { summarizeValueModelArena } from "@/game/self-play/arena-results";
import { readPersistedSelfPlayGames } from "@/server/self-play-storage";
import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

function increment(counts: Record<string, number>, key: string) {
  counts[key] = (counts[key] ?? 0) + 1;
}

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ generationId: string }> }
) {
  try {
    const { generationId } = await params;
    const games = await readPersistedSelfPlayGames<DurableSelfPlayGameResult>(
      generationId
    );
    const outcomes: Record<string, number> = {};
    const terminations: Record<string, number> = {};
    let decisions = 0;
    let fallbackDecisions = 0;
    let unverifiedFallbackDecisions = 0;
    for (const game of games) {
      increment(outcomes, game.outcome.winner ?? "DRAW");
      increment(terminations, game.outcome.termination);
      decisions += game.decisions.length;
      fallbackDecisions += game.quality.fallbackDecisions;
      unverifiedFallbackDecisions +=
        game.quality.unverifiedFallbackDecisions ??
        game.decisions.filter(
          (decision) =>
            decision.fallback === "seeded" ||
            (decision.fallback !== "none" && decision.completedDepth < 2)
        ).length;
    }
    return NextResponse.json({
      generationId,
      games: games.length,
      outcomes,
      terminations,
      decisions,
      fallbackRate: decisions ? fallbackDecisions / decisions : 0,
      unverifiedFallbackRate: decisions
        ? unverifiedFallbackDecisions / decisions
        : 0,
      valueModelArena: summarizeValueModelArena(games),
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
