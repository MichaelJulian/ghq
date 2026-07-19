import { NextResponse } from "next/server";
import { readPersistedSelfPlayGame } from "@/server/self-play-storage";
import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

export async function GET(
  _request: Request,
  {
    params,
  }: {
    params: Promise<{ generationId: string; gameId: string }>;
  }
) {
  try {
    const { generationId, gameId } = await params;
    const game = await readPersistedSelfPlayGame<DurableSelfPlayGameResult>(
      generationId,
      gameId
    );
    if (!game) {
      return NextResponse.json(
        { error: "Persisted self-play game not found" },
        { status: 404 }
      );
    }
    return NextResponse.json(game, {
      headers: {
        "Cache-Control": "private, no-store",
      },
    });
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unable to read self-play game";
    return NextResponse.json(
      { error: message },
      { status: message.includes("does not belong") ? 400 : 500 }
    );
  }
}
