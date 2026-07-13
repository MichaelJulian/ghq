import { NextResponse } from "next/server";
import type { FenAnalysisRequest } from "@/game/analysis/types";
import { AnalysisInputError, analyzeFen } from "@/server/fen-analysis";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 15;

export async function POST(request: Request) {
  try {
    const input = (await request.json()) as FenAnalysisRequest;
    return NextResponse.json(await analyzeFen(input));
  } catch (error) {
    const message =
      error instanceof Error ? error.message : "Unable to analyze position";
    return NextResponse.json(
      { error: message },
      { status: error instanceof AnalysisInputError ? 400 : 500 }
    );
  }
}
