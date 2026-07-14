import { NextResponse } from "next/server";
import { getRun } from "workflow/api";
import { selfPlayAuthorized } from "@/server/self-play-auth";
import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  request: Request,
  { params }: { params: Promise<{ runId: string }> }
) {
  if (!selfPlayAuthorized(request)) {
    return NextResponse.json(
      { error: "Self-play authorization is not configured or is invalid" },
      { status: process.env.SELF_PLAY_SECRET ? 401 : 503 }
    );
  }
  const { runId } = await params;
  const run = getRun<DurableSelfPlayGameResult>(runId);
  if (!(await run.exists)) {
    return NextResponse.json(
      { error: "Workflow run not found" },
      { status: 404 }
    );
  }
  const status = await run.status;
  return NextResponse.json({
    runId,
    status,
    workflowName: await run.workflowName,
    createdAt: (await run.createdAt).toISOString(),
    startedAt: (await run.startedAt)?.toISOString(),
    completedAt: (await run.completedAt)?.toISOString(),
    result: status === "completed" ? await run.returnValue : undefined,
  });
}
