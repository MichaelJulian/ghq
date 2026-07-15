import { NextResponse } from "next/server";
import { getRun } from "workflow/api";
import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET(
  _request: Request,
  { params }: { params: Promise<{ runId: string }> }
) {
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
