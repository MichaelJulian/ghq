import { NextResponse } from "next/server";
import { getRun } from "workflow/api";
import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

function workflowFailure(error: unknown): { name: string; message: string } {
  if (error instanceof Error) {
    return { name: error.name, message: error.message };
  }
  return { name: "WorkflowError", message: String(error) };
}

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
  let result: DurableSelfPlayGameResult | undefined;
  let failure: { name: string; message: string } | undefined;
  if (status === "completed") {
    result = await run.returnValue;
  } else if (status === "failed") {
    try {
      await run.returnValue;
    } catch (error) {
      failure = workflowFailure(error);
    }
  }
  return NextResponse.json({
    runId,
    status,
    workflowName: await run.workflowName,
    createdAt: (await run.createdAt).toISOString(),
    startedAt: (await run.startedAt)?.toISOString(),
    completedAt: (await run.completedAt)?.toISOString(),
    result,
    failure,
  });
}
