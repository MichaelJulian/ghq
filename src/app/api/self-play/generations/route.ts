import { NextResponse } from "next/server";
import {
  listPersistedSelfPlayGenerations,
  selfPlayStorageConfigured,
} from "@/server/self-play-storage";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";

export async function GET() {
  try {
    const configured = selfPlayStorageConfigured();
    return NextResponse.json({
      configured,
      generations: configured ? await listPersistedSelfPlayGenerations() : [],
    });
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to list self-play generations",
      },
      { status: 500 }
    );
  }
}
