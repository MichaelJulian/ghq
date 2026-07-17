import { NextResponse } from "next/server";
import { start } from "workflow/api";
import { PERSONALITIES } from "@/game/value-model/personalities";
import type { PersonalityId } from "@/game/analysis/types";
import { scheduleDurableCompetitors } from "@/game/self-play/durable-schedule";
import { persistSelfPlayGenerationManifest } from "@/server/self-play-storage";
import {
  playDurableSelfPlayGame,
  type DurableSelfPlayCompetitor,
} from "@/workflows/self-play-game";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

interface StartBatchRequest {
  games?: number;
  seed?: number;
  timeMs?: number;
  maxDepth?: number;
  beamWidth?: number;
  maxTurns?: number;
  repetitionLimit?: number;
  noProgressTurns?: number;
  redMaxActions?: number;
  blueMaxActions?: number;
  personalities?: PersonalityId[];
  /** Run color-swapped incumbent-vs-challenger value-model pairs. */
  valueModelArena?: boolean;
}

function integer(
  value: number | undefined,
  fallback: number,
  minimum: number,
  maximum: number,
  label: string
): number {
  const resolved = value ?? fallback;
  if (
    !Number.isSafeInteger(resolved) ||
    resolved < minimum ||
    resolved > maximum
  ) {
    throw new RangeError(
      `${label} must be an integer from ${minimum} to ${maximum}`
    );
  }
  return resolved;
}

export async function POST(request: Request) {
  try {
    const input = (await request.json()) as StartBatchRequest;
    const games = integer(input.games, 12, 1, 100, "games");
    const seed = integer(input.seed, 20260714, 0, 0xffff_ffff, "seed") >>> 0;
    const timeMs = integer(input.timeMs, 20_000, 50, 30_000, "timeMs");
    const maxDepth = integer(input.maxDepth, 2, 1, 3, "maxDepth");
    const beamWidth = integer(input.beamWidth, 6, 2, 16, "beamWidth");
    const maxTurns = integer(input.maxTurns, 160, 4, 400, "maxTurns");
    const repetitionLimit = integer(
      input.repetitionLimit,
      3,
      2,
      10,
      "repetitionLimit"
    );
    const noProgressTurns = integer(
      input.noProgressTurns,
      36,
      4,
      100,
      "noProgressTurns"
    );
    const redMaxActions = integer(
      input.redMaxActions,
      3,
      2,
      3,
      "redMaxActions"
    ) as 2 | 3;
    const blueMaxActions = integer(
      input.blueMaxActions,
      3,
      2,
      3,
      "blueMaxActions"
    ) as 2 | 3;
    const valueModelArena = input.valueModelArena === true;
    if (valueModelArena && games % 2 !== 0) {
      throw new RangeError("valueModelArena requires an even number of games");
    }
    const personalityIds =
      input.personalities ?? (Object.keys(PERSONALITIES) as PersonalityId[]);
    if (
      personalityIds.length < 2 ||
      personalityIds.some((id) => !(id in PERSONALITIES))
    ) {
      throw new RangeError(
        "personalities must contain at least two known personalities"
      );
    }
    const competitors: DurableSelfPlayCompetitor[] = personalityIds.map(
      (id) => ({
        id: `${id}-workflow-g0`,
        personality: id,
        timeMs,
        maxDepth,
        beamWidth,
        explorationTemperature: Math.max(
          0.18,
          PERSONALITIES[id].explorationTemperature
        ),
      })
    );
    const generationId = `vercel-${
      valueModelArena ? "arena-" : ""
    }r${redMaxActions}b${blueMaxActions}-${seed.toString(16)}-${Date.now().toString(
      36
    )}`;
    const codeVersion = process.env.VERCEL_GIT_COMMIT_SHA ?? "local";
    const runs = await Promise.all(
      Array.from({ length: games }, async (_, index) => {
        // Adjacent games form a controlled color-swapped pair: same matchup and
        // random seed, with only the personalities' colors reversed.
        const pairIndex = Math.floor(index / 2);
        const { red, blue } = scheduleDurableCompetitors({
          index,
          competitors,
          redMaxActions,
          blueMaxActions,
          valueModelArena,
        });
        const gameId = `${generationId}-${String(index + 1).padStart(4, "0")}`;
        const gameSeed = (seed + Math.imul(pairIndex + 1, 0x85ebca6b)) >>> 0;
        const run = await start(playDurableSelfPlayGame, [
          {
            generationId,
            gameId,
            seed: gameSeed,
            red,
            blue,
            maxTurns,
            repetitionLimit,
            noProgressTurns,
            codeVersion,
          },
        ]);
        return {
          gameId,
          runId: run.runId,
          red: red.id,
          blue: blue.id,
          redValueModel: red.valueModel ?? "incumbent",
          blueValueModel: blue.valueModel ?? "incumbent",
          redValueModelCheckpoint: red.valueModelCheckpoint ?? "unknown",
          blueValueModelCheckpoint: blue.valueModelCheckpoint ?? "unknown",
        };
      })
    );
    const checkpoints = {
      incumbentCheckpoints: new Set<string>(),
      challengerCheckpoints: new Set<string>(),
    };
    for (const run of runs) {
      for (const [model, checkpoint] of [
        [run.redValueModel, run.redValueModelCheckpoint],
        [run.blueValueModel, run.blueValueModelCheckpoint],
      ] as const) {
        checkpoints[
          model === "challenger"
            ? "challengerCheckpoints"
            : "incumbentCheckpoints"
        ].add(checkpoint);
      }
    }
    let manifestStorage: "saved" | "not-configured" | "failed";
    try {
      manifestStorage = await persistSelfPlayGenerationManifest({
        format: "ghq-self-play-generation-manifest-v1",
        generationId,
        createdAt: new Date().toISOString(),
        expectedGames: games,
        codeVersion,
        valueModelArena,
        settings: {
          timeMs,
          maxDepth,
          beamWidth,
          maxTurns,
          repetitionLimit,
          noProgressTurns,
          redMaxActions,
          blueMaxActions,
          seed,
        },
        expectedProvenance: {
          incumbentCheckpoints: [...checkpoints.incumbentCheckpoints].sort(),
          challengerCheckpoints: [...checkpoints.challengerCheckpoints].sort(),
        },
        runs: runs.map((run) => ({
          gameId: run.gameId,
          runId: run.runId,
          redAgentId: run.red,
          blueAgentId: run.blue,
        })),
      });
    } catch (error) {
      manifestStorage = "failed";
      console.error("Unable to persist self-play generation manifest", error);
    }
    return NextResponse.json(
      {
        generationId,
        games,
        redMaxActions,
        blueMaxActions,
        pairedColorSwap: true,
        valueModelArena,
        manifestStorage,
        runs,
      },
      { status: 202 }
    );
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error ? error.message : "Unable to start self-play",
      },
      { status: error instanceof RangeError ? 400 : 500 }
    );
  }
}
