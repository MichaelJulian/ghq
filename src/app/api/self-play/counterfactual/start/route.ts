import { NextResponse } from "next/server";
import { start } from "workflow/api";
import { PERSONALITIES } from "@/game/value-model/personalities";
import type { PersonalityId } from "@/game/analysis/types";
import { FENtoBoardState } from "@/game/notation";
import {
  counterfactualReplicateSeed,
  type CounterfactualBranch,
} from "@/game/self-play/counterfactual";
import { valueModelCheckpointId } from "@/game/value-model/inference";
import {
  persistSelfPlayGenerationManifest,
  type SelfPlayGenerationManifest,
} from "@/server/self-play-storage";
import {
  playDurableSelfPlayGame,
  type DurableSelfPlayCompetitor,
} from "@/workflows/self-play-game";

export const dynamic = "force-dynamic";
export const runtime = "nodejs";
export const maxDuration = 60;

interface CounterfactualStartRequest {
  sourceGenerationId?: string;
  seed?: number;
  timeMs?: number;
  maxDepth?: number;
  beamWidth?: number;
  rolloutTurns?: number;
  replicates?: number;
  explorationTemperature?: number;
  repetitionLimit?: number;
  noProgressTurns?: number;
  branches?: CounterfactualBranch[];
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

function nonEmptyString(value: unknown, label: string, maximum = 256): string {
  if (
    typeof value !== "string" ||
    value.trim().length === 0 ||
    value.length > maximum
  ) {
    throw new RangeError(`${label} must be a non-empty string`);
  }
  return value;
}

function requiredInteger(
  value: number | undefined,
  minimum: number,
  maximum: number,
  label: string
): number {
  if (value === undefined) {
    throw new RangeError(`${label} is required`);
  }
  return integer(value, value, minimum, maximum, label);
}

function personality(value: unknown, label: string): PersonalityId {
  if (typeof value !== "string" || !(value in PERSONALITIES)) {
    throw new RangeError(`${label} must be a known personality`);
  }
  return value as PersonalityId;
}

function validatedBranches(input: unknown): CounterfactualBranch[] {
  if (!Array.isArray(input) || input.length < 2 || input.length > 32) {
    throw new RangeError("branches must contain from 2 to 32 entries");
  }
  const branches = input.map((raw, index) => {
    if (!raw || typeof raw !== "object") {
      throw new RangeError(`branches[${index}] must be an object`);
    }
    const value = raw as Partial<CounterfactualBranch>;
    const initialFen = nonEmptyString(
      value.initialFen,
      `branches[${index}].initialFen`,
      1024
    );
    const rootPlayer = value.rootPlayer;
    if (rootPlayer !== "RED" && rootPlayer !== "BLUE") {
      throw new RangeError(`branches[${index}].rootPlayer must be RED or BLUE`);
    }
    const sideToMove = FENtoBoardState(initialFen).currentPlayerTurn;
    if (sideToMove === rootPlayer) {
      throw new RangeError(
        `branches[${index}] must start after the root player's completed turn`
      );
    }
    const candidateScore = Number(value.candidateScore);
    if (
      !Number.isFinite(candidateScore) ||
      Math.abs(candidateScore) >= 1_000_000
    ) {
      throw new RangeError(
        `branches[${index}].candidateScore must be a finite non-mate score`
      );
    }
    if (
      !Array.isArray(value.candidateMoves) ||
      value.candidateMoves.some((move) => typeof move !== "string")
    ) {
      throw new RangeError(
        `branches[${index}].candidateMoves must be a string array`
      );
    }
    return {
      rootId: nonEmptyString(value.rootId, `branches[${index}].rootId`),
      sourceGameId: nonEmptyString(
        value.sourceGameId,
        `branches[${index}].sourceGameId`
      ),
      sourceTurnNumber: requiredInteger(
        value.sourceTurnNumber,
        1,
        399,
        `branches[${index}].sourceTurnNumber`
      ),
      rootPlayer,
      candidateRank: requiredInteger(
        value.candidateRank,
        1,
        64,
        `branches[${index}].candidateRank`
      ),
      candidateScore,
      candidateMoves: [...value.candidateMoves],
      initialFen,
      initialTurnNumber: requiredInteger(
        value.initialTurnNumber,
        2,
        400,
        `branches[${index}].initialTurnNumber`
      ),
      redPersonality: personality(
        value.redPersonality,
        `branches[${index}].redPersonality`
      ),
      bluePersonality: personality(
        value.bluePersonality,
        `branches[${index}].bluePersonality`
      ),
    } satisfies CounterfactualBranch;
  });

  const roots = new Map<string, CounterfactualBranch[]>();
  const uniqueBranches = new Set<string>();
  for (const branch of branches) {
    const key = `${branch.rootId}:${branch.candidateRank}`;
    if (uniqueBranches.has(key)) {
      throw new RangeError(`duplicate counterfactual branch ${key}`);
    }
    uniqueBranches.add(key);
    const siblings = roots.get(branch.rootId) ?? [];
    siblings.push(branch);
    roots.set(branch.rootId, siblings);
  }
  for (const [rootId, siblings] of roots) {
    if (siblings.length < 2) {
      throw new RangeError(`counterfactual root ${rootId} needs two branches`);
    }
    const first = siblings[0];
    if (
      siblings.some(
        (branch) =>
          branch.sourceGameId !== first.sourceGameId ||
          branch.sourceTurnNumber !== first.sourceTurnNumber ||
          branch.rootPlayer !== first.rootPlayer ||
          branch.initialTurnNumber !== first.initialTurnNumber ||
          branch.redPersonality !== first.redPersonality ||
          branch.bluePersonality !== first.bluePersonality
      )
    ) {
      throw new RangeError(`counterfactual root ${rootId} has mixed metadata`);
    }
  }
  return branches;
}

function competitor(
  personalityId: PersonalityId,
  timeMs: number,
  maxDepth: number,
  beamWidth: number,
  explorationTemperature: number
): DurableSelfPlayCompetitor {
  return {
    id: `${personalityId}-counterfactual-incumbent-a3`,
    personality: personalityId,
    timeMs,
    maxDepth,
    beamWidth,
    // Candidate branches use the same random seed within each replicate, so
    // optional exploration remains a matched source of variation.
    explorationTemperature,
    maxActions: 3,
    valueModel: "incumbent",
    valueModelCheckpoint: valueModelCheckpointId("three-actions", "incumbent"),
  };
}

export async function POST(request: Request) {
  try {
    const input = (await request.json()) as CounterfactualStartRequest;
    const branches = validatedBranches(input.branches);
    const seed = integer(input.seed, 20260718, 0, 0xffff_ffff, "seed") >>> 0;
    const timeMs = integer(input.timeMs, 20_000, 50, 30_000, "timeMs");
    const maxDepth = integer(input.maxDepth, 2, 1, 3, "maxDepth");
    const beamWidth = integer(input.beamWidth, 6, 2, 16, "beamWidth");
    const rolloutTurns = integer(
      input.rolloutTurns,
      24,
      2,
      120,
      "rolloutTurns"
    );
    const replicates = integer(input.replicates, 1, 1, 4, "replicates");
    const explorationTemperature = Number(
      input.explorationTemperature ?? 0
    );
    if (
      !Number.isFinite(explorationTemperature) ||
      explorationTemperature < 0 ||
      explorationTemperature > 0.5
    ) {
      throw new RangeError(
        "explorationTemperature must be between zero and 0.5"
      );
    }
    const repetitionLimit = integer(
      input.repetitionLimit,
      3,
      2,
      10,
      "repetitionLimit"
    );
    const noProgressTurns = integer(
      input.noProgressTurns,
      24,
      4,
      100,
      "noProgressTurns"
    );
    if (
      branches.some(
        (branch) => branch.initialTurnNumber + rolloutTurns - 1 > 400
      )
    ) {
      throw new RangeError(
        "initialTurnNumber plus rolloutTurns may not exceed turn 400"
      );
    }
    if (branches.length * replicates > 32) {
      throw new RangeError("branches times replicates may not exceed 32 runs");
    }

    const generationId = `vercel-cf-r3b3-${seed.toString(
      16
    )}-${Date.now().toString(36)}`;
    const codeVersion = process.env.VERCEL_GIT_COMMIT_SHA ?? "local";
    const runSpecs = branches.flatMap((branch) =>
      Array.from({ length: replicates }, (_, replicate) => ({
        branch,
        replicate,
      }))
    );
    const runs = await Promise.all(
      runSpecs.map(async ({ branch, replicate }, index) => {
        const red = competitor(
          branch.redPersonality,
          timeMs,
          maxDepth,
          beamWidth,
          explorationTemperature
        );
        const blue = competitor(
          branch.bluePersonality,
          timeMs,
          maxDepth,
          beamWidth,
          explorationTemperature
        );
        const gameId = `${generationId}-${String(index + 1).padStart(4, "0")}`;
        const run = await start(playDurableSelfPlayGame, [
          {
            generationId,
            gameId,
            seed: counterfactualReplicateSeed(seed, branch.rootId, replicate),
            red,
            blue,
            initialFen: branch.initialFen,
            initialTurnNumber: branch.initialTurnNumber,
            dataRole: "counterfactual",
            maxTurns: branch.initialTurnNumber + rolloutTurns - 1,
            repetitionLimit,
            noProgressTurns,
            codeVersion,
          },
        ]);
        return { gameId, runId: run.runId, red, blue, branch, replicate };
      })
    );

    const checkpoint = valueModelCheckpointId("three-actions", "incumbent");
    const manifest: SelfPlayGenerationManifest = {
      format: "ghq-self-play-generation-manifest-v1",
      generationId,
      createdAt: new Date().toISOString(),
      expectedGames: runs.length,
      codeVersion,
      valueModelArena: false,
      settings: {
        timeMs,
        maxDepth,
        beamWidth,
        maxTurns: Math.max(
          ...branches.map(
            (branch) => branch.initialTurnNumber + rolloutTurns - 1
          )
        ),
        repetitionLimit,
        noProgressTurns,
        redMaxActions: 3,
        blueMaxActions: 3,
        seed,
        explorationTemperature,
      },
      expectedProvenance: {
        incumbentCheckpoints: [checkpoint],
        challengerCheckpoints: [],
      },
      runs: runs.map(({ gameId, runId, red, blue }) => ({
        gameId,
        runId,
        redAgentId: red.id,
        blueAgentId: blue.id,
      })),
      counterfactual: {
        sourceGenerationId: input.sourceGenerationId,
        rolloutTurns,
        replicates,
        explorationTemperature,
        branches: runs.map(({ gameId, branch, replicate }) => ({
          gameId,
          rootId: branch.rootId,
          sourceGameId: branch.sourceGameId,
          sourceTurnNumber: branch.sourceTurnNumber,
          rootPlayer: branch.rootPlayer,
          candidateRank: branch.candidateRank,
          candidateScore: branch.candidateScore,
          candidateMoves: branch.candidateMoves,
          initialFen: branch.initialFen,
          initialTurnNumber: branch.initialTurnNumber,
          replicate,
        })),
      },
    };
    let manifestStorage: "saved" | "not-configured" | "failed";
    try {
      manifestStorage = await persistSelfPlayGenerationManifest(manifest);
    } catch (error) {
      manifestStorage = "failed";
      console.error("Unable to persist counterfactual manifest", error);
    }
    return NextResponse.json(
      {
        generationId,
        games: runs.length,
        roots: new Set(branches.map((branch) => branch.rootId)).size,
        rolloutTurns,
        replicates,
        explorationTemperature,
        manifestStorage,
        runs: runs.map(({ gameId, runId, branch, replicate }) => ({
          gameId,
          runId,
          rootId: branch.rootId,
          candidateRank: branch.candidateRank,
          replicate,
        })),
      },
      { status: 202 }
    );
  } catch (error) {
    return NextResponse.json(
      {
        error:
          error instanceof Error
            ? error.message
            : "Unable to start counterfactual self-play",
      },
      { status: error instanceof RangeError ? 400 : 500 }
    );
  }
}
