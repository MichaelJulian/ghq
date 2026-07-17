import { get, list, put, type ListBlobResultBlob } from "@vercel/blob";

export const SELF_PLAY_BLOB_PREFIX = "self-play/generations/";

export interface PersistedSelfPlayArtifacts {
  status: "saved" | "not-configured";
  gamePathname?: string;
  trainingPathname?: string;
  trainingSamples: number;
}

export interface SelfPlayGenerationManifest {
  format: "ghq-self-play-generation-manifest-v1";
  generationId: string;
  createdAt: string;
  expectedGames: number;
  codeVersion: string;
  valueModelArena: boolean;
  settings: {
    timeMs: number;
    maxDepth: number;
    beamWidth: number;
    maxTurns: number;
    repetitionLimit: number;
    noProgressTurns: number;
    redMaxActions: number;
    blueMaxActions: number;
    seed: number;
  };
  expectedProvenance: {
    incumbentCheckpoints: string[];
    challengerCheckpoints: string[];
  };
  runs: Array<{
    gameId: string;
    runId: string;
    redAgentId: string;
    blueAgentId: string;
  }>;
}

export function selfPlayStorageConfigured(): boolean {
  return Boolean(
    process.env.BLOB_STORE_ID || process.env.BLOB_READ_WRITE_TOKEN
  );
}

function safeSegment(value: string): string {
  const safe = value.replace(/[^a-zA-Z0-9._-]/g, "-");
  if (!safe || safe === "." || safe === "..") {
    throw new Error("Invalid self-play storage identifier");
  }
  return safe;
}

function generationManifestPathname(generationId: string): string {
  return `${SELF_PLAY_BLOB_PREFIX}${safeSegment(generationId)}/manifest.json`;
}

export async function persistSelfPlayGenerationManifest(
  manifest: SelfPlayGenerationManifest
): Promise<"saved" | "not-configured"> {
  if (!selfPlayStorageConfigured()) return "not-configured";
  await put(
    generationManifestPathname(manifest.generationId),
    JSON.stringify(manifest),
    {
      access: "private",
      addRandomSuffix: false,
      contentType: "application/json",
    }
  );
  return "saved";
}

export async function readSelfPlayGenerationManifest(
  generationId: string
): Promise<SelfPlayGenerationManifest | undefined> {
  if (!selfPlayStorageConfigured()) return undefined;
  const response = await get(generationManifestPathname(generationId), {
    access: "private",
    useCache: false,
  });
  if (!response?.stream || response.statusCode !== 200) return undefined;
  return (await new Response(response.stream).json()) as SelfPlayGenerationManifest;
}

export async function persistSelfPlayArtifacts(input: {
  generationId: string;
  gameId: string;
  game: unknown;
  trainingSamples: unknown[];
}): Promise<PersistedSelfPlayArtifacts> {
  if (!selfPlayStorageConfigured()) {
    return {
      status: "not-configured",
      trainingSamples: input.trainingSamples.length,
    };
  }

  const generationId = safeSegment(input.generationId);
  const gameId = safeSegment(input.gameId);
  const base = `${SELF_PLAY_BLOB_PREFIX}${generationId}`;
  const gamePathname = `${base}/games/${gameId}.json`;
  const trainingPathname = `${base}/training/${gameId}.jsonl`;
  const trainingBody = input.trainingSamples
    .map((sample) => JSON.stringify(sample))
    .join("\n");

  const gameBlob = await put(gamePathname, JSON.stringify(input.game), {
    access: "private",
    addRandomSuffix: false,
    contentType: "application/json",
  });
  const trainingBlob = trainingBody
    ? await put(trainingPathname, trainingBody, {
        access: "private",
        addRandomSuffix: false,
        contentType: "application/x-ndjson",
      })
    : undefined;

  return {
    status: "saved",
    gamePathname: gameBlob.pathname,
    trainingPathname: trainingBlob?.pathname,
    trainingSamples: input.trainingSamples.length,
  };
}

export interface PersistedSelfPlayGeneration {
  generationId: string;
  manifestArtifacts: number;
  gameArtifacts: number;
  trainingArtifacts: number;
  bytes: number;
  updatedAt: string;
}

export async function listPersistedSelfPlayGenerations(): Promise<
  PersistedSelfPlayGeneration[]
> {
  if (!selfPlayStorageConfigured()) return [];

  const blobs: ListBlobResultBlob[] = [];
  let cursor: string | undefined;
  do {
    const page = await list({
      prefix: SELF_PLAY_BLOB_PREFIX,
      cursor,
      limit: 1000,
    });
    blobs.push(...page.blobs);
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor && blobs.length < 10_000);

  const generations = new Map<string, PersistedSelfPlayGeneration>();
  for (const blob of blobs) {
    const remainder = blob.pathname.slice(SELF_PLAY_BLOB_PREFIX.length);
    if (
      !remainder.endsWith("/manifest.json") &&
      !remainder.includes("/games/") &&
      !remainder.includes("/training/")
    ) {
      continue;
    }
    const generationId = remainder.split("/", 1)[0];
    if (!generationId) continue;
    const current = generations.get(generationId) ?? {
      generationId,
      manifestArtifacts: 0,
      gameArtifacts: 0,
      trainingArtifacts: 0,
      bytes: 0,
      updatedAt: blob.uploadedAt.toISOString(),
    };
    if (remainder.endsWith("/manifest.json")) current.manifestArtifacts++;
    if (remainder.includes("/games/")) current.gameArtifacts++;
    if (remainder.includes("/training/")) current.trainingArtifacts++;
    current.bytes += blob.size;
    if (blob.uploadedAt.toISOString() > current.updatedAt) {
      current.updatedAt = blob.uploadedAt.toISOString();
    }
    generations.set(generationId, current);
  }

  return [...generations.values()].sort((a, b) =>
    b.updatedAt.localeCompare(a.updatedAt)
  );
}

export async function readPersistedSelfPlayGames<T>(
  requestedGenerationId: string
): Promise<T[]> {
  if (!selfPlayStorageConfigured()) return [];
  const generationId = safeSegment(requestedGenerationId);
  const blobs: ListBlobResultBlob[] = [];
  let cursor: string | undefined;
  do {
    const page = await list({
      prefix: `${SELF_PLAY_BLOB_PREFIX}${generationId}/games/`,
      cursor,
      limit: 1000,
    });
    blobs.push(...page.blobs.filter((blob) => blob.pathname.endsWith(".json")));
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);

  const games: T[] = [];
  for (let index = 0; index < blobs.length; index += 12) {
    const chunk = await Promise.all(
      blobs.slice(index, index + 12).map(async (blob) => {
        const response = await get(blob.pathname, {
          access: "private",
          useCache: false,
        });
        if (!response?.stream || response.statusCode !== 200) return undefined;
        return (await new Response(response.stream).json()) as T;
      })
    );
    games.push(...chunk.filter((game): game is T => game !== undefined));
  }
  return games;
}
