import { list, put, type ListBlobResultBlob } from "@vercel/blob";

export const SELF_PLAY_BLOB_PREFIX = "self-play/generations/";

export interface PersistedSelfPlayArtifacts {
  status: "saved" | "not-configured";
  gamePathname?: string;
  trainingPathname?: string;
  trainingSamples: number;
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
    const generationId = remainder.split("/", 1)[0];
    if (!generationId) continue;
    const current = generations.get(generationId) ?? {
      generationId,
      gameArtifacts: 0,
      trainingArtifacts: 0,
      bytes: 0,
      updatedAt: blob.uploadedAt.toISOString(),
    };
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
