import { createHash } from "crypto";
import { get, put } from "@vercel/blob";
import type { GhqSearchResult, PersonalityId } from "@/game/analysis/types";
import type { ValueModelVersion } from "@/game/value-model/inference";
import { selfPlayStorageConfigured } from "@/server/self-play-storage";

const SEARCH_CACHE_VERSION = "early-depth-v3";
const SEARCH_CACHE_PREFIX = `self-play/search-cache/${SEARCH_CACHE_VERSION}/`;

export interface SearchCacheKey {
  serializedPosition: string;
  /** Exact deployed search implementation; prevents cross-code reuse. */
  searchCodeVersion: string;
  personality: PersonalityId;
  turnNumber: number;
  timeMs: number;
  maxDepth: number;
  beamWidth: number;
  maxActions: number;
  stagnationTurns: number;
  valueModel: ValueModelVersion;
  valueModelCheckpoint: string;
}

export function searchCachePathname(key: SearchCacheKey): string {
  const digest = createHash("sha256").update(JSON.stringify(key)).digest("hex");
  return `${SEARCH_CACHE_PREFIX}${digest}.json`;
}

/** The shared cache is deliberately limited to commonly repeated openings. */
export function shouldPersistSearch(
  key: SearchCacheKey,
  result: GhqSearchResult
): boolean {
  return Boolean(
    key.turnNumber > 4 &&
      key.turnNumber <= 16 &&
      result.search.completed_depth_in_turns >= key.maxDepth &&
      result.search.fallback_used === "none" &&
      result.candidate_turns.length > 0
  );
}

export async function readPersistedSearch(
  key: SearchCacheKey
): Promise<GhqSearchResult | undefined> {
  if (
    !selfPlayStorageConfigured() ||
    key.turnNumber <= 4 ||
    key.turnNumber > 16
  ) {
    return undefined;
  }
  try {
    const response = await get(searchCachePathname(key), {
      access: "private",
      useCache: true,
    });
    if (!response?.stream || response.statusCode !== 200) return undefined;
    return (await new Response(response.stream).json()) as GhqSearchResult;
  } catch {
    // Cache availability must never prevent analysis.
    return undefined;
  }
}

export async function persistSearch(
  key: SearchCacheKey,
  result: GhqSearchResult
): Promise<void> {
  if (!selfPlayStorageConfigured() || !shouldPersistSearch(key, result)) return;
  try {
    await put(searchCachePathname(key), JSON.stringify(result), {
      access: "private",
      addRandomSuffix: false,
      contentType: "application/json",
    });
  } catch {
    // Search remains authoritative if persistence is unavailable.
  }
}
