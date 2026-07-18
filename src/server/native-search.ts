import type {
  FenAnalysisRequest,
  FenAnalysisResponse,
  GhqSearchResult,
  PersonalityId,
  SearchEvaluationBreakdown,
} from "@/game/analysis/types";
import type { Player } from "@/game/engine";
import type { ValueModelVersion } from "@/game/value-model/inference";

interface NativeSearchRequest {
  fen?: string;
  serializedState?: string;
  personality: PersonalityId;
  turnNumber: number;
  timeMs: number;
  maxDepth: number;
  beamWidth: number;
  openingSeed: number;
  maxActions: 3;
  stagnationTurns: number;
  valueModel: ValueModelVersion;
}

export interface NativeSearchResponse {
  fen: string;
  sideToMove: Player;
  resultingFen: string;
  serializedState: string;
  outcome?: FenAnalysisResponse["outcome"];
  afterEvaluation: SearchEvaluationBreakdown;
  search: GhqSearchResult;
}

export interface NativeDescriptionResponse {
  fen: string;
  sideToMove: Player;
  serializedState: string;
  outcome?: FenAnalysisResponse["outcome"];
  evaluation: SearchEvaluationBreakdown;
}

/** Resolve the native function through the public production project domain. */
export function nativeSearchUrl(): string | undefined {
  const configured = process.env.GHQ_NATIVE_SEARCH_URL?.trim();
  if (configured) {
    return configured.endsWith("/") ? configured.slice(0, -1) : configured;
  }
  // Preview deployments are protected by default, so an internal fetch to
  // their public hostname is redirected to SSO. Explicit configuration can
  // still exercise native search in preview. The immutable VERCEL_URL may be
  // protected in production too, while VERCEL_PROJECT_PRODUCTION_URL is the
  // stable public project domain and is guaranteed by Vercel at runtime.
  if (process.env.VERCEL_ENV !== "production") return undefined;
  const production = process.env.VERCEL_PROJECT_PRODUCTION_URL?.trim();
  return production
    ? `https://${production.replace(/^https?:\/\//, "")}/api/native_search`
    : undefined;
}

async function postNative<T>(url: string, body: object): Promise<T> {
  const response = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
  });
  const text = await response.text();
  let payload: unknown;
  try {
    payload = JSON.parse(text);
  } catch {
    throw new Error(
      `Native GHQ search returned ${response.status} with a non-JSON body`
    );
  }
  if (!response.ok) {
    const message =
      typeof payload === "object" &&
      payload !== null &&
      "error" in payload &&
      typeof payload.error === "string"
        ? payload.error
        : `Native GHQ search failed with status ${response.status}`;
    throw new Error(message);
  }
  return payload as T;
}

export async function searchNatively(
  url: string,
  request: FenAnalysisRequest,
  config: Omit<NativeSearchRequest, "fen" | "serializedState">
): Promise<NativeSearchResponse> {
  const result = await postNative<NativeSearchResponse>(url, {
    fen: request.fen,
    serializedState: request.serializedState,
    ...config,
  });
  if (
    result.search?.search?.backend !== "native-python" ||
    result.search.search.value_model_backend !== "native-gbdt"
  ) {
    throw new Error("Native GHQ search did not prove its runtime provenance");
  }
  return result;
}

export async function describeNatively(
  url: string,
  fen: string,
  personality: PersonalityId,
  turnNumber: number
): Promise<NativeDescriptionResponse> {
  return postNative<NativeDescriptionResponse>(url, {
    mode: "describe",
    fen,
    personality,
    turnNumber,
  });
}
