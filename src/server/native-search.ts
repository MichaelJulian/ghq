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

/** Resolve the function in the exact deployment that is running this code. */
export function nativeSearchUrl(): string | undefined {
  const configured = process.env.GHQ_NATIVE_SEARCH_URL?.trim();
  if (configured) {
    return configured.endsWith("/") ? configured.slice(0, -1) : configured;
  }
  // Preview deployments are protected by default, so an internal fetch to
  // their public hostname is redirected to SSO. Explicit configuration can
  // still exercise native search in preview; production uses its exact
  // immutable deployment hostname automatically.
  if (process.env.VERCEL_ENV !== "production") return undefined;
  const deployment = process.env.VERCEL_URL?.trim();
  return deployment
    ? `https://${deployment.replace(/^https?:\/\//, "")}/api/native_search`
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
