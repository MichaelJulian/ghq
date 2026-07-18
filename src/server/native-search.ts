import type {
  FenAnalysisRequest,
  FenAnalysisResponse,
  GhqSearchResult,
  PersonalityId,
  SearchEvaluationBreakdown,
} from "@/game/analysis/types";
import type { Player } from "@/game/engine";
import type { ValueModelVersion } from "@/game/value-model/inference";
import { getVercelOidcTokenSync } from "@vercel/oidc";

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
  codeVersion: string;
  fen: string;
  sideToMove: Player;
  resultingFen: string;
  serializedState: string;
  outcome?: FenAnalysisResponse["outcome"];
  afterEvaluation: SearchEvaluationBreakdown;
  search: GhqSearchResult;
}

export interface NativeDescriptionResponse {
  codeVersion: string;
  fen: string;
  sideToMove: Player;
  serializedState: string;
  outcome?: FenAnalysisResponse["outcome"];
  evaluation: SearchEvaluationBreakdown;
}

function vercelDeploymentHost(): string | undefined {
  const deployment = process.env.VERCEL_URL?.trim();
  return deployment
    ? deployment.replace(/^https?:\/\//, "").replace(/\/$/, "")
    : undefined;
}

/**
 * Resolve native search through the immutable deployment that owns the
 * workflow step. This lets a durable run finish on one engine revision even
 * after a newer deployment takes over the production alias.
 */
export function nativeSearchUrl(): string | undefined {
  const configured = process.env.GHQ_NATIVE_SEARCH_URL?.trim();
  if (configured) {
    return configured.endsWith("/") ? configured.slice(0, -1) : configured;
  }
  // Preview workflows still require explicit configuration because their
  // environment-to-environment Trusted Sources policy is project-specific.
  if (process.env.VERCEL_ENV !== "production") return undefined;
  const deployment = vercelDeploymentHost();
  if (deployment) return `https://${deployment}/api/native_search`;
  const production = process.env.VERCEL_PROJECT_PRODUCTION_URL?.trim();
  return production
    ? `https://${production.replace(/^https?:\/\//, "")}/api/native_search`
    : undefined;
}

async function postNative<T>(url: string, body: object): Promise<T> {
  const deployment = vercelDeploymentHost();
  const deploymentProtected =
    process.env.VERCEL_ENV === "production" &&
    deployment !== undefined &&
    new URL(url).host === deployment;
  const headers: Record<string, string> = {
    "content-type": "application/json",
  };
  if (deploymentProtected) {
    headers["x-vercel-trusted-oidc-idp-token"] = getVercelOidcTokenSync();
  }
  const response = await fetch(url, {
    method: "POST",
    headers,
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

function assertNativeCodeVersion(result: {
  codeVersion?: string;
  search?: GhqSearchResult;
}): void {
  const expected =
    process.env.VERCEL_GIT_COMMIT_SHA?.trim() || "local-unversioned-search";
  const telemetryVersion = result.search?.search.code_version;
  if (
    !result.codeVersion ||
    result.codeVersion !== expected ||
    (telemetryVersion !== undefined && telemetryVersion !== result.codeVersion)
  ) {
    throw new Error(
      `Native GHQ search code mismatch: expected ${expected}, received ${result.codeVersion ?? "missing"}`
    );
  }
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
  assertNativeCodeVersion(result);
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
  const result = await postNative<NativeDescriptionResponse>(url, {
    mode: "describe",
    fen,
    personality,
    turnNumber,
  });
  assertNativeCodeVersion(result);
  return result;
}
