import type {
  DurableSelfPlayDecision,
  DurableSelfPlayGameResult,
} from "@/workflows/self-play-game";

export interface SearchRuntimeSummary {
  decisions: number;
  backendCounts: Record<string, number>;
  valueModelBackendCounts: Record<string, number>;
  completedDepthCounts: Record<string, number>;
  averageCompletedDepth: number;
  depthAtLeastTwoRate: number;
  zeroDepthRate: number;
  averageNodes?: number;
  averageCompleteTurnsGenerated?: number;
  hqExactReturnProbeDecisions: number;
  tacticalReturnGuardDecisions: number;
  tacticalReturnGuardRate: number;
  safeFallbackReplyVerifiedDecisions: number;
  safeFallbackReplyVerifiedRate: number;
}

function increment(counts: Record<string, number>, key: string): void {
  counts[key] = (counts[key] ?? 0) + 1;
}

/** Aggregate the exact behavior-search runtime used to create a generation. */
export function summarizeSearchRuntime(
  games: Pick<DurableSelfPlayGameResult, "decisions">[]
): SearchRuntimeSummary {
  const decisions = games.flatMap((game) => game.decisions);
  const backendCounts: Record<string, number> = {};
  const valueModelBackendCounts: Record<string, number> = {};
  const completedDepthCounts: Record<string, number> = {};
  let depthTotal = 0;
  let depthAtLeastTwo = 0;
  let zeroDepth = 0;
  let nodeTotal = 0;
  let nodeSamples = 0;
  let generatedTotal = 0;
  let generatedSamples = 0;
  let hqExactReturnProbeDecisions = 0;
  let tacticalReturnGuardDecisions = 0;
  let safeFallbackReplyVerifiedDecisions = 0;

  for (const decision of decisions as DurableSelfPlayDecision[]) {
    increment(backendCounts, decision.searchBackend ?? "unknown");
    increment(
      valueModelBackendCounts,
      decision.searchValueModelBackend ?? "unknown"
    );
    increment(completedDepthCounts, String(decision.completedDepth));
    depthTotal += decision.completedDepth;
    if (decision.completedDepth >= 2) depthAtLeastTwo++;
    if (decision.completedDepth === 0) zeroDepth++;
    if (decision.searchTelemetry) {
      nodeTotal += decision.searchTelemetry.nodes;
      nodeSamples++;
      generatedTotal += decision.searchTelemetry.completeTurnsGenerated;
      generatedSamples++;
      if (decision.searchTelemetry.hqExactReturnProbeUsed) {
        hqExactReturnProbeDecisions++;
      }
      if (decision.searchTelemetry.tacticalReturnGuardUsed) {
        tacticalReturnGuardDecisions++;
      }
      if (decision.searchTelemetry.safeFallbackReplyVerified) {
        safeFallbackReplyVerifiedDecisions++;
      }
    }
  }

  const count = decisions.length;
  return {
    decisions: count,
    backendCounts,
    valueModelBackendCounts,
    completedDepthCounts,
    averageCompletedDepth: count ? depthTotal / count : 0,
    depthAtLeastTwoRate: count ? depthAtLeastTwo / count : 0,
    zeroDepthRate: count ? zeroDepth / count : 0,
    averageNodes: nodeSamples ? nodeTotal / nodeSamples : undefined,
    averageCompleteTurnsGenerated: generatedSamples
      ? generatedTotal / generatedSamples
      : undefined,
    hqExactReturnProbeDecisions,
    tacticalReturnGuardDecisions,
    tacticalReturnGuardRate: count ? tacticalReturnGuardDecisions / count : 0,
    safeFallbackReplyVerifiedDecisions,
    safeFallbackReplyVerifiedRate: count
      ? safeFallbackReplyVerifiedDecisions / count
      : 0,
  };
}
