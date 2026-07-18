import type { SelfPlayProgressSnapshot } from "@/server/self-play-storage";

export interface ActiveProgressRuntimeSummary {
  games: number;
  decisions: number;
  depthAtLeastTwoDecisions: number;
  fallbackDecisions: number;
  unverifiedFallbackDecisions: number;
  timedOutDecisions: number;
  depthAtLeastTwoRate: number;
  fallbackRate: number;
  unverifiedFallbackRate: number;
  timedOutRate: number;
}

export function summarizeActiveProgressRuntime(
  snapshots: SelfPlayProgressSnapshot[]
): ActiveProgressRuntimeSummary {
  const totals = snapshots.reduce(
    (summary, snapshot) => {
      summary.decisions += snapshot.decisions;
      summary.depthAtLeastTwoDecisions += snapshot.depthAtLeastTwoDecisions;
      summary.fallbackDecisions += snapshot.fallbackDecisions;
      summary.unverifiedFallbackDecisions +=
        snapshot.unverifiedFallbackDecisions;
      summary.timedOutDecisions += snapshot.timedOutDecisions;
      return summary;
    },
    {
      decisions: 0,
      depthAtLeastTwoDecisions: 0,
      fallbackDecisions: 0,
      unverifiedFallbackDecisions: 0,
      timedOutDecisions: 0,
    }
  );
  const denominator = Math.max(1, totals.decisions);
  return {
    games: snapshots.length,
    ...totals,
    depthAtLeastTwoRate: totals.depthAtLeastTwoDecisions / denominator,
    fallbackRate: totals.fallbackDecisions / denominator,
    unverifiedFallbackRate: totals.unverifiedFallbackDecisions / denominator,
    timedOutRate: totals.timedOutDecisions / denominator,
  };
}
