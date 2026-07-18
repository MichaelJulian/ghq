export interface ParatrooperPolicyDecision {
  selectedPurpose?: {
    paratrooper_mission_penalty?: number;
  };
}

export interface ParatrooperPolicyAudit {
  decisions: number;
  missingTelemetryDecisions: number;
  violatingDecisions: number;
  telemetryComplete: boolean;
  eligible: boolean;
}

/**
 * Treat policy telemetry as part of a training label's provenance. Older or
 * malformed games cannot be assumed clean merely because the penalty field is
 * absent; they must be quarantined until they can be replayed and audited.
 */
export function auditParatrooperTrainingPolicy(
  decisions: readonly ParatrooperPolicyDecision[]
): ParatrooperPolicyAudit {
  const missingTelemetryDecisions = decisions.filter(
    (decision) =>
      typeof decision.selectedPurpose?.paratrooper_mission_penalty !== "number"
  ).length;
  const violatingDecisions = decisions.filter(
    (decision) =>
      (decision.selectedPurpose?.paratrooper_mission_penalty ?? 0) > 0
  ).length;
  const telemetryComplete =
    decisions.length > 0 && missingTelemetryDecisions === 0;
  return {
    decisions: decisions.length,
    missingTelemetryDecisions,
    violatingDecisions,
    telemetryComplete,
    eligible: telemetryComplete && violatingDecisions === 0,
  };
}
