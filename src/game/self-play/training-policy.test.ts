/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import { auditParatrooperTrainingPolicy } from "./training-policy";

describe("self-play paratrooper training policy", () => {
  it("accepts complete clean policy telemetry", () => {
    expect(
      auditParatrooperTrainingPolicy([
        { selectedPurpose: { paratrooper_mission_penalty: 0 } },
        { selectedPurpose: { paratrooper_mission_penalty: 0 } },
      ])
    ).toMatchObject({
      telemetryComplete: true,
      violatingDecisions: 0,
      eligible: true,
    });
  });

  it("quarantines a whole game after one policy violation", () => {
    expect(
      auditParatrooperTrainingPolicy([
        { selectedPurpose: { paratrooper_mission_penalty: 0 } },
        { selectedPurpose: { paratrooper_mission_penalty: 9 } },
      ])
    ).toMatchObject({
      telemetryComplete: true,
      violatingDecisions: 1,
      eligible: false,
    });
  });

  it("does not assume missing historical telemetry is clean", () => {
    expect(auditParatrooperTrainingPolicy([{}])).toMatchObject({
      missingTelemetryDecisions: 1,
      telemetryComplete: false,
      eligible: false,
    });
  });
});
