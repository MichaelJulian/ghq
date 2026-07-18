/** @jest-environment node */

import { describe, expect, it, jest } from "@jest/globals";
import { GHQ_STARTING_FEN } from "@/game/analysis/types";
import type { GhqCandidateTurn, GhqSearchResult } from "@/game/analysis/types";
import { analyzeFen } from "./fen-analysis";

const RESULTING_FEN =
  "qr↓6/iii5/8/8/8/8/5III/1P1H↑T↑1R↑Q IIIIIFFFRR iiiiifffprrth b";
const SERIALIZED_STATE =
  "eJxjZmcAgge3GCA0AxpgYoLQDlC+AJTmYITQDRDqwS1mdoQmjogIhE6w9Ken7PxICpAtwCqKCSD2sQIxMxadMHFGqDgjBAMAiAcI9w==";

function purpose() {
  return {
    capture_gain: 0,
    deployment_gain: 3,
    threat_gain: 0,
    protection_gain: 0,
    development_gain: 3,
    formation_gain: 0,
    dispersion_increase: 0,
    uncompensated_dispersion: 0,
    optionality_gain: 0,
    congestion_increase: 0,
    immobile_units: 0,
    relocation_options: 0,
    extension_increase: 0,
    frontier_rank: 1,
    frontier_limit: 2,
    forward_infantry_actions: 0,
    coordinated_overpush: 0,
    escape_actions: 0,
    purposeful_actions: 3,
    unpurposed_actions: 0,
    development_actions: 3,
    formation_actions: 0,
    quiet_actions: 0,
    counted_actions: 3,
    unused_actions: 0,
    backfills: 0,
    reversals: 0,
    pure_rotations: 0,
    forcing_gain: 3,
    net_purpose_penalty: 0,
    paratrooper_mission_penalty: 0,
    total_penalty: 0,
  };
}

function nativeSearchResult(): GhqSearchResult {
  const candidate = {
    rank: 1,
    automatic_captures: [],
    actions: ["rhd1", "rte1", "rpb1"],
    all_moves: ["rhd1", "rte1", "rpb1"],
    resulting_fen: RESULTING_FEN,
    score: 1,
    action_purposes: [],
    purpose: purpose(),
  } as GhqCandidateTurn;
  return {
    recommendation_label: "opening book",
    input_fen: GHQ_STARTING_FEN,
    side_to_move: "red",
    best_turn: candidate,
    principal_variation: candidate.all_moves,
    candidate_turns: [candidate],
    score: { current_player: 1, red: 1 },
    search: {
      completed_depth_in_turns: 1,
      requested_depth_in_turns: 1,
      base_complete_turn_width: 2,
      max_actions: 3,
      nodes: 1,
      elapsed_ms: 1,
      timed_out: false,
      fallback_used: "none",
      opening_book_used: true,
      early_game_focus: true,
      approximate: false,
      exhaustive_within_requested_horizon: false,
      rule_filtered_actions: 0,
      beam_pruned_actions: 0,
      partial_turns_pruned: 0,
      complete_turns_generated: 1,
      complete_turns_deduplicated: 0,
      complete_turns_pruned: 0,
      tactically_unsafe_turns: 0,
      rotation_quota_pruned: 0,
      purpose_filtered_turns: 0,
      value_model_evaluations: 1,
      turn_cache_hits: 0,
      transposition_hits: 0,
      backend: "native-python",
      value_model_backend: "native-gbdt",
      value_model_version: "incumbent",
      code_version: "local-unversioned-search",
    },
    evaluation: {
      before: {} as GhqSearchResult["evaluation"]["before"],
      after_best_turn: {} as GhqSearchResult["evaluation"]["after_best_turn"],
    },
  };
}

describe("native production FEN analysis", () => {
  it("preserves the public response contract and runtime provenance", async () => {
    process.env.GHQ_NATIVE_SEARCH_URL = "https://native.test/api";
    const nativeResponse = {
      codeVersion: "local-unversioned-search",
      fen: GHQ_STARTING_FEN,
      sideToMove: "RED",
      resultingFen: RESULTING_FEN,
      serializedState: SERIALIZED_STATE,
      outcome: undefined,
      afterEvaluation: {} as GhqSearchResult["evaluation"]["after_best_turn"],
      search: nativeSearchResult(),
    };
    const fetch = jest
      .spyOn(global, "fetch")
      .mockResolvedValue(
        new Response(JSON.stringify(nativeResponse), {
          status: 200,
          headers: { "content-type": "application/json" },
        })
      );

    const result = await analyzeFen({
      fen: GHQ_STARTING_FEN,
      turnNumber: 1,
      timeMs: 100,
      maxDepth: 1,
      beamWidth: 2,
      maxActions: 3,
      valueModel: "incumbent",
      explorationSeed: 17,
    });

    expect(result.resultingFen).toBe(RESULTING_FEN);
    expect(result.serializedState).toBe(SERIALIZED_STATE);
    expect(result.sideToMove).toBe("RED");
    expect(result.search.search.backend).toBe("native-python");
    expect(result.search.search.value_model_backend).toBe("native-gbdt");
    expect(result.search.search.value_model_version).toBe("incumbent");
    expect(result.model.before.redWinProbability).toBeGreaterThan(0);
    expect(fetch).toHaveBeenCalledWith(
      "https://native.test/api",
      expect.objectContaining({ method: "POST" })
    );
    const requestBody = JSON.parse(
      (fetch.mock.calls[0][1] as RequestInit).body as string
    );
    expect(requestBody).toEqual(
      expect.objectContaining({
        fen: GHQ_STARTING_FEN,
        maxActions: 3,
        valueModel: "incumbent",
      })
    );
    delete process.env.GHQ_NATIVE_SEARCH_URL;
  });
});
