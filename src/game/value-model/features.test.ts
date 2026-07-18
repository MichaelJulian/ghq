import { describe, expect, it } from "@jest/globals";
import type { Board, ReserveFleet } from "@/game/engine";
import { Blue, Red } from "@/game/tests/test-boards";
import {
  extractValueFeatures,
  extractValueFeaturesV2,
  extractValueFeaturesV3,
  VALUE_FEATURE_NAMES,
  VALUE_FEATURE_NAMES_V2,
  VALUE_FEATURE_NAMES_V3,
} from "./features";

const emptyBoard = (): Board =>
  Array.from({ length: 8 }, () =>
    Array.from({ length: 8 }, () => null)
  ) as Board;

const emptyReserve = (): ReserveFleet => ({
  INFANTRY: 0,
  ARMORED_INFANTRY: 0,
  AIRBORNE_INFANTRY: 0,
  ARTILLERY: 0,
  ARMORED_ARTILLERY: 0,
  HEAVY_ARTILLERY: 0,
});

function v2OwnFeature(board: Board, name: string): number {
  const features = extractValueFeaturesV2(
    {
      board,
      redReserve: emptyReserve(),
      blueReserve: emptyReserve(),
      currentPlayer: "RED",
      turnNumber: 5,
    },
    "RED"
  );
  const index = VALUE_FEATURE_NAMES_V2.indexOf(
    `own_${name}` as (typeof VALUE_FEATURE_NAMES_V2)[number]
  );
  expect(index).toBeGreaterThanOrEqual(0);
  return features[index];
}

function v3OwnFeature(board: Board, name: string): number {
  const features = extractValueFeaturesV3(
    {
      board,
      redReserve: emptyReserve(),
      blueReserve: emptyReserve(),
      currentPlayer: "RED",
      turnNumber: 40,
    },
    "RED"
  );
  const index = VALUE_FEATURE_NAMES_V3.indexOf(
    `own_${name}` as (typeof VALUE_FEATURE_NAMES_V3)[number]
  );
  expect(index).toBeGreaterThanOrEqual(0);
  return features[index];
}

describe("value feature schema v2", () => {
  it("preserves the incumbent schema while appending formation evidence", () => {
    const board = emptyBoard();
    board[7][7] = Red.HQ;
    board[0][0] = Blue.HQ;
    const position = {
      board,
      redReserve: emptyReserve(),
      blueReserve: emptyReserve(),
      currentPlayer: "RED" as const,
      turnNumber: 1,
    };

    expect(extractValueFeatures(position, "RED")).toHaveLength(
      VALUE_FEATURE_NAMES.length
    );
    expect(extractValueFeaturesV2(position, "RED")).toHaveLength(
      VALUE_FEATURE_NAMES_V2.length
    );
    expect(VALUE_FEATURE_NAMES_V2.length).toBeGreaterThan(
      VALUE_FEATURE_NAMES.length
    );
  });

  it("distinguishes a staggered infantry screen from a same-file column", () => {
    const vertical = emptyBoard();
    vertical[7][7] = Red.HQ;
    vertical[0][0] = Blue.HQ;
    vertical[5][5] = Red.INFANTRY;
    vertical[4][5] = Red.INFANTRY;
    vertical[3][5] = Red.INFANTRY;

    const staggered = emptyBoard();
    staggered[7][7] = Red.HQ;
    staggered[0][0] = Blue.HQ;
    staggered[5][5] = Red.INFANTRY;
    staggered[4][6] = Red.INFANTRY;
    staggered[5][7] = Red.INFANTRY;

    expect(v2OwnFeature(vertical, "infantry_vertical_adjacent_pairs")).toBe(2);
    expect(v2OwnFeature(staggered, "infantry_vertical_adjacent_pairs")).toBe(0);
    expect(v2OwnFeature(vertical, "infantry_diagonal_adjacent_pairs")).toBe(0);
    expect(v2OwnFeature(staggered, "infantry_diagonal_adjacent_pairs")).toBe(2);
    expect(v2OwnFeature(vertical, "infantry_same_file_run_excess")).toBe(1);
    expect(v2OwnFeature(staggered, "infantry_same_file_run_excess")).toBe(0);
    expect(v2OwnFeature(vertical, "infantry_distinct_files")).toBe(1);
    expect(v2OwnFeature(staggered, "infantry_distinct_files")).toBe(3);
    expect(v2OwnFeature(vertical, "infantry_frontier_count")).toBe(1);
    expect(v2OwnFeature(staggered, "infantry_frontier_count")).toBe(1);
  });
});

describe("value feature schema v3", () => {
  it("preserves v2 and exposes approaching HQ attackers before adjacency", () => {
    const board = emptyBoard();
    board[7][7] = Red.HQ;
    board[0][0] = Blue.HQ;
    board[5][7] = Blue.INFANTRY;
    board[7][5] = Blue.ARMORED_INF;
    board[4][4] = Blue.AIRBORNE;
    board[6][6] = Red.INFANTRY;

    expect(
      VALUE_FEATURE_NAMES_V3.slice(0, VALUE_FEATURE_NAMES_V2.length)
    ).toEqual([...VALUE_FEATURE_NAMES_V2]);
    expect(v3OwnFeature(board, "hq_enemy_infantry_distance_min")).toBe(2);
    expect(v3OwnFeature(board, "hq_enemy_armored_infantry_distance_min")).toBe(
      2
    );
    expect(v3OwnFeature(board, "hq_enemy_airborne_infantry_distance_min")).toBe(
      6
    );
    expect(v3OwnFeature(board, "hq_enemy_infantry_within_two")).toBe(2);
    expect(v3OwnFeature(board, "hq_enemy_infantry_within_three")).toBe(2);
    expect(v3OwnFeature(board, "hq_friendly_infantry_within_two")).toBe(1);
    expect(v3OwnFeature(board, "hq_friendly_infantry_within_three")).toBe(1);
    expect(v3OwnFeature(board, "hq_attack_pressure")).toBe(5);
    expect(v3OwnFeature(board, "hq_defense_density")).toBe(2);
  });
});
