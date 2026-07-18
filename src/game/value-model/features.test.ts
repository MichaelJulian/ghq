import { describe, expect, it } from "@jest/globals";
import type { Board, ReserveFleet } from "@/game/engine";
import { Blue, Red } from "@/game/tests/test-boards";
import {
  extractValueFeatures,
  extractValueFeaturesV2,
  VALUE_FEATURE_NAMES,
  VALUE_FEATURE_NAMES_V2,
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
