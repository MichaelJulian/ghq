import type { Player } from "@/game/engine-v2";
import { FENtoBoardState } from "@/game/notation";

export interface StrategicProgress {
  frontierRank: number;
  enemyHqDistance: number;
  enemyHqPressure: number;
}

/** Monotonic landmarks used to distinguish an approach from a quiet cycle. */
export function strategicProgress(
  fen: string,
  player: Player
): StrategicProgress {
  const { board } = FENtoBoardState(fen);
  const own: Array<[number, number]> = [];
  let enemyHq: [number, number] | undefined;

  for (let row = 0; row < board.length; row++) {
    for (let column = 0; column < board[row].length; column++) {
      const piece = board[row][column];
      if (!piece) continue;
      if (piece.player === player && piece.type !== "HQ") {
        own.push([row, column]);
      } else if (piece.player !== player && piece.type === "HQ") {
        enemyHq = [row, column];
      }
    }
  }

  const frontierRank = own.reduce(
    (best, [row]) => Math.max(best, player === "RED" ? 8 - row : row + 1),
    0
  );
  const enemyHqDistance =
    own.length && enemyHq
      ? Math.min(
          ...own.map(([row, column]) =>
            Math.max(Math.abs(row - enemyHq[0]), Math.abs(column - enemyHq[1]))
          )
        )
      : 8;
  const infantry = own.filter(([row, column]) => {
    const type = board[row][column]?.type;
    return (
      type === "INFANTRY" ||
      type === "ARMORED_INFANTRY" ||
      type === "AIRBORNE_INFANTRY"
    );
  });
  const pursuers = infantry.length ? infantry : own;
  const enemyHqPressure = enemyHq
    ? pursuers.reduce(
        (total, [row, column]) =>
          total +
          Math.max(
            0,
            5 -
              Math.max(
                Math.abs(row - enemyHq![0]),
                Math.abs(column - enemyHq![1])
              )
          ),
        0
      )
    : 0;
  return { frontierRank, enemyHqDistance, enemyHqPressure };
}

export function extendsStrategicBest(
  best: StrategicProgress,
  current: StrategicProgress
): boolean {
  return (
    current.frontierRank > best.frontierRank ||
    current.enemyHqDistance < best.enemyHqDistance ||
    current.enemyHqPressure > best.enemyHqPressure
  );
}

export function mergeStrategicBest(
  best: StrategicProgress,
  current: StrategicProgress
): StrategicProgress {
  return {
    frontierRank: Math.max(best.frontierRank, current.frontierRank),
    enemyHqDistance: Math.min(best.enemyHqDistance, current.enemyHqDistance),
    enemyHqPressure: Math.max(best.enemyHqPressure, current.enemyHqPressure),
  };
}
