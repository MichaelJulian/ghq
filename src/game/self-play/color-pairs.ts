export interface ColorSwapGame {
  generationId: string;
  gameId: string;
}

export function colorSwapGameNumber(gameId: string): number | undefined {
  const suffix = /-(\d+)$/.exec(gameId);
  if (!suffix) return undefined;
  const number = Number.parseInt(suffix[1], 10);
  return Number.isSafeInteger(number) && number >= 1 ? number : undefined;
}

export function partitionColorSwapPairs<T extends ColorSwapGame>(games: T[]): {
  pairs: Array<[T, T]>;
  orphans: T[];
} {
  const groups = new Map<string, T[]>();
  const orphans: T[] = [];
  for (const game of games) {
    const number = colorSwapGameNumber(game.gameId);
    if (number === undefined) {
      orphans.push(game);
      continue;
    }
    const pairNumber = Math.floor((number - 1) / 2) + 1;
    const key = `${game.generationId}:${pairNumber}`;
    const members = groups.get(key) ?? [];
    members.push(game);
    groups.set(key, members);
  }

  const pairs: Array<[T, T]> = [];
  for (const members of groups.values()) {
    const ordered = [...members].sort(
      (left, right) =>
        (colorSwapGameNumber(left.gameId) ?? 0) -
        (colorSwapGameNumber(right.gameId) ?? 0)
    );
    const first = ordered[0];
    const second = ordered[1];
    const firstNumber = first && colorSwapGameNumber(first.gameId);
    const secondNumber = second && colorSwapGameNumber(second.gameId);
    if (
      ordered.length === 2 &&
      firstNumber !== undefined &&
      firstNumber % 2 === 1 &&
      secondNumber === firstNumber + 1
    ) {
      pairs.push([first, second]);
    } else {
      orphans.push(...ordered);
    }
  }
  pairs.sort((left, right) => left[0].gameId.localeCompare(right[0].gameId));
  orphans.sort((left, right) => left.gameId.localeCompare(right.gameId));
  return { pairs, orphans };
}
