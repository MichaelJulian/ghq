import type { DurableSelfPlayCompetitor } from "@/workflows/self-play-game";

export interface DurableScheduleInput {
  index: number;
  competitors: DurableSelfPlayCompetitor[];
  redMaxActions: 2 | 3;
  blueMaxActions: 2 | 3;
  valueModelArena: boolean;
}

/** Build one member of an adjacent, color-swapped self-play pair. */
export function scheduleDurableCompetitors({
  index,
  competitors,
  redMaxActions,
  blueMaxActions,
  valueModelArena,
}: DurableScheduleInput): {
  red: DurableSelfPlayCompetitor;
  blue: DurableSelfPlayCompetitor;
} {
  const pairIndex = Math.floor(index / 2);
  const first = competitors[pairIndex % competitors.length];
  const second = valueModelArena
    ? first
    : competitors[
        (pairIndex +
          1 +
          (Math.floor(pairIndex / competitors.length) %
            (competitors.length - 1))) %
          competitors.length
      ];
  const [redBase, blueBase] = index % 2 ? [second, first] : [first, second];
  const redValueModel = valueModelArena
    ? index % 2
      ? "incumbent"
      : "challenger"
    : "incumbent";
  const blueValueModel = valueModelArena
    ? index % 2
      ? "challenger"
      : "incumbent"
    : "incumbent";
  return {
    red: {
      ...redBase,
      id: `${redBase.personality}-${redValueModel}-a${redMaxActions}`,
      maxActions: redMaxActions,
      valueModel: redValueModel,
    },
    blue: {
      ...blueBase,
      id: `${blueBase.personality}-${blueValueModel}-a${blueMaxActions}`,
      maxActions: blueMaxActions,
      valueModel: blueValueModel,
    },
  };
}
