import type {
  DurableSelfPlayCompetitor,
  DurableSelfPlayGameConfig,
} from "@/workflows/self-play-game";
import { valueModelCheckpointId } from "@/game/value-model/inference";

export const MAX_CONCURRENT_DURABLE_SEARCHES = 4;
export const DURABLE_SEARCH_SLOT_MS = 50_000;

export function scheduleDurableSearch(
  index: number,
  games: number,
  epochMs: number
): DurableSelfPlayGameConfig["searchSchedule"] {
  if (
    !Number.isSafeInteger(index) ||
    index < 0 ||
    !Number.isSafeInteger(games) ||
    games < 1 ||
    index >= games ||
    !Number.isSafeInteger(epochMs) ||
    epochMs < 0
  ) {
    throw new RangeError("Invalid durable search schedule input");
  }
  const laneCount = Math.ceil(games / MAX_CONCURRENT_DURABLE_SEARCHES);
  if (laneCount === 1) return undefined;
  return {
    epochMs,
    lane: Math.floor(index / MAX_CONCURRENT_DURABLE_SEARCHES),
    laneCount,
    slotMs: DURABLE_SEARCH_SLOT_MS,
  };
}

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
      valueModelCheckpoint: valueModelCheckpointId(
        redMaxActions === 2 ? "two-actions" : "three-actions",
        redValueModel
      ),
    },
    blue: {
      ...blueBase,
      id: `${blueBase.personality}-${blueValueModel}-a${blueMaxActions}`,
      maxActions: blueMaxActions,
      valueModel: blueValueModel,
      valueModelCheckpoint: valueModelCheckpointId(
        blueMaxActions === 2 ? "two-actions" : "three-actions",
        blueValueModel
      ),
    },
  };
}
