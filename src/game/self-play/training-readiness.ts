export const MINIMUM_AUDITED_TRAINING_PAIRS = 30;

export interface SelfPlayTrainingReadiness {
  qualityEligibleGames: number;
  preAuditCompletePairs: number;
  minimumAuditedPairs: number;
  preAuditPairDeficit: number;
}

function pairedGameIdentity(
  gameId: string
): { pairId: string; gameNumber: number } | undefined {
  const match = gameId.match(/^(.*)-(\d+)$/);
  if (!match) return undefined;
  const gameNumber = Number.parseInt(match[2], 10);
  if (!Number.isSafeInteger(gameNumber) || gameNumber < 1) return undefined;
  const pairNumber = Math.floor((gameNumber - 1) / 2) + 1;
  return {
    pairId: `${match[1]}-pair-${pairNumber}`,
    gameNumber,
  };
}

/** Count adjacent odd/even color swaps that survive the pre-audit gates. */
export function summarizeSelfPlayTrainingReadiness(
  qualityEligibleGameIds: Iterable<string>
): SelfPlayTrainingReadiness {
  const uniqueGameIds = new Set(qualityEligibleGameIds);
  const gamesByPair = new Map<string, Set<number>>();
  for (const gameId of uniqueGameIds) {
    const identity = pairedGameIdentity(gameId);
    if (!identity) continue;
    const gameNumbers = gamesByPair.get(identity.pairId) ?? new Set<number>();
    gameNumbers.add(identity.gameNumber);
    gamesByPair.set(identity.pairId, gameNumbers);
  }
  const preAuditCompletePairs = [...gamesByPair.values()].filter(
    (gameNumbers) => {
      if (gameNumbers.size !== 2) return false;
      const [first, second] = [...gameNumbers].sort(
        (left, right) => left - right
      );
      return first % 2 === 1 && second === first + 1;
    }
  ).length;
  return {
    qualityEligibleGames: uniqueGameIds.size,
    preAuditCompletePairs,
    minimumAuditedPairs: MINIMUM_AUDITED_TRAINING_PAIRS,
    preAuditPairDeficit: Math.max(
      0,
      MINIMUM_AUDITED_TRAINING_PAIRS - preAuditCompletePairs
    ),
  };
}
