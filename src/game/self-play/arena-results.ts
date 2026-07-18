import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";
import type { Player } from "@/game/engine-v2";

export interface ValueModelArenaSummary {
  games: number;
  pairs: number;
  provenance: {
    generationIds: string[];
    codeVersions: string[];
    incumbentCheckpoints: string[];
    challengerCheckpoints: string[];
  };
  searchQuality: {
    decisions: number;
    unverifiedFallbackDecisions: number;
    unverifiedFallbackRate: number;
    seededFallbackDecisions: number;
    incompleteTurnDecisions: number;
    missingRuntimeProvenanceDecisions: number;
    mismatchedSearchCodeDecisions: number;
    searchBackends: string[];
    valueModelBackends: string[];
  };
  challenger: {
    points: number;
    scoreRate: number;
    eloDifference: number;
    byColor: Record<
      Player,
      { games: number; points: number; scoreRate: number }
    >;
  };
  pairedOutcomes: { wins: number; ties: number; losses: number };
  pairBootstrap: {
    samples: number;
    ci95Low: number;
    ci95High: number;
  };
  promotionGate: { passed: boolean; reasons: string[] };
}

export interface ValueModelArenaExpectedProvenance {
  generationId: string;
  codeVersion: string;
  incumbentCheckpoints: string[];
  challengerCheckpoints: string[];
}

function modelFromAgent(agentId: string, recorded?: string): string {
  if (recorded === "challenger" || recorded === "incumbent") return recorded;
  return agentId.includes("-challenger-") ? "challenger" : "incumbent";
}

function competitorIdentity(agentId: string): string {
  return agentId.replace(/-(?:challenger|incumbent)-a[23]$/, "");
}

function challengerResult(game: DurableSelfPlayGameResult):
  | {
      color: Player;
      score: number;
    }
  | undefined {
  const red = modelFromAgent(game.redAgentId, game.redValueModel);
  const blue = modelFromAgent(game.blueAgentId, game.blueValueModel);
  if (red === blue) return undefined;
  const color: Player = red === "challenger" ? "RED" : "BLUE";
  return {
    color,
    score: !game.outcome.winner ? 0.5 : game.outcome.winner === color ? 1 : 0,
  };
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 0x1_0000_0000;
  };
}

function percentile(sorted: number[], fraction: number): number {
  return sorted[Math.round((sorted.length - 1) * fraction)] ?? 0;
}

export function summarizeValueModelArena(
  games: DurableSelfPlayGameResult[],
  bootstrapSamples = 5_000,
  expectedProvenance?: ValueModelArenaExpectedProvenance
): ValueModelArenaSummary | undefined {
  const scored = games
    .map((game) => ({ game, result: challengerResult(game) }))
    .filter(
      (
        entry
      ): entry is {
        game: DurableSelfPlayGameResult;
        result: { color: Player; score: number };
      } => Boolean(entry.result)
    )
    .sort((left, right) => left.game.gameId.localeCompare(right.game.gameId));
  if (!scored.length) return undefined;

  const generationIds = new Set<string>();
  const codeVersions = new Set<string>();
  const incumbentCheckpoints = new Set<string>();
  const challengerCheckpoints = new Set<string>();
  for (const { game } of scored) {
    generationIds.add(game.generationId || "unknown");
    codeVersions.add(game.codeVersion || "unknown");
    for (const [model, checkpoint] of [
      [
        modelFromAgent(game.redAgentId, game.redValueModel),
        game.redValueModelCheckpoint,
      ],
      [
        modelFromAgent(game.blueAgentId, game.blueValueModel),
        game.blueValueModelCheckpoint,
      ],
    ] as const) {
      const destination =
        model === "challenger" ? challengerCheckpoints : incumbentCheckpoints;
      destination.add(checkpoint || "unknown");
    }
  }

  const byColor: ValueModelArenaSummary["challenger"]["byColor"] = {
    RED: { games: 0, points: 0, scoreRate: 0 },
    BLUE: { games: 0, points: 0, scoreRate: 0 },
  };
  for (const { result } of scored) {
    byColor[result.color].games++;
    byColor[result.color].points += result.score;
  }
  for (const color of ["RED", "BLUE"] as const) {
    byColor[color].scoreRate = byColor[color].games
      ? byColor[color].points / byColor[color].games
      : 0;
  }

  const pairScores: number[] = [];
  const byPair = new Map<string, typeof scored>();
  for (const entry of scored) {
    const suffix = /-(\d+)$/.exec(entry.game.gameId);
    if (!suffix) continue;
    const pair = Math.floor((Number(suffix[1]) - 1) / 2);
    const pairKey = `${entry.game.generationId}:${pair}`;
    const members = byPair.get(pairKey) ?? [];
    members.push(entry);
    byPair.set(pairKey, members);
  }
  let mismatchedPairSeed = false;
  let mismatchedPairCompetitor = false;
  let mismatchedPairRules = false;
  for (const members of byPair.values()) {
    if (
      members.length !== 2 ||
      members[0].result.color === members[1].result.color
    ) {
      continue;
    }
    if (members[0].game.seed !== members[1].game.seed) {
      mismatchedPairSeed = true;
      continue;
    }
    const competitors = new Set(
      members.flatMap(({ game }) => [
        competitorIdentity(game.redAgentId),
        competitorIdentity(game.blueAgentId),
      ])
    );
    if (competitors.size !== 1) {
      mismatchedPairCompetitor = true;
      continue;
    }
    const rules = new Set(
      members.map(
        ({ game }) => `${game.redMaxActions}:${game.blueMaxActions}`
      )
    );
    if (
      rules.size !== 1 ||
      members.some(
        ({ game }) => game.redMaxActions !== game.blueMaxActions
      )
    ) {
      mismatchedPairRules = true;
      continue;
    }
    pairScores.push((members[0].result.score + members[1].result.score) / 2);
  }
  const pairedOutcomes = { wins: 0, ties: 0, losses: 0 };
  for (const score of pairScores) {
    if (score > 0.5) pairedOutcomes.wins++;
    else if (score < 0.5) pairedOutcomes.losses++;
    else pairedOutcomes.ties++;
  }

  const random = seededRandom(0x474851);
  const draws: number[] = [];
  if (pairScores.length) {
    for (let sample = 0; sample < bootstrapSamples; sample++) {
      let total = 0;
      for (let index = 0; index < pairScores.length; index++) {
        total += pairScores[Math.floor(random() * pairScores.length)];
      }
      draws.push(total / pairScores.length);
    }
    draws.sort((left, right) => left - right);
  }

  const points = scored.reduce((sum, entry) => sum + entry.result.score, 0);
  const decisions = scored.reduce(
    (sum, entry) => sum + entry.game.quality.decisions,
    0
  );
  const unverifiedFallbackDecisions = scored.reduce(
    (sum, entry) =>
      sum + (entry.game.quality.unverifiedFallbackDecisions ?? 0),
    0
  );
  const unverifiedFallbackRate = decisions
    ? unverifiedFallbackDecisions / decisions
    : 0;
  const arenaDecisions = scored.flatMap(({ game }) =>
    game.decisions.map((decision) => ({ game, decision }))
  );
  const seededFallbackDecisions = arenaDecisions.filter(
    ({ decision }) => decision.fallback === "seeded"
  ).length;
  const incompleteTurnDecisions = arenaDecisions.filter(
    ({ decision }) => !decision.completedTurn
  ).length;
  const missingRuntimeProvenanceDecisions = arenaDecisions.filter(
    ({ decision }) =>
      !decision.searchBackend ||
      !decision.searchValueModelBackend ||
      !decision.searchCodeVersion
  ).length;
  const mismatchedSearchCodeDecisions = arenaDecisions.filter(
    ({ game, decision }) =>
      Boolean(decision.searchCodeVersion) &&
      decision.searchCodeVersion !== game.codeVersion
  ).length;
  const searchBackends = new Set(
    arenaDecisions.map(({ decision }) => decision.searchBackend ?? "unknown")
  );
  const valueModelBackends = new Set(
    arenaDecisions.map(
      ({ decision }) => decision.searchValueModelBackend ?? "unknown"
    )
  );
  const scoreRate = points / scored.length;
  const clamped = Math.max(0.001, Math.min(0.999, scoreRate));
  const ci95Low = percentile(draws, 0.025);
  const ci95High = percentile(draws, 0.975);
  const reasons: string[] = [];
  if (generationIds.size !== 1 || generationIds.has("unknown"))
    reasons.push("mixed-or-missing-generation-provenance");
  if (codeVersions.size !== 1 || codeVersions.has("unknown"))
    reasons.push("mixed-or-missing-code-provenance");
  if (
    incumbentCheckpoints.size !== 1 ||
    incumbentCheckpoints.has("unknown")
  )
    reasons.push("mixed-or-missing-incumbent-checkpoint");
  if (
    challengerCheckpoints.size !== 1 ||
    challengerCheckpoints.has("unknown")
  )
    reasons.push("mixed-or-missing-challenger-checkpoint");
  if (
    incumbentCheckpoints.size === 1 &&
    challengerCheckpoints.size === 1 &&
    [...incumbentCheckpoints][0] === [...challengerCheckpoints][0]
  )
    reasons.push("identical-model-checkpoints");
  if (expectedProvenance) {
    const sameValues = (actual: Set<string>, expected: string[]) =>
      JSON.stringify([...actual].sort()) ===
      JSON.stringify([...expected].sort());
    if (!sameValues(generationIds, [expectedProvenance.generationId]))
      reasons.push("generation-does-not-match-manifest");
    if (!sameValues(codeVersions, [expectedProvenance.codeVersion]))
      reasons.push("code-does-not-match-manifest");
    if (
      !sameValues(
        incumbentCheckpoints,
        expectedProvenance.incumbentCheckpoints
      )
    )
      reasons.push("incumbent-checkpoint-does-not-match-manifest");
    if (
      !sameValues(
        challengerCheckpoints,
        expectedProvenance.challengerCheckpoints
      )
    )
      reasons.push("challenger-checkpoint-does-not-match-manifest");
  }
  if (scored.length < 100) reasons.push("fewer-than-100-games");
  if (unverifiedFallbackRate > 0.05)
    reasons.push("excessive-unverified-search-rate");
  // Aggregate rates are useful diagnostics, but promotion cannot tolerate a
  // single blind complete-turn seed: it can directly determine a game and
  // manufacture apparent Elo. Exact HQ-safe fallbacks are not seeds and stay
  // separately represented by the existing fallback telemetry.
  if (seededFallbackDecisions > 0)
    reasons.push("seeded-fallback-decision");
  if (incompleteTurnDecisions > 0)
    reasons.push("incomplete-turn-decision");
  if (missingRuntimeProvenanceDecisions > 0)
    reasons.push("missing-search-runtime-provenance");
  if (mismatchedSearchCodeDecisions > 0)
    reasons.push("search-code-does-not-match-game");
  if (arenaDecisions.length && searchBackends.size !== 1)
    reasons.push("mixed-search-backends");
  if (arenaDecisions.length && valueModelBackends.size !== 1)
    reasons.push("mixed-value-model-backends");
  if (mismatchedPairSeed) reasons.push("mismatched-pair-seed");
  if (mismatchedPairCompetitor) reasons.push("mismatched-pair-competitor");
  if (mismatchedPairRules) reasons.push("mismatched-pair-rules");
  if (pairScores.length * 2 !== scored.length)
    reasons.push("incomplete-color-pair");
  if (scoreRate <= 0.5) reasons.push("challenger-did-not-outscore-incumbent");
  if (ci95Low <= 0.5) reasons.push("paired-ci-does-not-clear-50-percent");
  for (const color of ["RED", "BLUE"] as const) {
    if (byColor[color].games < 25)
      reasons.push(`insufficient-${color.toLowerCase()}-games`);
    if (byColor[color].scoreRate < 0.45) {
      reasons.push(`challenger-regresses-as-${color.toLowerCase()}`);
    }
  }

  return {
    games: scored.length,
    pairs: pairScores.length,
    provenance: {
      generationIds: [...generationIds].sort(),
      codeVersions: [...codeVersions].sort(),
      incumbentCheckpoints: [...incumbentCheckpoints].sort(),
      challengerCheckpoints: [...challengerCheckpoints].sort(),
    },
    searchQuality: {
      decisions,
      unverifiedFallbackDecisions,
      unverifiedFallbackRate: Number(unverifiedFallbackRate.toFixed(4)),
      seededFallbackDecisions,
      incompleteTurnDecisions,
      missingRuntimeProvenanceDecisions,
      mismatchedSearchCodeDecisions,
      searchBackends: [...searchBackends].sort(),
      valueModelBackends: [...valueModelBackends].sort(),
    },
    challenger: {
      points,
      scoreRate: Number(scoreRate.toFixed(4)),
      eloDifference: Number(
        (400 * Math.log10(clamped / (1 - clamped))).toFixed(1)
      ),
      byColor: Object.fromEntries(
        Object.entries(byColor).map(([color, record]) => [
          color,
          { ...record, scoreRate: Number(record.scoreRate.toFixed(4)) },
        ])
      ) as ValueModelArenaSummary["challenger"]["byColor"],
    },
    pairedOutcomes,
    pairBootstrap: {
      samples: bootstrapSamples,
      ci95Low: Number(ci95Low.toFixed(4)),
      ci95High: Number(ci95High.toFixed(4)),
    },
    promotionGate: { passed: reasons.length === 0, reasons },
  };
}
