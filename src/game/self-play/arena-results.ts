import type { DurableSelfPlayGameResult } from "@/workflows/self-play-game";
import type { Player } from "@/game/engine-v2";

export interface ValueModelArenaSummary {
  games: number;
  pairs: number;
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

function modelFromAgent(agentId: string, recorded?: string): string {
  if (recorded === "challenger" || recorded === "incumbent") return recorded;
  return agentId.includes("-challenger-") ? "challenger" : "incumbent";
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
  bootstrapSamples = 5_000
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
  const byPair = new Map<number, typeof scored>();
  for (const entry of scored) {
    const suffix = /-(\d+)$/.exec(entry.game.gameId);
    if (!suffix) continue;
    const pair = Math.floor((Number(suffix[1]) - 1) / 2);
    const members = byPair.get(pair) ?? [];
    members.push(entry);
    byPair.set(pair, members);
  }
  for (const members of byPair.values()) {
    if (
      members.length !== 2 ||
      members[0].result.color === members[1].result.color
    ) {
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
  const scoreRate = points / scored.length;
  const clamped = Math.max(0.001, Math.min(0.999, scoreRate));
  const ci95Low = percentile(draws, 0.025);
  const ci95High = percentile(draws, 0.975);
  const reasons: string[] = [];
  if (scored.length < 100) reasons.push("fewer-than-100-games");
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
