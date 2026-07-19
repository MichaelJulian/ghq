#!/usr/bin/env tsx
/** Download quality-gated Vercel self-play samples into the trainer's JSONL format. */

import "dotenv/config";
import { createWriteStream } from "node:fs";
import { createHash } from "node:crypto";
import { mkdir, readFile } from "node:fs/promises";
import { dirname } from "node:path";
import { get, list, type ListBlobResultBlob } from "@vercel/blob";

import { FENtoBoardState } from "../src/game/notation";
import {
  extractValueFeaturesV2,
  extractValueFeaturesV3,
  VALUE_FEATURE_NAMES,
  VALUE_FEATURE_NAMES_V2,
  VALUE_FEATURE_NAMES_V3,
} from "../src/game/value-model/features";
import { auditParatrooperTrainingPolicy } from "../src/game/self-play/training-policy";
import { colorSwapPairIntegrityRejectionReasons } from "../src/game/self-play/color-pairs";
import {
  isDurableTrainingDecisionEligible,
  type DurableSelfPlayGameResult,
} from "../src/workflows/self-play-game";

interface DurableTrainingSample {
  generationId: string;
  gameId: string;
  turnNumber: number;
  player: "RED" | "BLUE";
  outcomeValue: number;
  features: number[];
  fen?: string;
  codeVersion: string;
  valueModel?: "incumbent" | "challenger";
  valueModelCheckpoint?: string;
  searchBackend?: "pyodide" | "native-python";
  searchValueModelBackend?: "typescript-callback" | "native-gbdt";
}

interface ExactHqAuditReport {
  format: "ghq-exact-hq-audit-v1";
  generationIds: string[];
  codeVersions: string[];
  games: number;
  immediateHqLosses: number;
  maxNodesPerAudit: number;
  approvedTrainingGameIds: string[];
  audits: Array<{
    gameId: string;
    losingTurn?: number;
    losingPlayer?: "RED" | "BLUE";
    searchScore?: number;
    losingWinProbability?: number;
    safe_turns: number;
    forced_hq_loss: boolean;
    inconclusive: boolean;
    exhaustive: boolean;
  }>;
}

function pairedGameNumber(gameId: string): number {
  const match = gameId.match(/-(\d+)$/);
  if (!match) {
    throw new Error(`Unable to derive color-swapped pair from ${gameId}`);
  }
  const gameNumber = Number.parseInt(match[1], 10);
  if (!Number.isSafeInteger(gameNumber) || gameNumber < 1) {
    throw new Error(`Invalid paired game number in ${gameId}`);
  }
  return gameNumber;
}

function pairedGameId(generationId: string, gameId: string): string {
  const gameNumber = pairedGameNumber(gameId);
  const pairNumber = Math.floor((gameNumber - 1) / 2) + 1;
  return `${generationId}-pair-${String(pairNumber).padStart(4, "0")}`;
}

function isCompleteColorSwap(gameIds: string[]): boolean {
  if (gameIds.length !== 2) return false;
  const numbers = gameIds.map(pairedGameNumber).sort((a, b) => a - b);
  return numbers[0] % 2 === 1 && numbers[1] === numbers[0] + 1;
}

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  if (index === -1 || !process.argv[index + 1]) {
    throw new Error(`Missing required argument ${name}`);
  }
  return process.argv[index + 1];
}

function optionalArgument(name: string): string | undefined {
  const index = process.argv.indexOf(name);
  return index === -1 ? undefined : process.argv[index + 1];
}

function argumentsFor(name: string): string[] {
  const values: string[] = [];
  for (let index = 0; index < process.argv.length - 1; index++) {
    if (process.argv[index] === name) values.push(process.argv[index + 1]);
  }
  return values;
}

async function readDownloadedGames(
  path: string
): Promise<DurableSelfPlayGameResult[]> {
  const body = await readFile(path, "utf8");
  return body
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as DurableSelfPlayGameResult);
}

function persistedTrainingSamples(
  game: DurableSelfPlayGameResult
): DurableTrainingSample[] {
  if (!game.quality.trainingEligible || !game.outcome.winner) return [];
  return game.decisions
    .filter((decision) =>
      isDurableTrainingDecisionEligible(decision, game.outcome)
    )
    .map((decision) => ({
      generationId: game.generationId,
      gameId: game.gameId,
      turnNumber: decision.turnNumber,
      player: decision.player,
      outcomeValue: decision.player === game.outcome.winner ? 1 : 0,
      features: decision.features,
      fen: decision.fen,
      codeVersion: decision.searchCodeVersion ?? game.codeVersion,
      valueModel: decision.valueModel ?? "incumbent",
      valueModelCheckpoint:
        decision.player === "RED"
          ? game.redValueModelCheckpoint
          : game.blueValueModelCheckpoint,
      searchBackend: decision.searchBackend,
      searchValueModelBackend: decision.searchValueModelBackend,
    }));
}

async function selectedBlobs(generationPrefix: string) {
  const blobs: ListBlobResultBlob[] = [];
  let cursor: string | undefined;
  do {
    const page = await list({
      prefix: `self-play/generations/${generationPrefix}`,
      cursor,
      limit: 1000,
    });
    blobs.push(
      ...page.blobs.filter(
        (blob) =>
          blob.pathname.includes("/training/") &&
          blob.pathname.endsWith(".jsonl")
      )
    );
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs.sort((a, b) => a.pathname.localeCompare(b.pathname));
}

async function selectedGameBlobs(generationId: string) {
  const blobs: ListBlobResultBlob[] = [];
  let cursor: string | undefined;
  do {
    const page = await list({
      prefix: `self-play/generations/${generationId}/games/`,
      cursor,
      limit: 1000,
    });
    blobs.push(...page.blobs.filter((blob) => blob.pathname.endsWith(".json")));
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs.sort((a, b) => a.pathname.localeCompare(b.pathname));
}

async function readBlob(
  blob: ListBlobResultBlob
): Promise<DurableTrainingSample[]> {
  const result = await get(blob.pathname, {
    access: "private",
    useCache: false,
  });
  if (!result?.stream || result.statusCode !== 200) {
    throw new Error(`Unable to read ${blob.pathname}`);
  }
  const body = await new Response(result.stream).text();
  return body
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line) as DurableTrainingSample);
}

async function readGameBlob(
  blob: ListBlobResultBlob
): Promise<DurableSelfPlayGameResult> {
  const result = await get(blob.pathname, {
    access: "private",
    useCache: false,
  });
  if (!result?.stream || result.statusCode !== 200) {
    throw new Error(`Unable to read ${blob.pathname}`);
  }
  return (await new Response(
    result.stream
  ).json()) as DurableSelfPlayGameResult;
}

async function main() {
  const generationPrefix = argument("--generation-prefix");
  const codeVersion = argument("--code-version");
  const valueModelCheckpoint = argument("--value-model-checkpoint");
  const searchBackend = optionalArgument("--search-backend") ?? "native-python";
  const searchValueModelBackend =
    optionalArgument("--value-model-backend") ?? "native-gbdt";
  const outputPath = argument("--output");
  const hqAuditPath = argument("--hq-audit-report");
  const inputPaths = argumentsFor("--input");
  const localCreatedAt = optionalArgument("--created-at");
  if (inputPaths.length && !localCreatedAt) {
    throw new Error("--created-at is required with --input");
  }
  if (localCreatedAt && !Number.isFinite(Date.parse(localCreatedAt))) {
    throw new Error("--created-at must be an ISO timestamp");
  }
  const featureSchema = optionalArgument("--feature-schema") ?? "v1";
  if (!["v1", "v2", "v3"].includes(featureSchema)) {
    throw new Error("--feature-schema must be v1, v2, or v3");
  }
  const featureNames =
    featureSchema === "v3"
      ? VALUE_FEATURE_NAMES_V3
      : featureSchema === "v2"
      ? VALUE_FEATURE_NAMES_V2
      : VALUE_FEATURE_NAMES;
  const hqAuditText = await readFile(hqAuditPath, "utf8");
  const hqAudit = JSON.parse(hqAuditText) as ExactHqAuditReport;
  if (hqAudit.format !== "ghq-exact-hq-audit-v1") {
    throw new Error("Unsupported exact HQ audit report format");
  }
  if (hqAudit.maxNodesPerAudit < 100_000) {
    throw new Error("Exact HQ audit must use at least 100000 nodes per loss");
  }
  if (
    !hqAudit.codeVersions.includes(codeVersion) ||
    hqAudit.codeVersions.some((version) => version !== codeVersion)
  ) {
    throw new Error(
      "Exact HQ audit code provenance does not match the dataset"
    );
  }
  const auditedGenerations = new Set(hqAudit.generationIds);
  const approvedByAudit = new Set(
    hqAudit.audits
      .filter(
        (audit) =>
          audit.forced_hq_loss &&
          audit.exhaustive &&
          !audit.inconclusive &&
          audit.safe_turns === 0
      )
      .map((audit) => audit.gameId)
  );
  const auditByGame = new Map(
    hqAudit.audits.map((audit) => [audit.gameId, audit] as const)
  );
  if (
    approvedByAudit.size !== hqAudit.approvedTrainingGameIds.length ||
    hqAudit.approvedTrainingGameIds.some(
      (gameId) => !approvedByAudit.has(gameId)
    )
  ) {
    throw new Error("Exact HQ audit approval summary is inconsistent");
  }
  const hqAuditSha256 = createHash("sha256").update(hqAuditText).digest("hex");
  const gameBlobs = inputPaths.length
    ? []
    : (
        await Promise.all([...auditedGenerations].map(selectedGameBlobs))
      ).flat();
  const persistedGames: DurableSelfPlayGameResult[] = (
    await Promise.all(inputPaths.map(readDownloadedGames))
  ).flat();
  for (let index = 0; index < gameBlobs.length; index += 12) {
    persistedGames.push(
      ...(await Promise.all(
        gameBlobs.slice(index, index + 12).map(readGameBlob)
      ))
    );
  }
  const policyAuditsByGame = new Map(
    persistedGames.map((game) => [
      game.gameId,
      auditParatrooperTrainingPolicy(game.decisions),
    ])
  );
  const persistedGamesById = new Map(
    persistedGames.map((game) => [game.gameId, game] as const)
  );
  if (persistedGamesById.size !== persistedGames.length) {
    throw new Error("Downloaded game inputs contain duplicate game IDs");
  }
  const searchQualityEligibleByGame = new Map(
    persistedGames.map((game) => [
      game.gameId,
      game.quality.trainingEligible &&
        (game.quality.unverifiedFallbackDecisions ??
          game.decisions.filter(
            (decision) =>
              decision.fallback === "seeded" ||
              (decision.fallback !== "none" && decision.completedDepth < 2)
          ).length) === 0,
    ])
  );
  const blobs = inputPaths.length ? [] : await selectedBlobs(generationPrefix);
  if (!inputPaths.length && !blobs.length) {
    throw new Error("No matching training artifacts found");
  }

  const recordsByGame = new Map<
    string,
    {
      createdAt: string;
      generationId: string;
      pairId: string;
      samples: DurableTrainingSample[];
    }
  >();
  for (const game of persistedGames) {
    if (!inputPaths.length) break;
    const samples = persistedTrainingSamples(game);
    if (!samples.length) continue;
    recordsByGame.set(game.gameId, {
      createdAt: localCreatedAt!,
      generationId: game.generationId,
      pairId: pairedGameId(game.generationId, game.gameId),
      samples,
    });
  }
  for (let index = 0; index < blobs.length; index += 12) {
    const batch = blobs.slice(index, index + 12);
    const records = await Promise.all(batch.map(readBlob));
    for (let offset = 0; offset < batch.length; offset++) {
      const createdAt = batch[offset].uploadedAt.toISOString();
      for (const sample of records[offset]) {
        if (sample.codeVersion !== codeVersion) {
          throw new Error(
            `Search provenance mismatch in ${
              sample.gameId
            }: expected ${codeVersion}, received ${
              sample.codeVersion || "missing"
            }`
          );
        }
        if (!auditedGenerations.has(sample.generationId)) {
          throw new Error(
            `Generation ${sample.generationId} is missing from the exact HQ audit`
          );
        }
        if (sample.valueModelCheckpoint !== valueModelCheckpoint) {
          throw new Error(
            `Behavior checkpoint mismatch in ${
              sample.gameId
            }: expected ${valueModelCheckpoint}, received ${
              sample.valueModelCheckpoint || "missing"
            }`
          );
        }
        if (sample.searchBackend !== searchBackend) {
          throw new Error(
            `Search runtime mismatch in ${
              sample.gameId
            }: expected ${searchBackend}, received ${
              sample.searchBackend || "missing"
            }`
          );
        }
        if (sample.searchValueModelBackend !== searchValueModelBackend) {
          throw new Error(
            `Value runtime mismatch in ${
              sample.gameId
            }: expected ${searchValueModelBackend}, received ${
              sample.searchValueModelBackend || "missing"
            }`
          );
        }
        if (
          featureSchema === "v1" &&
          sample.features.length !== VALUE_FEATURE_NAMES.length
        ) {
          throw new Error(`Feature mismatch in ${sample.gameId}`);
        }
        if (featureSchema !== "v1" && !sample.fen) {
          throw new Error(
            `Missing FEN for ${featureSchema} features in ${sample.gameId}`
          );
        }
        const pairId = pairedGameId(sample.generationId, sample.gameId);
        const record = recordsByGame.get(sample.gameId);
        if (record) {
          if (
            record.generationId !== sample.generationId ||
            record.pairId !== pairId
          ) {
            throw new Error(
              `Inconsistent game provenance for ${sample.gameId}`
            );
          }
          record.samples.push(sample);
        } else {
          recordsByGame.set(sample.gameId, {
            createdAt,
            generationId: sample.generationId,
            pairId,
            samples: [sample],
          });
        }
      }
    }
  }

  for (const [gameId, record] of recordsByGame) {
    if (!auditedGenerations.has(record.generationId)) {
      throw new Error(
        `Generation ${record.generationId} is missing from the exact HQ audit`
      );
    }
    for (const sample of record.samples) {
      if (
        sample.gameId !== gameId ||
        sample.generationId !== record.generationId
      ) {
        throw new Error(`Inconsistent game provenance for ${gameId}`);
      }
      if (sample.codeVersion !== codeVersion) {
        throw new Error(
          `Search provenance mismatch in ${gameId}: expected ${codeVersion}, received ${
            sample.codeVersion || "missing"
          }`
        );
      }
      if (sample.valueModelCheckpoint !== valueModelCheckpoint) {
        throw new Error(
          `Behavior checkpoint mismatch in ${gameId}: expected ${valueModelCheckpoint}, received ${
            sample.valueModelCheckpoint || "missing"
          }`
        );
      }
      if (sample.searchBackend !== searchBackend) {
        throw new Error(
          `Search runtime mismatch in ${gameId}: expected ${searchBackend}, received ${
            sample.searchBackend || "missing"
          }`
        );
      }
      if (sample.searchValueModelBackend !== searchValueModelBackend) {
        throw new Error(
          `Value runtime mismatch in ${gameId}: expected ${searchValueModelBackend}, received ${
            sample.searchValueModelBackend || "missing"
          }`
        );
      }
      if (
        featureSchema === "v1" &&
        sample.features.length !== VALUE_FEATURE_NAMES.length
      ) {
        throw new Error(`Feature mismatch in ${gameId}`);
      }
      if (featureSchema !== "v1" && !sample.fen) {
        throw new Error(
          `Missing FEN for ${featureSchema} features in ${gameId}`
        );
      }
    }
  }

  const gamesByPair = new Map<string, string[]>();
  for (const [gameId, record] of recordsByGame) {
    if (!approvedByAudit.has(gameId)) continue;
    const policyAudit = policyAuditsByGame.get(gameId);
    if (!policyAudit?.eligible) continue;
    if (!searchQualityEligibleByGame.get(gameId)) continue;
    const games = gamesByPair.get(record.pairId) ?? [];
    games.push(gameId);
    gamesByPair.set(record.pairId, games);
  }
  const invalidColorSwapPairs: Array<{
    pairId: string;
    reasons: string[];
  }> = [];
  const completePairs = new Set<string>();
  for (const [pairId, gameIds] of gamesByPair) {
    if (!isCompleteColorSwap(gameIds)) continue;
    const [firstId, secondId] = [...gameIds].sort((left, right) =>
      left.localeCompare(right)
    );
    const first = persistedGamesById.get(firstId);
    const second = persistedGamesById.get(secondId);
    const reasons =
      first && second
        ? colorSwapPairIntegrityRejectionReasons(first, second)
        : ["missing-persisted-game"];
    if (reasons.length) {
      invalidColorSwapPairs.push({ pairId, reasons });
    } else {
      completePairs.add(pairId);
    }
  }
  if (!completePairs.size) {
    throw new Error("No complete quality-eligible color-swapped pairs found");
  }

  await mkdir(dirname(outputPath), { recursive: true });
  const output = createWriteStream(outputPath, { encoding: "utf8" });
  output.write(
    `${JSON.stringify({
      type: "schema",
      format: "ghq-value-features-v1",
      feature_names: featureNames,
      feature_schema: featureSchema,
      ruleset: "three-actions",
      source: "vercel-self-play",
      generation_prefix: generationPrefix,
      code_version: codeVersion,
      behavior_value_model_checkpoint: valueModelCheckpoint,
      self_play_search_backend: searchBackend,
      self_play_value_model_backend: searchValueModelBackend,
      paired_complete_only: true,
      exact_hq_audit_required: true,
      paratrooper_policy_audit_required: true,
      zero_unverified_fallbacks_required: true,
      color_swap_integrity_verified: true,
      exact_hq_audit_sha256: hqAuditSha256,
      exact_hq_audit_max_nodes: hqAudit.maxNodesPerAudit,
    })}\n`
  );

  let samples = 0;
  const games = new Set<string>();
  const generations = new Set<string>();
  for (const [gameId, record] of [...recordsByGame].sort(([left], [right]) =>
    left.localeCompare(right)
  )) {
    if (!completePairs.has(record.pairId)) continue;
    games.add(gameId);
    generations.add(record.generationId);
    for (const sample of record.samples) {
      const audit = auditByGame.get(sample.gameId);
      const forcedHqLossPredecessor = Boolean(
        audit?.forced_hq_loss &&
          audit.exhaustive &&
          !audit.inconclusive &&
          audit.safe_turns === 0 &&
          audit.losingTurn === sample.turnNumber &&
          audit.losingPlayer === sample.player
      );
      const tacticalValueContradiction = Boolean(
        forcedHqLossPredecessor &&
          typeof audit?.searchScore === "number" &&
          audit.searchScore <= -50_000 &&
          typeof audit.losingWinProbability === "number" &&
          audit.losingWinProbability > 0.5
      );
      let features = sample.features;
      if (featureSchema !== "v1") {
        const state = FENtoBoardState(sample.fen!);
        const extractor =
          featureSchema === "v3"
            ? extractValueFeaturesV3
            : extractValueFeaturesV2;
        features = extractor(
          {
            board: state.board,
            redReserve: state.redReserve,
            blueReserve: state.blueReserve,
            currentPlayer: state.currentPlayerTurn ?? sample.player,
            turnNumber: sample.turnNumber,
          },
          sample.player
        );
      }
      output.write(
        `${JSON.stringify({
          type: "sample",
          game_id: sample.gameId,
          generation_id: sample.generationId,
          pair_id: record.pairId,
          source: "vercel_self_play",
          code_version: sample.codeVersion,
          behavior_value_model: sample.valueModel ?? "unknown",
          behavior_value_model_checkpoint:
            sample.valueModelCheckpoint ?? "unknown",
          behavior_search_backend: sample.searchBackend ?? "unknown",
          behavior_value_model_backend:
            sample.searchValueModelBackend ?? "unknown",
          created_at: record.createdAt,
          outcome_reason: "hq-capture",
          turn: sample.turnNumber,
          perspective: sample.player,
          label: sample.outcomeValue,
          forced_hq_loss_predecessor: forcedHqLossPredecessor,
          tactical_value_contradiction: tacticalValueContradiction,
          audited_search_score: forcedHqLossPredecessor
            ? audit?.searchScore
            : undefined,
          audited_static_win_probability: forcedHqLossPredecessor
            ? audit?.losingWinProbability
            : undefined,
          features,
        })}\n`
      );
      samples++;
    }
  }
  await new Promise<void>((resolve, reject) => {
    output.end(resolve);
    output.on("error", reject);
  });
  process.stderr.write(
    `${JSON.stringify({
      generationPrefix,
      codeVersion,
      valueModelCheckpoint,
      searchBackend,
      searchValueModelBackend,
      generations: generations.size,
      artifacts: blobs.length,
      games: games.size,
      pairs: completePairs.size,
      excludedGames: recordsByGame.size - games.size,
      excludedByExactHqAudit: [...recordsByGame.keys()].filter(
        (gameId) => !approvedByAudit.has(gameId)
      ).length,
      excludedByParatrooperPolicy: [...recordsByGame.keys()].filter(
        (gameId) => !policyAuditsByGame.get(gameId)?.eligible
      ).length,
      excludedByUnverifiedFallback: [...recordsByGame.keys()].filter(
        (gameId) => !searchQualityEligibleByGame.get(gameId)
      ).length,
      excludedByMissingParatrooperPolicyTelemetry: [
        ...recordsByGame.keys(),
      ].filter((gameId) => !policyAuditsByGame.get(gameId)?.telemetryComplete)
        .length,
      excludedIncompletePairs: [...gamesByPair.values()].filter(
        (gameIds) => !isCompleteColorSwap(gameIds)
      ).length,
      excludedInvalidColorSwapPairs: invalidColorSwapPairs.length,
      invalidColorSwapPairs,
      samples,
      featureSchema,
      hqAuditPath,
      hqAuditSha256,
      outputPath,
    })}\n`
  );
}

void main();
