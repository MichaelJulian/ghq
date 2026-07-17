"use client";

import { useEffect, useMemo, useState } from "react";
import { Activity, Database, Download, RefreshCw, Rocket } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

const STORAGE_KEY = "ghq-self-play-batches-v1";

interface RunReference {
  gameId: string;
  runId: string;
  red: string;
  blue: string;
}

interface StoredBatch {
  generationId: string;
  createdAt: string;
  redMaxActions?: number;
  blueMaxActions?: number;
  runs: RunReference[];
}

interface GameResult {
  gameId: string;
  decisions: unknown[];
  outcome: { winner?: "RED" | "BLUE"; termination: string };
  trainingPositions: number;
  storage?: { status: "saved" | "not-configured" };
}

interface PersistedGeneration {
  generationId: string;
  gameArtifacts: number;
  trainingArtifacts: number;
  bytes: number;
  updatedAt: string;
}

interface RunStatus {
  runId: string;
  status: string;
  completedAt?: string;
  result?: GameResult;
}

interface StartResponse {
  generationId: string;
  redMaxActions: number;
  blueMaxActions: number;
  runs: RunReference[];
  error?: string;
}

interface GenerationSummary {
  generationId: string;
  games: number;
  outcomes: Record<string, number>;
  terminations: Record<string, number>;
  fallbackRate: number;
  unverifiedFallbackRate: number;
  timedOutRate?: number;
  persistentCacheHitRate?: number;
  provenance?: {
    codeVersions: string[];
    valueModelCheckpoints: string[];
  };
  valueModelArena?: {
    challenger: {
      scoreRate: number;
      eloDifference: number;
      byColor: Record<string, { scoreRate: number }>;
    };
    pairBootstrap: { ci95Low: number; ci95High: number };
    promotionGate: { passed: boolean; reasons: string[] };
  };
}

function loadBatches(): StoredBatch[] {
  try {
    const value = window.localStorage.getItem(STORAGE_KEY);
    return value ? (JSON.parse(value) as StoredBatch[]) : [];
  } catch {
    return [];
  }
}

export function SelfPlayRuns() {
  const [games, setGames] = useState(24);
  const [timeMs, setTimeMs] = useState(20_000);
  const [depth, setDepth] = useState(2);
  const [beam, setBeam] = useState(6);
  const [batches, setBatches] = useState<StoredBatch[]>([]);
  const [statuses, setStatuses] = useState<Record<string, RunStatus>>({});
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string>();
  const [savedGenerations, setSavedGenerations] = useState<
    PersistedGeneration[]
  >([]);
  const [storageConfigured, setStorageConfigured] = useState<boolean>();
  const [generationSummary, setGenerationSummary] =
    useState<GenerationSummary>();

  useEffect(() => {
    setBatches(loadBatches());
    void refreshSavedGenerations();
  }, []);

  const latest = batches[0];
  const latestStatuses = useMemo(
    () => latest?.runs.map((run) => statuses[run.runId]).filter(Boolean) ?? [],
    [latest, statuses]
  );
  const completed = latestStatuses.filter(
    (run) => run.status === "completed"
  ).length;

  const saveBatches = (next: StoredBatch[]) => {
    const retained = next.slice(0, 12);
    setBatches(retained);
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(retained));
  };

  const refreshSavedGenerations = async () => {
    try {
      const response = await fetch("/api/self-play/generations");
      const body = (await response.json()) as {
        configured?: boolean;
        generations?: PersistedGeneration[];
        error?: string;
      };
      if (!response.ok) throw new Error(body.error ?? "Storage lookup failed");
      setStorageConfigured(Boolean(body.configured));
      setSavedGenerations(body.generations ?? []);
    } catch {
      setStorageConfigured(false);
    }
  };

  const startBatch = async () => {
    setBusy(true);
    setMessage(undefined);
    try {
      const response = await fetch("/api/self-play/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          games,
          timeMs,
          maxDepth: depth,
          beamWidth: beam,
          maxTurns: 160,
          repetitionLimit: 3,
          noProgressTurns: 24,
          redMaxActions: 3,
          blueMaxActions: 3,
          seed: Date.now() >>> 0,
        }),
      });
      const body = (await response.json()) as StartResponse;
      if (!response.ok) throw new Error(body.error ?? "Batch launch failed");
      saveBatches([
        {
          generationId: body.generationId,
          createdAt: new Date().toISOString(),
          redMaxActions: body.redMaxActions,
          blueMaxActions: body.blueMaxActions,
          runs: body.runs,
        },
        ...batches,
      ]);
      setMessage(
        `${body.runs.length} durable 3-action games launched on Vercel.`
      );
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Batch launch failed"
      );
    } finally {
      setBusy(false);
    }
  };

  const summarizeGeneration = async (generationId: string) => {
    setBusy(true);
    setMessage(undefined);
    try {
      const response = await fetch(
        `/api/self-play/generations/${encodeURIComponent(generationId)}/summary`
      );
      const body = (await response.json()) as GenerationSummary & {
        error?: string;
      };
      if (!response.ok) throw new Error(body.error ?? "Summary failed");
      setGenerationSummary(body);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Summary failed");
    } finally {
      setBusy(false);
    }
  };

  const refreshBatch = async () => {
    if (!latest) return;
    setBusy(true);
    setMessage(undefined);
    try {
      const responses = await Promise.all(
        latest.runs.map(async (run) => {
          const response = await fetch(`/api/self-play/runs/${run.runId}`);
          const body = (await response.json()) as RunStatus & {
            error?: string;
          };
          if (!response.ok)
            throw new Error(body.error ?? "Result lookup failed");
          return body;
        })
      );
      setStatuses((current) => ({
        ...current,
        ...Object.fromEntries(responses.map((run) => [run.runId, run])),
      }));
      await refreshSavedGenerations();
      setMessage(
        `${responses.filter((run) => run.status === "completed").length}/${
          responses.length
        } games complete.`
      );
    } catch (error) {
      setMessage(
        error instanceof Error ? error.message : "Result lookup failed"
      );
    } finally {
      setBusy(false);
    }
  };

  const exportResults = () => {
    if (!latest) return;
    const payload = latest.runs.map((run) => ({
      ...run,
      status: statuses[run.runId],
    }));
    const blob = new Blob([JSON.stringify(payload, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${latest.generationId}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <Card className="mt-5 border-indigo-200 bg-white/90">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2">
          <Rocket className="text-indigo-700" /> Vercel self-play batches
        </CardTitle>
        <p className="text-xs leading-5 text-slate-600">
          Launch independent durable games. Completed games and quality-gated
          training samples are saved to private Vercel Blob storage; run IDs
          also persist in this browser. Access is controlled by Vercel
          Authentication. Self-play uses the standard three actions per side.
        </p>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid gap-2 md:grid-cols-4">
          <BatchNumber
            label="Games"
            value={games}
            min={1}
            max={100}
            onChange={setGames}
          />
          <BatchNumber
            label="Time (ms)"
            value={timeMs}
            min={50}
            max={30_000}
            onChange={setTimeMs}
          />
          <div className="grid grid-cols-2 gap-2">
            <BatchNumber
              label="Depth"
              value={depth}
              min={1}
              max={3}
              onChange={setDepth}
            />
            <BatchNumber
              label="Beam"
              value={beam}
              min={2}
              max={16}
              onChange={setBeam}
            />
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={startBatch} disabled={busy}>
            <Rocket /> Launch batch
          </Button>
          <Button
            variant="outline"
            onClick={refreshBatch}
            disabled={busy || !latest}
          >
            <RefreshCw /> Refresh latest
          </Button>
          <Button
            variant="outline"
            onClick={exportResults}
            disabled={!latest || !latestStatuses.length}
          >
            <Download /> Export results
          </Button>
          <Button
            variant="outline"
            onClick={() => void refreshSavedGenerations()}
            disabled={busy}
          >
            <Database /> Refresh saved
          </Button>
        </div>
        {message && (
          <div className="rounded border border-slate-200 bg-slate-50 px-3 py-2 text-sm text-slate-700">
            {message}
          </div>
        )}
        {latest && (
          <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <div className="font-mono text-xs font-bold text-slate-800">
                  {latest.generationId}
                </div>
                <div className="mt-1 text-xs text-slate-500">
                  {new Date(latest.createdAt).toLocaleString()} ·{" "}
                  {latest.runs.length} games
                </div>
              </div>
              <div className="flex items-center gap-2 text-sm font-bold text-indigo-700">
                <Activity className="h-4 w-4" /> {completed}/
                {latest.runs.length} complete
              </div>
            </div>
            <div className="mt-3 max-h-52 space-y-1 overflow-y-auto">
              {latest.runs.map((run) => {
                const status = statuses[run.runId];
                const result = status?.result;
                return (
                  <div
                    key={run.runId}
                    className="grid gap-1 rounded bg-white px-2 py-2 text-xs md:grid-cols-[1fr_1fr_auto]"
                  >
                    <span className="font-mono text-slate-600">
                      {run.gameId}
                    </span>
                    <span>
                      {run.red} vs {run.blue}
                    </span>
                    <span className="font-semibold text-slate-700">
                      {result
                        ? `${result.outcome.winner ?? "DRAW"} · ${
                            result.outcome.termination
                          } · ${result.decisions.length} turns`
                        : status?.status ?? "not refreshed"}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        )}
        <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
          <div className="flex items-center gap-2 text-sm font-bold text-slate-800">
            <Database className="h-4 w-4 text-indigo-700" /> Persistent training
            store
          </div>
          {storageConfigured === false ? (
            <p className="mt-2 text-xs text-amber-800">
              No private Vercel Blob store is connected yet. Games still run,
              but permanent storage is disabled until the project has a Blob
              store.
            </p>
          ) : savedGenerations.length ? (
            <div className="mt-2 max-h-40 space-y-1 overflow-y-auto">
              {savedGenerations.slice(0, 12).map((generation) => (
                <button
                  type="button"
                  key={generation.generationId}
                  onClick={() =>
                    void summarizeGeneration(generation.generationId)
                  }
                  disabled={busy}
                  className="grid w-full gap-1 rounded bg-white px-2 py-2 text-left text-xs hover:bg-indigo-50 disabled:opacity-60 md:grid-cols-[1fr_auto]"
                >
                  <span className="font-mono text-slate-600">
                    {generation.generationId}
                  </span>
                  <span className="font-semibold text-slate-700">
                    {generation.gameArtifacts} games ·{" "}
                    {generation.trainingArtifacts} training files ·{" "}
                    {(generation.bytes / 1_000_000).toFixed(1)} MB
                  </span>
                </button>
              ))}
            </div>
          ) : (
            <p className="mt-2 text-xs text-slate-500">
              {storageConfigured
                ? "No completed Vercel games have been saved yet."
                : "Checking storage…"}
            </p>
          )}
          {generationSummary && (
            <div className="mt-3 rounded border border-indigo-200 bg-indigo-50 p-3 text-xs text-slate-700">
              <div className="font-mono font-bold">
                {generationSummary.generationId}
              </div>
              <div className="mt-1">
                {generationSummary.games} games · outcomes{" "}
                {JSON.stringify(generationSummary.outcomes)} · terminations{" "}
                {JSON.stringify(generationSummary.terminations)}
              </div>
              <div className="mt-1">
                fallback {(100 * generationSummary.fallbackRate).toFixed(1)}% ·
                unverified{" "}
                {(100 * generationSummary.unverifiedFallbackRate).toFixed(1)}%
                {generationSummary.timedOutRate !== undefined && (
                  <>
                    {" "}· timeout {(
                      100 * generationSummary.timedOutRate
                    ).toFixed(1)}%
                  </>
                )}
                {generationSummary.persistentCacheHitRate !== undefined && (
                  <>
                    {" "}· shared cache {(
                      100 * generationSummary.persistentCacheHitRate
                    ).toFixed(1)}%
                  </>
                )}
              </div>
              {generationSummary.provenance &&
                (generationSummary.provenance.codeVersions.length > 0 ||
                  generationSummary.provenance.valueModelCheckpoints.length >
                    0) && (
                  <div className="mt-1 break-all font-mono text-[10px] text-slate-500">
                    code {generationSummary.provenance.codeVersions.join(", ")} ·
                    models{" "}
                    {generationSummary.provenance.valueModelCheckpoints.join(
                      ", "
                    )}
                  </div>
                )}
              {generationSummary.valueModelArena && (
                <div className="mt-2 font-semibold text-indigo-900">
                  Challenger{" "}
                  {(
                    100 * generationSummary.valueModelArena.challenger.scoreRate
                  ).toFixed(1)}
                  % (
                  {generationSummary.valueModelArena.challenger.eloDifference >=
                  0
                    ? "+"
                    : ""}
                  {generationSummary.valueModelArena.challenger.eloDifference}{" "}
                  Elo) · paired 95% CI{" "}
                  {(
                    100 *
                    generationSummary.valueModelArena.pairBootstrap.ci95Low
                  ).toFixed(1)}
                  –
                  {(
                    100 *
                    generationSummary.valueModelArena.pairBootstrap.ci95High
                  ).toFixed(1)}
                  % · promotion{" "}
                  {generationSummary.valueModelArena.promotionGate.passed
                    ? "PASS"
                    : "HOLD"}
                  {!generationSummary.valueModelArena.promotionGate.passed && (
                    <span className="mt-1 block font-normal text-slate-600">
                      {generationSummary.valueModelArena.promotionGate.reasons.join(
                        ", "
                      )}
                    </span>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

function BatchNumber({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="text-xs font-semibold text-slate-600">
      {label}
      <Input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-1"
      />
    </label>
  );
}
