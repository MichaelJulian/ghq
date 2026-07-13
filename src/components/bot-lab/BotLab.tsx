"use client";

import { useRef, useState } from "react";
import {
  Activity,
  Beaker,
  Bot,
  ChevronLeft,
  ChevronRight,
  Pause,
  Play,
  RotateCcw,
  Search,
} from "lucide-react";
import Header from "@/components/Header";
import { PositionBoard } from "@/components/bot-lab/PositionBoard";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import {
  GHQ_STARTING_FEN,
  type FenAnalysisRequest,
  type FenAnalysisResponse,
  type PersonalityId,
} from "@/game/analysis/types";
import { FENtoBoardState } from "@/game/notation";
import { PERSONALITIES } from "@/game/value-model/personalities";
import { cn } from "@/lib/utils";

type LabMode = "arena" | "fen";

interface ArenaSnapshot {
  fen: string;
  serializedState?: string;
  turnNumber: number;
  title: string;
  analysis?: FenAnalysisResponse;
}

const PERSONALITY_IDS = Object.keys(PERSONALITIES) as PersonalityId[];

export default function BotLab() {
  const [mode, setMode] = useState<LabMode>("arena");
  const [fenInput, setFenInput] = useState(GHQ_STARTING_FEN);
  const [redPersonality, setRedPersonality] =
    useState<PersonalityId>("battery_commander");
  const [bluePersonality, setBluePersonality] =
    useState<PersonalityId>("mobile_raider");
  const [timeMs, setTimeMs] = useState(1500);
  const [maxDepth, setMaxDepth] = useState(1);
  const [beamWidth, setBeamWidth] = useState(6);
  const [snapshots, setSnapshots] = useState<ArenaSnapshot[]>([
    initialSnapshot(GHQ_STARTING_FEN),
  ]);
  const [snapshotIndex, setSnapshotIndex] = useState(0);
  const [fenAnalysis, setFenAnalysis] = useState<FenAnalysisResponse>();
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>();
  const stopRequested = useRef(false);

  const snapshot = snapshots[snapshotIndex];
  const displayedFen =
    mode === "arena" ? snapshot.fen : fenAnalysis?.fen ?? fenInput;
  const activeAnalysis = mode === "arena" ? snapshot.analysis : fenAnalysis;

  const loadArenaFen = () => {
    try {
      FENtoBoardState(fenInput);
      setSnapshots([initialSnapshot(fenInput)]);
      setSnapshotIndex(0);
      setError(undefined);
    } catch (error) {
      setError(error instanceof Error ? error.message : "Invalid GHQ FEN");
    }
  };

  const runArenaTurns = async (count: number) => {
    if (running) return;
    stopRequested.current = false;
    setRunning(true);
    setError(undefined);
    let workingSnapshots = snapshots.slice(0, snapshotIndex + 1);
    let working = workingSnapshots[workingSnapshots.length - 1];
    try {
      for (let index = 0; index < count; index++) {
        if (stopRequested.current || working.analysis?.outcome) break;
        const side = FENtoBoardState(working.fen).currentPlayerTurn ?? "RED";
        const personality = side === "RED" ? redPersonality : bluePersonality;
        const analysis = await requestAnalysis({
          fen: working.serializedState ? undefined : working.fen,
          serializedState: working.serializedState,
          personality,
          turnNumber: working.turnNumber,
          timeMs,
          maxDepth,
          beamWidth,
        });
        const moves = analysis.search.best_turn.actions;
        const next: ArenaSnapshot = {
          fen: analysis.resultingFen,
          serializedState: analysis.serializedState,
          turnNumber: working.turnNumber + 1,
          title: `${side} · ${moves.length ? moves.join(" ") : "no action"}`,
          analysis,
        };
        workingSnapshots = [...workingSnapshots, next];
        working = next;
        setSnapshots(workingSnapshots);
        setSnapshotIndex(workingSnapshots.length - 1);
        if (analysis.outcome || moves.length === 0) break;
      }
    } catch (error) {
      setError(error instanceof Error ? error.message : "Arena turn failed");
    } finally {
      setRunning(false);
    }
  };

  const analyzeInputFen = async () => {
    if (running) return;
    setRunning(true);
    setError(undefined);
    try {
      const side = FENtoBoardState(fenInput).currentPlayerTurn ?? "RED";
      setFenAnalysis(
        await requestAnalysis({
          fen: fenInput,
          personality: side === "RED" ? redPersonality : bluePersonality,
          turnNumber: snapshot.turnNumber,
          timeMs,
          maxDepth,
          beamWidth,
        })
      );
    } catch (error) {
      setError(error instanceof Error ? error.message : "Analysis failed");
    } finally {
      setRunning(false);
    }
  };

  return (
    <main className="min-h-screen bg-[radial-gradient(circle_at_top_left,_#eef5ff,_#f8fafc_45%,_#fff7ed)] px-3 pb-16 text-slate-900 lg:px-8">
      <div className="mx-auto max-w-7xl">
        <Header />
        <div className="mb-5 flex flex-col justify-between gap-3 border-b border-slate-300 pb-5 md:flex-row md:items-end">
          <div>
            <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.22em] text-blue-700">
              <Beaker className="h-4 w-4" /> GHQ Bot Lab
            </div>
            <h1 className="text-3xl font-black tracking-tight md:text-4xl">
              Watch the characters think.
            </h1>
            <p className="mt-2 max-w-2xl text-sm text-slate-600">
              Run production-rules matchups, inspect model probabilities, and
              ask for the best turn found from any GHQ FEN.
            </p>
          </div>
          <div className="flex rounded-lg border bg-white p-1 shadow-sm">
            <ModeButton
              active={mode === "arena"}
              onClick={() => setMode("arena")}
            >
              <Bot /> Character arena
            </ModeButton>
            <ModeButton active={mode === "fen"} onClick={() => setMode("fen")}>
              <Search /> FEN analysis
            </ModeButton>
          </div>
        </div>

        <div className="grid gap-5 xl:grid-cols-[390px_minmax(0,1fr)]">
          <div className="flex flex-col items-center gap-3 xl:items-start">
            <PositionBoard fen={displayedFen} />
            {mode === "arena" && (
              <ReplayControls
                index={snapshotIndex}
                count={snapshots.length}
                running={running}
                onPrevious={() =>
                  setSnapshotIndex((value) => Math.max(0, value - 1))
                }
                onNext={() =>
                  setSnapshotIndex((value) =>
                    Math.min(snapshots.length - 1, value + 1)
                  )
                }
              />
            )}
          </div>

          <div className="min-w-0 space-y-4">
            <Card className="border-slate-300 bg-white/90">
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2">
                  {mode === "arena" ? <Bot /> : <Search />}
                  {mode === "arena"
                    ? "Character matchup"
                    : "Analyze a position"}
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-3 md:grid-cols-2">
                  <PersonalitySelect
                    label="Red character"
                    value={redPersonality}
                    onChange={setRedPersonality}
                    accent="red"
                  />
                  <PersonalitySelect
                    label="Blue character"
                    value={bluePersonality}
                    onChange={setBluePersonality}
                    accent="blue"
                  />
                </div>
                <Textarea
                  value={fenInput}
                  onChange={(event) => setFenInput(event.target.value)}
                  spellCheck={false}
                  className="min-h-20 resize-y font-mono text-xs"
                  aria-label="GHQ FEN"
                />
                <SearchControls
                  timeMs={timeMs}
                  maxDepth={maxDepth}
                  beamWidth={beamWidth}
                  setTimeMs={setTimeMs}
                  setMaxDepth={setMaxDepth}
                  setBeamWidth={setBeamWidth}
                />
                {error && (
                  <div className="rounded border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {error}
                  </div>
                )}
                {mode === "arena" ? (
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="outline"
                      onClick={loadArenaFen}
                      disabled={running}
                    >
                      <RotateCcw /> Load / reset FEN
                    </Button>
                    <Button onClick={() => runArenaTurns(1)} disabled={running}>
                      <Play /> Play one turn
                    </Button>
                    <Button onClick={() => runArenaTurns(8)} disabled={running}>
                      <Activity /> Play 8 turns
                    </Button>
                    {running && (
                      <Button
                        variant="secondary"
                        onClick={() => {
                          stopRequested.current = true;
                        }}
                      >
                        <Pause /> Pause after this turn
                      </Button>
                    )}
                  </div>
                ) : (
                  <div className="flex flex-wrap gap-2">
                    <Button onClick={analyzeInputFen} disabled={running}>
                      <Search /> {running ? "Searching…" : "Find best turn"}
                    </Button>
                    {fenAnalysis && (
                      <Button
                        variant="outline"
                        onClick={() => {
                          setFenInput(fenAnalysis.resultingFen);
                          setFenAnalysis(undefined);
                        }}
                      >
                        Apply best turn
                      </Button>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>

            {mode === "arena" && (
              <TurnHistory
                snapshots={snapshots}
                selected={snapshotIndex}
                onSelect={setSnapshotIndex}
              />
            )}
            {activeAnalysis ? (
              <AnalysisDetails analysis={activeAnalysis} />
            ) : (
              <EmptyAnalysis mode={mode} />
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

function initialSnapshot(fen: string): ArenaSnapshot {
  return { fen, turnNumber: 1, title: "Starting position" };
}

async function requestAnalysis(
  input: FenAnalysisRequest
): Promise<FenAnalysisResponse> {
  const response = await fetch("/api/ai/analyze", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(input),
  });
  const body = await response.json();
  if (!response.ok) throw new Error(body.error ?? "Analysis request failed");
  return body as FenAnalysisResponse;
}

function ModeButton({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "flex items-center gap-2 rounded-md px-3 py-2 text-sm font-semibold transition",
        active ? "bg-slate-900 text-white" : "text-slate-600 hover:bg-slate-100"
      )}
    >
      {children}
    </button>
  );
}

function PersonalitySelect({
  label,
  value,
  onChange,
  accent,
}: {
  label: string;
  value: PersonalityId;
  onChange: (value: PersonalityId) => void;
  accent: "red" | "blue";
}) {
  return (
    <label className="block">
      <span
        className={cn(
          "mb-1 block text-xs font-bold uppercase tracking-wider",
          accent === "red" ? "text-red-700" : "text-blue-700"
        )}
      >
        {label}
      </span>
      <select
        className="h-10 w-full rounded-md border bg-white px-3 text-sm font-semibold"
        value={value}
        onChange={(event) => onChange(event.target.value as PersonalityId)}
      >
        {PERSONALITY_IDS.map((id) => (
          <option key={id} value={id}>
            {PERSONALITIES[id].name}
          </option>
        ))}
      </select>
      <span className="mt-1 block text-xs text-slate-500">
        {PERSONALITIES[value].description}
      </span>
    </label>
  );
}

function SearchControls({
  timeMs,
  maxDepth,
  beamWidth,
  setTimeMs,
  setMaxDepth,
  setBeamWidth,
}: {
  timeMs: number;
  maxDepth: number;
  beamWidth: number;
  setTimeMs: (value: number) => void;
  setMaxDepth: (value: number) => void;
  setBeamWidth: (value: number) => void;
}) {
  return (
    <div>
      <div className="grid grid-cols-3 gap-2">
        <NumericControl
          label="Time (ms)"
          value={timeMs}
          min={50}
          max={3000}
          onChange={setTimeMs}
        />
        <NumericControl
          label="Depth"
          value={maxDepth}
          min={1}
          max={3}
          onChange={setMaxDepth}
        />
        <NumericControl
          label="Beam"
          value={beamWidth}
          min={2}
          max={16}
          onChange={setBeamWidth}
        />
      </div>
      <p className="mt-2 text-[11px] leading-4 text-slate-500">
        Beam controls how many diverse complete turns survive at each search
        node. A larger beam considers more plans but leaves less time to finish
        the requested reply depth. For opponent-aware analysis, start with
        depth 2 and beam 6–8; depth 3 with beam 15 is usually too wide for a
        three-second budget.
      </p>
    </div>
  );
}

function NumericControl({
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
        min={min}
        max={max}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-1"
      />
    </label>
  );
}

function ReplayControls({
  index,
  count,
  running,
  onPrevious,
  onNext,
}: {
  index: number;
  count: number;
  running: boolean;
  onPrevious: () => void;
  onNext: () => void;
}) {
  return (
    <div className="flex w-[360px] items-center justify-between rounded border bg-white px-2 py-1 shadow-sm">
      <Button
        variant="ghost"
        size="icon"
        onClick={onPrevious}
        disabled={index === 0 || running}
        aria-label="Previous position"
      >
        <ChevronLeft />
      </Button>
      <span className="text-xs font-semibold text-slate-600">
        Position {index + 1} / {count}
      </span>
      <Button
        variant="ghost"
        size="icon"
        onClick={onNext}
        disabled={index === count - 1 || running}
        aria-label="Next position"
      >
        <ChevronRight />
      </Button>
    </div>
  );
}

function TurnHistory({
  snapshots,
  selected,
  onSelect,
}: {
  snapshots: ArenaSnapshot[];
  selected: number;
  onSelect: (index: number) => void;
}) {
  return (
    <Card className="bg-white/90">
      <CardHeader className="pb-2">
        <CardTitle className="text-base">Turn log</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="max-h-44 space-y-1 overflow-y-auto pr-1">
          {snapshots.map((snapshot, index) => (
            <button
              key={index}
              onClick={() => onSelect(index)}
              className={cn(
                "flex w-full items-center justify-between rounded px-3 py-2 text-left text-xs",
                selected === index
                  ? "bg-slate-900 text-white"
                  : "bg-slate-50 hover:bg-slate-100"
              )}
            >
              <span className="font-semibold">{snapshot.title}</span>
              {snapshot.analysis && (
                <span className="font-mono opacity-70">
                  {Math.round(
                    snapshot.analysis.model.after.redWinProbability * 100
                  )}
                  % R
                </span>
              )}
            </button>
          ))}
        </div>
      </CardContent>
    </Card>
  );
}

function AnalysisDetails({ analysis }: { analysis: FenAnalysisResponse }) {
  const before = analysis.model.before;
  const after = analysis.model.after;
  const moverProbability =
    analysis.sideToMove === "RED"
      ? before.redWinProbability
      : before.blueWinProbability;
  const moverAfter =
    analysis.sideToMove === "RED"
      ? after.redWinProbability
      : after.blueWinProbability;
  const contributions = before.personality.styleContributions.slice(0, 6);
  const componentEntries = Object.entries(
    analysis.search.evaluation.before.weighted_components
  ).sort((left, right) => Math.abs(right[1]) - Math.abs(left[1]));

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card className="border-slate-300 bg-slate-950 text-slate-50 lg:col-span-2">
        <CardContent className="pt-5">
          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-center">
            <div>
              <div className="mb-1 text-xs font-semibold uppercase tracking-[0.2em] text-amber-300">
                {analysis.search.recommendation_label}
              </div>
              <div className="flex flex-wrap gap-2 font-mono text-lg font-bold">
                {analysis.search.best_turn.actions.length ? (
                  analysis.search.best_turn.actions.map((move) => (
                    <span key={move} className="rounded bg-white/10 px-2 py-1">
                      {move}
                    </span>
                  ))
                ) : (
                  <span>No player action</span>
                )}
              </div>
              {analysis.search.best_turn.automatic_captures.length > 0 && (
                <div className="mt-2 text-xs text-slate-400">
                  Forced first:{" "}
                  {analysis.search.best_turn.automatic_captures.join(" ")}
                </div>
              )}
            </div>
            <div className="grid grid-cols-3 gap-4 text-center text-xs">
              <Metric
                label="Nodes"
                value={analysis.search.search.nodes.toLocaleString()}
              />
              <Metric
                label="Depth"
                value={`${analysis.search.search.completed_depth_in_turns}/${analysis.search.search.requested_depth_in_turns}`}
              />
              <Metric
                label="Time"
                value={`${Math.round(analysis.search.search.elapsed_ms)} ms`}
              />
            </div>
          </div>
          {!analysis.search.search.exhaustive_within_requested_horizon && (
            <div className="mt-4 rounded border border-amber-400/30 bg-amber-300/10 px-3 py-2 text-xs text-amber-100">
              This is the strongest line found inside the time and beam budget,
              not a proof of the globally optimal turn.
            </div>
          )}
          {analysis.search.search.completed_depth_in_turns < 2 && (
            <div className="mt-2 rounded border border-red-400/30 bg-red-300/10 px-3 py-2 text-xs text-red-100">
              No full opponent reply was completed. Treat this turn as
              tactically unverified; increase time or reduce the beam.
            </div>
          )}
          {analysis.search.search.fallback_used !== "none" && (
            <div className="mt-2 text-xs text-slate-300">
              {analysis.search.search.fallback_used === "safe"
                ? "Search timed out after producing this tactically screened root turn, but before completing the requested depth."
                : "Search timed out before a tactically screened root turn was ready, so this came from the emergency positional policy and has no opponent-response verification."}
            </div>
          )}
        </CardContent>
      </Card>

      <Card className="bg-white/90">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            Gradient-boosted value model
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <ProbabilityBar
            label="Red before"
            value={before.redWinProbability}
            color="red"
          />
          <ProbabilityBar
            label="Red after"
            value={after.redWinProbability}
            color="red"
          />
          <ProbabilityBar
            label="Blue before"
            value={before.blueWinProbability}
            color="blue"
          />
          <ProbabilityBar
            label="Blue after"
            value={after.blueWinProbability}
            color="blue"
          />
          <div className="rounded bg-slate-50 p-3 text-sm">
            <span className="font-semibold">Mover change:</span>{" "}
            <span
              className={
                moverAfter >= moverProbability
                  ? "text-emerald-700"
                  : "text-red-700"
              }
            >
              {signedPercent(moverAfter - moverProbability)}
            </span>
          </div>
        </CardContent>
      </Card>

      <Card className="bg-white/90">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            {PERSONALITIES[analysis.personality].name} preferences
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="mb-3 grid grid-cols-2 gap-2 text-xs">
            <MetricBox
              label="Objective"
              value={percent(before.personality.objectiveWinProbability)}
            />
            <MetricBox
              label="Style bonus"
              value={before.personality.styleBonus.toFixed(3)}
            />
          </div>
          <div className="space-y-2">
            {contributions.length ? (
              contributions.map((item) => (
                <div
                  key={item.feature}
                  className="flex items-center justify-between gap-3 text-xs"
                >
                  <span className="truncate text-slate-600">
                    {prettyName(item.feature)}
                  </span>
                  <span
                    className={cn(
                      "font-mono font-semibold",
                      item.contribution >= 0
                        ? "text-emerald-700"
                        : "text-red-700"
                    )}
                  >
                    {item.contribution >= 0 ? "+" : ""}
                    {item.contribution.toFixed(3)}
                  </span>
                </div>
              ))
            ) : (
              <div className="text-sm text-slate-500">
                Balanced uses objective value without a style preference.
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="bg-white/90 lg:col-span-2">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">
            Search evaluator contributions
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid gap-x-6 gap-y-2 md:grid-cols-2">
            {componentEntries.map(([name, value]) => (
              <div
                key={name}
                className="flex items-center justify-between border-b border-slate-100 py-1 text-xs"
              >
                <span className="text-slate-600">{prettyName(name)}</span>
                <span
                  className={cn(
                    "font-mono font-semibold",
                    value >= 0 ? "text-red-700" : "text-blue-700"
                  )}
                >
                  {value >= 0 ? "+" : ""}
                  {value.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
          <div className="mt-3 text-xs text-slate-500">
            Positive values favor Red; negative values favor Blue. These are the
            search heuristic inputs, shown separately from the trained value
            model.
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function EmptyAnalysis({ mode }: { mode: LabMode }) {
  return (
    <Card className="border-dashed bg-white/60">
      <CardContent className="flex min-h-40 flex-col items-center justify-center pt-5 text-center text-slate-500">
        <Activity className="mb-2 h-7 w-7" />
        <div className="font-semibold text-slate-700">No model output yet</div>
        <div className="mt-1 max-w-md text-sm">
          {mode === "arena"
            ? "Play a turn to see the chosen line, win probabilities, personality preferences, and search diagnostics."
            : "Analyze the FEN to get the best turn found and its model breakdown."}
        </div>
        <code className="mt-3 rounded bg-slate-100 px-2 py-1 text-xs">
          POST /api/ai/analyze
        </code>
      </CardContent>
    </Card>
  );
}

function ProbabilityBar({
  label,
  value,
  color,
}: {
  label: string;
  value: number;
  color: "red" | "blue";
}) {
  const width = `${Math.max(0, Math.min(100, value * 100))}%`;
  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span>{label}</span>
        <span className="font-mono font-semibold">{percent(value)}</span>
      </div>
      <div className="h-2 overflow-hidden rounded bg-slate-100">
        <div
          className={cn(
            "h-full rounded",
            color === "red" ? "bg-red-400" : "bg-blue-400"
          )}
          style={{ width }}
        />
      </div>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="font-mono text-base font-bold">{value}</div>
      <div className="text-slate-400">{label}</div>
    </div>
  );
}

function MetricBox({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-slate-50 p-2">
      <div className="text-slate-500">{label}</div>
      <div className="font-mono text-sm font-bold">{value}</div>
    </div>
  );
}

function percent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}
function signedPercent(value: number): string {
  return `${value >= 0 ? "+" : ""}${(value * 100).toFixed(1)} points`;
}
function prettyName(value: string): string {
  return value.replaceAll("_", " ");
}
