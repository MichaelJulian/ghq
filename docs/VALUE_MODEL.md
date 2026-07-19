# GHQ value model

The value model predicts the probability that a specified player eventually
wins from a completed-turn position. It is trained from saved human games but
served as plain TypeScript and JSON, so production inference does not require
Python or a native machine-learning runtime.

## Data policy

- Human training uses decisive HQ-capture and resignation games. The legacy
  two-action experiment also admitted repetition draws with a neutral 0.5
  target. Production three-action self-play training admits only quality-gated
  HQ captures; timeouts, no-progress stops, repetitions, and max-turn games are
  excluded.
- Undo/redo actions are resolved before replay.
- Start-of-turn captures are restored from `historyLog`.
- Samples are created only after committed turns.
- Every position is represented from both players' perspectives.
- Train, calibration, validation, and test sets are split chronologically by
  whole game. Positions from one game can never leak across splits, and
  color-swapped self-play games sharing a seed remain in the same split.
- Probability calibration is fitted only on the calibration split. Model and
  self-play-share selection use validation; promotion evidence uses test.
- Each game receives equal aggregate sample weight, so unusually long games do
  not dominate training.

## Train locally

Training is an offline build step. Runtime inference remains Vercel-compatible.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-train.txt

pnpm value:positions \
  --games-csv /path/to/Games_rows.csv \
  --output .data/value-positions.jsonl

pnpm value:features \
  --positions .data/value-positions.jsonl \
  --output .data/value-features.jsonl

pnpm value:train \
  --dataset .data/value-features.jsonl \
  --output src/game/value-model/model.generated.json \
  --report .data/value-model-report.json
```

The committed `model.generated.json` is the only training artifact needed by
the deployed app. A future self-play job can write games in the same canonical
position format and reuse this trainer without changing production inference.

Two-action Vercel games can be extracted directly from private Blob storage:

```bash
pnpm value:download:2a \
  --generation-prefix vercel-r2b2- \
  --output .data/value-2a.jsonl
```

The trainer represents each 0.5 target as half-weight win and loss observations,
preserving the same exported tree format and TypeScript inference path.

Three-action training must pin both the exact search revision and the exact
value-model checkpoint that generated the behavior policy. It also requires an
exhaustive HQ-loss audit, complete adjacent color-swapped pairs, complete
paratrooper-policy telemetry, zero unverified fallbacks, and per-position
behavior-quality telemetry. The latter records the acting agents and
personality, selected turn, completed reply depth, fallback class, and timeout
status. Download the completed games first so auditing and extraction do not
depend on local Blob credentials and the V3 exporter can read that full decision
telemetry:

```bash
pnpm self-play:download \
  --generation <generation-id> \
  --output .data/<generation-id>.jsonl

pnpm self-play:audit-hq \
  --input .data/<generation-id>.jsonl \
  --max-nodes 2000000 \
  --output .data/<generation-id>-hq-audit.json \
  --fail-on-avoidable \
  --fail-on-inconclusive

pnpm value:download:vercel \
  --generation-prefix <generation-id> \
  --input .data/<generation-id>.jsonl \
  --created-at <manifest-created-at-iso> \
  --code-version <full-git-sha> \
  --value-model-checkpoint <checkpoint-id> \
  --search-backend native-python \
  --value-model-backend native-gbdt \
  --feature-schema v3 \
  --hq-audit-report .data/<generation-id>-hq-audit.json \
  --output .data/value-selfplay.jsonl

pnpm value:merge \
  --human .data/value-features.jsonl \
  --self-play .data/value-selfplay.jsonl \
  --code-version <full-git-sha> \
  --value-model-checkpoint <checkpoint-id> \
  --output .data/value-mixed.jsonl

pnpm value:inspect \
  --dataset .data/value-mixed.jsonl \
  --output .data/value-readiness.json
```

`--input` may be repeated when every cohort has the same search and behavior
provenance and the audit report covers all of them. The trainer requires at
least 30 independent audited color-swapped pairs from the self-play source
before it will create train, calibration, validation, and test splits. Do not
weaken that gate or train on a collection of unrelated search revisions. The
trainer independently revalidates the audit attestations, exact search/model
backends, revision and checkpoint binding, pair completeness, generation
boundaries, and duplicate sample identities; hand-assembled self-play JSONL
cannot bypass those gates by skipping the exporter or merger. It also hashes
each pair's canonical feature trajectory: repeated trajectories stay in the
same split and count only once toward the 30-unit minimum, even when they have
different pair IDs, seeds, or generation IDs.
`value:inspect` runs all trainer-boundary checks without fitting a model and
reports raw pairs, distinct trajectory units, duplicate pairs, and the exact
remaining deficit for each data source. It also reports fallback counts,
timed-out samples, and how many samples completed depth three or deeper. The V3
exporter, merger, and trainer all fail closed when behavior-quality telemetry is
missing, a reply was not completed to depth two, or the move came from a seeded
or otherwise unverified fallback. New compact Blob training artifacts preserve
the same behavior-quality fields as full-game downloads; historical artifacts
without those fields must be re-exported from their full persisted games.
Once the admission gate is met, validation and test reports stratify self-play
accuracy by fallback class, deadline status, and completed depth. Use those
results—not timeout frequency alone—to decide whether a behavior class should
be downweighted or excluded from a later experiment.

## Personalities

All personalities share the calibrated objective value model. A personality
does not get to redefine whether a position is winning; it only ranks positions
that are already inside its allowed objective-value envelope.

The runtime currently includes:

- `balanced`
- `fortress`
- `mobile_raider`
- `battery_commander`
- `para_specialist`
- `tactical_gambler`

Each profile defines normalized style weights, a maximum permitted sacrifice in
objective win probability, a bounded log-odds bonus, risk aversion, exploration
temperature, and a tactical search bias. Style is automatically suppressed as
a line becomes forcing, and a proven win always overrides personality.

```ts
const ranked = rankPersonalityCandidates(
  candidateTurns,
  "RED",
  "battery_commander"
);

const best = ranked[0];
console.log(best.evaluation.objectiveWinProbability);
console.log(best.evaluation.styleContributions);
```

`styleContributions` provides an explanation of which normalized preferences
raised or lowered a candidate. The self-play search will supply tacticality,
risk, and search-derived objective values when those are available.

## Self-play game runner

`playOneGame(config)` in `src/game/self-play/play-one-game.ts` runs a complete,
headless game through the same `public/engine.py` board used by production. It
does not duplicate GHQ rules: agents only choose from `generate_legal_moves()`,
and the production board applies captures, the three-action limit, turn changes,
and terminal outcomes.

Load the Pyodide engine once per worker and reuse it across a batch:

```ts
const engine = await loadV2Engine();
const result = await playOneGame({
  engine,
  red: createRandomAgent("red-random"),
  blue: createRandomAgent("blue-random"),
  seed: 42,
});
```

The result contains only serializable FEN, UCI, agent, seed, turn, and outcome
data. Supplying the same seed and deterministic agents reproduces the game.

## Bot Lab and FEN API

`/bot-lab` provides three views over the same production engine:

Run `pnpm dev:bot-lab`, then open `http://localhost:3000/bot-lab` to use the
dashboard locally.

- Play the Bot uses the production interactive board against the same search
  endpoint used by FEN analysis. Both sides use the production limit of three
  voluntary actions per turn.
- Character Arena plays one or several complete player turns, preserves the
  serialized engine state between requests, and allows every resulting position
  to be replayed.
- FEN Analysis returns the best turn found within a chosen time, depth, and beam
  budget. It shows the gradient-boosted model probabilities, personality style
  output, heuristic contributions, principal variation, and search statistics.

The underlying endpoint is public and accepts JSON:

```http
POST /api/ai/analyze
Content-Type: application/json

{
  "fen": "qr↓6/iii5/8/8/8/8/5III/6R↑Q IIIIIFFFPRRTH iiiiifffprrth r",
  "personality": "balanced",
  "turnNumber": 1,
  "timeMs": 750,
  "maxDepth": 2,
  "beamWidth": 8
}
```

The response includes `serializedState`; send that value instead of `fen` for
the next arena turn so draw offers and other state not represented by FEN are
preserved. Recommendations are labeled `best found` unless the requested search
horizon was exhaustive.
