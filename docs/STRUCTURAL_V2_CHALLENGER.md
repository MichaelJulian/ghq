# Structural v2 challenger checkpoint

Status: rejected. It is not the production incumbent and failed the durable
promotion arena.

## Artifact

- Generated: `2026-07-18T09:20:46.768643+00:00`
- Training seed: `11`
- Dataset SHA-256: `ca192875684079802f65f38c71ffdb64b7f81540b725f1167b1265d832a6bc56`
- Artifact SHA-256: `4ef9abbbfc7a7a0ca340873008ad7f9bc5149ffe4bbcb4987f9e89ca966a9935`
- Schema: 167 append-only v2 features
- Model: 200 depth-3 gradient-boosted trees, learning rate 0.03,
  minimum leaf size 20
- Checkpoint commit: `1a4b6f7`

The authoritative copies are `api/_model_challenger.json` and
`src/game/value-model/model.challenger.generated.json`. Mirror and exact
TypeScript/Python prediction tests must pass whenever this checkpoint changes.

## Offline evaluation

| Split            | Incumbent log loss | Challenger log loss | Difference |
| ---------------- | -----------------: | ------------------: | ---------: |
| Human validation |           0.458895 |            0.450391 |  -0.008504 |
| Human test       |           0.423262 |            0.417520 |  -0.005742 |

The paired test bootstrap interval for challenger-minus-incumbent log loss is
`[-0.025933, 0.014607]`. The point estimate is favorable, but the interval
crosses zero; offline evidence alone does not authorize promotion.

## Production-engine screens

All screens used the production Python engine, paired color swaps, depth 2,
beam 6, and a two-second search budget. All quality gates passed.

| Games | Max turns | Challenger points | Pair W-T-L | Verified decisions |
| ----: | --------: | ----------------: | ---------: | -----------------: |
|     4 |        30 |           2.0 / 4 |      0-2-0 |             96.67% |
|     4 |        80 |           2.5 / 4 |      1-1-0 |             98.75% |
|     8 |        80 |           4.5 / 8 |      1-3-0 |             97.78% |
|     8 |       100 |           5.0 / 8 |      2-2-0 |             96.15% |

The three outcome-length screens total 12.0/20 points with a 4-6-0 paired
record. Including the 30-turn policy-divergence pilot gives 14.0/24 points and
a 4-8-0 paired record. This is a screening signal, not an Elo estimate.

## Durable Vercel arena rejection

Generation `vercel-arena-r3b3-6a5b54fe-mrq83cmf` ran 16 games as eight exact
color-swapped pairs on production revision
`f238b913be625c12b48707ab2ddf57d25ff11d21`. All 16 games ended by HQ capture.

- The challenger scored 6/16 points (37.5%), an observed Elo difference of
  `-88.7`.
- Its paired record was 0 wins, 6 ties, and 2 losses.
- It scored only 1/8 as Red (12.5%) and 5/8 as Blue (62.5%).
- The paired bootstrap interval was `[0.1875, 0.5]`.
- Search provenance was exact, but 26 of 1,438 decisions were unverified
  fallbacks, concentrated in one pathological game. That independently
  prevents promotion.
- A two-million-node exact audit proved all 16 immediate HQ losses were
  already forced: zero avoidable and zero inconclusive losses.
- The raw challenger value model nevertheless produced three high-confidence
  tactical contradictions, including losing-side win probabilities of 91.6%,
  97.4%, and 80.3% in exactly forced positions.

The exact loss audit confirms that the search layer avoided horizon blunders;
it does not rescue the evaluator. The challenger lost the paired arena and is
rejected. The incumbent remains production.

## Rejected sparse-correction alternative

An append-only logistic correction using the three material-dispersion
features looked stronger offline than the staged tree challenger: its human
test log loss was `0.414428` versus `0.437159` for its aligned incumbent
baseline, and its paired bootstrap interval favored the correction. Production
engine play contradicted that proxy:

- The 4-game, 80-turn pilot scored 2.0/4 with two tied pairs; every game hit the
  turn cap.
- The outcome-length 8-game, 100-turn screen scored 2.0/8 (25%) with a 0-2-2
  pair record. All eight games ended by HQ capture and the search-quality gate
  passed at 96.9% reply-verified decisions.

The sparse correction is therefore rejected. This is a concrete example of why
holdout prediction gains are only a screening signal and cannot replace the
paired production-engine arena.

## Rejected full structural correction

An append-only logistic correction using all 33 newly derived structural
features also improved the human-data proxy substantially. Its human test log
loss was `0.401612` versus `0.423262` for the incumbent, with a point-estimate
improvement of `-0.021650`; however, the paired bootstrap interval
`[-0.056380, 0.016083]` crossed zero.

The production-engine screens did not support advancing it:

- The 4-game, 80-turn pilot scored 2.0/4 with two tied pairs. All four games
  reached the turn cap despite both pairs showing policy divergence; the
  reply-verification rate was 99.1%.
- The outcome-length 8-game, 100-turn screen scored 3.5/8 (43.75%) with a
  1-2-1 pair record. Seven games ended by HQ capture, all four pairs diverged,
  and the search-quality gate passed at a 97.5% reply-verification rate.

This correction is rejected. Broad structural feature coverage improved
outcome prediction but did not improve move selection against the incumbent,
reinforcing the requirement that every value checkpoint pass color-paired
production-engine play before durable Vercel evaluation.

## Rejected symmetric-difference correction

To remove the collinearity between own, opponent, and difference versions of
the same structural measurements, a second logistic correction was restricted
to the 11 symmetric `own - opponent` features. It produced the strongest
human-data proxy result of the correction experiments: test log loss
`0.399163` versus `0.423262` for the incumbent. Its paired bootstrap point
estimate was `-0.024099`, but the interval `[-0.062561, 0.017690]` still
crossed zero.

The 4-game, 80-turn production-engine screen rejected it immediately. The
candidate scored 1.0/4 (25%), lost both color-swapped pairs, and changed play
in both pairs. Two games ended by HQ capture and two reached the turn cap. The
search-quality gate passed with a 99.7% reply-verification rate, so poor search
coverage does not explain the result.

This correction is rejected without a longer screen. Symmetry removed the
redundant-feature pathology but did not close the objective mismatch between
retrospective human-game outcome prediction and prospective move selection.

## Rejected tactical v3 retrain

An append-only v3 schema added 27 own/opponent/difference features describing
enemy infantry distance to HQ, nearby HQ attackers and defenders, and
speed-weighted HQ pressure. TypeScript and native Python extraction agree on
the schema, and both runtimes retain exact v1/v2 compatibility.

The resulting tree model failed the offline gate before production-engine
play. Human validation log loss was `0.483972` versus `0.458895` for the
incumbent, and test log loss was `0.452660` versus `0.423262`. The paired test
bootstrap point estimate for candidate-minus-incumbent was `+0.029398`, with a
95% interval of `[-0.011948, 0.073990]`. The candidate also failed every
human-retention validation constraint.

Raw airborne-infantry distance dominated the new tactical features, indicating
that sparse retrospective games let the model use HQ-approach geometry as a
game-state proxy instead of learning reliable forced-loss semantics. The
checkpoint is rejected without an arena. The schema remains available for a
future model trained on exact-audited tactical hard negatives rather than on
eventual game outcomes alone.

## Promotion-gate outcome

This artifact did not earn the planned 100-game arena. The 16-game durable
screen already showed an adverse paired result, severe Red regression,
high-confidence tactical value contradictions, and contaminated fallback
quality. Retain the incumbent and require a new checkpoint to start again at
the local production-engine screen.
