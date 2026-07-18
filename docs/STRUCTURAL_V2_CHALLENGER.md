# Structural v2 challenger checkpoint

Status: staged challenger only. It is not the production incumbent and has not
passed the durable promotion arena.

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

| Split | Incumbent log loss | Challenger log loss | Difference |
| --- | ---: | ---: | ---: |
| Human validation | 0.458895 | 0.450391 | -0.008504 |
| Human test | 0.423262 | 0.417520 | -0.005742 |

The paired test bootstrap interval for challenger-minus-incumbent log loss is
`[-0.025933, 0.014607]`. The point estimate is favorable, but the interval
crosses zero; offline evidence alone does not authorize promotion.

## Production-engine screens

All screens used the production Python engine, paired color swaps, depth 2,
beam 6, and a two-second search budget. All quality gates passed.

| Games | Max turns | Challenger points | Pair W-T-L | Verified decisions |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 30 | 2.0 / 4 | 0-2-0 | 96.67% |
| 4 | 80 | 2.5 / 4 | 1-1-0 | 98.75% |
| 8 | 80 | 4.5 / 8 | 1-3-0 | 97.78% |
| 8 | 100 | 5.0 / 8 | 2-2-0 | 96.15% |

The three outcome-length screens total 12.0/20 points with a 4-6-0 paired
record. Including the 30-turn policy-divergence pilot gives 14.0/24 points and
a 4-8-0 paired record. This is a screening signal, not an Elo estimate.

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

## Promotion gate

Before promotion, deploy this artifact as the challenger and run at least 100
durable, color-paired arena games. Reject the arena if it contains seeded or
shallow fallbacks, incomplete turns, broken pairs, mixed code/runtime/model
provenance, avoidable HQ losses, or inconclusive HQ audits. Retain the current
incumbent unless the clean arena's confidence and paired results clear the
promotion thresholds.
