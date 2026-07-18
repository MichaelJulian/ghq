# Exact tactical HQ supervision

Status: dataset and training pipeline validated; first classifier rejected.
No tactical classifier is active in production.

## Why this dataset exists

Eventual game outcomes are a noisy target for immediate tactics. A player can
lose 50 turns after a sound position, or win despite previously allowing an
avoidable HQ capture. The production value model also assigned a 97.3% win
probability to one position that exact search proved was already lost.

`scripts/build_tactical_hq_dataset.ts` supplies a direct target. For the last
configured number of turns by each player in every persisted HQ-capture game,
it exhaustively asks whether **any** complete defender turn survives every
same-turn HQ-capture reply. Each record retains the FEN, search decision,
value-model probability, v3 feature vector, exact node counts, and one of:

- `forced_hq_loss`: every legal defender turn loses the HQ on the reply;
- `safe`: at least one defender turn demonstrably survives;
- `inconclusive`: the exact search exhausted its node cap without proving
  either result.

The builder supports a fast first node cap followed by a larger retry of only
inconclusive positions. Safe labels remain proofs even when the complete tree
was not exhausted, because one explicit surviving turn is sufficient.

## First 100-game audit

Source generation: `vercel-arena-r3b3-724fa155-mrpjkg8u`.

- 100 games, 96 HQ captures, 7,718 decisions;
- exact code provenance `7d8d0f636e285d8b4c0fed71b9714dc186f65019`;
- zero unverified fallbacks in the recorded search telemetry;
- 384 audited positions using two own-turn lookbacks per player;
- 383 decisive exact labels: 15 forced losses and 368 safe positions;
- one position remained inconclusive after a two-million-node retry.

The most important finding is not the class balance. Among the 96 immediate
losing decisions, only 15 were already forced, 80 still had at least one exact
HQ-saving turn, and one was inconclusive. All 96 positions one losing-player
turn earlier were safe. This historical generation therefore contains at
least 80 avoidable immediate HQ losses and is unsuitable for naive outcome
training. By comparison, all four losses in the recent durable pilot were
exhaustively forced, demonstrating a substantial search-reliability gain.

## Rejected logistic detector

`scripts/train_tactical_hq_model.py` groups color-swapped games into indivisible
train, validation, and test units and selects a sparse, regularized tactical
feature subset. It requires at least 30 independent pairs and chooses a risk
threshold on validation only.

The first 61-feature logistic detector reached 100% precision and 75% recall
on validation, but only 50% precision and 33.3% recall on the untouched test
split. Its test ROC AUC was `0.939394`, but only three forced examples occurred
in that split. This is useful ranking evidence, not a safe tactical veto. The
artifact is rejected and must not be deployed.

The subsequent 16-game production arena supplied a strict external test set:
64 positions with 16 forced losses and 48 safe positions, all from revision
`f238b913be625c12b48707ab2ddf57d25ff11d21`. Training on the old control plus
the four-game pilot improved external ranking to ROC AUC `0.932292` and average
precision `0.847638`. At the validation-selected 95%-precision threshold it
flagged four forced losses with no false positives, but recall was only 25%; it
also flagged none of the three forced examples in the internal untouched test
split. The high-precision deployment gate therefore remains failed.

The next classifier needs more exact forced-loss examples from post-fix,
three-action games. Any runtime integration must clear a held-out
high-precision gate and then improve color-paired production-engine play; a
retrospective metric alone is insufficient.
