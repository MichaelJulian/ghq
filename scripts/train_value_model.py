#!/usr/bin/env python3
"""Train and export a Vercel-compatible gradient-boosted GHQ value model."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    share_group = parser.add_mutually_exclusive_group()
    share_group.add_argument("--self-play-train-share", type=float)
    share_group.add_argument(
        "--self-play-train-shares",
        help="comma-separated validation-selected training-share grid",
    )
    parser.add_argument("--self-play-code-version")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def requested_self_play_shares(
    single: Optional[float], grid: Optional[str]
) -> List[float]:
    if grid is not None:
        try:
            shares = [float(value.strip()) for value in grid.split(",") if value.strip()]
        except ValueError as error:
            raise ValueError("self-play training shares must be numbers") from error
    else:
        shares = [0.5 if single is None else float(single)]
    if not shares:
        raise ValueError("at least one self-play training share is required")
    if any(not 0.0 < share < 1.0 for share in shares):
        raise ValueError("self-play training shares must be between zero and one")
    if len(set(shares)) != len(shares):
        raise ValueError("self-play training shares must be unique")
    return shares


def load_dataset(path: Path) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray, np.ndarray]:
    feature_names: List[str] | None = None
    rows: List[Dict[str, Any]] = []
    vectors: List[List[float]] = []
    labels: List[float] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            if item.get("type") == "schema":
                feature_names = list(item["feature_names"])
                continue
            if item.get("type") != "sample":
                continue
            vector = item.pop("features")
            if feature_names is None or len(vector) != len(feature_names):
                raise ValueError("feature schema mismatch")
            rows.append(item)
            vectors.append(vector)
            label = float(item["label"])
            if label not in (0.0, 0.5, 1.0):
                raise ValueError(f"unsupported outcome label {label}")
            labels.append(label)
    if not feature_names or not rows:
        raise ValueError("empty value-model dataset")
    return feature_names, rows, np.asarray(vectors, dtype=np.float64), np.asarray(labels, dtype=np.float64)


def data_source(row: Dict[str, Any]) -> str:
    return str(row.get("source") or ("vercel_self_play" if row.get("generation_id") else "human"))


def evaluation_unit(row: Dict[str, Any]) -> Tuple[str, str]:
    """Keep correlated color-swapped self-play games in one data split."""
    source = data_source(row)
    if source == "vercel_self_play":
        pair_id = str(row.get("pair_id") or "").strip()
        if not pair_id:
            raise ValueError(
                f"self-play row {row.get('game_id', '<missing>')} requires pair_id"
            )
        return source, pair_id
    return source, str(row["game_id"])


def validate_self_play_code_version(
    rows: List[Dict[str, Any]], required: Optional[str] = None
) -> Optional[str]:
    """Require one exact search revision for every self-play training row."""
    versions: set[str] = set()
    for row in rows:
        if data_source(row) != "vercel_self_play":
            continue
        version = str(row.get("code_version") or "").strip()
        if not version or version == "unknown":
            raise ValueError(
                "self-play rows require exact code_version provenance"
            )
        versions.add(version)
    if not versions:
        return None
    if len(versions) != 1:
        raise ValueError(
            f"self-play rows mix search revisions: {sorted(versions)}"
        )
    observed = next(iter(versions))
    if required is not None and observed != required:
        raise ValueError(
            f"self-play code version mismatch: expected {required}, received {observed}"
        )
    return observed


def chronological_split(rows: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    unit_dates: Dict[Tuple[str, str], str] = {}
    for row in rows:
        key = evaluation_unit(row)
        created_at = str(row["created_at"])
        unit_dates[key] = min(unit_dates.get(key, created_at), created_at)
    sets: Dict[str, set[Tuple[str, str]]] = {
        "train": set(),
        "validation": set(),
        "test": set(),
    }
    sources = sorted({source for source, _ in unit_dates})
    for source in sources:
        units = sorted(
            (key for key in unit_dates if key[0] == source),
            key=lambda key: (unit_dates[key], key[1]),
        )
        if len(units) < 30:
            raise ValueError(
                f"source {source} requires at least 30 independent evaluation units for splits"
            )
        train_end = max(1, int(len(units) * 0.70))
        validation_end = max(train_end + 1, int(len(units) * 0.85))
        sets["train"].update(units[:train_end])
        sets["validation"].update(units[train_end:validation_end])
        sets["test"].update(units[validation_end:])
    return {
        name: np.asarray(
            [
                index
                for index, row in enumerate(rows)
                if evaluation_unit(row) in game_ids
            ]
        )
        for name, game_ids in sets.items()
    }


def game_balanced_weights(
    rows: List[Dict[str, Any]],
    indices: np.ndarray,
    self_play_share: float = 0.5,
    balance_outcomes: bool = False,
    outcome_balance_sources: Optional[set[str]] = None,
) -> np.ndarray:
    counts = Counter(
        (data_source(rows[index]), rows[index]["game_id"]) for index in indices
    )
    source_games: Dict[str, set[str]] = {}
    for index in indices:
        source_games.setdefault(data_source(rows[index]), set()).add(rows[index]["game_id"])
    if len(source_games) == 1:
        source_weight = {next(iter(source_games)): 1.0}
    elif set(source_games) == {"human", "vercel_self_play"}:
        source_weight = {
            "human": 1.0 - self_play_share,
            "vercel_self_play": self_play_share,
        }
    else:
        source_weight = {source: 1.0 / len(source_games) for source in source_games}
    outcome_games: Dict[Tuple[str, str], set[str]] = {}
    balanced_sources = (
        set(source_games)
        if balance_outcomes and outcome_balance_sources is None
        else set(outcome_balance_sources or ())
    )
    if balance_outcomes:
        for index in indices:
            row = rows[index]
            source = data_source(row)
            if source not in balanced_sources:
                continue
            label = float(row["label"])
            if label == 0.5:
                outcome = "draw"
            elif (row["perspective"] == "RED") == (label == 1.0):
                outcome = "red_win"
            else:
                outcome = "blue_win"
            outcome_games.setdefault((source, outcome), set()).add(row["game_id"])

    weights_list: List[float] = []
    for index in indices:
        row = rows[index]
        source = data_source(row)
        game = row["game_id"]
        denominator = len(source_games[source])
        if source in balanced_sources:
            label = float(row["label"])
            if label == 0.5:
                outcome = "draw"
            elif (row["perspective"] == "RED") == (label == 1.0):
                outcome = "red_win"
            else:
                outcome = "blue_win"
            source_outcomes = {
                bucket for candidate_source, bucket in outcome_games if candidate_source == source
            }
            denominator = len(source_outcomes) * len(outcome_games[(source, outcome)])
        weights_list.append(
            source_weight[source] / counts[(source, game)] / denominator
        )
    weights = np.asarray(weights_list)
    return weights / weights.mean()


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    return np.log(clipped / (1 - clipped))


def sigmoid(values: np.ndarray) -> np.ndarray:
    positive = values >= 0
    result = np.empty_like(values, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def metrics(labels: np.ndarray, probabilities: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    log_scores = -(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    result = {
        "log_loss": round(float(np.average(log_scores, weights=weights)), 6),
        "brier": round(float(np.average((probabilities - labels) ** 2, weights=weights)), 6),
    }
    decisive = labels != 0.5
    if np.any(decisive) and len(np.unique(labels[decisive])) == 2:
        result["decisive_accuracy"] = round(
            float(
                accuracy_score(
                    labels[decisive],
                    probabilities[decisive] >= 0.5,
                    sample_weight=weights[decisive],
                )
            ),
            6,
        )
        result["decisive_roc_auc"] = round(
            float(
                roc_auc_score(
                    labels[decisive],
                    probabilities[decisive],
                    sample_weight=weights[decisive],
                )
            ),
            6,
        )
    result["draw_samples"] = int(np.sum(labels == 0.5))
    return result


def expand_soft_labels(
    vectors: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Represent a 0.5 target as half-weight loss and win observations."""
    draw = labels == 0.5
    if not np.any(draw):
        return vectors, labels.astype(np.int8), weights
    expanded_vectors = np.concatenate([vectors[~draw], vectors[draw], vectors[draw]])
    expanded_labels = np.concatenate(
        [labels[~draw], np.zeros(np.sum(draw)), np.ones(np.sum(draw))]
    ).astype(np.int8)
    expanded_weights = np.concatenate(
        [weights[~draw], weights[draw] * 0.5, weights[draw] * 0.5]
    )
    return expanded_vectors, expanded_labels, expanded_weights


def latest_position_indices(rows: List[Dict[str, Any]], indices: np.ndarray) -> np.ndarray:
    latest: Dict[Tuple[str, str, str], Tuple[int, int]] = {}
    for index in indices:
        row = rows[index]
        key = (data_source(row), row["game_id"], row["perspective"])
        candidate = (int(row["turn"]), int(index))
        if key not in latest or candidate[0] > latest[key][0]:
            latest[key] = candidate
    return np.asarray([item[1] for item in latest.values()])


def metrics_by_source(
    rows: List[Dict[str, Any]],
    indices: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> Dict[str, Any]:
    probability_by_index = {
        int(index): probabilities[offset] for offset, index in enumerate(indices)
    }
    result: Dict[str, Any] = {}
    for source in sorted({data_source(rows[index]) for index in indices}):
        source_indices = np.asarray(
            [index for index in indices if data_source(rows[index]) == source]
        )
        source_probabilities = np.asarray(
            [probability_by_index[int(index)] for index in source_indices]
        )
        source_weights = game_balanced_weights(rows, source_indices)
        result[source] = metrics(
            labels[source_indices], source_probabilities, source_weights
        )
        result[source]["games"] = len(
            {rows[index]["game_id"] for index in source_indices}
        )
        result[source]["samples"] = len(source_indices)
        result[source]["by_perspective"] = {}
        for perspective in ("RED", "BLUE"):
            perspective_indices = np.asarray(
                [
                    index
                    for index in source_indices
                    if rows[index]["perspective"] == perspective
                ],
                dtype=np.int64,
            )
            if not len(perspective_indices):
                raise ValueError(
                    f"source {source} split is missing {perspective} perspective samples"
                )
            perspective_probabilities = np.asarray(
                [probability_by_index[int(index)] for index in perspective_indices]
            )
            result[source]["by_perspective"][perspective] = metrics(
                labels[perspective_indices],
                perspective_probabilities,
                game_balanced_weights(rows, perspective_indices),
            )
    return result


def validation_selection_gates(
    candidate: Dict[str, Any], baseline: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Mirror promotion's source/color retention rules during model selection."""
    gates: List[Dict[str, Any]] = []

    def add(name: str, source: str, maximum: float, perspective: Optional[str] = None) -> None:
        candidate_record = candidate[source]
        baseline_record = baseline[source]
        if perspective is not None:
            candidate_record = candidate_record["by_perspective"][perspective]
            baseline_record = baseline_record["by_perspective"][perspective]
        delta = float(candidate_record["log_loss"]) - float(
            baseline_record["log_loss"]
        )
        gates.append(
            {
                "name": name,
                "passed": delta <= maximum,
                "candidate_minus_baseline": round(delta, 6),
                "maximum": maximum,
            }
        )

    add("human-log-loss", "human", 0.01)
    add("self-play-log-loss", "vercel_self_play", 0.0)
    for perspective in ("RED", "BLUE"):
        add(
            f"human-{perspective.lower()}-log-loss",
            "human",
            0.015,
            perspective,
        )
        add(
            f"self-play-{perspective.lower()}-log-loss",
            "vercel_self_play",
            0.0,
            perspective,
        )
    return gates


def select_validation_candidate(
    candidates: List[Dict[str, Any]],
    *,
    baseline_constrained: bool,
    share_key: str = "share",
    score_key: str = "score",
) -> Tuple[Dict[str, Any], bool]:
    """Select a candidate without spending avoidable human-data tolerance.

    Once a baseline exists, a self-play share is a risk budget rather than an
    ordinary hyperparameter: larger shares permit more distribution shift away
    from observed human play.  Prefer the smallest share that clears every
    validation constraint, then use validation loss only to choose among model
    shapes at that share.  If no candidate clears the constraints, return the
    best diagnostic candidate but mark the selection infeasible so promotion
    remains impossible.

    Unconstrained first-time training still minimizes validation loss normally.
    """
    if not candidates:
        raise ValueError("candidate selection requires at least one candidate")
    feasible = [item for item in candidates if item["constraints_passed"]]
    if baseline_constrained and feasible:
        minimum_share = min(float(item[share_key]) for item in feasible)
        least_shifted = [
            item
            for item in feasible
            if float(item[share_key]) == minimum_share
        ]
        return min(least_shifted, key=lambda item: item[score_key]), True
    selection_pool = feasible if feasible else candidates
    return min(selection_pool, key=lambda item: item[score_key]), bool(feasible)


def paired_game_bootstrap(
    rows: List[Dict[str, Any]],
    indices: np.ndarray,
    labels: np.ndarray,
    candidate_probabilities: np.ndarray,
    baseline_probabilities: np.ndarray,
    random_state: int,
    samples: int = 2000,
) -> Dict[str, Any]:
    """Estimate candidate-minus-baseline loss by resampling independent units.

    Positions inside a game are highly correlated. Treating them as
    independent would produce falsely narrow confidence intervals. The two
    color-swapped self-play games sharing a seed are correlated as well, so
    each bootstrap draw resamples pair-level means within each source.
    """
    candidate_by_index = {
        int(index): candidate_probabilities[offset]
        for offset, index in enumerate(indices)
    }
    baseline_by_index = {
        int(index): baseline_probabilities[offset]
        for offset, index in enumerate(indices)
    }
    by_source_game: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for index in indices:
        row = rows[int(index)]
        source = data_source(row)
        unit = evaluation_unit(row)[1]
        label = labels[int(index)]
        candidate = np.clip(candidate_by_index[int(index)], 1e-7, 1 - 1e-7)
        baseline = np.clip(baseline_by_index[int(index)], 1e-7, 1 - 1e-7)
        record = by_source_game.setdefault(source, {}).setdefault(
            unit,
            {
                "log_loss": [],
                "brier": [],
                "by_perspective": {},
            },
        )
        candidate_log = -(label * math.log(candidate) + (1 - label) * math.log(1 - candidate))
        baseline_log = -(label * math.log(baseline) + (1 - label) * math.log(1 - baseline))
        record["log_loss"].append(candidate_log - baseline_log)
        record["brier"].append((candidate - label) ** 2 - (baseline - label) ** 2)
        perspective_record = record["by_perspective"].setdefault(
            row["perspective"], {"log_loss": [], "brier": []}
        )
        perspective_record["log_loss"].append(candidate_log - baseline_log)
        perspective_record["brier"].append(
            (candidate - label) ** 2 - (baseline - label) ** 2
        )

    rng = np.random.default_rng(random_state)

    def summarize(source_arrays: Dict[str, np.ndarray]) -> Dict[str, float]:
        point = float(np.mean([values.mean() for values in source_arrays.values()]))
        draws = np.empty(samples, dtype=np.float64)
        for sample in range(samples):
            source_means = []
            for values in source_arrays.values():
                selected = rng.integers(0, len(values), size=len(values))
                source_means.append(float(np.mean(values[selected])))
            draws[sample] = float(np.mean(source_means))
        return {
            "candidate_minus_baseline": round(point, 6),
            "ci95_low": round(float(np.quantile(draws, 0.025)), 6),
            "ci95_high": round(float(np.quantile(draws, 0.975)), 6),
        }

    result: Dict[str, Any] = {
        "bootstrap_samples": samples,
        "by_source": {},
        "by_perspective": {},
        "by_source_and_perspective": {},
    }
    for metric in ("log_loss", "brier"):
        source_arrays = {
            source: np.asarray(
                [np.mean(game[metric]) for game in games.values()],
                dtype=np.float64,
            )
            for source, games in by_source_game.items()
        }
        result[metric] = summarize(source_arrays)
        for source, values in source_arrays.items():
            source_record = result["by_source"].setdefault(source, {})
            source_record[metric] = summarize({source: values})
            source_record["resampling_units"] = len(values)
        for perspective in ("RED", "BLUE"):
            perspective_arrays = {
                source: np.asarray(
                    [
                        np.mean(game["by_perspective"][perspective][metric])
                        for game in games.values()
                        if perspective in game["by_perspective"]
                    ],
                    dtype=np.float64,
                )
                for source, games in by_source_game.items()
            }
            result["by_perspective"].setdefault(perspective, {})[metric] = summarize(
                perspective_arrays
            )
            for source, values in perspective_arrays.items():
                result["by_source_and_perspective"].setdefault(source, {}).setdefault(
                    perspective, {}
                )[metric] = summarize({source: values})
    return result


def export_tree(tree: Any) -> Dict[str, Any]:
    raw = tree.tree_
    return {
        "children_left": raw.children_left.astype(int).tolist(),
        "children_right": raw.children_right.astype(int).tolist(),
        "feature": raw.feature.astype(int).tolist(),
        "threshold": [round(float(value), 12) for value in raw.threshold],
        "value": [round(float(value), 12) for value in raw.value[:, 0, 0]],
    }


def exported_raw_scores(artifact: Dict[str, Any], vectors: np.ndarray) -> np.ndarray:
    raw = np.full(len(vectors), artifact["base_raw_score"], dtype=np.float64)
    for tree in artifact["trees"]:
        values = np.empty(len(vectors), dtype=np.float64)
        for row_index, vector in enumerate(vectors):
            node = 0
            while tree["children_left"][node] != -1:
                feature = tree["feature"][node]
                node = (
                    tree["children_left"][node]
                    if vector[feature] <= tree["threshold"][node]
                    else tree["children_right"][node]
                )
            values[row_index] = tree["value"][node]
        raw += artifact["learning_rate"] * values
    return raw


def exported_probabilities(artifact: Dict[str, Any], vectors: np.ndarray) -> np.ndarray:
    raw = exported_raw_scores(artifact, vectors)
    calibration = artifact["calibration"]
    return sigmoid(calibration["scale"] * raw + calibration["intercept"])


def main() -> None:
    args = parse_args()
    self_play_train_shares = requested_self_play_shares(
        args.self_play_train_share, args.self_play_train_shares
    )
    feature_names, rows, vectors, labels = load_dataset(args.dataset)
    self_play_code_version = validate_self_play_code_version(
        rows, args.self_play_code_version
    )
    splits = chronological_split(rows)
    weights = {name: game_balanced_weights(rows, indices) for name, indices in splits.items()}
    fit_weights_by_share = {
        share: {
            name: game_balanced_weights(
                rows,
                indices,
                share,
                balance_outcomes=True,
                outcome_balance_sources={"vercel_self_play"},
            )
            for name, indices in splits.items()
        }
        for share in self_play_train_shares
    }

    candidates = [
        {"n_estimators": 120, "max_depth": 2, "learning_rate": 0.05, "min_samples_leaf": 30},
        {"n_estimators": 160, "max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 30},
        {"n_estimators": 220, "max_depth": 2, "learning_rate": 0.04, "min_samples_leaf": 20},
        {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "min_samples_leaf": 20},
    ]
    train_indices = splits["train"]
    validation_indices = splits["validation"]
    baseline_artifact: Optional[Dict[str, Any]] = None
    baseline_validation_by_source: Optional[Dict[str, Any]] = None
    if args.baseline:
        baseline_artifact = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline_artifact.get("feature_names") != feature_names:
            raise ValueError("baseline feature schema does not match dataset")
        baseline_validation_by_source = metrics_by_source(
            rows,
            validation_indices,
            labels,
            exported_probabilities(
                baseline_artifact, vectors[validation_indices]
            ),
        )

    trained: List[Dict[str, Any]] = []
    candidate_validation: List[Dict[str, Any]] = []
    # When selecting among shares, compare every candidate against one fixed
    # 50/50 source-balanced validation objective. Letting each share redefine
    # its own validation weights would make scores incomparable.
    selection_weights = (
        weights["validation"]
        if len(self_play_train_shares) > 1
        else fit_weights_by_share[self_play_train_shares[0]]["validation"]
    )
    for share in self_play_train_shares:
        train_vectors, train_labels, train_weights = expand_soft_labels(
            vectors[train_indices],
            labels[train_indices],
            fit_weights_by_share[share]["train"],
        )
        for parameters in candidates:
            model = GradientBoostingClassifier(
                **parameters,
                loss="log_loss",
                subsample=0.85,
                random_state=args.random_state,
            )
            model.fit(
                train_vectors,
                train_labels,
                sample_weight=train_weights,
            )
            validation_probability = model.predict_proba(
                vectors[validation_indices]
            )[:, 1]
            score = metrics(
                labels[validation_indices],
                validation_probability,
                selection_weights,
            )["log_loss"]
            source_metrics = metrics_by_source(
                rows,
                validation_indices,
                labels,
                validation_probability,
            )
            selection_gates = (
                validation_selection_gates(
                    source_metrics, baseline_validation_by_source
                )
                if baseline_validation_by_source is not None
                else []
            )
            constraints_passed = all(
                gate["passed"] for gate in selection_gates
            )
            print(
                json.dumps(
                    {
                        "self_play_train_share": share,
                        "candidate": parameters,
                        "validation_log_loss": round(score, 6),
                        "validation_constraints_passed": constraints_passed,
                    }
                )
            )
            candidate_validation.append(
                {
                    "self_play_train_share": share,
                    "hyperparameters": parameters,
                    "validation_log_loss": round(score, 6),
                    "validation_by_source": source_metrics,
                    "validation_selection_gates": selection_gates,
                    "validation_constraints_passed": constraints_passed,
                }
            )
            trained.append(
                {
                    "score": score,
                    "share": share,
                    "parameters": parameters,
                    "model": model,
                    "constraints_passed": constraints_passed,
                }
            )
    selected, feasible_selection = select_validation_candidate(
        trained,
        baseline_constrained=baseline_artifact is not None,
    )
    selected_self_play_train_share = selected["share"]
    best_parameters = selected["parameters"]
    model = selected["model"]
    validation_constraints_passed = (
        feasible_selection or baseline_artifact is None
    )

    validation_raw_probability = model.predict_proba(vectors[validation_indices])[:, 1]
    calibrator = LogisticRegression(random_state=args.random_state)
    calibration_vectors, calibration_labels, calibration_weights = expand_soft_labels(
        safe_logit(validation_raw_probability).reshape(-1, 1),
        labels[validation_indices],
        selection_weights,
    )
    calibrator.fit(
        calibration_vectors,
        calibration_labels,
        sample_weight=calibration_weights,
    )
    calibration_scale = float(calibrator.coef_[0, 0])
    calibration_intercept = float(calibrator.intercept_[0])

    split_metrics: Dict[str, Any] = {}
    all_probabilities: Dict[str, np.ndarray] = {}
    for name, indices in splits.items():
        raw_probability = model.predict_proba(vectors[indices])[:, 1]
        probability = sigmoid(calibration_scale * safe_logit(raw_probability) + calibration_intercept)
        all_probabilities[name] = probability
        split_metrics[name] = metrics(labels[indices], probability, weights[name])
        latest = latest_position_indices(rows, indices)
        probability_by_index = {int(index): probability[offset] for offset, index in enumerate(indices)}
        latest_probability = np.asarray([probability_by_index[int(index)] for index in latest])
        latest_weights = game_balanced_weights(rows, latest)
        split_metrics[name]["latest_position"] = metrics(
            labels[latest], latest_probability, latest_weights
        )
        split_metrics[name]["games"] = len({rows[index]["game_id"] for index in indices})
        split_metrics[name]["samples"] = len(indices)
        split_metrics[name]["by_source"] = metrics_by_source(
            rows, indices, labels, probability
        )

    baseline_metrics: Dict[str, Any] | None = None
    paired_bootstrap_test: Dict[str, Any] | None = None
    if baseline_artifact is not None:
        baseline_metrics = {}
        for name, indices in splits.items():
            probability = exported_probabilities(baseline_artifact, vectors[indices])
            baseline_metrics[name] = metrics(labels[indices], probability, weights[name])
            baseline_metrics[name]["games"] = len(
                {rows[index]["game_id"] for index in indices}
            )
            baseline_metrics[name]["samples"] = len(indices)
            baseline_metrics[name]["by_source"] = metrics_by_source(
                rows, indices, labels, probability
            )
        paired_bootstrap_test = paired_game_bootstrap(
            rows,
            splits["test"],
            labels,
            all_probabilities["test"],
            exported_probabilities(baseline_artifact, vectors[splits["test"]]),
            args.random_state,
        )

    priors = model.init_.class_prior_
    base_raw_score = math.log(float(priors[1]) / float(priors[0]))
    importances = sorted(
        zip(feature_names, model.feature_importances_), key=lambda item: item[1], reverse=True
    )
    artifact = {
        "format": "ghq-gradient-boosted-value-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "feature_names": feature_names,
        "base_raw_score": base_raw_score,
        "learning_rate": float(model.learning_rate),
        "calibration": {
            "kind": "platt",
            "scale": calibration_scale,
            "intercept": calibration_intercept,
        },
        "trees": [export_tree(stage[0]) for stage in model.estimators_],
        "metadata": {
            "target": "expected eventual score for the perspective player (win=1, draw=0.5, loss=0)",
            "eligible_outcomes": sorted({row["outcome_reason"] for row in rows}),
            "split": "source-stratified chronological 70/15/15 by game; color-swapped self-play pairs are indivisible",
            "hyperparameters": best_parameters,
            "self_play_train_share": selected_self_play_train_share,
            "self_play_train_share_candidates": self_play_train_shares,
            "model_selection": "validation-only; fixed 50/50 source-balanced validation weights when comparing multiple self-play shares",
            "validation_constraints": "require human retention and self-play improvement overall and by color; choose the smallest feasible self-play share before minimizing validation loss",
            "validation_constraints_passed": validation_constraints_passed,
            "candidate_validation": candidate_validation,
            "self_play_code_version": self_play_code_version,
            "metrics": split_metrics,
            "feature_importance": [
                {"feature": name, "importance": round(float(importance), 8)}
                for name, importance in importances[:25]
            ],
            "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            "data_start": min(row["created_at"] for row in rows),
            "data_end": max(row["created_at"] for row in rows),
        },
    }
    test_indices = splits["test"]
    sample_size = min(250, len(test_indices))
    exported = exported_probabilities(artifact, vectors[test_indices[:sample_size]])
    reference = all_probabilities["test"][:sample_size]
    max_export_error = float(np.max(np.abs(exported - reference)))
    if max_export_error > 1e-9:
        raise RuntimeError(f"exported tree inference differs by {max_export_error}")

    report = {
        "format": artifact["format"],
        "best_hyperparameters": best_parameters,
        "metrics": split_metrics,
        "baseline_metrics": baseline_metrics,
        "paired_bootstrap_test": paired_bootstrap_test,
        "feature_importance": artifact["metadata"]["feature_importance"],
        "export_parity_max_absolute_error": max_export_error,
        "feature_count": len(feature_names),
        "tree_count": len(artifact["trees"]),
        "games": len({(data_source(row), row["game_id"]) for row in rows}),
        "positions": len(
            {(data_source(row), row["game_id"], row["turn"]) for row in rows}
        ),
        "samples": len(rows),
        "self_play_train_share": selected_self_play_train_share,
        "self_play_train_share_candidates": self_play_train_shares,
        "candidate_validation": candidate_validation,
        "validation_constraints_passed": validation_constraints_passed,
        "self_play_code_version": self_play_code_version,
        "sources": dict(Counter(data_source(row) for row in rows)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
