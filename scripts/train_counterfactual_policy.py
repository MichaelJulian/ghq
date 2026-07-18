#!/usr/bin/env python3
"""Train an append-only policy correction from paired branch continuations."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize

try:
    from scripts.train_value_model import (
        align_append_only_baseline_schema,
        exported_probabilities,
        safe_logit,
        sigmoid,
    )
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from train_value_model import (
        align_append_only_baseline_schema,
        exported_probabilities,
        safe_logit,
        sigmoid,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=48)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--feature-scope",
        choices=("all", "appended", "difference"),
        default="all",
        help=(
            "Features eligible for the sparse residual correction. The "
            "difference scope only admits perspective-antisymmetric diff_* "
            "features."
        ),
    )
    parser.add_argument(
        "--correction-target",
        choices=("policy", "value"),
        default="policy",
        help=(
            "Export a move-ranking policy head by default. The legacy value "
            "target directly changes calibrated win probabilities."
        ),
    )
    parser.add_argument(
        "--policy-scale",
        type=float,
        default=1.0,
        help="Multiplier in [0, 1] applied only to an exported policy head.",
    )
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def offset_probabilities(
    offsets: np.ndarray, vectors: np.ndarray, coefficients: np.ndarray
) -> np.ndarray:
    return sigmoid(offsets + vectors @ coefficients)


def fit_offset_logistic_correction(
    offsets: np.ndarray,
    vectors: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    l2: float,
) -> np.ndarray:
    """Fit feature coefficients while keeping incumbent logits as an offset."""
    if vectors.ndim != 2 or len(vectors) != len(offsets):
        raise ValueError("offset correction inputs have incompatible shapes")
    if len(labels) != len(offsets) or len(weights) != len(offsets):
        raise ValueError("offset correction labels or weights are misaligned")
    if l2 <= 0:
        raise ValueError("l2 must be positive")
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights = normalized_weights / normalized_weights.sum()

    def objective(coefficients: np.ndarray) -> Tuple[float, np.ndarray]:
        logits = offsets + vectors @ coefficients
        losses = np.logaddexp(0.0, logits) - labels * logits
        probability = sigmoid(logits)
        value = float(np.dot(normalized_weights, losses))
        value += 0.5 * l2 * float(np.dot(coefficients, coefficients))
        gradient = vectors.T @ (normalized_weights * (probability - labels))
        gradient += l2 * coefficients
        return value, gradient

    result = minimize(
        objective,
        np.zeros(vectors.shape[1], dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": 2_000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"counterfactual correction failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def rank_correction_features(
    offsets: np.ndarray,
    vectors: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Rank correction features from training residuals without holdout leakage."""
    if vectors.ndim != 2 or len(vectors) != len(offsets):
        raise ValueError("feature-ranking inputs have incompatible shapes")
    normalized_weights = np.asarray(weights, dtype=np.float64)
    normalized_weights = normalized_weights / normalized_weights.sum()
    residual = labels - sigmoid(offsets)
    signal = np.abs(vectors.T @ (normalized_weights * residual))
    return np.argsort(-signal, kind="stable")


def correction_feature_indices(
    feature_names: Sequence[str], baseline_feature_count: int, scope: str
) -> np.ndarray:
    """Select correction features without weakening perspective symmetry."""
    if not 0 <= baseline_feature_count <= len(feature_names):
        raise ValueError("baseline feature count is outside the feature schema")
    if scope == "all":
        indices = range(len(feature_names))
    elif scope == "appended":
        indices = range(baseline_feature_count, len(feature_names))
    elif scope == "difference":
        indices = (
            index
            for index, name in enumerate(feature_names)
            if name.startswith("diff_")
        )
    else:
        raise ValueError(f"unsupported correction feature scope: {scope}")
    selected = np.fromiter(indices, dtype=np.int64)
    if not len(selected):
        raise ValueError(f"no features are eligible for {scope} correction scope")
    return selected


def export_policy_correction(
    baseline: Dict[str, Any],
    feature_names: List[str],
    feature_indices: np.ndarray,
    feature_scales: np.ndarray,
    standardized_coefficients: np.ndarray,
    dataset_hash: str,
    metadata: Dict[str, Any],
    correction_target: str = "value",
    policy_scale: float = 1.0,
) -> Dict[str, Any]:
    if correction_target not in ("policy", "value"):
        raise ValueError(f"unsupported correction target: {correction_target}")
    if correction_target == "policy" and not 0.0 <= policy_scale <= 1.0:
        raise ValueError("policy scale must be between zero and one")
    correction_key = (
        "policy_correction" if correction_target == "policy" else "linear_correction"
    )
    if baseline.get(correction_key):
        raise ValueError(f"nested {correction_target} corrections are not supported")
    baseline_names = list(baseline.get("feature_names") or [])
    if feature_names[: len(baseline_names)] != baseline_names:
        raise ValueError("counterfactual features are not append-only")
    if len(feature_indices) != len(standardized_coefficients):
        raise ValueError("counterfactual coefficient schema mismatch")
    coefficients = standardized_coefficients / feature_scales
    artifact = copy.deepcopy(baseline)
    artifact["generated_at"] = datetime.now(timezone.utc).isoformat()
    artifact["feature_names"] = feature_names
    artifact[correction_key] = {
        "feature_indices": feature_indices.astype(int).tolist(),
        "coefficients": coefficients.astype(float).tolist(),
    }
    if correction_target == "policy":
        artifact[correction_key]["scale"] = float(policy_scale)
    artifact["metadata"] = {
        **baseline.get("metadata", {}),
        **metadata,
        "correction_dataset_sha256": dataset_hash,
        "correction_kind": (
            "paired-counterfactual-policy-logistic"
            if correction_target == "policy"
            else "paired-counterfactual-offset-logistic"
        ),
        "correction_target": correction_target,
        "correction_base_feature_count": len(baseline_names),
        "correction_feature_count": len(feature_indices),
    }
    return artifact


def grouped_split(
    records: Sequence[Dict[str, Any]], random_state: int
) -> Dict[str, np.ndarray]:
    groups = sorted({str(record["source_game_id"]) for record in records})
    if len(groups) < 12:
        raise ValueError("counterfactual training needs at least 12 source games")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{random_state}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    train_end = max(1, int(len(ordered) * 0.7))
    validation_end = max(train_end + 1, int(len(ordered) * 0.85))
    validation_end = min(validation_end, len(ordered) - 1)
    group_split = {
        group: (
            "train"
            if index < train_end
            else "validation"
            if index < validation_end
            else "test"
        )
        for index, group in enumerate(ordered)
    }
    return {
        name: np.asarray(
            [
                index
                for index, record in enumerate(records)
                if group_split[str(record["source_game_id"])] == name
            ],
            dtype=np.int64,
        )
        for name in ("train", "validation", "test")
    }


def load_counterfactual_reports(
    paths: Iterable[Path],
) -> Tuple[List[str], List[Dict[str, Any]], str]:
    feature_names: List[str] | None = None
    records: List[Dict[str, Any]] = []
    digest = hashlib.sha256()
    seen_roots: set[str] = set()
    seen_root_fingerprints: set[str] = set()
    for path in paths:
        raw = path.read_bytes()
        digest.update(raw)
        report = json.loads(raw)
        if report.get("format") != "ghq-counterfactual-rollout-report-v1":
            raise ValueError(f"unsupported counterfactual report {path}")
        expected_branches = report.get("expectedBranches")
        completed_branches = report.get("completedBranches")
        missing_branches = report.get("missingBranches")
        if (
            not isinstance(expected_branches, int)
            or not isinstance(completed_branches, int)
            or not isinstance(missing_branches, int)
            or completed_branches != expected_branches
            or missing_branches != 0
        ):
            raise ValueError(
                f"counterfactual report is incomplete in {path}: "
                f"{completed_branches}/{expected_branches} branches completed"
            )
        names = report.get("featureSchema")
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError(f"counterfactual feature schema is missing in {path}")
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("counterfactual reports use different feature schemas")
        for pair in report.get("pairs", []):
            if not pair.get("confident"):
                continue
            # Newer analyzers distinguish raw separation from a trustworthy
            # training label (replicate agreement, fallback quality, etc.).
            # Preserve compatibility with older reports that predate the flag.
            if "trainingEligible" in pair and not pair.get("trainingEligible"):
                continue
            root_id = str(pair["rootId"])
            if root_id in seen_roots:
                continue
            branches = sorted(
                [
                    branch
                    for branch in pair.get("branches", [])
                    if branch.get("status") == "completed"
                    and isinstance(branch.get("featuresV3"), list)
                    and int(branch.get("unverifiedFallbackDecisions", 0)) == 0
                ],
                key=lambda branch: int(branch["candidateRank"]),
            )
            if len(branches) < 2:
                continue
            left, right = branches[:2]
            candidate_fens = sorted(
                {
                    str(branch["initialFen"])
                    for branch in (left, right)
                    if isinstance(branch.get("initialFen"), str)
                }
            )
            root_fingerprint = (
                f'{pair["rootPlayer"]}:{"||".join(candidate_fens)}'
                if len(candidate_fens) == 2
                else f"legacy-root-id:{root_id}"
            )
            if root_fingerprint in seen_root_fingerprints:
                continue
            left_value = float(left["rolloutValue"])
            right_value = float(right["rolloutValue"])
            delta = abs(left_value - right_value)
            if delta <= 0:
                continue
            left_features = np.asarray(left["featuresV3"], dtype=np.float64)
            right_features = np.asarray(right["featuresV3"], dtype=np.float64)
            if len(left_features) != len(names) or len(right_features) != len(names):
                raise ValueError(f"feature vector length mismatch at {root_id}")
            seen_roots.add(root_id)
            seen_root_fingerprints.add(root_fingerprint)
            records.append(
                {
                    "root_id": root_id,
                    "root_fingerprint": root_fingerprint,
                    "source_game_id": str(pair["sourceGameId"]),
                    "root_player": str(pair["rootPlayer"]),
                    "source_turn_number": int(pair["sourceTurnNumber"]),
                    "left_rank": int(left["candidateRank"]),
                    "right_rank": int(right["candidateRank"]),
                    "left_features": left_features,
                    "right_features": right_features,
                    "label": 1.0 if left_value > right_value else 0.0,
                    "rollout_delta": delta,
                    "terminal_pair": (
                        left.get("valueSource") == "terminal"
                        or right.get("valueSource") == "terminal"
                    ),
                }
            )
    if feature_names is None:
        raise ValueError("no counterfactual reports were loaded")
    return feature_names, records, digest.hexdigest()


def augmented_pair_arrays(
    records: Sequence[Dict[str, Any]],
    baseline: Dict[str, Any],
    feature_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.vstack([record["left_features"] for record in records])
    right = np.vstack([record["right_features"] for record in records])
    left_probability = exported_probabilities(baseline, left)
    right_probability = exported_probabilities(baseline, right)
    offset = safe_logit(left_probability) - safe_logit(right_probability)
    vector = left[:, feature_indices] - right[:, feature_indices]
    label = np.asarray([record["label"] for record in records], dtype=np.float64)
    weight = np.asarray(
        [
            max(0.02, min(0.5, float(record["rollout_delta"])))
            * (2.0 if record["terminal_pair"] else 1.0)
            for record in records
        ],
        dtype=np.float64,
    )
    return (
        np.concatenate([offset, -offset]),
        np.vstack([vector, -vector]),
        np.concatenate([label, 1.0 - label]),
        np.concatenate([weight, weight]),
    )


def grouped_folds(
    records: Sequence[Dict[str, Any]], random_state: int, fold_count: int = 5
) -> List[np.ndarray]:
    """Deterministic source-game folds for leakage-free model selection."""
    groups = sorted({str(record["source_game_id"]) for record in records})
    if len(groups) < fold_count * 2:
        raise ValueError("counterfactual cross-validation needs two games per fold")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"linear:{random_state}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    assignment = {group: index % fold_count for index, group in enumerate(ordered)}
    return [
        np.asarray(
            [
                index
                for index, record in enumerate(records)
                if assignment[str(record["source_game_id"])] == fold
            ],
            dtype=np.int64,
        )
        for fold in range(fold_count)
    ]


def stability_random_states(primary: int) -> List[int]:
    """Use the same independent fold assignments for every model candidate."""
    states: List[int] = []
    for value in (primary, 42, 23, 101):
        if value not in states:
            states.append(value)
    return states


def cross_validated_linear_candidate(
    records: Sequence[Dict[str, Any]],
    baseline: Dict[str, Any],
    feature_indices: np.ndarray,
    feature_count: int,
    l2: float,
    folds: Sequence[np.ndarray],
) -> Dict[str, Any]:
    probabilities: List[float] = []
    baseline_probabilities: List[float] = []
    labels: List[float] = []
    all_indices = np.arange(len(records), dtype=np.int64)
    for validation_indices in folds:
        training_indices = np.setdiff1d(all_indices, validation_indices)
        training_records = records_at(records, training_indices)
        train_offset, train_vector, train_label, train_weight = (
            augmented_pair_arrays(training_records, baseline, feature_indices)
        )
        scales = np.std(train_vector, axis=0)
        scales[scales < 1e-8] = 1.0
        standardized = train_vector / scales
        ranking = rank_correction_features(
            train_offset,
            standardized,
            train_label,
            train_weight,
        )
        selected_positions = ranking[:feature_count]
        coefficients = fit_offset_logistic_correction(
            train_offset,
            standardized[:, selected_positions],
            train_label,
            train_weight,
            l2,
        )
        validation_records = records_at(records, validation_indices)
        val_offset, val_vector, val_label, _ = augmented_pair_arrays(
            validation_records, baseline, feature_indices
        )
        probability = offset_probabilities(
            val_offset,
            (val_vector / scales)[:, selected_positions],
            coefficients,
        )
        probabilities.extend(probability.tolist())
        baseline_probabilities.extend(sigmoid(val_offset).tolist())
        labels.extend(val_label.tolist())
    label_array = np.asarray(labels, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    baseline_array = np.asarray(baseline_probabilities, dtype=np.float64)
    metrics = binary_metrics(label_array, probability_array)
    baseline_metrics = binary_metrics(label_array, baseline_array)
    return {
        "feature_count": feature_count,
        "l2": l2,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "passed": (
            metrics["log_loss"] < baseline_metrics["log_loss"]
            and metrics["accuracy"] >= baseline_metrics["accuracy"]
        ),
    }


def safe_cross_validated_linear_candidate(
    records: Sequence[Dict[str, Any]],
    baseline: Dict[str, Any],
    feature_indices: np.ndarray,
    feature_count: int,
    l2: float,
    folds: Sequence[np.ndarray],
) -> Dict[str, Any]:
    """Reject an unstable optimizer configuration without aborting the sweep."""
    try:
        return cross_validated_linear_candidate(
            records,
            baseline,
            feature_indices,
            feature_count,
            l2,
            folds,
        )
    except RuntimeError as error:
        offset, _vector, label, _weight = augmented_pair_arrays(
            records, baseline, feature_indices
        )
        baseline_metrics = binary_metrics(label, sigmoid(offset))
        return {
            "feature_count": feature_count,
            "l2": l2,
            "metrics": {
                "log_loss": 1_000_000.0,
                "accuracy": 0.0,
                "samples": int(len(label)),
            },
            "baseline_metrics": baseline_metrics,
            "passed": False,
            "error": str(error),
        }


def binary_metrics(labels: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    clipped = np.clip(probabilities, 1e-7, 1 - 1e-7)
    log_loss = -np.mean(
        labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)
    )
    return {
        "log_loss": round(float(log_loss), 6),
        "accuracy": round(
            float(np.mean((probabilities >= 0.5) == (labels >= 0.5))), 6
        ),
        "samples": int(len(labels)),
    }


def metrics_by_player(
    records: Sequence[Dict[str, Any]],
    probabilities: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    labels = np.asarray([record["label"] for record in records], dtype=np.float64)
    result: Dict[str, Dict[str, float]] = {}
    for player in ("RED", "BLUE"):
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if record["root_player"] == player
            ],
            dtype=np.int64,
        )
        if len(indices):
            result[player] = binary_metrics(labels[indices], probabilities[indices])
    return result


def bootstrap_loss_delta(
    records: Sequence[Dict[str, Any]],
    labels: np.ndarray,
    candidate: np.ndarray,
    baseline: np.ndarray,
    random_state: int,
    samples: int = 2_000,
) -> Dict[str, float]:
    candidate = np.clip(candidate, 1e-7, 1 - 1e-7)
    baseline = np.clip(baseline, 1e-7, 1 - 1e-7)
    deltas = -(
        labels * np.log(candidate) + (1.0 - labels) * np.log(1.0 - candidate)
    ) + (
        labels * np.log(baseline) + (1.0 - labels) * np.log(1.0 - baseline)
    )
    by_game: Dict[str, List[float]] = {}
    for record, delta in zip(records, deltas):
        by_game.setdefault(str(record["source_game_id"]), []).append(float(delta))
    units = np.asarray(
        [np.mean(values) for values in by_game.values()], dtype=np.float64
    )
    rng = np.random.default_rng(random_state)
    draws = np.asarray(
        [
            np.mean(units[rng.integers(0, len(units), size=len(units))])
            for _ in range(samples)
        ],
        dtype=np.float64,
    )
    return {
        "candidate_minus_baseline": round(float(units.mean()), 6),
        "ci95_low": round(float(np.quantile(draws, 0.025)), 6),
        "ci95_high": round(float(np.quantile(draws, 0.975)), 6),
        "source_games": int(len(units)),
    }


def records_at(
    records: Sequence[Dict[str, Any]], indices: np.ndarray
) -> List[Dict[str, Any]]:
    return [records[int(index)] for index in indices]


def main() -> None:
    args = parse_args()
    if args.minimum_pairs < 30:
        raise ValueError("--minimum-pairs must be at least 30")
    feature_names, records, dataset_hash = load_counterfactual_reports(args.report)
    if len(records) < args.minimum_pairs:
        raise ValueError(
            f"need {args.minimum_pairs} confident pairs, found {len(records)}"
        )
    root_player_counts = Counter(record["root_player"] for record in records)
    if any(root_player_counts[player] < 12 for player in ("RED", "BLUE")):
        raise ValueError(
            "counterfactual training needs at least 12 trustworthy roots "
            "for each player"
        )
    phase_counts = Counter(
        "early"
        if record["source_turn_number"] <= 24
        else "middle"
        if record["source_turn_number"] <= 59
        else "late"
        for record in records
    )
    if any(phase_counts[phase] < 8 for phase in ("early", "middle", "late")):
        raise ValueError(
            "counterfactual training needs at least 8 trustworthy roots "
            "in each phase"
        )
    raw_baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_names = list(raw_baseline.get("feature_names") or [])
    baseline = align_append_only_baseline_schema(raw_baseline, feature_names)
    feature_indices = correction_feature_indices(
        feature_names,
        len(baseline_names),
        args.feature_scope,
    )
    cv_random_states = stability_random_states(args.random_state)
    folds_by_state = {
        state: grouped_folds(records, state) for state in cv_random_states
    }
    l2_candidates = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    feature_count_candidates = sorted(
        {
            min(len(feature_indices), count)
            for count in (4, 8, 16, 32, len(feature_indices))
        }
    )
    candidates = []
    for feature_count in feature_count_candidates:
        for l2 in l2_candidates:
            runs = [
                safe_cross_validated_linear_candidate(
                    records,
                    baseline,
                    feature_indices,
                    feature_count,
                    l2,
                    folds_by_state[state],
                )
                for state in cv_random_states
            ]
            candidates.append(
                {
                    "feature_count": feature_count,
                    "l2": l2,
                    "runs": [
                        {
                            "random_state": state,
                            "metrics": run["metrics"],
                            "baseline_metrics": run["baseline_metrics"],
                            "passed": run["passed"],
                            **({"error": run["error"]} if "error" in run else {}),
                        }
                        for state, run in zip(cv_random_states, runs)
                    ],
                    "mean_log_loss": float(
                        np.mean([run["metrics"]["log_loss"] for run in runs])
                    ),
                    "mean_baseline_log_loss": float(
                        np.mean(
                            [run["baseline_metrics"]["log_loss"] for run in runs]
                        )
                    ),
                    "mean_accuracy": float(
                        np.mean([run["metrics"]["accuracy"] for run in runs])
                    ),
                    "mean_baseline_accuracy": float(
                        np.mean(
                            [run["baseline_metrics"]["accuracy"] for run in runs]
                        )
                    ),
                    "passed": all(run["passed"] for run in runs),
                }
            )
    feasible = [candidate for candidate in candidates if candidate["passed"]]
    pool = feasible or candidates
    best_loss = min(candidate["mean_log_loss"] for candidate in pool)
    selected = max(
        [
            candidate
            for candidate in pool
            if candidate["mean_log_loss"] <= best_loss + 0.001
        ],
        key=lambda candidate: (
            candidate["l2"],
            -candidate["feature_count"],
        ),
    )

    training_offset, training_vector, training_label, training_weight = (
        augmented_pair_arrays(records, baseline, feature_indices)
    )
    feature_scales = np.std(training_vector, axis=0)
    feature_scales[feature_scales < 1e-8] = 1.0
    training_standardized = training_vector / feature_scales
    feature_ranking = rank_correction_features(
        training_offset,
        training_standardized,
        training_label,
        training_weight,
    )
    selected_positions = feature_ranking[: selected["feature_count"]]
    selected_coefficients = fit_offset_logistic_correction(
        training_offset,
        training_standardized[:, selected_positions],
        training_label,
        training_weight,
        selected["l2"],
    )
    ready_for_external_holdout = bool(feasible)
    report = {
        "format": "ghq-counterfactual-policy-training-v1",
        "reports": [str(path) for path in args.report],
        "dataset_sha256": dataset_hash,
        "pairs": len(records),
        "source_games": len({record["source_game_id"] for record in records}),
        "source_game_ids": sorted(
            {str(record["source_game_id"]) for record in records}
        ),
        "root_ids": sorted({str(record["root_id"]) for record in records}),
        "root_fingerprints": sorted(
            {str(record["root_fingerprint"]) for record in records}
        ),
        "root_players": dict(root_player_counts),
        "phases": dict(phase_counts),
        "split_pairs": {
            "training": len(records),
        },
        "cross_validation_folds": 5,
        "cross_validation_random_states": cv_random_states,
        "feature_count": len(feature_names),
        "correction_feature_scope": args.feature_scope,
        "correction_target": args.correction_target,
        "correction_feature_count": len(feature_indices),
        "candidate_cross_validation": [
            {
                "l2": candidate["l2"],
                "feature_count": candidate["feature_count"],
                "runs": candidate["runs"],
                "mean_log_loss": round(candidate["mean_log_loss"], 6),
                "mean_baseline_log_loss": round(
                    candidate["mean_baseline_log_loss"], 6
                ),
                "mean_accuracy": round(candidate["mean_accuracy"], 6),
                "mean_baseline_accuracy": round(
                    candidate["mean_baseline_accuracy"], 6
                ),
                "passed": candidate["passed"],
            }
            for candidate in candidates
        ],
        "selected_l2": selected["l2"],
        "selected_feature_count": selected["feature_count"],
        "selected_features": [
            feature_names[int(feature_indices[position])]
            for position in selected_positions
        ],
        "validation_constraints_passed": bool(feasible),
        "approved_for_external_holdout": ready_for_external_holdout,
        "approved_for_arena": False,
    }
    artifact = export_policy_correction(
        raw_baseline,
        feature_names,
        feature_indices[selected_positions],
        feature_scales[selected_positions],
        selected_coefficients,
        dataset_hash,
        {
            "counterfactual_policy_report": str(args.training_report),
            "counterfactual_pairs": len(records),
            "counterfactual_source_games": report["source_games"],
            "counterfactual_selected_l2": selected["l2"],
            "counterfactual_selected_feature_count": selected["feature_count"],
            "counterfactual_feature_scope": args.feature_scope,
            "counterfactual_correction_target": args.correction_target,
            "counterfactual_policy_scale": args.policy_scale,
            "counterfactual_approved_for_external_holdout": (
                ready_for_external_holdout
            ),
            "counterfactual_approved_for_arena": False,
            "counterfactual_training_root_ids": report["root_ids"],
            "counterfactual_training_root_fingerprints": report[
                "root_fingerprints"
            ],
            "counterfactual_training_source_game_ids": report[
                "source_game_ids"
            ],
        },
        correction_target=args.correction_target,
        policy_scale=args.policy_scale,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.training_report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.training_report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if args.require_pass and not ready_for_external_holdout:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
