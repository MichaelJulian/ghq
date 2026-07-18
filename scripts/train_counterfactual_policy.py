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
    parser.add_argument("--minimum-pairs", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
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


def export_policy_correction(
    baseline: Dict[str, Any],
    feature_names: List[str],
    feature_indices: np.ndarray,
    feature_scales: np.ndarray,
    standardized_coefficients: np.ndarray,
    dataset_hash: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    if baseline.get("linear_correction"):
        raise ValueError("nested linear corrections are not supported")
    baseline_names = list(baseline.get("feature_names") or [])
    if feature_names[: len(baseline_names)] != baseline_names:
        raise ValueError("counterfactual features are not append-only")
    if len(feature_indices) != len(standardized_coefficients):
        raise ValueError("counterfactual coefficient schema mismatch")
    coefficients = standardized_coefficients / feature_scales
    artifact = copy.deepcopy(baseline)
    artifact["generated_at"] = datetime.now(timezone.utc).isoformat()
    artifact["feature_names"] = feature_names
    artifact["linear_correction"] = {
        "feature_indices": feature_indices.astype(int).tolist(),
        "coefficients": coefficients.astype(float).tolist(),
    }
    artifact["metadata"] = {
        **baseline.get("metadata", {}),
        **metadata,
        "correction_dataset_sha256": dataset_hash,
        "correction_kind": "paired-counterfactual-offset-logistic",
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
            records.append(
                {
                    "root_id": root_id,
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
    if args.minimum_pairs < 12:
        raise ValueError("--minimum-pairs must be at least 12")
    feature_names, records, dataset_hash = load_counterfactual_reports(args.report)
    if len(records) < args.minimum_pairs:
        raise ValueError(
            f"need {args.minimum_pairs} confident pairs, found {len(records)}"
        )
    raw_baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_names = list(raw_baseline.get("feature_names") or [])
    baseline = align_append_only_baseline_schema(raw_baseline, feature_names)
    feature_indices = np.arange(len(baseline_names), len(feature_names))
    if not len(feature_indices):
        raise ValueError("counterfactual policy training needs appended features")
    splits = grouped_split(records, args.random_state)
    train_records = records_at(records, splits["train"])
    train_offset, train_vector, train_label, train_weight = augmented_pair_arrays(
        train_records, baseline, feature_indices
    )
    feature_scales = np.std(train_vector, axis=0)
    feature_scales[feature_scales < 1e-8] = 1.0
    train_standardized = train_vector / feature_scales

    l2_candidates = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    validation_records = records_at(records, splits["validation"])
    validation_offset, validation_vector, validation_label, _ = augmented_pair_arrays(
        validation_records, baseline, feature_indices
    )
    validation_standardized = validation_vector / feature_scales
    baseline_validation_probability = sigmoid(validation_offset)
    candidates = []
    for l2 in l2_candidates:
        coefficients = fit_offset_logistic_correction(
            train_offset,
            train_standardized,
            train_label,
            train_weight,
            l2,
        )
        probability = offset_probabilities(
            validation_offset, validation_standardized, coefficients
        )
        candidate_metrics = binary_metrics(validation_label, probability)
        baseline_metrics = binary_metrics(
            validation_label, baseline_validation_probability
        )
        candidates.append(
            {
                "l2": l2,
                "coefficients": coefficients,
                "probability": probability,
                "metrics": candidate_metrics,
                "baseline_metrics": baseline_metrics,
                "passed": (
                    candidate_metrics["log_loss"] < baseline_metrics["log_loss"]
                    and candidate_metrics["accuracy"] >= baseline_metrics["accuracy"]
                ),
            }
        )
    feasible = [candidate for candidate in candidates if candidate["passed"]]
    pool = feasible or candidates
    best_loss = min(candidate["metrics"]["log_loss"] for candidate in pool)
    selected = max(
        [
            candidate
            for candidate in pool
            if candidate["metrics"]["log_loss"] <= best_loss + 0.001
        ],
        key=lambda candidate: candidate["l2"],
    )

    test_records = records_at(records, splits["test"])
    test_offset, test_vector, test_label, _ = augmented_pair_arrays(
        test_records, baseline, feature_indices
    )
    test_probability = offset_probabilities(
        test_offset, test_vector / feature_scales, selected["coefficients"]
    )
    baseline_test_probability = sigmoid(test_offset)
    test_metrics = binary_metrics(test_label, test_probability)
    baseline_test_metrics = binary_metrics(test_label, baseline_test_probability)
    bootstrap = bootstrap_loss_delta(
        test_records,
        np.asarray([record["label"] for record in test_records]),
        test_probability[: len(test_records)],
        baseline_test_probability[: len(test_records)],
        args.random_state,
    )
    candidate_player = metrics_by_player(
        test_records, test_probability[: len(test_records)]
    )
    baseline_player = metrics_by_player(
        test_records, baseline_test_probability[: len(test_records)]
    )
    player_gates = {
        player: (
            player not in candidate_player
            or candidate_player[player]["log_loss"]
            <= baseline_player[player]["log_loss"] + 0.02
        )
        for player in ("RED", "BLUE")
    }
    approved = bool(feasible) and all(
        [
            test_metrics["log_loss"] < baseline_test_metrics["log_loss"],
            test_metrics["accuracy"] >= baseline_test_metrics["accuracy"],
            bootstrap["ci95_high"] <= 0.02,
            *player_gates.values(),
        ]
    )
    phase_counts = Counter(
        "early"
        if record["source_turn_number"] <= 24
        else "middle"
        if record["source_turn_number"] <= 59
        else "late"
        for record in records
    )
    report = {
        "format": "ghq-counterfactual-policy-training-v1",
        "reports": [str(path) for path in args.report],
        "dataset_sha256": dataset_hash,
        "pairs": len(records),
        "source_games": len({record["source_game_id"] for record in records}),
        "root_players": dict(Counter(record["root_player"] for record in records)),
        "phases": dict(phase_counts),
        "split_pairs": {name: int(len(indices)) for name, indices in splits.items()},
        "feature_count": len(feature_names),
        "correction_feature_count": len(feature_indices),
        "candidate_validation": [
            {
                "l2": candidate["l2"],
                "metrics": candidate["metrics"],
                "baseline_metrics": candidate["baseline_metrics"],
                "passed": candidate["passed"],
            }
            for candidate in candidates
        ],
        "selected_l2": selected["l2"],
        "validation_constraints_passed": bool(feasible),
        "test": {
            "candidate": test_metrics,
            "baseline": baseline_test_metrics,
            "candidate_by_root_player": candidate_player,
            "baseline_by_root_player": baseline_player,
            "paired_game_bootstrap": bootstrap,
        },
        "approved_for_arena": approved,
    }
    artifact = export_policy_correction(
        raw_baseline,
        feature_names,
        feature_indices,
        feature_scales,
        selected["coefficients"],
        dataset_hash,
        {
            "counterfactual_policy_report": str(args.training_report),
            "counterfactual_pairs": len(records),
            "counterfactual_source_games": report["source_games"],
            "counterfactual_selected_l2": selected["l2"],
            "counterfactual_approved_for_arena": approved,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.training_report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.training_report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if args.require_pass and not approved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
