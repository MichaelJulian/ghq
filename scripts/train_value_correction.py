#!/usr/bin/env python3
"""Fit an append-only structural correction without discarding incumbent trees."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    from scripts.train_value_model import (
        align_append_only_baseline_schema,
        data_source,
        expand_soft_labels,
        exported_probabilities,
        game_balanced_weights,
        latest_position_indices,
        load_dataset,
        metrics,
        metrics_by_source,
        paired_game_bootstrap,
        requested_self_play_shares,
        safe_logit,
        select_validation_candidate,
        sigmoid,
        validate_self_play_behavior_checkpoint,
        validate_self_play_code_version,
        validation_selection_gates,
        chronological_split,
    )
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from train_value_model import (
        align_append_only_baseline_schema,
        chronological_split,
        data_source,
        expand_soft_labels,
        exported_probabilities,
        game_balanced_weights,
        latest_position_indices,
        load_dataset,
        metrics,
        metrics_by_source,
        paired_game_bootstrap,
        requested_self_play_shares,
        safe_logit,
        select_validation_candidate,
        sigmoid,
        validate_self_play_behavior_checkpoint,
        validate_self_play_code_version,
        validation_selection_gates,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    share_group = parser.add_mutually_exclusive_group()
    share_group.add_argument("--self-play-train-share", type=float)
    share_group.add_argument("--self-play-train-shares")
    parser.add_argument("--self-play-code-version")
    parser.add_argument("--self-play-behavior-checkpoint")
    parser.add_argument(
        "--correction-feature-mode",
        choices=("material-diff", "diff", "infantry-diff", "all"),
        default="material-diff",
        help="use symmetric own-minus-opponent additions or every append-only feature",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def split_metrics(
    rows: List[Dict[str, Any]],
    labels: np.ndarray,
    splits: Dict[str, np.ndarray],
    probabilities: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for name, indices in splits.items():
        weights = game_balanced_weights(rows, indices)
        probability = probabilities[name]
        result[name] = metrics(labels[indices], probability, weights)
        latest = latest_position_indices(rows, indices)
        probability_by_index = {
            int(index): probability[offset] for offset, index in enumerate(indices)
        }
        latest_probability = np.asarray(
            [probability_by_index[int(index)] for index in latest]
        )
        result[name]["latest_position"] = metrics(
            labels[latest],
            latest_probability,
            game_balanced_weights(rows, latest),
        )
        result[name]["games"] = len({rows[index]["game_id"] for index in indices})
        result[name]["samples"] = len(indices)
        result[name]["by_source"] = metrics_by_source(
            rows, indices, labels, probability
        )
    return result


def select_stable_correction_candidate(
    candidates: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], bool]:
    """Prefer the simplest regularization within a negligible loss tolerance."""
    feasible = [
        candidate for candidate in candidates if candidate["constraints_passed"]
    ]
    if not feasible:
        return select_validation_candidate(candidates, baseline_constrained=True)
    minimum_share = min(float(candidate["share"]) for candidate in feasible)
    least_shifted = [
        candidate
        for candidate in feasible
        if float(candidate["share"]) == minimum_share
    ]
    best_score = min(float(candidate["score"]) for candidate in least_shifted)
    statistically_tied = [
        candidate
        for candidate in least_shifted
        if float(candidate["score"]) <= best_score + 0.0005
    ]
    return min(
        statistically_tied,
        key=lambda candidate: float(candidate["regularization_c"]),
    ), True


def exported_correction(
    baseline: Dict[str, Any],
    feature_names: List[str],
    new_feature_indices: np.ndarray,
    scaler: StandardScaler,
    model: LogisticRegression,
    dataset_hash: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Translate standardized logistic coefficients into runtime feature space."""
    if baseline.get("linear_correction"):
        raise ValueError("nested linear corrections are not supported")
    baseline_coefficient = float(model.coef_[0, 0])
    standardized_coefficients = np.asarray(model.coef_[0, 1:], dtype=np.float64)
    coefficients = standardized_coefficients / scaler.scale_
    calibration = baseline["calibration"]
    intercept = (
        float(model.intercept_[0])
        + baseline_coefficient * float(calibration["intercept"])
        - float(np.dot(coefficients, scaler.mean_))
    )
    artifact = copy.deepcopy(baseline)
    artifact["generated_at"] = datetime.now(timezone.utc).isoformat()
    artifact["feature_names"] = feature_names
    artifact["calibration"] = {
        "kind": "platt",
        "scale": baseline_coefficient * float(calibration["scale"]),
        "intercept": intercept,
    }
    artifact["linear_correction"] = {
        "feature_indices": new_feature_indices.astype(int).tolist(),
        "coefficients": coefficients.astype(float).tolist(),
    }
    artifact["metadata"] = {
        **baseline.get("metadata", {}),
        **metadata,
        "correction_dataset_sha256": dataset_hash,
        "correction_kind": "regularized-logistic-append-only",
        "correction_base_feature_count": len(baseline["feature_names"]),
        "correction_feature_count": len(new_feature_indices),
    }
    return artifact


def main() -> None:
    args = parse_args()
    shares = requested_self_play_shares(
        args.self_play_train_share, args.self_play_train_shares
    )
    feature_names, rows, vectors, labels = load_dataset(args.dataset)
    code_version = validate_self_play_code_version(
        rows, args.self_play_code_version
    )
    behavior_checkpoint = validate_self_play_behavior_checkpoint(
        rows, args.self_play_behavior_checkpoint
    )
    raw_baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if raw_baseline.get("linear_correction"):
        raise ValueError("baseline already contains a linear correction")
    baseline_feature_names = list(raw_baseline.get("feature_names") or [])
    baseline = align_append_only_baseline_schema(raw_baseline, feature_names)
    if len(baseline_feature_names) >= len(feature_names):
        raise ValueError("correction training requires append-only features")
    appended_indices = np.arange(len(baseline_feature_names), len(feature_names))
    if args.correction_feature_mode == "all":
        new_indices = appended_indices
    else:
        prefix = {
            "diff": "diff_",
            "material-diff": "diff_material_",
            "infantry-diff": "diff_infantry_",
        }[args.correction_feature_mode]
        new_indices = np.asarray(
            [
                index
                for index in appended_indices
                if feature_names[index].startswith(prefix)
            ],
            dtype=np.int64,
        )
    if not len(new_indices):
        raise ValueError("correction feature selection is empty")
    splits = chronological_split(rows)
    baseline_probability = exported_probabilities(baseline, vectors)
    baseline_logit = safe_logit(baseline_probability)
    baseline_validation = metrics_by_source(
        rows,
        splits["validation"],
        labels,
        baseline_probability[splits["validation"]],
    )
    ordinary_weights = {
        name: game_balanced_weights(rows, indices)
        for name, indices in splits.items()
    }
    fit_weights = {
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
        for share in shares
    }
    selection_weights = (
        ordinary_weights["validation"]
        if len(shares) > 1
        else fit_weights[shares[0]]["validation"]
    )
    regularization_candidates = [
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0,
        3.0,
        10.0,
    ]
    trained: List[Dict[str, Any]] = []
    candidate_validation: List[Dict[str, Any]] = []
    train_indices = splits["train"]
    validation_indices = splits["validation"]
    for share in shares:
        scaler = StandardScaler().fit(
            vectors[train_indices][:, new_indices],
            sample_weight=fit_weights[share]["train"],
        )
        design = np.column_stack(
            [baseline_logit, scaler.transform(vectors[:, new_indices])]
        )
        train_vectors, train_labels, train_weights = expand_soft_labels(
            design[train_indices],
            labels[train_indices],
            fit_weights[share]["train"],
        )
        for regularization in regularization_candidates:
            model = LogisticRegression(
                C=regularization,
                max_iter=5000,
                solver="lbfgs",
                random_state=args.random_state,
            )
            model.fit(
                train_vectors,
                train_labels,
                sample_weight=train_weights,
            )
            probability = model.predict_proba(design[validation_indices])[:, 1]
            score = metrics(
                labels[validation_indices], probability, selection_weights
            )["log_loss"]
            source_metrics = metrics_by_source(
                rows, validation_indices, labels, probability
            )
            gates = validation_selection_gates(source_metrics, baseline_validation)
            constraints_passed = all(gate["passed"] for gate in gates)
            candidate_validation.append(
                {
                    "self_play_train_share": share,
                    "regularization_c": regularization,
                    "validation_log_loss": round(score, 6),
                    "validation_by_source": source_metrics,
                    "validation_selection_gates": gates,
                    "validation_constraints_passed": constraints_passed,
                }
            )
            trained.append(
                {
                    "score": score,
                    "share": share,
                    "regularization_c": regularization,
                    "model": model,
                    "scaler": scaler,
                    "design": design,
                    "constraints_passed": constraints_passed,
                }
            )
    selected, feasible = select_stable_correction_candidate(trained)
    model = selected["model"]
    scaler = selected["scaler"]
    design = selected["design"]
    probabilities = {
        name: model.predict_proba(design[indices])[:, 1]
        for name, indices in splits.items()
    }
    baseline_probabilities = {
        name: baseline_probability[indices] for name, indices in splits.items()
    }
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    coefficients = sorted(
        zip(
            [feature_names[index] for index in new_indices],
            (model.coef_[0, 1:] / scaler.scale_).astype(float),
        ),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    artifact = exported_correction(
        raw_baseline,
        feature_names,
        new_indices,
        scaler,
        model,
        dataset_hash,
        {
            "correction_regularization_c": selected["regularization_c"],
            "correction_feature_mode": args.correction_feature_mode,
            "self_play_train_share": selected["share"],
            "self_play_train_share_candidates": shares,
            "validation_constraints_passed": feasible,
            "self_play_code_version": code_version,
            "self_play_behavior_value_model_checkpoint": behavior_checkpoint,
            "correction_feature_coefficients": [
                {"feature": name, "coefficient": round(float(value), 10)}
                for name, value in coefficients
            ],
            "candidate_validation": candidate_validation,
            "data_start": min(row["created_at"] for row in rows),
            "data_end": max(row["created_at"] for row in rows),
        },
    )
    exported_test = exported_probabilities(artifact, vectors[splits["test"]])
    parity_error = float(
        np.max(np.abs(exported_test - probabilities["test"]))
    )
    if parity_error > 1e-9:
        raise RuntimeError(f"exported correction differs by {parity_error}")
    candidate_metrics = split_metrics(rows, labels, splits, probabilities)
    baseline_metrics = split_metrics(rows, labels, splits, baseline_probabilities)
    bootstrap = paired_game_bootstrap(
        rows,
        splits["test"],
        labels,
        probabilities["test"],
        baseline_probabilities["test"],
        args.random_state,
    )
    artifact["metadata"]["metrics"] = candidate_metrics
    artifact["metadata"]["baseline_metrics"] = baseline_metrics
    artifact["metadata"]["paired_bootstrap_test"] = bootstrap
    report = {
        "format": artifact["format"],
        "model_kind": artifact["metadata"]["correction_kind"],
        "regularization_c": selected["regularization_c"],
        "metrics": candidate_metrics,
        "baseline_metrics": baseline_metrics,
        "paired_bootstrap_test": bootstrap,
        "candidate_validation": candidate_validation,
        "validation_constraints_passed": feasible,
        "export_parity_max_absolute_error": parity_error,
        "feature_count": len(feature_names),
        "base_feature_count": len(baseline_feature_names),
        "correction_feature_count": len(new_indices),
        "correction_feature_mode": args.correction_feature_mode,
        "games": len({(data_source(row), row["game_id"]) for row in rows}),
        "samples": len(rows),
        "self_play_train_share": selected["share"],
        "self_play_train_share_candidates": shares,
        "self_play_code_version": code_version,
        "self_play_behavior_value_model_checkpoint": behavior_checkpoint,
        "sources": dict(Counter(data_source(row) for row in rows)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
