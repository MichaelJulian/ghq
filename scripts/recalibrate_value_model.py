#!/usr/bin/env python3
"""Recalibrate an incumbent value model without changing its learned trees."""

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

from train_value_model import (
    chronological_split,
    data_source,
    expand_soft_labels,
    exported_probabilities,
    exported_raw_scores,
    game_balanced_weights,
    load_dataset,
    metrics,
    metrics_by_source,
    paired_game_bootstrap,
    requested_self_play_shares,
    select_validation_candidate,
    sigmoid,
    validate_self_play_code_version,
    validation_selection_gates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--self-play-train-shares", required=True)
    parser.add_argument("--self-play-code-version", required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shares = requested_self_play_shares(None, args.self_play_train_shares)
    feature_names, rows, vectors, labels = load_dataset(args.dataset)
    code_version = validate_self_play_code_version(
        rows, args.self_play_code_version
    )
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    if baseline.get("feature_names") != feature_names:
        raise ValueError("baseline feature schema does not match dataset")

    splits = chronological_split(rows)
    calibration_indices = splits["calibration"]
    validation_indices = splits["validation"]
    raw_scores = exported_raw_scores(baseline, vectors)
    baseline_probabilities = {
        name: exported_probabilities(baseline, vectors[indices])
        for name, indices in splits.items()
    }
    baseline_validation_by_source = metrics_by_source(
        rows,
        validation_indices,
        labels,
        baseline_probabilities["validation"],
    )
    selection_weights = game_balanced_weights(rows, validation_indices)

    candidates: List[Dict[str, Any]] = []
    for share in shares:
        calibration_weights = game_balanced_weights(
            rows,
            calibration_indices,
            share,
            balance_outcomes=True,
            outcome_balance_sources={"vercel_self_play"},
        )
        fit_vectors, fit_labels, fit_weights = expand_soft_labels(
            raw_scores[calibration_indices].reshape(-1, 1),
            labels[calibration_indices],
            calibration_weights,
        )
        calibrator = LogisticRegression(random_state=args.random_state)
        calibrator.fit(fit_vectors, fit_labels, sample_weight=fit_weights)
        scale = float(calibrator.coef_[0, 0])
        intercept = float(calibrator.intercept_[0])
        validation_probability = sigmoid(
            scale * raw_scores[validation_indices] + intercept
        )
        score = metrics(
            labels[validation_indices],
            validation_probability,
            selection_weights,
        )["log_loss"]
        by_source = metrics_by_source(
            rows,
            validation_indices,
            labels,
            validation_probability,
        )
        gates = validation_selection_gates(
            by_source, baseline_validation_by_source
        )
        candidates.append(
            {
                "self_play_train_share": share,
                "validation_log_loss": score,
                "validation_by_source": by_source,
                "validation_selection_gates": gates,
                "validation_constraints_passed": all(
                    gate["passed"] for gate in gates
                ),
                "scale": scale,
                "intercept": intercept,
            }
        )
        print(
            json.dumps(
                {
                    "self_play_train_share": share,
                    "validation_log_loss": score,
                    "validation_constraints_passed": candidates[-1][
                        "validation_constraints_passed"
                    ],
                }
            )
        )

    selectable = [
        {
            **candidate,
            "share": candidate["self_play_train_share"],
            "score": candidate["validation_log_loss"],
            "constraints_passed": candidate["validation_constraints_passed"],
        }
        for candidate in candidates
    ]
    selected, validation_constraints_passed = select_validation_candidate(
        selectable,
        baseline_constrained=True,
    )

    artifact = copy.deepcopy(baseline)
    artifact["generated_at"] = datetime.now(timezone.utc).isoformat()
    artifact["calibration"] = {
        "kind": "platt",
        "scale": selected["scale"],
        "intercept": selected["intercept"],
    }
    metadata = dict(artifact.get("metadata") or {})
    metadata.update(
        {
            "model_selection": "fixed incumbent trees are calibrated on a dedicated chronological calibration split, then the smallest feasible self-play share is selected on validation only",
            "validation_constraints": "require human retention and self-play improvement overall and by color",
            "validation_constraints_passed": validation_constraints_passed,
            "self_play_train_share": selected["self_play_train_share"],
            "self_play_train_share_candidates": shares,
            "candidate_validation": candidates,
            "self_play_code_version": code_version,
            "baseline_dataset_sha256": metadata.get("dataset_sha256"),
            "calibration_dataset_sha256": hashlib.sha256(
                args.dataset.read_bytes()
            ).hexdigest(),
        }
    )
    artifact["metadata"] = metadata

    candidate_probabilities: Dict[str, np.ndarray] = {}
    split_metrics: Dict[str, Any] = {}
    baseline_metrics: Dict[str, Any] = {}
    for name, indices in splits.items():
        probability = exported_probabilities(artifact, vectors[indices])
        candidate_probabilities[name] = probability
        split_weights = game_balanced_weights(rows, indices)
        split_metrics[name] = metrics(labels[indices], probability, split_weights)
        split_metrics[name]["by_source"] = metrics_by_source(
            rows, indices, labels, probability
        )
        baseline_metrics[name] = metrics(
            labels[indices], baseline_probabilities[name], split_weights
        )
        baseline_metrics[name]["by_source"] = metrics_by_source(
            rows, indices, labels, baseline_probabilities[name]
        )

    bootstrap = paired_game_bootstrap(
        rows,
        splits["test"],
        labels,
        candidate_probabilities["test"],
        baseline_probabilities["test"],
        args.random_state,
    )
    parity = float(
        np.max(
            np.abs(
                exported_probabilities(artifact, vectors[splits["test"][:250]])
                - candidate_probabilities["test"][:250]
            )
        )
    )
    report = {
        "format": artifact["format"],
        "model_kind": "calibration-only",
        "metrics": split_metrics,
        "baseline_metrics": baseline_metrics,
        "paired_bootstrap_test": bootstrap,
        "validation_constraints_passed": validation_constraints_passed,
        "self_play_train_share": selected["self_play_train_share"],
        "self_play_train_share_candidates": shares,
        "candidate_validation": candidates,
        "self_play_code_version": code_version,
        "export_parity_max_absolute_error": parity,
        "sources": dict(Counter(data_source(row) for row in rows)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, separators=(",", ":")), encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
