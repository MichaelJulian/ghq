#!/usr/bin/env python3
"""Train a high-precision forced-HQ-loss detector from exact audit labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--external-test-dataset", action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--random-state", type=int, default=19)
    parser.add_argument("--minimum-precision", type=float, default=0.95)
    return parser.parse_args()


def pair_id(record: Dict[str, Any]) -> str:
    game_id = str(record["gameId"])
    try:
        number = int(game_id.rsplit("-", 1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"game id lacks numeric suffix: {game_id}") from error
    return f"{record['generationId']}:pair-{(number - 1) // 2 + 1:04d}"


def tactical_feature_indices(feature_names: Sequence[str]) -> List[int]:
    unit_suffixes = (
        "board_infantry",
        "board_armored_infantry",
        "board_airborne_infantry",
        "reserve_infantry",
        "reserve_armored_infantry",
        "reserve_airborne_infantry",
        "paratrooper_ready",
        "paratrooper_deployed",
        "paratrooper_distance_home",
        "pseudo_mobility",
    )
    return [
        index
        for index, name in enumerate(feature_names)
        if name == "turn_progress"
        or "_hq_" in name
        or name.endswith(unit_suffixes)
    ]


def load_datasets(
    paths: Iterable[Path],
) -> Tuple[List[str], List[Dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]:
    feature_names: List[str] | None = None
    records: List[Dict[str, Any]] = []
    digests: List[str] = []
    for path in paths:
        raw = path.read_bytes()
        digests.append(hashlib.sha256(raw).hexdigest())
        report = json.loads(raw)
        if report.get("format") != "ghq-tactical-hq-dataset-v1":
            raise ValueError(f"unsupported tactical dataset: {path}")
        names = list(report["featureNames"])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise ValueError("tactical datasets use different feature schemas")
        records.extend(
            record for record in report["records"] if record.get("label") in (0, 1)
        )
    if feature_names is None or not records:
        raise ValueError("no eligible tactical samples")
    indices = tactical_feature_indices(feature_names)
    if not indices:
        raise ValueError("no tactical features selected")
    matrix = np.asarray(
        [[float(record["features"][index]) for index in indices] for record in records],
        dtype=np.float64,
    )
    labels = np.asarray([int(record["label"]) for record in records], dtype=np.int64)
    groups = np.asarray([pair_id(record) for record in records], dtype=object)
    return [feature_names[index] for index in indices], records, matrix, labels, groups


def split_indices(
    labels: np.ndarray, groups: np.ndarray, random_state: int
) -> Dict[str, np.ndarray]:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 30:
        raise ValueError("at least 30 independent color-swapped pairs are required")
    outer = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=random_state)
    train_validation, test = next(outer.split(labels, labels, groups))
    inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + 1)
    relative_train, relative_validation = next(
        inner.split(
            labels[train_validation],
            labels[train_validation],
            groups[train_validation],
        )
    )
    split = {
        "train": train_validation[relative_train],
        "validation": train_validation[relative_validation],
        "test": test,
    }
    for name, indices in split.items():
        if len(np.unique(labels[indices])) != 2:
            raise ValueError(f"{name} split does not contain both labels")
    return split


def probabilities(model: Pipeline, matrix: np.ndarray) -> np.ndarray:
    return np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)


def metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> dict:
    clipped = np.clip(scores, 1e-9, 1 - 1e-9)
    predictions = clipped >= threshold
    return {
        "samples": int(len(labels)),
        "forced": int(np.sum(labels)),
        "safe": int(len(labels) - np.sum(labels)),
        "log_loss": round(float(log_loss(labels, clipped)), 6),
        "roc_auc": round(float(roc_auc_score(labels, clipped)), 6),
        "average_precision": round(float(average_precision_score(labels, clipped)), 6),
        "threshold": round(float(threshold), 9),
        "precision": round(float(precision_score(labels, predictions, zero_division=0)), 6),
        "recall": round(float(recall_score(labels, predictions, zero_division=0)), 6),
        "flagged": int(np.sum(predictions)),
        "false_positives": int(np.sum(predictions & (labels == 0))),
        "false_negatives": int(np.sum((~predictions) & (labels == 1))),
    }


def high_precision_threshold(
    labels: np.ndarray, scores: np.ndarray, minimum_precision: float
) -> float | None:
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    candidates = [
        (float(recall[index]), float(precision[index]), float(thresholds[index]))
        for index in range(len(thresholds))
        if precision[index] >= minimum_precision
        and math.isfinite(thresholds[index])
        and np.sum(scores >= thresholds[index]) >= 2
    ]
    if not candidates:
        return None
    return max(candidates)[2]


def main() -> None:
    args = parse_args()
    if not 0.5 < args.minimum_precision <= 1:
        raise ValueError("--minimum-precision must be in (0.5, 1]")
    feature_names, records, matrix, labels, groups = load_datasets(args.dataset)
    split = split_indices(labels, groups, args.random_state)

    candidates = []
    for class_weight in (None, "balanced"):
        for regularization in (0.03, 0.1, 0.3, 1.0, 3.0):
            model = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            C=regularization,
                            class_weight=class_weight,
                            max_iter=5000,
                            random_state=args.random_state,
                        ),
                    ),
                ]
            )
            model.fit(matrix[split["train"]], labels[split["train"]])
            validation_scores = probabilities(model, matrix[split["validation"]])
            threshold = high_precision_threshold(
                labels[split["validation"]],
                validation_scores,
                args.minimum_precision,
            )
            candidates.append(
                {
                    "model": model,
                    "class_weight": class_weight,
                    "regularization": regularization,
                    "threshold": threshold,
                    "validation": metrics(
                        labels[split["validation"]],
                        validation_scores,
                        0.5 if threshold is None else threshold,
                    ),
                }
            )
    eligible = [candidate for candidate in candidates if candidate["threshold"] is not None]
    if not eligible:
        raise ValueError("no candidate reached the validation precision requirement")
    selected = max(
        eligible,
        key=lambda candidate: (
            candidate["validation"]["recall"],
            candidate["validation"]["average_precision"],
            -candidate["validation"]["log_loss"],
        ),
    )
    model = selected["model"]
    threshold = float(selected["threshold"])
    test_scores = probabilities(model, matrix[split["test"]])
    incumbent_risk = np.asarray(
        [1 - float(records[index]["valueModelWinProbability"]) for index in split["test"]],
        dtype=np.float64,
    )
    external_report = None
    if args.external_test_dataset:
        (
            external_feature_names,
            external_records,
            external_matrix,
            external_labels,
            external_groups,
        ) = load_datasets(args.external_test_dataset)
        if external_feature_names != feature_names:
            raise ValueError("external test feature schema does not match training")
        external_scores = probabilities(model, external_matrix)
        external_incumbent_risk = np.asarray(
            [
                1 - float(record["valueModelWinProbability"])
                for record in external_records
            ],
            dtype=np.float64,
        )
        external_metrics = metrics(external_labels, external_scores, threshold)
        external_report = {
            "datasets": [str(path) for path in args.external_test_dataset],
            "samples": len(external_records),
            "independentPairs": len(np.unique(external_groups)),
            "candidate": external_metrics,
            "incumbentRawValueAsRisk": metrics(
                external_labels, external_incumbent_risk, threshold=0.5
            ),
            "highPrecisionGatePassed": bool(
                external_metrics["precision"] >= args.minimum_precision
                and external_metrics["recall"] >= 0.5
                and external_metrics["false_positives"] == 0
            ),
        }

    scaler: StandardScaler = model.named_steps["scale"]
    classifier: LogisticRegression = model.named_steps["classifier"]
    artifact = {
        "format": "ghq-tactical-hq-logistic-v1",
        "feature_names": feature_names,
        "mean": [round(float(value), 12) for value in scaler.mean_],
        "scale": [round(float(value), 12) for value in scaler.scale_],
        "coefficients": [round(float(value), 12) for value in classifier.coef_[0]],
        "intercept": round(float(classifier.intercept_[0]), 12),
        "threshold": round(threshold, 12),
        "metadata": {
            "target": "exact forced HQ loss after every legal defender turn",
            "dataset_sha256": [
                hashlib.sha256(path.read_bytes()).hexdigest() for path in args.dataset
            ],
            "random_state": args.random_state,
            "minimum_validation_precision": args.minimum_precision,
            "regularization": selected["regularization"],
            "class_weight": selected["class_weight"],
        },
    }
    report = {
        "format": "ghq-tactical-hq-training-report-v1",
        "datasets": [str(path) for path in args.dataset],
        "samples": len(records),
        "independentPairs": len(np.unique(groups)),
        "features": len(feature_names),
        "split": {
            name: {
                "samples": int(len(indices)),
                "pairs": int(len(np.unique(groups[indices]))),
            }
            for name, indices in split.items()
        },
        "selected": {
            "regularization": selected["regularization"],
            "class_weight": selected["class_weight"],
            "threshold": threshold,
        },
        "validation": selected["validation"],
        "test": metrics(labels[split["test"]], test_scores, threshold),
        "incumbentRawValueAsRiskTest": metrics(
            labels[split["test"]], incumbent_risk, threshold=0.5
        ),
        "externalTest": external_report,
        "allCandidates": [
            {
                "regularization": candidate["regularization"],
                "class_weight": candidate["class_weight"],
                "threshold": candidate["threshold"],
                "validation": candidate["validation"],
            }
            for candidate in candidates
        ],
    }
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
