#!/usr/bin/env python3
"""Train a decomposable tree correction from paired GHQ continuations.

Each rollout says which of two candidate positions produced the stronger
continuation.  The learner keeps the incumbent's calibrated log-odds fixed and
fits a small scalar tree ensemble g(position) under the pairwise objective

  sigmoid((incumbent(left) + g(left)) - (incumbent(right) + g(right))).

Unlike a generic classifier over feature differences, g remains a normal
position evaluator that can be consumed by minimax search.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from sklearn.tree import DecisionTreeRegressor

try:
    from scripts.train_counterfactual_policy import (
        binary_metrics,
        bootstrap_loss_delta,
        grouped_split,
        load_counterfactual_reports,
        metrics_by_player,
        records_at,
    )
    from scripts.train_value_model import (
        align_append_only_baseline_schema,
        export_tree,
        exported_probabilities,
        safe_logit,
        sigmoid,
    )
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from train_counterfactual_policy import (
        binary_metrics,
        bootstrap_loss_delta,
        grouped_split,
        load_counterfactual_reports,
        metrics_by_player,
        records_at,
    )
    from train_value_model import (
        align_append_only_baseline_schema,
        export_tree,
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
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def pair_arrays(
    records: Sequence[Dict[str, Any]], baseline: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = np.vstack([record["left_features"] for record in records])
    right = np.vstack([record["right_features"] for record in records])
    offset = safe_logit(exported_probabilities(baseline, left)) - safe_logit(
        exported_probabilities(baseline, right)
    )
    labels = np.asarray([record["label"] for record in records], dtype=np.float64)
    weights = np.asarray(
        [
            max(0.02, min(0.5, float(record["rollout_delta"])))
            * (2.0 if record["terminal_pair"] else 1.0)
            for record in records
        ],
        dtype=np.float64,
    )
    return left, right, offset, labels, weights


def rank_pairwise_features(
    left: np.ndarray,
    right: np.ndarray,
    offsets: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Rank features from training residuals without looking at holdouts."""
    normalized = weights / weights.sum()
    residual = labels - sigmoid(offsets)
    signal = np.abs((left - right).T @ (normalized * residual))
    return np.argsort(-signal, kind="stable")


def fit_pairwise_tree_boosting(
    left: np.ndarray,
    right: np.ndarray,
    offsets: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    n_estimators: int,
    max_depth: int,
    min_samples_leaf: int,
    learning_rate: float,
    random_state: int,
) -> List[DecisionTreeRegressor]:
    """Functional-gradient boosting for a decomposable pairwise score."""
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("pairwise tree features have incompatible shapes")
    if len(left) != len(offsets) or len(left) != len(labels):
        raise ValueError("pairwise tree labels are misaligned")
    left_score = np.zeros(len(left), dtype=np.float64)
    right_score = np.zeros(len(right), dtype=np.float64)
    stacked = np.vstack([left, right])
    sample_weight = np.concatenate([weights, weights])
    trees: List[DecisionTreeRegressor] = []
    for index in range(n_estimators):
        probability = sigmoid(offsets + left_score - right_score)
        residual = labels - probability
        target = np.concatenate([residual, -residual])
        tree = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + index,
        )
        tree.fit(stacked, target, sample_weight=sample_weight)
        left_score += learning_rate * tree.predict(left)
        right_score += learning_rate * tree.predict(right)
        trees.append(tree)
    return trees


def correction_scores(
    trees: Sequence[DecisionTreeRegressor],
    vectors: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    score = np.zeros(len(vectors), dtype=np.float64)
    for tree in trees:
        score += learning_rate * tree.predict(vectors)
    return score


def pair_probabilities(
    trees: Sequence[DecisionTreeRegressor],
    left: np.ndarray,
    right: np.ndarray,
    offsets: np.ndarray,
    learning_rate: float,
) -> np.ndarray:
    return sigmoid(
        offsets
        + correction_scores(trees, left, learning_rate)
        - correction_scores(trees, right, learning_rate)
    )


def remapped_tree(
    tree: DecisionTreeRegressor, feature_indices: np.ndarray
) -> Dict[str, Any]:
    rendered = export_tree(tree)
    rendered["feature"] = [
        int(feature_indices[index]) if index >= 0 else int(index)
        for index in rendered["feature"]
    ]
    return rendered


def export_pairwise_tree_correction(
    baseline: Dict[str, Any],
    feature_names: List[str],
    feature_indices: np.ndarray,
    trees: Sequence[DecisionTreeRegressor],
    learning_rate: float,
    dataset_hash: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    if baseline.get("linear_correction") or baseline.get("tree_correction"):
        raise ValueError("nested value-model corrections are not supported")
    baseline_names = list(baseline.get("feature_names") or [])
    if feature_names[: len(baseline_names)] != baseline_names:
        raise ValueError("pairwise features are not append-only")
    artifact = copy.deepcopy(baseline)
    artifact["generated_at"] = datetime.now(timezone.utc).isoformat()
    artifact["feature_names"] = list(feature_names)
    artifact["tree_correction"] = {
        "learning_rate": learning_rate,
        "trees": [remapped_tree(tree, feature_indices) for tree in trees],
    }
    artifact["metadata"] = {
        **baseline.get("metadata", {}),
        **metadata,
        "correction_dataset_sha256": dataset_hash,
        "correction_kind": "paired-counterfactual-tree-boosting",
        "correction_base_feature_count": len(baseline_names),
        "correction_feature_count": len(feature_indices),
    }
    return artifact


def phase_metrics(
    records: Sequence[Dict[str, Any]], probabilities: np.ndarray
) -> Dict[str, Dict[str, float]]:
    labels = np.asarray([record["label"] for record in records], dtype=np.float64)
    result: Dict[str, Dict[str, float]] = {}
    for phase, lower, upper in (
        ("early", 0, 24),
        ("middle", 25, 59),
        ("late", 60, 10_000),
    ):
        indices = np.asarray(
            [
                index
                for index, record in enumerate(records)
                if lower <= int(record["source_turn_number"]) <= upper
            ],
            dtype=np.int64,
        )
        if len(indices):
            result[phase] = binary_metrics(labels[indices], probabilities[indices])
    return result


def validate_coverage(records: Sequence[Dict[str, Any]], minimum_pairs: int) -> None:
    if minimum_pairs < 30:
        raise ValueError("--minimum-pairs must be at least 30")
    if len(records) < minimum_pairs:
        raise ValueError(f"need {minimum_pairs} confident pairs, found {len(records)}")
    players = Counter(record["root_player"] for record in records)
    if any(players[player] < 12 for player in ("RED", "BLUE")):
        raise ValueError("pairwise training needs at least 12 roots for each player")
    phases = Counter(
        "early"
        if record["source_turn_number"] <= 24
        else "middle"
        if record["source_turn_number"] <= 59
        else "late"
        for record in records
    )
    if any(phases[phase] < 8 for phase in ("early", "middle", "late")):
        raise ValueError("pairwise training needs at least 8 roots in each phase")


def grouped_folds(
    records: Sequence[Dict[str, Any]], random_state: int, fold_count: int = 5
) -> List[np.ndarray]:
    """Deterministic source-game folds for leakage-free model selection."""
    groups = sorted({str(record["source_game_id"]) for record in records})
    if len(groups) < fold_count * 2:
        raise ValueError("pairwise cross-validation needs at least two games per fold")
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"pairwise:{random_state}:{group}".encode("utf-8")
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


def cross_validated_candidate(
    records: Sequence[Dict[str, Any]],
    baseline: Dict[str, Any],
    config: Dict[str, Any],
    random_state: int,
    folds: Sequence[np.ndarray],
) -> Dict[str, Any]:
    probabilities: List[float] = []
    baseline_probabilities: List[float] = []
    labels: List[float] = []
    all_indices = np.arange(len(records), dtype=np.int64)
    for fold_number, validation_indices in enumerate(folds):
        training_indices = np.setdiff1d(all_indices, validation_indices)
        training_records = records_at(records, training_indices)
        left, right, offsets, training_labels, weights = pair_arrays(
            training_records, baseline
        )
        ranking = rank_pairwise_features(
            left, right, offsets, training_labels, weights
        )
        selected = ranking[: config["feature_count"]]
        trees = fit_pairwise_tree_boosting(
            left[:, selected],
            right[:, selected],
            offsets,
            training_labels,
            weights,
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_leaf=config["min_samples_leaf"],
            learning_rate=config["learning_rate"],
            random_state=random_state + 1_000 * fold_number,
        )
        validation_records = records_at(records, validation_indices)
        val_left, val_right, val_offset, val_label, _ = pair_arrays(
            validation_records, baseline
        )
        probability = pair_probabilities(
            trees,
            val_left[:, selected],
            val_right[:, selected],
            val_offset,
            config["learning_rate"],
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
        **config,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "passed": (
            metrics["log_loss"] < baseline_metrics["log_loss"]
            and metrics["accuracy"] >= baseline_metrics["accuracy"]
        ),
    }


def main() -> None:
    args = parse_args()
    feature_names, records, dataset_hash = load_counterfactual_reports(args.report)
    validate_coverage(records, args.minimum_pairs)
    raw_baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline = align_append_only_baseline_schema(raw_baseline, feature_names)
    splits = grouped_split(records, args.random_state)

    development_indices = np.concatenate([splits["train"], splits["validation"]])
    development_records = records_at(records, development_indices)
    folds = grouped_folds(development_records, args.random_state)

    candidates: List[Dict[str, Any]] = []
    feature_counts = sorted(
        {min(len(feature_names), count) for count in (8, 16, 32, 64)}
    )
    for feature_count in feature_counts:
        for max_depth in (1, 2):
            for min_leaf in (4, 8):
                for learning_rate in (0.05, 0.1):
                    for n_estimators in (8, 16, 32):
                        candidates.append(
                            cross_validated_candidate(
                                development_records,
                                baseline,
                                {
                                    "feature_count": feature_count,
                                    "max_depth": max_depth,
                                    "min_samples_leaf": min_leaf,
                                    "learning_rate": learning_rate,
                                    "n_estimators": n_estimators,
                                },
                                args.random_state,
                                folds,
                            )
                        )
    feasible = [candidate for candidate in candidates if candidate["passed"]]
    pool = feasible or candidates
    best_loss = min(candidate["metrics"]["log_loss"] for candidate in pool)
    selected_candidate = min(
        [
            candidate
            for candidate in pool
            if candidate["metrics"]["log_loss"] <= best_loss + 0.002
        ],
        key=lambda candidate: (
            candidate["max_depth"],
            candidate["n_estimators"],
            candidate["feature_count"],
            candidate["learning_rate"],
            -candidate["min_samples_leaf"],
        ),
    )

    dev_left, dev_right, dev_offset, dev_label, dev_weight = pair_arrays(
        development_records, baseline
    )
    development_ranking = rank_pairwise_features(
        dev_left, dev_right, dev_offset, dev_label, dev_weight
    )
    chosen_features = development_ranking[: selected_candidate["feature_count"]]
    final_trees = fit_pairwise_tree_boosting(
        dev_left[:, chosen_features],
        dev_right[:, chosen_features],
        dev_offset,
        dev_label,
        dev_weight,
        n_estimators=selected_candidate["n_estimators"],
        max_depth=selected_candidate["max_depth"],
        min_samples_leaf=selected_candidate["min_samples_leaf"],
        learning_rate=selected_candidate["learning_rate"],
        random_state=args.random_state,
    )

    test_records = records_at(records, splits["test"])
    test_left, test_right, test_offset, test_label, _ = pair_arrays(
        test_records, baseline
    )
    test_probability = pair_probabilities(
        final_trees,
        test_left[:, chosen_features],
        test_right[:, chosen_features],
        test_offset,
        selected_candidate["learning_rate"],
    )
    baseline_test_probability = sigmoid(test_offset)
    test_metrics = binary_metrics(test_label, test_probability)
    baseline_test_metrics = binary_metrics(test_label, baseline_test_probability)
    bootstrap = bootstrap_loss_delta(
        test_records,
        test_label,
        test_probability,
        baseline_test_probability,
        args.random_state,
    )
    candidate_players = metrics_by_player(test_records, test_probability)
    baseline_players = metrics_by_player(test_records, baseline_test_probability)
    player_gates = {
        player: (
            player not in candidate_players
            or candidate_players[player]["log_loss"]
            <= baseline_players[player]["log_loss"] + 0.02
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

    root_players = Counter(record["root_player"] for record in records)
    root_ids = sorted({str(record["root_id"]) for record in records})
    phases = Counter(
        "early"
        if record["source_turn_number"] <= 24
        else "middle"
        if record["source_turn_number"] <= 59
        else "late"
        for record in records
    )
    report = {
        "format": "ghq-counterfactual-pairwise-tree-training-v1",
        "reports": [str(path) for path in args.report],
        "dataset_sha256": dataset_hash,
        "pairs": len(records),
        "source_games": len({record["source_game_id"] for record in records}),
        "root_ids": root_ids,
        "terminal_pairs": sum(bool(record["terminal_pair"]) for record in records),
        "root_players": dict(root_players),
        "phases": dict(phases),
        "split_pairs": {
            "development": len(development_records),
            "test": len(test_records),
        },
        "cross_validation_folds": len(folds),
        "candidate_count": len(candidates),
        "validation_constraints_passed": bool(feasible),
        "selected": {
            key: selected_candidate[key]
            for key in (
                "feature_count",
                "max_depth",
                "min_samples_leaf",
                "learning_rate",
                "n_estimators",
                "metrics",
                "baseline_metrics",
            )
        },
        "selected_features": [feature_names[int(index)] for index in chosen_features],
        "test": {
            "candidate": test_metrics,
            "baseline": baseline_test_metrics,
            "candidate_by_root_player": candidate_players,
            "baseline_by_root_player": baseline_players,
            "candidate_by_phase": phase_metrics(test_records, test_probability),
            "baseline_by_phase": phase_metrics(
                test_records, baseline_test_probability
            ),
            "paired_game_bootstrap": bootstrap,
        },
        "approved_for_arena": approved,
    }
    artifact = export_pairwise_tree_correction(
        raw_baseline,
        feature_names,
        chosen_features,
        final_trees,
        selected_candidate["learning_rate"],
        dataset_hash,
        {
            "counterfactual_policy_report": str(args.training_report),
            "counterfactual_pairs": len(records),
            "counterfactual_terminal_pairs": report["terminal_pairs"],
            "counterfactual_approved_for_arena": approved,
            "counterfactual_training_root_ids": root_ids,
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.training_report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, separators=(",", ":")) + "\n", encoding="utf-8"
    )
    args.training_report.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    if args.require_pass and not approved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
