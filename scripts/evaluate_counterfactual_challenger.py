#!/usr/bin/env python3
"""Evaluate a frozen GHQ value challenger on untouched paired rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

try:
    from scripts.train_counterfactual_policy import (
        binary_metrics,
        bootstrap_loss_delta,
        load_counterfactual_reports,
        metrics_by_player,
    )
    from scripts.train_pairwise_tree_correction import phase_metrics
    from scripts.train_value_model import (
        align_append_only_baseline_schema,
        exported_policy_adjustments,
        exported_probabilities,
        safe_logit,
        sigmoid,
    )
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from train_counterfactual_policy import (
        binary_metrics,
        bootstrap_loss_delta,
        load_counterfactual_reports,
        metrics_by_player,
    )
    from train_pairwise_tree_correction import phase_metrics
    from train_value_model import (
        align_append_only_baseline_schema,
        exported_policy_adjustments,
        exported_probabilities,
        safe_logit,
        sigmoid,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--challenger", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-pairs", type=int, default=8)
    parser.add_argument("--minimum-terminal-pairs", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=1784390403)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def artifact_pair_probabilities(
    records: Sequence[Dict[str, Any]], artifact: Dict[str, Any]
) -> np.ndarray:
    left = np.vstack([record["left_features"] for record in records])
    right = np.vstack([record["right_features"] for record in records])
    left_logit = safe_logit(exported_probabilities(artifact, left))
    right_logit = safe_logit(exported_probabilities(artifact, right))
    left_logit += exported_policy_adjustments(artifact, left)
    right_logit += exported_policy_adjustments(artifact, right)
    return sigmoid(left_logit - right_logit)


def subset_metrics(
    records: Sequence[Dict[str, Any]], probabilities: np.ndarray
) -> Dict[str, Any] | None:
    if not records:
        return None
    labels = np.asarray([record["label"] for record in records], dtype=np.float64)
    return binary_metrics(labels, probabilities)


def terminal_improvement_gate(
    baseline: Dict[str, Any] | None,
    challenger: Dict[str, Any] | None,
    pair_count: int,
    minimum_pairs: int,
) -> bool:
    return bool(
        baseline is not None
        and challenger is not None
        and pair_count >= minimum_pairs
        and challenger["log_loss"] < baseline["log_loss"]
        and challenger["accuracy"] >= baseline["accuracy"]
    )


def training_root_overlap(
    records: Sequence[Dict[str, Any]], artifact: Dict[str, Any]
) -> tuple[bool, List[str]]:
    raw = artifact.get("metadata", {}).get("counterfactual_training_root_ids")
    if not isinstance(raw, list) or not all(isinstance(root, str) for root in raw):
        return False, []
    evaluation_roots = {str(record["root_id"]) for record in records}
    return True, sorted(evaluation_roots.intersection(raw))


def training_source_game_overlap(
    records: Sequence[Dict[str, Any]], artifact: Dict[str, Any]
) -> tuple[bool, List[str]]:
    raw = artifact.get("metadata", {}).get(
        "counterfactual_training_source_game_ids"
    )
    if not isinstance(raw, list) or not all(isinstance(game, str) for game in raw):
        return False, []
    evaluation_games = {str(record["source_game_id"]) for record in records}
    return True, sorted(evaluation_games.intersection(raw))


def main() -> None:
    args = parse_args()
    feature_names, records, dataset_hash = load_counterfactual_reports(args.report)
    if args.minimum_pairs < 4:
        raise ValueError("--minimum-pairs must be at least 4")
    if args.minimum_terminal_pairs < 2:
        raise ValueError("--minimum-terminal-pairs must be at least 2")
    if len(records) < args.minimum_pairs:
        raise ValueError(
            f"need {args.minimum_pairs} trustworthy holdout pairs, found {len(records)}"
        )
    baseline_raw = json.loads(args.baseline.read_text(encoding="utf-8"))
    challenger_raw = json.loads(args.challenger.read_text(encoding="utf-8"))
    provenance_available, overlapping_roots = training_root_overlap(
        records, challenger_raw
    )
    source_provenance_available, overlapping_source_games = (
        training_source_game_overlap(records, challenger_raw)
    )
    baseline = align_append_only_baseline_schema(baseline_raw, feature_names)
    challenger = align_append_only_baseline_schema(challenger_raw, feature_names)
    labels = np.asarray([record["label"] for record in records], dtype=np.float64)
    baseline_probability = artifact_pair_probabilities(records, baseline)
    challenger_probability = artifact_pair_probabilities(records, challenger)
    baseline_metrics = binary_metrics(labels, baseline_probability)
    challenger_metrics = binary_metrics(labels, challenger_probability)
    bootstrap = bootstrap_loss_delta(
        records,
        labels,
        challenger_probability,
        baseline_probability,
        args.random_state,
    )
    baseline_players = metrics_by_player(records, baseline_probability)
    challenger_players = metrics_by_player(records, challenger_probability)
    player_gates = {
        player: (
            player not in challenger_players
            or challenger_players[player]["log_loss"]
            <= baseline_players[player]["log_loss"] + 0.05
        )
        for player in ("RED", "BLUE")
    }
    terminal_indices = np.asarray(
        [index for index, record in enumerate(records) if record["terminal_pair"]],
        dtype=np.int64,
    )
    baseline_decision = baseline_probability >= 0.5
    challenger_decision = challenger_probability >= 0.5
    changed = baseline_decision != challenger_decision
    improved = changed & (challenger_decision == (labels >= 0.5))
    worsened = changed & (baseline_decision == (labels >= 0.5))
    terminal_baseline = subset_metrics(
        [records[int(index)] for index in terminal_indices],
        baseline_probability[terminal_indices],
    )
    terminal_challenger = subset_metrics(
        [records[int(index)] for index in terminal_indices],
        challenger_probability[terminal_indices],
    )
    terminal_gate = terminal_improvement_gate(
        terminal_baseline,
        terminal_challenger,
        len(terminal_indices),
        args.minimum_terminal_pairs,
    )
    approved = all(
        [
            challenger_metrics["log_loss"] < baseline_metrics["log_loss"],
            challenger_metrics["accuracy"] >= baseline_metrics["accuracy"],
            bootstrap["ci95_high"] <= 0.02,
            int(changed.sum()) >= 1,
            int(improved.sum()) >= int(worsened.sum()),
            terminal_gate,
            provenance_available,
            not overlapping_roots,
            source_provenance_available,
            not overlapping_source_games,
            *player_gates.values(),
        ]
    )
    report = {
        "format": "ghq-counterfactual-challenger-evaluation-v1",
        "reports": [str(path) for path in args.report],
        "dataset_sha256": dataset_hash,
        "pairs": len(records),
        "source_games": len({record["source_game_id"] for record in records}),
        "terminal_pairs": int(len(terminal_indices)),
        "minimum_terminal_pairs": args.minimum_terminal_pairs,
        "baseline": baseline_metrics,
        "challenger": challenger_metrics,
        "baseline_by_root_player": baseline_players,
        "challenger_by_root_player": challenger_players,
        "baseline_by_phase": phase_metrics(records, baseline_probability),
        "challenger_by_phase": phase_metrics(records, challenger_probability),
        "terminal_baseline": terminal_baseline,
        "terminal_challenger": terminal_challenger,
        "ranking_changes": {
            "changed": int(changed.sum()),
            "improved": int(improved.sum()),
            "worsened": int(worsened.sum()),
        },
        "paired_game_bootstrap": bootstrap,
        "gates": {
            "root_players": player_gates,
            "terminal_retention": terminal_gate,
            "training_root_provenance_available": provenance_available,
            "training_evaluation_roots_disjoint": not overlapping_roots,
            "training_source_game_provenance_available": (
                source_provenance_available
            ),
            "training_evaluation_source_games_disjoint": (
                not overlapping_source_games
            ),
        },
        "training_evaluation_root_overlap": overlapping_roots,
        "training_evaluation_source_game_overlap": overlapping_source_games,
        "approved_for_arena": approved,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_pass and not approved:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
