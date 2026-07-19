#!/usr/bin/env python3
"""Summarize structural quality in persisted Vercel self-play snapshots."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.parse import quote
from urllib.request import Request, urlopen

import ghq_ai
import engine


METRIC_NAMES = (
    "support_penalty",
    "overextension_penalty",
    "phase_extension_penalty",
    "dispersion_penalty",
    "infantry_isolation_penalty",
    "airborne_survival_penalty",
    "infantry_shape_score",
    "artillery_formation_score",
    "development_score",
    "frontier_rank",
    "largest_component_ratio",
    "home_rank_occupancy",
    "immobile_units",
    "mean_relocation_options",
    "tactical_risk_value",
    "forced_loss_value",
    "critical_exposure_value",
)

DANGER_METRICS = (
    "support_penalty",
    "phase_extension_penalty",
    "dispersion_penalty",
    "infantry_isolation_penalty",
    "airborne_survival_penalty",
)

GAME_NUMBER_PATTERN = re.compile(r"^(.*)-(\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base-url", default="https://ghq-one.vercel.app")
    return parser.parse_args()


def read_summary(base_url: str, generation_id: str) -> Dict[str, Any]:
    generation = quote(generation_id, safe="")
    url = (
        f"{base_url.rstrip('/')}/api/self-play/generations/"
        f"{generation}/summary"
    )
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=90) as response:  # noqa: S310 - explicit API URL
        return json.load(response)


def snapshot_metric_rows(snapshot: Dict[str, Any]) -> List[Dict[str, Any]]:
    board = engine.BaseBoard(str(snapshot["currentFen"]))
    turn_number = int(snapshot["completedTurns"]) + 1
    tactical_probe = ghq_ai.Searcher(
        "balanced", time_ms=60_000, beam_width=1, turn_number=turn_number
    )
    rows: List[Dict[str, Any]] = []
    for side, color in (("RED", engine.RED), ("BLUE", engine.BLUE)):
        structure = ghq_ai.structure_metrics(board, color)
        optionality = ghq_ai.optionality_metrics(board, color)
        tactical_risk, forced_loss, critical_exposure = (
            tactical_probe.tactical_risk(
                board, color, check_hq_combinations=False
            )
        )
        rows.append(
            {
                "gameId": snapshot["gameId"],
                "completedTurns": int(snapshot["completedTurns"]),
                "side": side,
                "to_move": board.turn == color,
                "support_penalty": ghq_ai.support_penalty(board, color),
                "overextension_penalty": ghq_ai.overextension_penalty(
                    board, color
                ),
                "phase_extension_penalty": ghq_ai.phase_extension_penalty(
                    board, color, turn_number
                ),
                "dispersion_penalty": ghq_ai.dispersion_penalty(board, color),
                "infantry_isolation_penalty": (
                    ghq_ai.infantry_isolation_penalty(board, color)
                ),
                "airborne_survival_penalty": (
                    ghq_ai.airborne_survival_penalty(board, color)
                ),
                "infantry_shape_score": ghq_ai.infantry_shape_score(
                    board, color
                ),
                "artillery_formation_score": ghq_ai.artillery_formation(
                    board, color
                ),
                "development_score": ghq_ai.development_for(
                    board, color, turn_number
                ),
                "frontier_rank": structure["frontier_rank"],
                "largest_component_ratio": structure[
                    "largest_component_ratio"
                ],
                "home_rank_occupancy": optionality["home_rank_occupancy"],
                "immobile_units": optionality["immobile_units"],
                "mean_relocation_options": optionality[
                    "mean_relocation_options"
                ],
                "tactical_risk_value": tactical_risk,
                "forced_loss_value": forced_loss,
                "critical_exposure_value": critical_exposure,
            }
        )
    return rows


def summarize_metric_rows(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    materialized = list(rows)
    metrics = (
        {
            metric: {
                "mean": round(
                    statistics.mean(
                        float(row[metric]) for row in materialized
                    ),
                    4,
                ),
                "min": round(
                    min(float(row[metric]) for row in materialized), 4
                ),
                "max": round(
                    max(float(row[metric]) for row in materialized), 4
                ),
            }
            for metric in METRIC_NAMES
        }
        if materialized
        else {}
    )
    structural_debt_rows = [
        row
        for row in materialized
        if any(float(row[metric]) > 0.0 for metric in DANGER_METRICS)
    ]
    tactical_danger_rows = [
        row
        for row in materialized
        if float(row["tactical_risk_value"]) > 0.0
    ]
    repair_required_rows = [
        row
        for row in tactical_danger_rows
        if bool(row.get("to_move"))
    ]
    immediate_danger_rows = [
        row
        for row in tactical_danger_rows
        if not bool(row.get("to_move"))
    ]
    constrained_rows = [
        row
        for row in materialized
        if float(row["overextension_penalty"]) > 0.0
        or float(row["immobile_units"]) > 0.0
    ]
    return {
        "sidePositions": len(materialized),
        "metrics": metrics,
        "structuralDebtPositions": len(structural_debt_rows),
        "structuralDebtExamples": structural_debt_rows[:12],
        # Retain the old keys for readers of v1 reports. Structural debt is
        # not proof that a piece can actually be captured on the next turn.
        "structuralDangerPositions": len(structural_debt_rows),
        "structuralDangerExamples": structural_debt_rows[:12],
        "tacticalDangerPositions": len(tactical_danger_rows),
        "tacticalDangerExamples": tactical_danger_rows[:12],
        "repairRequiredPositions": len(repair_required_rows),
        "repairRequiredExamples": repair_required_rows[:12],
        "immediateForcedCapturePositions": len(immediate_danger_rows),
        "immediateForcedCaptureExamples": immediate_danger_rows[:12],
        "immediateTacticalDangerPositions": len(immediate_danger_rows),
        "immediateTacticalDangerExamples": immediate_danger_rows[:12],
        "constrainedPositions": len(constrained_rows),
        "constrainedExamples": constrained_rows[:12],
    }


def summarize_pair_diversity(
    snapshots: Iterable[Dict[str, Any]],
) -> Dict[str, Any]:
    positions_by_pair: Dict[str, List[str]] = {}
    unpaired_games = 0
    for snapshot in snapshots:
        game_id = str(snapshot["gameId"])
        match = GAME_NUMBER_PATTERN.match(game_id)
        if match is None:
            unpaired_games += 1
            continue
        game_number = int(match.group(2))
        if game_number < 1:
            unpaired_games += 1
            continue
        pair_number = (game_number - 1) // 2 + 1
        pair_id = f"{match.group(1)}-pair-{pair_number}"
        positions_by_pair.setdefault(pair_id, []).append(
            str(snapshot["currentFen"])
        )
    complete_signatures = [
        tuple(sorted(positions))
        for positions in positions_by_pair.values()
        if len(positions) == 2
    ]
    signature_counts: Dict[tuple[str, ...], int] = {}
    for signature in complete_signatures:
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
    return {
        "completeColorSwapPairs": len(complete_signatures),
        "incompleteColorSwapPairs": sum(
            len(positions) != 2 for positions in positions_by_pair.values()
        ),
        "unpairedSnapshotGames": unpaired_games,
        "uniquePairTrajectories": len(signature_counts),
        "maxPairTrajectoryMultiplicity": max(
            signature_counts.values(), default=0
        ),
        "pairTrajectoryMultiplicity": sorted(
            signature_counts.values(), reverse=True
        ),
    }


def analyze_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    snapshots = summary.get("progress", {}).get("snapshots", [])
    position_counts: Dict[str, int] = {}
    for snapshot in snapshots:
        fen = str(snapshot["currentFen"])
        position_counts[fen] = position_counts.get(fen, 0) + 1
    rows = [
        row
        for snapshot in snapshots
        for row in snapshot_metric_rows(snapshot)
    ]
    return {
        "format": "ghq-progress-structure-v1",
        "generationId": summary.get("generationId"),
        "snapshotGames": len(snapshots),
        "uniqueSnapshotPositions": len(position_counts),
        "maxPositionMultiplicity": max(position_counts.values(), default=0),
        "positionMultiplicity": sorted(
            position_counts.values(), reverse=True
        ),
        "pairDiversity": summarize_pair_diversity(snapshots),
        "completedTurns": sorted(
            {int(snapshot["completedTurns"]) for snapshot in snapshots}
        ),
        **summarize_metric_rows(rows),
    }


def main() -> None:
    args = parse_args()
    report = analyze_summary(read_summary(args.base_url, args.generation))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
