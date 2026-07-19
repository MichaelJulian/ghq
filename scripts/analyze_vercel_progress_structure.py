#!/usr/bin/env python3
"""Summarize structural quality in persisted Vercel self-play snapshots."""

from __future__ import annotations

import argparse
import json
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
)

DANGER_METRICS = (
    "support_penalty",
    "phase_extension_penalty",
    "dispersion_penalty",
    "infantry_isolation_penalty",
    "airborne_survival_penalty",
)


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
    rows: List[Dict[str, Any]] = []
    for side, color in (("RED", engine.RED), ("BLUE", engine.BLUE)):
        structure = ghq_ai.structure_metrics(board, color)
        optionality = ghq_ai.optionality_metrics(board, color)
        rows.append(
            {
                "gameId": snapshot["gameId"],
                "completedTurns": int(snapshot["completedTurns"]),
                "side": side,
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
    danger_rows = [
        row
        for row in materialized
        if any(float(row[metric]) > 0.0 for metric in DANGER_METRICS)
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
        "structuralDangerPositions": len(danger_rows),
        "structuralDangerExamples": danger_rows[:12],
        "constrainedPositions": len(constrained_rows),
        "constrainedExamples": constrained_rows[:12],
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
