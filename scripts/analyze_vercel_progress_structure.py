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
PIECE_VALUE_BY_NAME = {
    str(engine.PIECE_NAMES[piece_type]): float(
        ghq_ai.PIECE_VALUES[piece_type]
    )
    for piece_type in ghq_ai.NON_HQ_TYPES
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--quiet", action="store_true")
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
                "piece_inventory": piece_inventory(board, color),
                "forced_capture_targets": forced_capture_targets(
                    board, color
                ),
            }
        )
    return rows


def piece_inventory(board: engine.BaseBoard, color: bool) -> Dict[str, int]:
    return {
        str(engine.PIECE_NAMES[piece_type]): (
            engine.popcount(board.pieces_mask(piece_type, color))
            + board.get_reserve_count(piece_type, color)
        )
        for piece_type in ghq_ai.NON_HQ_TYPES
    }


def inventory_value(inventory: Dict[str, Any]) -> float:
    return sum(
        float(count) * PIECE_VALUE_BY_NAME.get(str(piece_type), 0.0)
        for piece_type, count in inventory.items()
    )


def forced_capture_targets(
    board: engine.BaseBoard, defender: bool
) -> List[Dict[str, Any]]:
    probe = ghq_ai.Searcher.board_as_turn(board, not defender)
    frontier = [probe]
    targets: Dict[tuple[int, int], Dict[str, Any]] = {}
    for _ in range(8):
        next_frontier = []
        for position in frontier[:12]:
            legal = list(position.generate_legal_moves())
            if not legal or not all(
                move.name == "AutoCapture" for move in legal
            ):
                continue
            for move in legal[:12]:
                target = move.capture_preference
                if target is not None and (
                    engine.BB_SQUARES[target]
                    & position.occupied_co[defender]
                ):
                    piece_type = position.piece_type_at(target)
                    if piece_type is not None:
                        targets[(target, piece_type)] = {
                            "square": engine.square_name(target),
                            "pieceType": str(
                                engine.PIECE_NAMES[piece_type]
                            ),
                            "value": ghq_ai.PIECE_VALUES[piece_type],
                        }
                child = position.copy()
                child.push(move)
                next_frontier.append(child)
        if not next_frontier:
            break
        frontier = next_frontier[:12]
    return sorted(
        targets.values(), key=lambda target: (target["square"], target["pieceType"])
    )


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
        "immediateCaptureThreatPositions": len(immediate_danger_rows),
        "immediateCaptureThreatExamples": immediate_danger_rows[:12],
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
        "workflowRuns": summary.get("workflowRuns", {}),
        "activeProgressRuntime": summary.get(
            "activeProgressRuntime", {}
        ),
        "snapshotTelemetry": [
            {
                field: snapshot.get(field)
                for field in (
                    "gameId",
                    "seed",
                    "codeVersion",
                    "redAgentId",
                    "blueAgentId",
                    "completedTurns",
                    "currentPlayer",
                    "currentFen",
                    "decisions",
                    "depthAtLeastTwoDecisions",
                    "fallbackDecisions",
                    "unverifiedFallbackDecisions",
                    "timedOutDecisions",
                    "status",
                    "updatedAt",
                )
            }
            for snapshot in snapshots
        ],
        "positionMetrics": rows,
        "completedTurns": sorted(
            {int(snapshot["completedTurns"]) for snapshot in snapshots}
        ),
        **summarize_metric_rows(rows),
    }


def compare_checkpoint_reports(
    before: Dict[str, Any], after: Dict[str, Any]
) -> Dict[str, Any]:
    if before.get("generationId") != after.get("generationId"):
        raise ValueError("checkpoint reports belong to different generations")
    before_games = {
        str(item["gameId"]): item
        for item in before.get("snapshotTelemetry", [])
    }
    after_games = {
        str(item["gameId"]): item
        for item in after.get("snapshotTelemetry", [])
    }
    shared_games = sorted(set(before_games).intersection(after_games))
    counter_fields = (
        "decisions",
        "depthAtLeastTwoDecisions",
        "fallbackDecisions",
        "unverifiedFallbackDecisions",
        "timedOutDecisions",
    )
    counter_deltas = {
        field: sum(
            int(after_games[game].get(field) or 0)
            - int(before_games[game].get(field) or 0)
            for game in shared_games
        )
        for field in counter_fields
    }
    before_positions = {
        (str(item["gameId"]), str(item["side"])): item
        for item in before.get("positionMetrics", [])
    }
    after_positions = {
        (str(item["gameId"]), str(item["side"])): item
        for item in after.get("positionMetrics", [])
    }
    repair_keys = sorted(
        key
        for key, item in before_positions.items()
        if bool(item.get("to_move"))
        and float(item.get("tactical_risk_value") or 0.0) > 0.0
    )
    repair_outcomes = []
    for key in repair_keys:
        prior = before_positions[key]
        current = after_positions.get(key)
        current_risk = (
            float(current.get("tactical_risk_value") or 0.0)
            if current is not None
            else None
        )
        before_snapshot = before_games.get(key[0])
        after_snapshot = after_games.get(key[0])
        color = engine.RED if key[1] == "RED" else engine.BLUE
        prior_board = (
            engine.BaseBoard(str(before_snapshot["currentFen"]))
            if before_snapshot and before_snapshot.get("currentFen")
            else None
        )
        current_board = (
            engine.BaseBoard(str(after_snapshot["currentFen"]))
            if after_snapshot and after_snapshot.get("currentFen")
            else None
        )
        targets = (
            forced_capture_targets(prior_board, color)
            if prior_board is not None
            else []
        )
        prior_inventory = (
            piece_inventory(prior_board, color)
            if prior_board is not None
            else {}
        )
        current_inventory = (
            piece_inventory(current_board, color)
            if current_board is not None
            else {}
        )
        target_types = sorted(
            {str(target["pieceType"]) for target in targets}
        )
        retained_by_type = {
            piece_type: (
                current_inventory.get(piece_type, 0)
                >= prior_inventory.get(piece_type, 0)
            )
            for piece_type in target_types
        }
        opponent_key = (key[0], "BLUE" if key[1] == "RED" else "RED")
        prior_opponent = before_positions.get(opponent_key)
        current_opponent = after_positions.get(opponent_key)
        prior_own_value = inventory_value(
            dict(prior.get("piece_inventory") or {})
        )
        current_own_value = (
            inventory_value(dict(current.get("piece_inventory") or {}))
            if current is not None
            else None
        )
        prior_opponent_value = (
            inventory_value(
                dict(prior_opponent.get("piece_inventory") or {})
            )
            if prior_opponent is not None
            else None
        )
        current_opponent_value = (
            inventory_value(
                dict(current_opponent.get("piece_inventory") or {})
            )
            if current_opponent is not None
            else None
        )
        own_material_lost = (
            max(0.0, prior_own_value - current_own_value)
            if current_own_value is not None
            else None
        )
        opponent_material_lost = (
            max(0.0, prior_opponent_value - current_opponent_value)
            if prior_opponent_value is not None
            and current_opponent_value is not None
            else None
        )
        net_material_exchange = (
            opponent_material_lost - own_material_lost
            if own_material_lost is not None
            and opponent_material_lost is not None
            else None
        )
        repair_outcomes.append(
            {
                "gameId": key[0],
                "side": key[1],
                "priorRiskValue": float(prior["tactical_risk_value"]),
                "laterCheckpointRiskValue": current_risk,
                "sameSideRiskFreeAtLaterCheckpoint": (
                    current_risk == 0.0 if current_risk is not None else None
                ),
                "priorForcedCaptureTargets": targets,
                "threatenedInventoryRetainedByType": retained_by_type,
                "allThreatenedInventoryRetained": (
                    all(retained_by_type.values())
                    if retained_by_type
                    else None
                ),
                "ownMaterialLost": own_material_lost,
                "opponentMaterialLost": opponent_material_lost,
                "netMaterialExchange": net_material_exchange,
                "materialExchangeAssessment": (
                    "favorable"
                    if net_material_exchange is not None
                    and net_material_exchange > 0.0
                    else "unfavorable"
                    if net_material_exchange is not None
                    and net_material_exchange < 0.0
                    else "even"
                    if net_material_exchange == 0.0
                    else None
                ),
            }
        )
    return {
        "beforeCompletedTurns": before.get("completedTurns", []),
        "afterCompletedTurns": after.get("completedTurns", []),
        "sharedGames": len(shared_games),
        "counterDeltas": counter_deltas,
        "repairObligations": len(repair_outcomes),
        "sameSideRiskFreeAtLaterCheckpoint": sum(
            outcome["sameSideRiskFreeAtLaterCheckpoint"] is True
            for outcome in repair_outcomes
        ),
        "threatenedInventoryRetained": sum(
            outcome["allThreatenedInventoryRetained"] is True
            for outcome in repair_outcomes
        ),
        "favorableMaterialExchanges": sum(
            outcome["materialExchangeAssessment"] == "favorable"
            for outcome in repair_outcomes
        ),
        "unfavorableMaterialExchanges": sum(
            outcome["materialExchangeAssessment"] == "unfavorable"
            for outcome in repair_outcomes
        ),
        "repairOutcomes": repair_outcomes,
        "structuralDebtPositionDelta": int(
            after.get("structuralDebtPositions", 0)
        )
        - int(before.get("structuralDebtPositions", 0)),
        "immediateForcedCapturePositionDelta": int(
            after.get("immediateForcedCapturePositions", 0)
        )
        - int(before.get("immediateForcedCapturePositions", 0)),
        "immediateCaptureThreatPositionDelta": int(
            after.get(
                "immediateCaptureThreatPositions",
                after.get("immediateForcedCapturePositions", 0),
            )
        )
        - int(
            before.get(
                "immediateCaptureThreatPositions",
                before.get("immediateForcedCapturePositions", 0),
            )
        ),
    }


def main() -> None:
    args = parse_args()
    report = analyze_summary(read_summary(args.base_url, args.generation))
    if args.previous:
        previous = json.loads(args.previous.read_text(encoding="utf-8"))
        report["checkpointComparison"] = compare_checkpoint_reports(
            previous, report
        )
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    if args.quiet:
        print(
            json.dumps(
                {
                    "generationId": report["generationId"],
                    "completedTurns": report["completedTurns"],
                    "snapshotGames": report["snapshotGames"],
                    "activeProgressRuntime": report[
                        "activeProgressRuntime"
                    ],
                    "pairDiversity": report["pairDiversity"],
                    "structuralDebtPositions": report[
                        "structuralDebtPositions"
                    ],
                    "repairRequiredPositions": report[
                        "repairRequiredPositions"
                    ],
                    "immediateForcedCapturePositions": report[
                        "immediateForcedCapturePositions"
                    ],
                    "immediateCaptureThreatPositions": report[
                        "immediateCaptureThreatPositions"
                    ],
                    "checkpointComparison": report.get(
                        "checkpointComparison"
                    ),
                    "output": str(args.output) if args.output else None,
                },
                indent=2,
            )
        )
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
