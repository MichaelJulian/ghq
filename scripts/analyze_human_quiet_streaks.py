#!/usr/bin/env python3
"""Measure capture/deployment-free stretches in saved human GHQ games."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from scripts.extract_value_positions import committed_turns, extract_game
except ModuleNotFoundError:  # Direct `python scripts/...` execution.
    from extract_value_positions import committed_turns, extract_game


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-csv", required=True, type=Path)
    parser.add_argument("--since", help="ISO date/time cutoff")
    return parser.parse_args()


def maximum_quiet_streak(
    turns: Dict[int, List[Dict[str, Any]]], capture_turns: Iterable[int]
) -> int:
    captures = set(capture_turns)
    longest = 0
    current = 0
    for turn, actions in sorted(turns.items()):
        made_progress = turn in captures or any(
            action.get("type") == "Reinforce" for action in actions
        )
        current = 0 if made_progress else current + 1
        longest = max(longest, current)
    return longest


def strategic_metrics(board: List[List[Any]], player: str) -> tuple[int, int]:
    own = []
    enemy_hq = None
    for row, rank in enumerate(board):
        for column, piece in enumerate(rank):
            if not piece:
                continue
            if piece.get("player") == player and piece.get("type") != "HQ":
                own.append((row, column))
            elif piece.get("player") != player and piece.get("type") == "HQ":
                enemy_hq = (row, column)
    frontier = max(
        (8 - row if player == "RED" else row + 1 for row, _ in own),
        default=0,
    )
    hq_distance = min(
        (
            max(abs(row - enemy_hq[0]), abs(column - enemy_hq[1]))
            for row, column in own
        ),
        default=8,
    ) if enemy_hq else 8
    return frontier, hq_distance


def maximum_strategic_quiet_streak(
    turns: Dict[int, List[Dict[str, Any]]],
    capture_turns: Iterable[int],
    initial_board: List[List[Any]],
    snapshots: List[Dict[str, Any]],
) -> int:
    captures = set(capture_turns)
    by_turn = {snapshot["turn"]: snapshot for snapshot in snapshots}
    best = {
        player: strategic_metrics(initial_board, player)
        for player in ("RED", "BLUE")
    }
    longest = 0
    current = 0
    for turn, actions in sorted(turns.items()):
        snapshot = by_turn.get(turn)
        if not snapshot:
            continue
        mover = "BLUE" if snapshot["current_player"] == "RED" else "RED"
        frontier, hq_distance = strategic_metrics(snapshot["board"], mover)
        best_frontier, best_hq_distance = best[mover]
        strategic = frontier > best_frontier or hq_distance < best_hq_distance
        best[mover] = (
            max(best_frontier, frontier),
            min(best_hq_distance, hq_distance),
        )
        made_progress = (
            strategic
            or turn in captures
            or any(action.get("type") == "Reinforce" for action in actions)
        )
        current = 0 if made_progress else current + 1
        longest = max(longest, current)
    return longest


def percentile(values: List[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: List[int]) -> Dict[str, Any]:
    return {
        "games": len(values),
        "median": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values, default=0),
        "at_least_24": round(sum(value >= 24 for value in values) / len(values), 4)
        if values
        else 0,
        "at_least_36": round(sum(value >= 36 for value in values) / len(values), 4)
        if values
        else 0,
        "at_least_48": round(sum(value >= 48 for value in values) / len(values), 4)
        if values
        else 0,
    }


def parse_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def main() -> None:
    args = parse_args()
    cutoff = parse_datetime(args.since) if args.since else None
    csv.field_size_limit(300_000_000)
    by_mode: Dict[str, List[int]] = defaultdict(list)
    strategic_by_mode: Dict[str, List[int]] = defaultdict(list)
    skipped = 0
    with args.games_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            try:
                created_at = parse_datetime(row["createdAt"])
                gameover = json.loads(row.get("gameover") or "null")
                state = json.loads(row["state"])
                log = json.loads(row["log"])
                if cutoff and created_at < cutoff:
                    continue
                if not isinstance(gameover, dict) or gameover.get("status") != "WIN":
                    continue
                if gameover.get("reason") not in {"by HQ capture", "by resignation"}:
                    continue
                turns = committed_turns(log)
                if len(turns) < 8:
                    continue
                capture_turns = {
                    item["turn"]
                    for item in state["G"].get("historyLog") or []
                    if item.get("captured") and isinstance(item.get("turn"), int)
                }
                streak = maximum_quiet_streak(turns, capture_turns)
                initial_board = json.loads(row["initialState"])["G"]["board"]
                snapshots = extract_game(row, 8)
                strategic_streak = maximum_strategic_quiet_streak(
                    turns,
                    capture_turns,
                    initial_board,
                    snapshots,
                )
                time_control = state["G"].get("timeControl")
                mode = "timed" if isinstance(time_control, (int, float)) and time_control > 0 else "correspondence"
                by_mode[mode].append(streak)
                by_mode["all"].append(streak)
                strategic_by_mode[mode].append(strategic_streak)
                strategic_by_mode["all"].append(strategic_streak)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError):
                skipped += 1
    print(
        json.dumps(
            {
                "since": cutoff.isoformat() if cutoff else None,
                "definition": "maximum consecutive committed turns without a capture or reinforcement",
                "groups": {mode: summarize(values) for mode, values in by_mode.items()},
                "strategic_reset_definition": "capture, reinforcement, new frontier, or new closest distance to enemy HQ",
                "strategic_reset_groups": {
                    mode: summarize(values)
                    for mode, values in strategic_by_mode.items()
                },
                "skipped_rows": skipped,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
