#!/usr/bin/env python3
"""Mine common, symmetry-normalized two-turn opening plans from GHQ games."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import engine
from extract_value_positions import committed_turns


ELIGIBLE_REASONS = {"by HQ capture", "by resignation"}
UNIT_TYPES = {
    name: index
    for index, name in enumerate(engine.PIECE_NAMES)
    if name is not None
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games-csv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--minimum-game-turns", type=int, default=12)
    parser.add_argument("--minimum-plan-count", type=int, default=2)
    parser.add_argument("--maximum-plans", type=int, default=24)
    parser.add_argument(
        "--since",
        help="inclusive creation date in YYYY-MM-DD format",
    )
    return parser.parse_args()


def coordinate_square(value: Any) -> Optional[int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or not all(isinstance(part, int) and 0 <= part < 8 for part in value)
    ):
        return None
    row, column = value
    return (7 - row) * 8 + column


def payload_move_matches(move: engine.Move, payload: Dict[str, Any]) -> bool:
    move_type = payload.get("type")
    args = payload.get("args") or []
    if move.name != move_type:
        return False
    if move_type == "Reinforce":
        return (
            len(args) >= 2
            and UNIT_TYPES.get(args[0]) == move.unit_type
            and coordinate_square(args[1]) == move.to_square
            and (
                coordinate_square(args[2]) if len(args) > 2 else None
            ) == move.capture_preference
        )
    if move_type in {"Move", "MoveAndOrient"}:
        if len(args) < 2:
            return False
        if (
            coordinate_square(args[0]) != move.from_square
            or coordinate_square(args[1]) != move.to_square
        ):
            return False
        if move_type == "MoveAndOrient":
            return (args[2] if len(args) > 2 else None) == move.orientation
        return (
            coordinate_square(args[2]) if len(args) > 2 else None
        ) == move.capture_preference
    return False


def payload_uci(payload: Dict[str, Any]) -> str:
    move_type = payload.get("type")
    args = payload.get("args") or []
    if move_type == "Reinforce":
        if len(args) < 2 or args[0] not in UNIT_TYPES:
            raise ValueError("invalid reinforcement payload")
        to_square = coordinate_square(args[1])
        capture = coordinate_square(args[2]) if len(args) > 2 else None
        if to_square is None:
            raise ValueError("invalid reinforcement square")
        return engine.Move.reinforce(UNIT_TYPES[args[0]], to_square, capture).uci()
    if move_type in {"Move", "MoveAndOrient"}:
        if len(args) < 2:
            raise ValueError(f"invalid {move_type} payload")
        from_square = coordinate_square(args[0])
        to_square = coordinate_square(args[1])
        if from_square is None or to_square is None:
            raise ValueError(f"invalid {move_type} coordinates")
        if move_type == "MoveAndOrient":
            orientation = args[2] if len(args) > 2 else None
            if orientation is not None and orientation >= 8:
                if orientation % 45:
                    raise ValueError(f"invalid legacy orientation {orientation}")
                orientation = (orientation // 45) % 8
            return engine.Move.move_and_orient(
                from_square, to_square, orientation
            ).uci()
        capture = coordinate_square(args[2]) if len(args) > 2 else None
        return engine.Move.move(from_square, to_square, capture).uci()
    raise ValueError(f"unsupported opening action {move_type}")


def apply_payload_turn(
    board: engine.BaseBoard, payloads: Sequence[Dict[str, Any]]
) -> List[engine.Move]:
    mover = board.turn
    result: List[engine.Move] = []
    for payload in payloads:
        legal = list(board.generate_legal_moves())
        matches = [move for move in legal if payload_move_matches(move, payload)]
        if len(matches) != 1:
            raise ValueError(
                f"expected one legal match for {payload.get('type')}, got {len(matches)}"
            )
        board.push(matches[0])
        result.append(matches[0])
    if board.turn == mover:
        skip = next(
            (move for move in board.generate_legal_moves() if move.name == "Skip"),
            None,
        )
        if skip is None:
            raise ValueError("committed turn did not end and Skip was unavailable")
        board.push(skip)
    return result


def rotate_square(square: Optional[int]) -> Optional[int]:
    return None if square is None else 63 - square


def rotate_move(move: engine.Move) -> engine.Move:
    """Rotate a move 180 degrees so a Blue plan is viewed as Red."""
    return engine.Move(
        name=move.name,
        from_square=rotate_square(move.from_square),
        to_square=rotate_square(move.to_square),
        unit_type=move.unit_type,
        orientation=(
            None if move.orientation is None else (move.orientation + 4) % 8
        ),
        capture_preference=rotate_square(move.capture_preference),
        auto_capture_type=move.auto_capture_type,
    )


def normalized_uci(moves: Iterable[engine.Move], color: bool) -> Tuple[str, ...]:
    return tuple(
        (move if color == engine.RED else rotate_move(move)).uci() for move in moves
    )


def opening_signature(board: engine.BaseBoard, color: bool) -> str:
    pieces = []
    for square in engine.scan_forward(board.occupied_co[color]):
        piece = board.piece_at(square)
        if piece is None:
            continue
        normalized_square = square if color == engine.RED else 63 - square
        orientation = piece.orientation
        if orientation is not None and color == engine.BLUE:
            orientation = (orientation + 4) % 8
        pieces.append((piece.piece_type, normalized_square, orientation))
    reserve = tuple(count for _, count in board.reserves[color])
    return json.dumps([sorted(pieces), reserve], separators=(",", ":"))


def normalized_first_signature(first_turn: Sequence[str]) -> str:
    board = engine.BaseBoard()
    for uci in first_turn:
        move = engine.Move.from_uci(uci)
        if not board.is_legal(move):
            raise ValueError(f"normalized opening contains illegal move {uci}")
        board.push(move)
    return opening_signature(board, engine.RED)


def game_plan_rows(
    row: Dict[str, str], minimum_game_turns: int
) -> List[Tuple[Tuple[str, ...], Tuple[str, ...], bool]]:
    gameover = json.loads(row.get("gameover") or "null")
    if not isinstance(gameover, dict):
        return []
    winner = gameover.get("winner")
    if (
        gameover.get("status") != "WIN"
        or winner not in {"RED", "BLUE"}
        or gameover.get("reason") not in ELIGIBLE_REASONS
    ):
        return []
    turns = committed_turns(json.loads(row["log"]))
    if len(turns) < minimum_game_turns or not all(turn in turns for turn in range(1, 5)):
        return []
    replayed: Dict[int, Tuple[bool, Tuple[str, ...]]] = {}
    for turn in range(1, 5):
        color = engine.RED if turn % 2 else engine.BLUE
        moves = tuple(engine.Move.from_uci(payload_uci(payload)) for payload in turns[turn])
        replayed[turn] = (color, normalized_uci(moves, color))
    return [
        (replayed[1][1], replayed[3][1], winner == "RED"),
        (replayed[2][1], replayed[4][1], winner == "BLUE"),
    ]


def main() -> None:
    args = parse_args()
    csv.field_size_limit(300_000_000)
    counts: Counter[Tuple[Tuple[str, ...], Tuple[str, ...]]] = Counter()
    wins: Counter[Tuple[Tuple[str, ...], Tuple[str, ...]]] = Counter()
    first_counts: Counter[Tuple[str, ...]] = Counter()
    first_wins: Counter[Tuple[str, ...]] = Counter()
    summary: Dict[str, Any] = {
        "rows": 0,
        "eligible_games": 0,
        "player_plans": 0,
        "replay_errors": 0,
        "error_examples": [],
    }
    with args.games_csv.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            summary["rows"] += 1
            if args.since and (row.get("createdAt") or "")[:10] < args.since:
                continue
            try:
                plans = game_plan_rows(row, args.minimum_game_turns)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                summary["replay_errors"] += 1
                if len(summary["error_examples"]) < 10:
                    summary["error_examples"].append(
                        {"game_id": row.get("id"), "error": str(error)}
                    )
                continue
            if not plans:
                continue
            summary["eligible_games"] += 1
            for first, second, won in plans:
                key = (first, second)
                counts[key] += 1
                wins[key] += int(won)
                first_counts[first] += 1
                first_wins[first] += int(won)
                summary["player_plans"] += 1

    grouped_counts: Counter[Tuple[Tuple[str, ...], Tuple[str, ...]]] = Counter()
    grouped_wins: Counter[Tuple[Tuple[str, ...], Tuple[str, ...]]] = Counter()
    grouped_representatives: Dict[
        Tuple[Tuple[str, ...], Tuple[str, ...]],
        Counter[Tuple[Tuple[str, ...], Tuple[str, ...]]],
    ] = defaultdict(Counter)
    for (first, second), count in counts.items():
        group = (tuple(sorted(first)), tuple(sorted(second)))
        grouped_counts[group] += count
        grouped_wins[group] += wins[(first, second)]
        grouped_representatives[group][(first, second)] += count
    selected = [
        key
        for key, count in grouped_counts.most_common()
        if count >= args.minimum_plan_count
    ][: args.maximum_plans]
    plans = []
    for group in selected:
        (first, second), _ = grouped_representatives[group].most_common(1)[0]
        try:
            signature = normalized_first_signature(first)
        except ValueError:
            continue
        plans.append(
            {
                "first": list(first),
                "second": list(second),
                "first_signature": signature,
                "count": grouped_counts[group],
                "wins": grouped_wins[group],
                "win_rate": round(
                    grouped_wins[group] / grouped_counts[group], 4
                ),
            }
        )
    grouped_first_counts: Counter[Tuple[str, ...]] = Counter()
    grouped_first_wins: Counter[Tuple[str, ...]] = Counter()
    first_representatives: Dict[
        Tuple[str, ...], Counter[Tuple[str, ...]]
    ] = defaultdict(Counter)
    for first, count in first_counts.items():
        group = tuple(sorted(first))
        grouped_first_counts[group] += count
        grouped_first_wins[group] += first_wins[first]
        first_representatives[group][first] += count
    first_turns = []
    for group, count in grouped_first_counts.most_common(20):
        first, _ = first_representatives[group].most_common(1)[0]
        first_turns.append(
            {
                "moves": list(first),
                "first_signature": normalized_first_signature(first),
                "count": count,
                "wins": grouped_first_wins[group],
                "win_rate": round(grouped_first_wins[group] / count, 4),
            }
        )
    result = {
        "format": "ghq-human-opening-plans-v1",
        "source": str(args.games_csv),
        "filters": {
            "reasons": sorted(ELIGIBLE_REASONS),
            "since": args.since,
            "minimum_game_turns": args.minimum_game_turns,
            "minimum_plan_count": args.minimum_plan_count,
        },
        "summary": summary,
        "first_turns": first_turns,
        "plans": plans,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**summary, "selected_plans": len(plans)}, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
