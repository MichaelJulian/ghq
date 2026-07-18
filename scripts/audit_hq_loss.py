#!/usr/bin/env python3
"""Exhaustively test whether a GHQ position can survive the opponent's next turn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public"))

import engine  # noqa: E402


MoveLine = Tuple[str, ...]


def complete_turn_states(
    root: engine.BaseBoard,
) -> Iterator[Tuple[engine.BaseBoard, MoveLine]]:
    """Yield each distinct state after the side to move completes its turn."""
    mover = root.turn
    frontier: list[Tuple[engine.BaseBoard, MoveLine]] = [(root, ())]
    seen: set[str] = set()
    while frontier:
        board, moves = frontier.pop()
        key = board.serialize()
        if key in seen:
            continue
        seen.add(key)
        if board.is_game_over() or board.turn != mover:
            yield board, moves
            continue
        for move in board.generate_legal_moves():
            child = board.copy()
            child.push(move)
            frontier.append((child, (*moves, move.uci())))


def same_turn_hq_win(
    board: engine.BaseBoard, attacker: bool
) -> MoveLine | None:
    """Return one complete same-turn HQ capture line, if one exists."""
    mover = board.turn
    frontier: list[Tuple[engine.BaseBoard, MoveLine]] = [(board, ())]
    seen: set[str] = set()
    while frontier:
        current, moves = frontier.pop()
        key = current.serialize()
        if key in seen:
            continue
        seen.add(key)
        outcome = current.outcome()
        if outcome is not None:
            if outcome.termination == "hq-capture" and outcome.winner == attacker:
                return moves
            continue
        if current.turn != mover:
            continue
        for move in current.generate_legal_moves():
            child = current.copy()
            child.push(move)
            frontier.append((child, (*moves, move.uci())))
    return None


def audit_hq_loss(fen: str, example_limit: int = 10) -> dict[str, object]:
    position = engine.BaseBoard(fen)
    defender = position.turn
    attacker = not defender
    complete_states = 0
    terminal_states = 0
    safe_count = 0
    safe_turns: list[dict[str, object]] = []

    for result, moves in complete_turn_states(position):
        complete_states += 1
        outcome = result.outcome()
        if outcome is not None:
            terminal_states += 1
            if outcome.winner != attacker:
                safe_count += 1
                if len(safe_turns) < example_limit:
                    safe_turns.append(
                        {
                            "moves": moves,
                            "reason": f"terminal-{outcome.termination}",
                        }
                    )
            continue
        reply = same_turn_hq_win(result, attacker)
        if reply is None:
            safe_count += 1
            if len(safe_turns) < example_limit:
                safe_turns.append(
                    {"moves": moves, "reason": "no-same-turn-hq-capture"}
                )

    return {
        "fen": fen,
        "defender": "RED" if defender == engine.RED else "BLUE",
        "complete_turn_states": complete_states,
        "safe_turns": safe_count,
        "forced_hq_loss": safe_count == 0,
        "safe_examples": safe_turns,
        "terminal_states": terminal_states,
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fen", help="GHQ FEN before the losing player's turn")
    parser.add_argument("--examples", type=int, default=10)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    print(json.dumps(audit_hq_loss(args.fen, max(0, args.examples))))


if __name__ == "__main__":
    main()
