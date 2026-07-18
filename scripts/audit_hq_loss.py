#!/usr/bin/env python3
"""Exhaustively test whether a GHQ position can survive the opponent's next turn."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Iterator, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public"))

import engine  # noqa: E402


MoveLine = Tuple[str, ...]
DEFAULT_MAX_NODES = 2_000_000


class AuditLimit(RuntimeError):
    pass


@dataclass
class NodeBudget:
    maximum: int
    visited: int = 0

    def consume(self) -> None:
        if self.visited >= self.maximum:
            raise AuditLimit
        self.visited += 1


def complete_turn_states(
    root: engine.BaseBoard,
    budget: NodeBudget,
) -> Iterator[Tuple[engine.BaseBoard, MoveLine]]:
    """Yield each distinct state after the side to move completes its turn."""
    mover = root.turn
    frontier: list[Tuple[engine.BaseBoard, MoveLine]] = [(root, ())]
    seen: set[str] = set()
    while frontier:
        budget.consume()
        board, moves = frontier.pop()
        key = board.serialize()
        if key in seen:
            continue
        seen.add(key)
        if board.is_game_over() or board.turn != mover:
            yield board, moves
            continue
        hq_squares = list(board.pieces(engine.HQ, mover))
        hq_square = hq_squares[0] if hq_squares else None
        children: list[
            tuple[tuple[int, int, str], engine.BaseBoard, engine.Move]
        ] = []
        for move in board.generate_legal_moves():
            child = board.copy()
            child.push(move)
            children.append(
                (
                    defensive_turn_move_priority(
                        board, child, move, mover, hq_square
                    ),
                    child,
                    move,
                )
            )
        # The frontier is LIFO, so ascending insertion explores the greatest
        # defensive-effect key first. Membership is unchanged: this improves
        # time-to-first-safe-line without weakening a forced-loss proof.
        children.sort(key=lambda item: item[0])
        for _, child, move in children:
            frontier.append((child, (*moves, move.uci())))


def defensive_turn_move_priority(
    board: engine.BaseBoard,
    child: engine.BaseBoard,
    move: engine.Move,
    mover: bool,
    hq_square: int | None,
) -> tuple[int, int, str]:
    """Order exact defender turns by their potential effect near the HQ."""
    if move.name == "AutoCapture":
        return (6, 0, move.uci())
    if hq_square is None:
        return (0, 0, move.uci())
    piece_type = (
        move.unit_type
        if move.name == "Reinforce"
        else board.piece_type_at(move.from_square)
        if move.from_square is not None
        else None
    )
    if piece_type == engine.HQ:
        return (5, 0, move.uci())

    hq_file = engine.square_file(hq_square)
    hq_rank = engine.square_rank(hq_square)
    defense_zone = engine.BB_EMPTY
    for file_index in range(max(0, hq_file - 2), min(7, hq_file + 2) + 1):
        for rank_index in range(max(0, hq_rank - 2), min(7, hq_rank + 2) + 1):
            defense_zone |= engine.BB_SQUARES[
                engine.square(file_index, rank_index)
            ]
    newly_bombarded = (
        child.bombarded_co[mover]
        & defense_zone
        & ~board.bombarded_co[mover]
    )
    if move.name == "MoveAndOrient" and newly_bombarded:
        return (4, engine.popcount(newly_bombarded), move.uci())
    if move.capture_preference is not None:
        return (3, 0, move.uci())
    distances = [
        max(
            abs(engine.square_file(square) - hq_file),
            abs(engine.square_rank(square) - hq_rank),
        )
        for square in (move.from_square, move.to_square)
        if square is not None
    ]
    if distances and min(distances) <= 3:
        return (2, -min(distances), move.uci())
    if move.name == "Skip":
        return (1, 0, move.uci())
    return (0, 0, move.uci())


def same_turn_hq_win(
    board: engine.BaseBoard, attacker: bool, budget: NodeBudget
) -> MoveLine | None:
    """Return one complete same-turn HQ capture line, if one exists."""
    mover = board.turn
    frontier: list[Tuple[engine.BaseBoard, MoveLine]] = [(board, ())]
    seen: set[str] = set()
    while frontier:
        budget.consume()
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
        moves_to_try = list(exact_hq_capture_moves(current))
        moves_to_try.sort(key=lambda move: exact_hq_capture_move_priority(current, move))
        for move in moves_to_try:
            child = current.copy()
            child.push(move)
            frontier.append((child, (*moves, move.uci())))
    return None


def exact_hq_capture_moves(board: engine.BaseBoard) -> Iterator[engine.Move]:
    """Collapse only actions provably irrelevant to same-turn HQ capture."""
    enemy_hq_mask = board.hq & board.occupied_co[not board.turn]
    surviving_non_hq = engine.popcount(board.occupied & ~board.hq)
    preserve_artillery_orientations = surviving_non_hq <= 12
    artillery_relocations: set[tuple[int, int]] = set()
    for move in board.generate_legal_moves():
        if move.name == "Skip":
            continue
        if (
            move.name == "Reinforce"
            and move.unit_type is not None
            and not engine.is_infantry(move.unit_type)
        ):
            continue
        if move.name == "MoveAndOrient":
            if (
                move.from_square is None
                or move.to_square is None
                or move.orientation is None
            ):
                continue
            piece_type = board.piece_type_at(move.from_square)
            distance = 3 if piece_type == engine.HEAVY_ARTILLERY else 2
            target = board.get_bombardment_target(
                move.to_square, move.orientation, distance
            )
            directly_bombards_hq = bool(
                target is not None
                and engine.between_inclusive_end(move.to_square, target)
                & enemy_hq_mask
            )
            if move.from_square == move.to_square:
                if directly_bombards_hq:
                    yield move
                continue
            relocation = (move.from_square, move.to_square)
            if not (
                directly_bombards_hq or preserve_artillery_orientations
            ):
                if relocation in artillery_relocations:
                    continue
                artillery_relocations.add(relocation)
        yield move


def exact_hq_capture_move_priority(
    board: engine.BaseBoard, move: engine.Move
) -> tuple[int, int, int, str]:
    enemy_hqs = list(board.pieces(engine.HQ, not board.turn))
    target = move.capture_preference
    captures_hq = bool(
        target is not None and board.piece_type_at(target) == engine.HQ
    )
    captures_piece = target is not None
    piece_type = (
        move.unit_type
        if move.name == "Reinforce"
        else board.piece_type_at(move.from_square)
        if move.from_square is not None
        else None
    )
    is_infantry_action = bool(
        piece_type is not None and engine.is_infantry(piece_type)
    )
    destination_distance = min(
        (
            max(
                abs(engine.square_file(move.to_square) - engine.square_file(hq_square)),
                abs(engine.square_rank(move.to_square) - engine.square_rank(hq_square)),
            )
            for hq_square in enemy_hqs
            if move.to_square is not None
        ),
        default=9,
    )
    return (
        3 if captures_hq else 2 if captures_piece else 1 if is_infantry_action else 0,
        -destination_distance,
        1 if move.name in ("Move", "Reinforce", "AutoCapture") else 0,
        move.uci(),
    )


def audit_hq_loss(
    fen: str, example_limit: int = 10, max_nodes: int = DEFAULT_MAX_NODES
) -> dict[str, object]:
    if max_nodes < 1:
        raise ValueError("max_nodes must be positive")
    position = engine.BaseBoard(fen)
    defender = position.turn
    attacker = not defender
    complete_states = 0
    terminal_states = 0
    safe_count = 0
    safe_turns: list[dict[str, object]] = []
    budget = NodeBudget(max_nodes)
    truncated = False
    stopped_after_safe = False

    try:
        for result, moves in complete_turn_states(position, budget):
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
                    stopped_after_safe = True
                    break
                continue
            reply = same_turn_hq_win(result, attacker, budget)
            if reply is None:
                safe_count += 1
                if len(safe_turns) < example_limit:
                    safe_turns.append(
                        {"moves": moves, "reason": "no-same-turn-hq-capture"}
                    )
                stopped_after_safe = True
                break
    except AuditLimit:
        truncated = True

    forced = safe_count == 0 and not truncated
    inconclusive = safe_count == 0 and truncated

    return {
        "fen": fen,
        "defender": "RED" if defender == engine.RED else "BLUE",
        "complete_turn_states": complete_states,
        "safe_turns": safe_count,
        "forced_hq_loss": forced,
        "inconclusive": inconclusive,
        "exhaustive": not truncated and not stopped_after_safe,
        "nodes_visited": budget.visited,
        "max_nodes": budget.maximum,
        "safe_examples": safe_turns,
        "terminal_states": terminal_states,
    }


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fen", help="GHQ FEN before the losing player's turn")
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--max-nodes", type=int, default=DEFAULT_MAX_NODES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    print(
        json.dumps(
            audit_hq_loss(
                args.fen,
                max(0, args.examples),
                max_nodes=args.max_nodes,
            )
        )
    )


if __name__ == "__main__":
    main()
