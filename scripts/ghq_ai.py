#!/usr/bin/env python3
"""Explainable, headless GHQ search engine.

Search depth is measured in complete player turns. A turn can contain automatic
captures followed by up to three player actions, matching public/engine.py.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "public"))

import engine  # noqa: E402


PIECE_VALUES = {
    engine.HQ: 100.0,
    engine.INFANTRY: 1.0,
    engine.ARMORED_INFANTRY: 3.0,
    engine.AIRBORNE_INFANTRY: 5.0,
    engine.ARTILLERY: 3.0,
    engine.ARMORED_ARTILLERY: 5.0,
    engine.HEAVY_ARTILLERY: 6.0,
}
UNIT_TYPES = tuple(PIECE_VALUES)
NON_HQ_TYPES = tuple(piece for piece in UNIT_TYPES if piece != engine.HQ)
ARTILLERY_TYPES = (engine.ARTILLERY, engine.ARMORED_ARTILLERY, engine.HEAVY_ARTILLERY)
INFANTRY_TYPES = (engine.INFANTRY, engine.ARMORED_INFANTRY, engine.AIRBORNE_INFANTRY)

# Component weights are intentionally editable. A personality changes priorities,
# not game rules or the underlying feature definitions.
PERSONALITIES: Dict[str, Dict[str, float]] = {
    "balanced": {},
    "fortress": {
        "support": 1.45,
        "overextension": 1.40,
        "artillery_formation": 1.25,
        "hq_safety": 1.35,
        "mobility": 0.80,
        "optionality": 0.90,
    },
    "mobile": {
        "material": 0.90,
        "support": 0.85,
        "mobility": 1.60,
        "optionality": 1.45,
        "open_board_armored_infantry": 1.55,
        "artillery_formation": 0.70,
    },
    "artillery": {
        "artillery_formation": 1.55,
        "artillery_pressure": 1.45,
        "support": 1.20,
        "open_board_armored_infantry": 0.70,
    },
    "mobile_raider": {
        "material": 0.90,
        "support": 0.85,
        "mobility": 1.60,
        "optionality": 1.45,
        "open_board_armored_infantry": 1.55,
        "artillery_formation": 0.70,
    },
    "battery_commander": {
        "artillery_formation": 1.55,
        "artillery_pressure": 1.45,
        "optionality": 0.90,
        "support": 1.20,
        "open_board_armored_infantry": 0.70,
    },
    "para_specialist": {
        "airborne_survival": 1.65,
        "support": 1.15,
        "mobility": 1.10,
        "artillery_pressure": 1.10,
    },
    "tactical_gambler": {
        "material": 0.82,
        "support": 0.72,
        "overextension": 0.65,
        "artillery_pressure": 1.55,
        "mobility": 1.35,
        "hq_safety": 0.85,
    },
}

BASE_WEIGHTS = {
    "material": 1.0,
    "support": 1.0,
    "overextension": 1.0,
    "phase_extension": 1.0,
    "cohesion": 1.0,
    "optionality": 1.0,
    "artillery_formation": 1.0,
    "artillery_pressure": 1.0,
    "artillery_protection": 1.0,
    "infantry_shape": 1.0,
    "infantry_isolation": 1.0,
    "mobility": 1.0,
    "open_board_armored_infantry": 1.0,
    "airborne_survival": 1.0,
    "hq_safety": 1.0,
    "development": 1.0,
}

MATE_SCORE = 1_000_000.0
MISSIONLESS_PARATROOPER_PENALTY = 9.0
EARLY_GAME_LAST_TURN = 12

# In 194 reconstructed complete games (12,083 positions), the 90th-95th
# percentile non-paratrooper frontier was rank 2 on turns 1-2, rank 3 on
# turns 3-4, and rank 4 on turns 5-6.  Continue that two-turn cadence through
# the opening.  The value is a rank from the moving side's perspective: its
# deployment rank is 1.
EARLY_FRONTIER_RANKS: Tuple[Tuple[int, int], ...] = (
    (2, 2),
    (4, 3),
    (6, 4),
    (8, 5),
    (10, 6),
    (12, 7),
)

# Human opening book mined from completed games created 2025-10-14 or later,
# plus one explicitly coached infantry-screen plan. Only empirical formations
# seen at least ten times are admitted. Counts become sampling weights,
# tempered with sqrt so variety survives; the coached plan's value is a
# sampling weight rather than a claim about observed frequency.
OPENING_FIRST_TURNS: Tuple[Tuple[Tuple[str, ...], int], ...] = (
    (("rhd1", "rte1", "rpb1"), 76),
    (("rhe1", "rtd1", "rpa1"), 15),
    (("rhe1", "rtd1", "rpb1"), 10),
    (("ric1", "rid1", "rie1"), 25),
)

OPENING_CONTINUATIONS: Dict[str, Tuple[Tuple[Tuple[str, ...], int], ...]] = {
    "A": ((("e1e3↑", "d1d2↑", "rfc1"), 69),),
    "B": ((("d1f3↑", "e1e2↑", "rfd1"), 12),),
    "C": (
        (("e1e2↑", "g2g3", "rfe1"), 3),
        (("d1f3↑", "e1e2↑", "rfd1"), 2),
        (("e1e2↑", "d1d3↑", "rfd1"), 2),
    ),
    # Infantry Screen: build a broad ordinary-infantry front, then step the
    # center forward and put regular artillery behind it. Armored infantry
    # stays in reserve as a rapid answer to an enemy paradrop.
    "D": ((("d1d2", "e1e2", "rre1"), 25),),
}

OPENING_SIGNATURE_KEYS: Dict[str, Tuple[Tuple[Any, ...], Tuple[int, ...]]] = {
    "A": (
        ((1, 7, None), (2, 13, None), (2, 14, None), (2, 15, None),
         (4, 1, None), (5, 6, 0), (6, 4, 0), (7, 3, 0)),
        (5, 3, 2),
    ),
    "B": (
        ((1, 7, None), (2, 13, None), (2, 14, None), (2, 15, None),
         (4, 0, None), (5, 6, 0), (6, 3, 0), (7, 4, 0)),
        (5, 3, 2),
    ),
    "C": (
        ((1, 7, None), (2, 13, None), (2, 14, None), (2, 15, None),
         (4, 1, None), (5, 6, 0), (6, 3, 0), (7, 4, 0)),
        (5, 3, 2),
    ),
    "D": (
        ((1, 7, None), (2, 2, None), (2, 3, None), (2, 4, None),
         (2, 13, None), (2, 14, None), (2, 15, None), (5, 6, 0)),
        (2, 3, 1, 2, 1, 1),
    ),
}

OPENING_BOOK_ANCHORS: Dict[Tuple[bool, int], Tuple[Tuple[str, int], ...]] = {
    (engine.RED, 1): (
        ("h1", engine.HQ),
        ("g1", engine.ARTILLERY),
        ("f2", engine.INFANTRY),
        ("g2", engine.INFANTRY),
        ("h2", engine.INFANTRY),
    ),
    (engine.BLUE, 2): (
        ("a8", engine.HQ),
        ("b8", engine.ARTILLERY),
        ("a7", engine.INFANTRY),
        ("b7", engine.INFANTRY),
        ("c7", engine.INFANTRY),
    ),
}


def squares(mask: int) -> List[int]:
    return list(engine.scan_reversed(mask))


def chebyshev(a: int, b: int) -> int:
    return max(abs(engine.square_file(a) - engine.square_file(b)), abs(engine.square_rank(a) - engine.square_rank(b)))


def color_name(color: bool) -> str:
    return "blue" if color == engine.BLUE else "red"


def normalized_move_uci(move: engine.Move, color: bool) -> str:
    """Use the same lexical tie-break from either color's perspective."""
    if color == engine.RED:
        return move.uci()
    rotated = engine.Move(
        name=move.name,
        from_square=None if move.from_square is None else 63 - move.from_square,
        to_square=None if move.to_square is None else 63 - move.to_square,
        unit_type=move.unit_type,
        orientation=None if move.orientation is None else (move.orientation + 4) % 8,
        capture_preference=(
            None if move.capture_preference is None else 63 - move.capture_preference
        ),
        auto_capture_type=move.auto_capture_type,
    )
    return rotated.uci()


def normalized_turn_key(moves: Sequence[engine.Move], color: bool) -> Tuple[str, ...]:
    return tuple(normalized_move_uci(move, color) for move in moves)


def material_for(board: engine.BaseBoard, color: bool) -> float:
    total = 0.0
    for piece_type, value in PIECE_VALUES.items():
        total += value * engine.popcount(board.pieces_mask(piece_type, color))
        if piece_type != engine.HQ:
            total += value * board.get_reserve_count(piece_type, color)
    return total


def board_material_for(board: engine.BaseBoard, color: bool) -> float:
    return sum(
        value * engine.popcount(board.pieces_mask(piece_type, color))
        for piece_type, value in PIECE_VALUES.items()
    )


def surviving_non_hq_count(board: engine.BaseBoard) -> int:
    on_board = engine.popcount(board.occupied & ~board.hq)
    in_reserve = sum(
        board.get_reserve_count(piece_type, color)
        for color in (engine.RED, engine.BLUE)
        for piece_type in NON_HQ_TYPES
    )
    return on_board + in_reserve


def support_penalty(board: engine.BaseBoard, color: bool) -> float:
    friendly = squares(board.occupied_co[color] & ~board.hq)
    penalty = 0.0
    for square in friendly:
        piece_type = board.piece_type_at(square)
        if piece_type is None:
            continue
        # Airborne survival is modeled separately. Applying generic support
        # and overextension penalties as well would double-count its risk and
        # can perversely value leaving it trapped over actually capturing it.
        if piece_type == engine.AIRBORNE_INFANTRY:
            continue
        neighbours = [other for other in friendly if other != square]
        nearest = min((chebyshev(square, other) for other in neighbours), default=8)
        value = PIECE_VALUES[piece_type]
        if nearest > 1:
            penalty += 0.50 * value
        if nearest >= 3:
            penalty += 0.25 * value
    return penalty


def infantry_shape_score(board: engine.BaseBoard, color: bool) -> float:
    """Reward interlocking diagonals and penalize penetrable infantry files."""
    infantry = squares(
        board.occupied_co[color] & (board.infantry | board.armored_infantry)
    )
    score = 0.0
    for index, first in enumerate(infantry):
        first_file = engine.square_file(first)
        first_rank = engine.square_rank(first)
        for second in infantry[index + 1 :]:
            file_distance = abs(first_file - engine.square_file(second))
            rank_distance = abs(first_rank - engine.square_rank(second))
            if file_distance == 1 and rank_distance == 1:
                score += 0.50
            elif file_distance == 0 and rank_distance == 1:
                score -= 0.60

    for file_index in range(8):
        ranks = sorted(
            engine.square_rank(square)
            for square in infantry
            if engine.square_file(square) == file_index
        )
        run = 1
        for index in range(1, len(ranks)):
            if ranks[index] == ranks[index - 1] + 1:
                run += 1
                if run >= 3:
                    score -= 1.25
            else:
                run = 1
    return score


def infantry_isolation_penalty(board: engine.BaseBoard, color: bool) -> float:
    """Do not let a same-file chain count as its own connection to the army."""
    infantry = squares(
        board.occupied_co[color] & (board.infantry | board.armored_infantry)
    )
    friendly = squares(
        board.occupied_co[color] & ~board.hq & ~board.airborne_infantry
    )
    penalty = 0.0
    for square in infantry:
        file_index = engine.square_file(square)
        anchors = [
            other
            for other in friendly
            if other != square
            and (
                board.piece_type_at(other) not in (engine.INFANTRY, engine.ARMORED_INFANTRY)
                or engine.square_file(other) != file_index
            )
        ]
        nearest = min((chebyshev(square, other) for other in anchors), default=8)
        if nearest > 1:
            piece_type = board.piece_type_at(square)
            penalty += 0.18 * (nearest - 1) * PIECE_VALUES.get(piece_type, 0.0)
    return penalty


def airborne_survival_penalty(board: engine.BaseBoard, color: bool) -> float:
    """Penalize committed paratroopers that are deep, unsupported, or engaged.

    A staged paratrooper on its home rank is deliberately held as a threat and
    receives no penalty. Once deployed, its nominal material value is only
    real if it has a plausible extraction route. Engagement is especially
    dangerous because the opponent can often use the other two actions of the
    turn to clear a capture lane and take it.
    """
    penalty = 0.0
    home_rank = 0 if color == engine.RED else 7
    enemy_home_rank = 7 if color == engine.RED else 0
    enemy = not color
    enemy_infantry_reserve = (
        board.get_reserve_count(engine.INFANTRY, enemy)
        + board.get_reserve_count(engine.ARMORED_INFANTRY, enemy)
    )
    friendly_support = board.occupied_co[color] & ~board.hq
    enemy_infantry = board.occupied_co[not color] & (
        board.infantry | board.armored_infantry | board.airborne_infantry
    )
    for square in squares(board.airborne_infantry & board.occupied_co[color]):
        rank = engine.square_rank(square)
        if rank == home_rank:
            continue
        distance_home = abs(rank - home_rank)
        adjacent = engine.BB_ADJACENT_SQUARES[square]
        penalty += 0.20 * distance_home
        if rank == enemy_home_rank and enemy_infantry_reserve > 0:
            # A paradrop onto the opponent's deployment rank is not an
            # extraction plan. Fresh infantry can be introduced around it,
            # so the unit is treated as tactically trapped unless the search
            # has an immediate capture mission that justifies the commitment.
            penalty += 10.0
        if engine.BB_SQUARES[square] & board.get_bombarded_squares(not color):
            # A deployed para in a live bombardment is not merely awkward: it
            # is an automatic capture at the opponent's next turn boundary.
            penalty += 8.0
        if not adjacent & friendly_support:
            penalty += 1.25
        if adjacent & enemy_infantry:
            # Engagement is a major warning, but not itself a capture. Search
            # must still prove the remaining setup actions that take the unit.
            penalty += 2.0
    return penalty


def development_for(board: engine.BaseBoard, color: bool, turn_number: int) -> float:
    """Reward useful early deployment and building a broad connected rank."""
    phase = max(0.0, min(1.0, (16.0 - max(1, turn_number)) / 12.0))
    if phase <= 0.0:
        return 0.0
    deployed = engine.popcount(board.occupied_co[color] & ~board.hq)
    rank_counts = [
        engine.popcount(board.occupied_co[color] & engine.BB_RANKS[rank])
        for rank in range(8)
    ]
    broad_rank = max(rank_counts, default=0)
    home_rank = 0 if color == engine.RED else 7
    home_and_front = rank_counts[home_rank] + rank_counts[home_rank + (1 if color == engine.RED else -1)]
    return phase * (0.18 * deployed + 0.14 * broad_rank + 0.06 * home_and_front)


def overextension_penalty(board: engine.BaseBoard, color: bool) -> float:
    units = squares(board.occupied_co[color] & ~board.hq & ~board.airborne_infantry)
    if not units:
        return 0.0
    rank_power = [0.0] * 8
    for square in units:
        piece_type = board.piece_type_at(square)
        if piece_type is not None:
            rank_power[engine.square_rank(square)] += PIECE_VALUES[piece_type]
    # Ties must be resolved from the moving side's perspective. The old
    # physical-rank tie-break always selected the lowest board rank, which
    # made otherwise mirrored positions evaluate differently and materially
    # favored Red in self-play. Prefer the rearmost equally powerful rank.
    anchor = max(
        range(8),
        key=lambda rank: (
            rank_power[rank],
            -(rank if color == engine.RED else 7 - rank),
        ),
    )
    penalty = 0.0
    for square in units:
        advance = engine.square_rank(square) - anchor if color == engine.RED else anchor - engine.square_rank(square)
        if advance >= 2:
            piece_type = board.piece_type_at(square)
            if piece_type is not None:
                penalty += 0.35 * PIECE_VALUES[piece_type] * (advance - 1)
    return penalty


def relative_rank(square: int, color: bool) -> int:
    """Return rank 1-8 measured outward from ``color``'s home rank."""
    rank = engine.square_rank(square)
    return rank + 1 if color == engine.RED else 8 - rank


def early_frontier_rank(turn_number: int) -> int:
    """Data-backed maximum quiet frontier for the opening."""
    for last_turn, frontier in EARLY_FRONTIER_RANKS:
        if turn_number <= last_turn:
            return frontier
    return 8


def phase_extension_penalty(
    board: engine.BaseBoard, color: bool, turn_number: int
) -> float:
    """Penalize ordinary material beyond the opening's supported frontier."""
    if turn_number > EARLY_GAME_LAST_TURN:
        return 0.0
    limit = early_frontier_rank(turn_number)
    material = board.occupied_co[color] & ~board.hq & ~board.airborne_infantry
    penalty = 0.0
    for square in squares(material):
        excess = max(0, relative_rank(square, color) - limit)
        if excess:
            piece_type = board.piece_type_at(square)
            penalty += 0.90 * excess * PIECE_VALUES.get(piece_type, 0.0)
    return penalty


def structure_metrics(board: engine.BaseBoard, color: bool) -> Dict[str, float]:
    """Measure army dispersion without punishing a broad connected front."""
    units = squares(
        board.occupied_co[color] & ~board.hq & ~board.airborne_infantry
    )
    if not units:
        return {
            "frontier_rank": 1.0,
            "rank_span": 0.0,
            "mean_nearest_distance": 0.0,
            "isolated_units": 0.0,
            "components": 0.0,
            "largest_component_ratio": 1.0,
        }

    ranks = [relative_rank(square, color) for square in units]
    nearest = [
        min((chebyshev(square, other) for other in units if other != square), default=0)
        for square in units
    ]
    unseen = set(units)
    component_sizes: List[int] = []
    while unseen:
        stack = [unseen.pop()]
        size = 1
        while stack:
            square = stack.pop()
            connected = {
                other for other in unseen if chebyshev(square, other) <= 1
            }
            unseen -= connected
            stack.extend(connected)
            size += len(connected)
        component_sizes.append(size)

    return {
        "frontier_rank": float(max(ranks)),
        "rank_span": float(max(ranks) - min(ranks)),
        "mean_nearest_distance": sum(nearest) / len(nearest),
        "isolated_units": float(sum(distance > 1 for distance in nearest)),
        "components": float(len(component_sizes)),
        "largest_component_ratio": max(component_sizes) / len(units),
    }


def dispersion_penalty(board: engine.BaseBoard, color: bool) -> float:
    """Penalize detached groups and unsupported pieces, not horizontal width."""
    metrics = structure_metrics(board, color)
    return (
        0.65 * metrics["isolated_units"]
        + 0.55 * max(0.0, metrics["components"] - 1.0)
        + 1.20 * (1.0 - metrics["largest_component_ratio"])
        + 0.30 * max(0.0, metrics["mean_nearest_distance"] - 1.0)
    )


def optionality_metrics(board: engine.BaseBoard, color: bool) -> Dict[str, float]:
    """Estimate how much deployed material can actually relocate.

    This deliberately ignores in-place artillery rotations: a boxed-in gun
    may still rotate, but it does not create the positional optionality the
    user is describing. Paratroopers are evaluated separately as a held threat.
    """
    material = squares(
        board.occupied_co[color] & ~board.hq & ~board.airborne_infantry
    )
    home_rank = 0 if color == engine.RED else 7
    total_options = 0
    immobile = 0
    home_immobile = 0
    for square in material:
        piece_type = board.piece_type_at(square)
        speed = 2 if piece_type in (
            engine.ARMORED_INFANTRY,
            engine.ARMORED_ARTILLERY,
        ) else 1
        file_index = engine.square_file(square)
        rank_index = engine.square_rank(square)
        options = 0
        for rank in range(max(0, rank_index - speed), min(7, rank_index + speed) + 1):
            for file in range(max(0, file_index - speed), min(7, file_index + speed) + 1):
                if rank == rank_index and file == file_index:
                    continue
                destination = engine.square(file, rank)
                if chebyshev(square, destination) <= speed and not (
                    engine.BB_SQUARES[destination] & board.occupied
                ):
                    options += 1
        total_options += options
        if options == 0:
            immobile += 1
            if rank_index == home_rank:
                home_immobile += 1

    home_occupancy = engine.popcount(
        board.occupied_co[color] & engine.BB_RANKS[home_rank]
    )
    mean_options = total_options / len(material) if material else 0.0
    return {
        "relocation_options": float(total_options),
        "mean_relocation_options": mean_options,
        "immobile_units": float(immobile),
        "home_rank_immobile_units": float(home_immobile),
        "home_rank_occupancy": float(home_occupancy),
    }


def congestion_penalty(board: engine.BaseBoard, color: bool) -> float:
    metrics = optionality_metrics(board, color)
    excess_home = max(0.0, metrics["home_rank_occupancy"] - 4.0)
    return (
        0.90 * metrics["immobile_units"]
        + 0.45 * metrics["home_rank_immobile_units"]
        + 0.16 * excess_home * excess_home
        + 0.15 * max(0.0, 1.5 - metrics["mean_relocation_options"])
    )


def optionality_score(board: engine.BaseBoard, color: bool) -> float:
    metrics = optionality_metrics(board, color)
    return math.log1p(metrics["relocation_options"]) - congestion_penalty(
        board, color
    )


def artillery_formation(board: engine.BaseBoard, color: bool) -> float:
    artillery_mask = board.occupied_co[color] & (board.artillery | board.armored_artillery | board.heavy_artillery)
    guns = squares(artillery_mask)
    score = 0.0
    for index, a in enumerate(guns):
        for b in guns[index + 1 :]:
            if chebyshev(a, b) == 1:
                score += 0.35
    # Reward the user's preferred battery: heavy artillery between two guns.
    for heavy in squares(artillery_mask & board.heavy_artillery):
        hf, hr = engine.square_file(heavy), engine.square_rank(heavy)
        same_rank = [g for g in guns if g != heavy and engine.square_rank(g) == hr]
        same_file = [g for g in guns if g != heavy and engine.square_file(g) == hf]
        if any(engine.square_file(g) < hf for g in same_rank) and any(engine.square_file(g) > hf for g in same_rank):
            score += 1.25
        if any(engine.square_rank(g) < hr for g in same_file) and any(engine.square_rank(g) > hr for g in same_file):
            score += 1.25
    return score


def artillery_pressure(board: engine.BaseBoard, color: bool) -> float:
    controlled = board.get_bombarded_squares(color)
    enemy_targets = controlled & board.occupied_co[not color]
    target_value = 0.0
    for square in squares(enemy_targets):
        piece_type = board.piece_type_at(square)
        if piece_type is not None:
            target_value += 0.22 * PIECE_VALUES[piece_type]
    return target_value + 0.025 * engine.popcount(controlled)


def artillery_exposure_penalty(board: engine.BaseBoard, color: bool) -> float:
    """Value diagonal infantry cover against a staged enemy paratrooper."""
    enemy = not color
    enemy_home = engine.BB_RANK_1 if enemy == engine.RED else engine.BB_RANK_8
    staged = board.airborne_infantry & board.occupied_co[enemy] & enemy_home
    reserve_ready = board.get_reserve_count(engine.AIRBORNE_INFANTRY, enemy) > 0
    readiness = 1.0 if staged else 0.45 if reserve_ready else 0.0
    if readiness == 0.0:
        return 0.0
    guns = board.occupied_co[color] & (
        board.artillery | board.armored_artillery | board.heavy_artillery
    )
    friendly_infantry = board.occupied_co[color] & (
        board.infantry | board.armored_infantry | board.airborne_infantry
    )
    penalty = 0.0
    for gun in squares(guns):
        gf, gr = engine.square_file(gun), engine.square_rank(gun)
        diagonal_cover = 0
        cardinal_cover = 0
        for infantry in squares(engine.BB_REGULAR_MOVES[gun] & friendly_infantry):
            diagonal = (
                engine.square_file(infantry) != gf
                and engine.square_rank(infantry) != gr
            )
            diagonal_cover += 1 if diagonal else 0
            cardinal_cover += 0 if diagonal else 1
        coverage = 2.0 * diagonal_cover + cardinal_cover
        uncovered_fraction = max(0.0, 1.0 - coverage / 3.0)
        piece_type = board.piece_type_at(gun)
        penalty += readiness * uncovered_fraction * 0.45 * PIECE_VALUES.get(piece_type, 0.0)
    return penalty


def action_mobility(board: engine.BaseBoard, color: bool) -> float:
    probe = board.copy()
    probe.turn = color
    probe.turn_moves = 0
    probe.turn_auto_moves = 0
    probe.turn_pieces = engine.BB_EMPTY
    probe.did_offer_draw = False
    probe.did_accept_draw = False
    probe._clear_free_captures()
    list(probe._generate_free_captures(color))
    return math.log1p(sum(1 for move in probe.generate_legal_moves() if move.name != "Skip"))


def hq_safety(board: engine.BaseBoard, color: bool) -> float:
    hq = board.hq & board.occupied_co[color]
    if not hq:
        return -MATE_SCORE
    score = 0.0
    if hq & board.get_bombarded_squares(not color):
        score -= 40.0
    hq_square = squares(hq)[0]
    enemy_infantry = board.occupied_co[not color] & (board.infantry | board.armored_infantry | board.airborne_infantry)
    score -= 1.5 * engine.popcount(engine.BB_ADJACENT_SQUARES[hq_square] & enemy_infantry)
    return score


def evaluation_breakdown(
    board: engine.BaseBoard,
    personality: str = "balanced",
    turn_number: int = 1,
) -> Dict[str, Any]:
    if personality not in PERSONALITIES:
        raise ValueError(f"unknown personality: {personality}")

    surviving_units = surviving_non_hq_count(board)
    raw = {
        "material": material_for(board, engine.RED) - material_for(board, engine.BLUE),
        "support": support_penalty(board, engine.BLUE) - support_penalty(board, engine.RED),
        "overextension": overextension_penalty(board, engine.BLUE) - overextension_penalty(board, engine.RED),
        "phase_extension": phase_extension_penalty(board, engine.BLUE, turn_number)
        - phase_extension_penalty(board, engine.RED, turn_number),
        "cohesion": dispersion_penalty(board, engine.BLUE)
        - dispersion_penalty(board, engine.RED),
        "optionality": optionality_score(board, engine.RED)
        - optionality_score(board, engine.BLUE),
        "artillery_formation": artillery_formation(board, engine.RED) - artillery_formation(board, engine.BLUE),
        "artillery_pressure": artillery_pressure(board, engine.RED) - artillery_pressure(board, engine.BLUE),
        "artillery_protection": artillery_exposure_penalty(board, engine.BLUE)
        - artillery_exposure_penalty(board, engine.RED),
        "infantry_shape": infantry_shape_score(board, engine.RED)
        - infantry_shape_score(board, engine.BLUE),
        "infantry_isolation": infantry_isolation_penalty(board, engine.BLUE)
        - infantry_isolation_penalty(board, engine.RED),
        "mobility": 0.30 * (action_mobility(board, engine.RED) - action_mobility(board, engine.BLUE)),
        "open_board_armored_infantry": 0.0,
        "airborne_survival": airborne_survival_penalty(board, engine.BLUE)
        - airborne_survival_penalty(board, engine.RED),
        "hq_safety": hq_safety(board, engine.RED) - hq_safety(board, engine.BLUE),
        "development": development_for(board, engine.RED, turn_number)
        - development_for(board, engine.BLUE, turn_number),
    }
    if surviving_units <= 18:
        raw["open_board_armored_infantry"] = 0.65 * (
            engine.popcount(board.armored_infantry & board.occupied_co[engine.RED])
            - engine.popcount(board.armored_infantry & board.occupied_co[engine.BLUE])
        )

    profile = PERSONALITIES[personality]
    weights = {name: BASE_WEIGHTS[name] * profile.get(name, 1.0) for name in raw}
    weighted = {name: raw[name] * weights[name] for name in raw}
    return {
        "perspective": "red-positive",
        "personality": personality,
        "components": {key: round(value, 4) for key, value in raw.items()},
        "weights": weights,
        "weighted_components": {key: round(value, 4) for key, value in weighted.items()},
        "total_red": round(sum(weighted.values()), 4),
    }


def quick_evaluation(board: engine.BaseBoard, turn_number: int) -> float:
    """Cheap red-positive score that never enumerates legal actions."""
    return (
        material_for(board, engine.RED)
        - material_for(board, engine.BLUE)
        + development_for(board, engine.RED, turn_number)
        - development_for(board, engine.BLUE, turn_number)
        + artillery_formation(board, engine.RED)
        - artillery_formation(board, engine.BLUE)
        + artillery_exposure_penalty(board, engine.BLUE)
        - artillery_exposure_penalty(board, engine.RED)
        + infantry_shape_score(board, engine.RED)
        - infantry_shape_score(board, engine.BLUE)
        + infantry_isolation_penalty(board, engine.BLUE)
        - infantry_isolation_penalty(board, engine.RED)
        + support_penalty(board, engine.BLUE)
        - support_penalty(board, engine.RED)
        + overextension_penalty(board, engine.BLUE)
        - overextension_penalty(board, engine.RED)
        + phase_extension_penalty(board, engine.BLUE, turn_number)
        - phase_extension_penalty(board, engine.RED, turn_number)
        + dispersion_penalty(board, engine.BLUE)
        - dispersion_penalty(board, engine.RED)
        + optionality_score(board, engine.RED)
        - optionality_score(board, engine.BLUE)
        + airborne_survival_penalty(board, engine.BLUE)
        - airborne_survival_penalty(board, engine.RED)
        + hq_safety(board, engine.RED)
        - hq_safety(board, engine.BLUE)
    )


class SearchTimeout(Exception):
    pass


@dataclass
class SearchResult:
    score: float
    pv: List[engine.Move]


@dataclass
class TurnCandidate:
    moves: List[engine.Move]
    board: engine.BaseBoard
    priority: float
    static_score: float = 0.0
    safety_penalty: float = 0.0
    tactically_safe: bool = True
    purpose_penalty: float = 0.0
    paratrooper_mission_penalty: float = 0.0
    action_purposes: List[Dict[str, Any]] = field(default_factory=list)
    early_plan_score: float = 0.0
    progress_score: float = 0.0
    conveyor_actions: float = 0.0
    skip_actions: float = 0.0


@dataclass
class PartialTurn:
    moves: List[engine.Move]
    board: engine.BaseBoard
    priority: float


@dataclass
class TacticalSafety:
    risk_value: float
    new_risk_value: float
    compensation_value: float
    forced_loss_value: float
    para_or_artillery_loss_value: float
    tactically_safe: bool


class Searcher:
    def __init__(
        self,
        personality: str,
        time_ms: int,
        beam_width: int,
        turn_number: int = 1,
        value_function: Optional[Any] = None,
        max_actions: int = 3,
        stagnation_turns: int = 0,
    ) -> None:
        self.personality = personality
        self.time_ms = max(1, time_ms)
        self.turn_number = max(1, turn_number)
        self.value_function = value_function
        self.max_actions = max(2, min(3, max_actions))
        self.stagnation_turns = max(0, stagnation_turns)
        self.deadline = time.monotonic() + max(1, time_ms) / 1000.0
        self.beam_width = max(1, beam_width)
        self.nodes = 0
        self.table: Dict[Tuple[str, int], SearchResult] = {}
        self.transposition_hits = 0
        self.turn_cache_hits = 0
        self.exhaustive_within_horizon = True
        self.rule_filtered_actions = 0
        self.beam_pruned_actions = 0
        self.partial_turns_pruned = 0
        self.complete_turns_generated = 0
        self.complete_turns_deduplicated = 0
        self.complete_turns_pruned = 0
        self.purposeful_early_stops_generated = 0
        self.tactically_unsafe_turns = 0
        self.rotation_quota_pruned = 0
        self.purpose_filtered_turns = 0
        self.value_model_evaluations = 0
        self.turn_cache: Dict[str, List[TurnCandidate]] = {}
        self.value_cache: Dict[str, float] = {}
        self.safety_cache: Dict[Tuple[str, bool], Tuple[float, float, float]] = {}
        self.immediate_hq_capture_cache: Dict[str, bool] = {}
        self.same_turn_hq_capture_cache: Dict[str, bool] = {}
        self.hq_defense_move_cache: Dict[Tuple[str, str], bool] = {}
        self.capture_setup_move_cache: Dict[Tuple[str, str], bool] = {}
        self.root_key: Optional[str] = None
        self.root_fallback: Optional[TurnCandidate] = None
        self.root_ranked_turns: List[Tuple[float, TurnCandidate]] = []
        # Each entry has completed the opponent reply at the requested root
        # horizon. Keep it incrementally: a later timeout must not erase root
        # lines that were already tactically verified.
        self.root_verified_lines: List[
            Tuple[float, TurnCandidate, List[engine.Move]]
        ] = []
        # Search begins with a deliberately narrow depth-two pass. Only after
        # every retained root has a complete opponent reply may the broader
        # user-requested beam consume the remaining budget.
        self.verification_mode = False
        # Enabled by the production depth-two path. Depth-one analysis keeps
        # its historical cost envelope instead of silently becoming a
        # selective depth-two search.
        self.hq_leaf_extension_enabled = False

    def stagnation_factor(self) -> float:
        """Escalate only after several consecutive non-progress turns."""
        return max(0.0, min(1.0, (self.stagnation_turns - 4) / 20.0))

    def stagnation_value(self, candidate: TurnCandidate) -> float:
        # One backfill is often how a cohesive formation advances. Forgive it
        # only when the complete turn demonstrably closes on contact/HQ; pure
        # conveyor shuffles still pay the full penalty.
        useful_conveyor = 1.0 if candidate.progress_score >= 3.0 else 0.0
        return (
            candidate.progress_score
            - 2.5 * max(0.0, candidate.conveyor_actions - useful_conveyor)
            - 5.0 * candidate.skip_actions
        )

    def check_time(self, count_node: bool = True) -> None:
        if count_node:
            self.nodes += 1
        if time.monotonic() >= self.deadline:
            raise SearchTimeout

    def best_verified_root_result(self, mover: bool) -> Optional[SearchResult]:
        if not self.root_verified_lines:
            return None
        score, turn, reply = min(
            self.root_verified_lines,
            key=lambda item: (
                -item[0] if mover == engine.RED else item[0],
                normalized_turn_key(item[1].moves, mover),
            ),
        )
        return SearchResult(score, list(turn.moves) + list(reply))

    def terminal_score(self, board: engine.BaseBoard, turns_left: int) -> Optional[float]:
        outcome = board.outcome()
        if outcome is None:
            return None
        if outcome.winner is None:
            return 0.0
        tempo = max(0, turns_left)
        return (MATE_SCORE + tempo) if outcome.winner == engine.RED else -(MATE_SCORE + tempo)

    def heuristic_score(self, board: engine.BaseBoard) -> float:
        return float(
            evaluation_breakdown(board, self.personality, self.turn_number)["total_red"]
        )

    def quick_score(self, board: engine.BaseBoard) -> float:
        """Cheap deadline score with no legal-move enumeration."""
        return quick_evaluation(board, self.turn_number)

    @staticmethod
    def infantry_pressure(board: engine.BaseBoard, color: bool) -> float:
        """Value distinct enemy pieces currently engaged by friendly infantry."""
        attackers = board.occupied_co[color] & (
            board.infantry | board.armored_infantry | board.airborne_infantry
        )
        targets = engine.BB_EMPTY
        for square in squares(attackers):
            targets |= engine.BB_ADJACENT_SQUARES[square] & board.occupied_co[not color]
        return 0.22 * Searcher.mask_value(board, targets)

    def positional_risk(self, board: engine.BaseBoard, color: bool) -> float:
        """Cheap, non-search risk used to decide whether a turn accomplished work."""
        return (
            support_penalty(board, color)
            + overextension_penalty(board, color)
            + phase_extension_penalty(board, color, self.turn_number)
            + dispersion_penalty(board, color)
            + congestion_penalty(board, color)
            + artillery_exposure_penalty(board, color)
            + airborne_survival_penalty(board, color)
            - hq_safety(board, color)
        )

    def is_paradrop(self, board: engine.BaseBoard, move: engine.Move) -> bool:
        """Whether a staged para is leaving its own deployment rank."""
        return (
            self.move_piece_type(board, move) == engine.AIRBORNE_INFANTRY
            and move.name != "Reinforce"
            and move.from_square is not None
            and move.to_square is not None
            and self.home_distance(move.from_square, board.turn) == 0
            and self.home_distance(move.to_square, board.turn) > 0
        )

    def paradrop_capture_targets(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> Dict[int, float]:
        """Return pieces the para could legally capture from its landing square."""
        if not self.is_paradrop(board, move) or move.to_square is None:
            return {}
        mover = board.turn
        child = board.copy()
        child.push(move)
        probe = self.board_as_turn(child, mover)
        # We are inspecting the para's voluntary move options, not unrelated
        # start-of-turn bombardments or automatic captures elsewhere.
        probe.turn_moves = 1
        source_mask = engine.BB_SQUARES[move.to_square]
        targets: Dict[int, float] = {}
        for candidate in probe.generate_legal_moves(from_mask=source_mask):
            if (
                candidate.from_square == move.to_square
                and candidate.capture_preference is not None
            ):
                target = candidate.capture_preference
                targets[target] = PIECE_VALUES.get(probe.piece_type_at(target), 0.0)
        return targets

    def paradrop_allowed(self, board: engine.BaseBoard, move: engine.Move) -> bool:
        """A paradrop is legal for search only when it arrives with a capture."""
        if not self.is_paradrop(board, move):
            return True
        if move.capture_preference is not None:
            return True
        return bool(self.paradrop_capture_targets(board, move))

    def early_extension_allowed(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Reject quiet opening moves beyond the data-backed frontier."""
        if self.turn_number > EARLY_GAME_LAST_TURN:
            return True
        piece_type = self.move_piece_type(board, move)
        if (
            piece_type in (None, engine.HQ, engine.AIRBORNE_INFANTRY)
            or move.name == "Reinforce"
            or move.capture_preference is not None
            or move.from_square is None
            or move.to_square is None
        ):
            return True
        if relative_rank(move.to_square, board.turn) <= early_frontier_rank(
            self.turn_number
        ):
            return True

        # A gun or infantry that is already under automatic bombardment may
        # cross the frontier to survive. This is an escape, not speculative
        # development.
        source = engine.BB_SQUARES[move.from_square]
        destination = engine.BB_SQUARES[move.to_square]
        return bool(
            source & board.get_bombarded_squares(not board.turn)
            and not destination & board.get_bombarded_squares(not board.turn)
        )

    @staticmethod
    def forward_infantry_actions(
        before: engine.BaseBoard,
        moves: Sequence[engine.Move],
        mover: bool,
    ) -> int:
        """Count non-capturing infantry actions that increase relative rank."""
        working = before.copy()
        count = 0
        for move in moves:
            piece_type = Searcher.move_piece_type(working, move)
            if (
                piece_type in (engine.INFANTRY, engine.ARMORED_INFANTRY)
                and move.name != "Reinforce"
                and move.capture_preference is None
                and move.from_square is not None
                and move.to_square is not None
                and relative_rank(move.to_square, mover)
                > relative_rank(move.from_square, mover)
            ):
                count += 1
            working.push(move)
        return count

    def early_structure_allowed(
        self,
        before: engine.BaseBoard,
        after: engine.BaseBoard,
        moves: Sequence[engine.Move],
        mover: bool,
        purposes: Sequence[Dict[str, Any]],
    ) -> bool:
        """Keep early turns cohesive and reject purposeless mass advances."""
        if self.turn_number > EARLY_GAME_LAST_TURN:
            return True
        forward_actions = self.forward_infantry_actions(before, moves, mover)
        forcing_roles = {"capture", "save", "threat", "protect"}
        forcing_actions = sum(
            bool(forcing_roles.intersection(item["roles"])) for item in purposes
        )
        captures = any(
            move.capture_preference is not None or move.name == "AutoCapture"
            for move in moves
        )

        # Spending the whole turn pushing three infantry is not development.
        # It is allowed only when at least two actions have concrete tactical
        # jobs, rather than merely acquiring a "form" label after the fact.
        if forward_actions >= 3 and forcing_actions < 2:
            return False

        dispersion_gain = dispersion_penalty(after, mover) - dispersion_penalty(before, mover)
        optionality_gain = optionality_score(after, mover) - optionality_score(
            before, mover
        )
        has_save = any("save" in item["roles"] for item in purposes)
        # Cohesion is a band, not a mandate to pack the home rank. A modest
        # increase in spacing is welcome when it unlocks real relocation
        # options; splitting without gaining optionality is still rejected.
        if dispersion_gain > 1.50 and not captures and not has_save:
            return False
        if (
            dispersion_gain > 0.50
            and optionality_gain <= 0.10
            and not captures
            and not has_save
        ):
            return False
        # Even a two-piece push may not split the army unless the turn captures
        # material or saves something already in danger.
        if forward_actions >= 2 and dispersion_gain > 0.20 and not captures:
            if not has_save:
                return False
        return True

    def paratrooper_mission_penalty(
        self,
        before: engine.BaseBoard,
        after: engine.BaseBoard,
        moves: Sequence[engine.Move],
        mover: bool,
    ) -> float:
        """Flag any paradrop that did not arrive with a legal capture mission."""
        working = before.copy()
        penalty = 0.0
        for move in moves:
            if self.is_paradrop(working, move) and not self.paradrop_allowed(working, move):
                penalty += MISSIONLESS_PARATROOPER_PENALTY
            working.push(move)
        return penalty

    @staticmethod
    def formation_quality(board: engine.BaseBoard, color: bool) -> float:
        """Broad formation quality used for explainable early-game plans."""
        return (
            artillery_formation(board, color)
            + infantry_shape_score(board, color)
            - 0.35 * support_penalty(board, color)
            - 0.35 * infantry_isolation_penalty(board, color)
            - 0.75 * dispersion_penalty(board, color)
            + 0.45 * optionality_score(board, color)
            - artillery_exposure_penalty(board, color)
        )

    def action_purpose_labels(
        self,
        before: engine.BaseBoard,
        moves: Sequence[engine.Move],
        mover: bool,
        retrospective: bool = True,
        include_tactical_roles: bool = True,
    ) -> List[Dict[str, Any]]:
        """Explain the concrete job performed by every action in a turn."""
        working = before.copy()
        result: List[Dict[str, Any]] = []
        for move in moves:
            piece_type = self.move_piece_type(working, move)
            roles: List[str] = []
            if move.capture_preference is not None or move.name == "AutoCapture":
                roles.append("capture")
            if move.name == "Reinforce":
                roles.append("develop")
            if self.unlocks_airborne_extraction(working, move):
                roles.append("unlock")
            if self.unlocks_capture_this_turn(working, move):
                roles.append("capture_setup")
            if include_tactical_roles and self.unlocks_hq_escape(working, move):
                roles.append("hq_escape_unlock")
            if include_tactical_roles and self.resolves_hq_threat(working, move):
                roles.append("hq_defense")
            if include_tactical_roles and self.unlocks_immediate_hq_capture(
                working, move
            ):
                roles.append("hq_capture_unlock")
            if move.from_square is not None and move.to_square is not None:
                if (
                    piece_type in ARTILLERY_TYPES
                    and engine.BB_SQUARES[move.from_square]
                    & working.get_bombarded_squares(not mover)
                    and not engine.BB_SQUARES[move.to_square]
                    & working.get_bombarded_squares(not mover)
                ):
                    roles.append("save")
                if (
                    piece_type == engine.AIRBORNE_INFANTRY
                    and self.home_distance(move.to_square, mover)
                    < self.home_distance(move.from_square, mover)
                ):
                    roles.append("extract")

            child = working.copy()
            child.push(move)
            pressure_gain = (
                artillery_pressure(child, mover)
                + self.infantry_pressure(child, mover)
                - artillery_pressure(working, mover)
                - self.infantry_pressure(working, mover)
            )
            protection_gain = self.positional_risk(working, mover) - self.positional_risk(child, mover)
            formation_gain = self.formation_quality(child, mover) - self.formation_quality(working, mover)
            optionality_gain = optionality_score(child, mover) - optionality_score(
                working, mover
            )
            contact_gain = self.approach_distance(
                working, mover
            ) - self.approach_distance(child, mover)
            hq_approach_gain = self.approach_distance(
                working, mover, hq_only=True
            ) - self.approach_distance(child, mover, hq_only=True)
            if pressure_gain > 0.05:
                roles.append("threat")
            if protection_gain > 0.25:
                roles.append("protect")
            if formation_gain > 0.10:
                roles.append("form")
            if optionality_gain > 0.15:
                roles.append("mobilize")
            if (
                self.turn_number > EARLY_GAME_LAST_TURN
                and max(contact_gain, hq_approach_gain) > 0.0
            ):
                roles.append("advance")
            if move.name == "Skip":
                roles.append("end_turn")
            if not roles:
                roles.append("no_new_effect")
            result.append({"move": move.uci(), "roles": list(dict.fromkeys(roles))})
            working = child

        if retrospective:
            # Purpose can be relational.  A quiet first action may appear to
            # do nothing in isolation while creating the geometry that makes
            # a later protection, threat, or formation action work.  Replay
            # the turn without each apparent filler and credit such moves as
            # setup instead of letting the no-effect filter discard them.
            meaningful_roles = {
                "capture",
                "save",
                "develop",
                "unlock",
                "capture_setup",
                "extract",
                "threat",
                "protect",
                "form",
                "mobilize",
                "advance",
            }
            for omitted, item in enumerate(result):
                if item["roles"] != ["no_new_effect"]:
                    continue
                counter = before.copy()
                counter_moves: List[engine.Move] = []
                counter_index: Dict[int, int] = {}
                replayable = True
                for original_index, original_move in enumerate(moves):
                    if original_index == omitted:
                        continue
                    replay_move = next(
                        (
                            legal
                            for legal in counter.generate_legal_moves()
                            if legal.uci() == original_move.uci()
                        ),
                        None,
                    )
                    if replay_move is None:
                        replayable = False
                        break
                    counter_index[original_index] = len(counter_moves)
                    counter_moves.append(replay_move)
                    counter.push(replay_move)
                if not replayable:
                    # Keep this separate from the concrete airborne/HQ
                    # ``unlock`` roles detected above.  It is retrospective,
                    # dependent purpose and should not receive forcing-line
                    # priority merely because a later quiet move becomes
                    # illegal without it.
                    item["roles"] = ["setup"]
                    continue
                counter_labels = self.action_purpose_labels(
                    before,
                    counter_moves,
                    mover,
                    retrospective=False,
                    include_tactical_roles=False,
                )
                for original_index in range(omitted + 1, len(moves)):
                    replay_index = counter_index.get(original_index)
                    if replay_index is None:
                        continue
                    original_roles = meaningful_roles.intersection(
                        result[original_index]["roles"]
                    )
                    replay_roles = meaningful_roles.intersection(
                        counter_labels[replay_index]["roles"]
                    )
                    if original_roles - replay_roles:
                        item["roles"] = ["setup"]
                        break
        return result

    def early_plan_score(self, action_purposes: Sequence[Dict[str, Any]]) -> float:
        """Prefer development and formation while the board is still building."""
        role_values = {
            "capture": 4.0,
            "save": 3.0,
            "develop": 2.5 if self.turn_number <= EARLY_GAME_LAST_TURN else 1.0,
            "form": 2.0 if self.turn_number <= EARLY_GAME_LAST_TURN else 0.8,
            "threat": 2.0,
            "protect": 1.5,
            "extract": 2.5,
            "unlock": 1.5,
            "capture_setup": 2.5,
            "hq_escape_unlock": 5.0,
            "hq_defense": 5.0,
            "hq_capture_unlock": 5.0,
            "mobilize": 1.8,
            "advance": 1.5,
        }
        return sum(
            max((role_values.get(role, 0.0) for role in item["roles"]), default=0.0)
            for item in action_purposes
        )

    def turn_purpose_breakdown(
        self,
        before: engine.BaseBoard,
        after: engine.BaseBoard,
        moves: Sequence[engine.Move],
        mover: bool,
        retrospective: bool = True,
    ) -> Dict[str, float]:
        """Measure what a full turn changed, beyond merely spending actions."""
        action_purposes = self.action_purpose_labels(
            before, moves, mover, retrospective=retrospective
        )
        counted_actions = sum(
            move.name not in ("AutoCapture", "Skip") for move in moves
        )
        unused_actions = (
            0 if after.is_game_over() else max(0, self.max_actions - counted_actions)
        )
        purposeful_actions = sum(
            1
            for item in action_purposes
            if any(role not in ("no_new_effect", "end_turn") for role in item["roles"])
        )
        unpurposed_actions = sum(
            1 for item in action_purposes if item["roles"] == ["no_new_effect"]
        )
        setup_actions = sum(
            1 for item in action_purposes if item["roles"] == ["setup"]
        )
        development_actions = sum(
            1 for item in action_purposes if "develop" in item["roles"]
        )
        formation_actions = sum(
            1 for item in action_purposes if "form" in item["roles"]
        )
        working = before.copy()
        vacated: set[int] = set()
        quiet_actions = 0
        backfills = 0
        reversals = 0
        pure_rotations = 0
        extractions = 0
        for move in moves:
            piece_type = self.move_piece_type(working, move)
            is_quiet = move.capture_preference is None and move.name not in ("AutoCapture", "Skip")
            if is_quiet:
                quiet_actions += 1
            if move.to_square is not None and move.to_square in vacated and is_quiet:
                backfills += 1
            if move.from_square is not None and move.to_square is not None:
                if move.from_square == move.to_square:
                    pure_rotations += 1
                else:
                    if (
                        piece_type == engine.AIRBORNE_INFANTRY
                        and self.home_distance(move.to_square, mover)
                        < self.home_distance(move.from_square, mover)
                    ):
                        extractions += 1
                    elif (
                        piece_type != engine.AIRBORNE_INFANTRY
                        and self.home_distance(move.to_square, mover)
                        < self.home_distance(move.from_square, mover)
                        and move.capture_preference is None
                    ):
                        reversals += 1
                    vacated.add(move.from_square)
            working.push(move)

        opponent = not mover
        capture_gain = max(
            0.0,
            board_material_for(before, opponent) - board_material_for(after, opponent),
        )
        deployment_gain = max(
            0.0,
            board_material_for(after, mover) - board_material_for(before, mover),
        )
        pressure_before = artillery_pressure(before, mover) + self.infantry_pressure(before, mover)
        pressure_after = artillery_pressure(after, mover) + self.infantry_pressure(after, mover)
        threat_gain = max(0.0, pressure_after - pressure_before)
        protection_gain = max(
            0.0, self.positional_risk(before, mover) - self.positional_risk(after, mover)
        )
        development_gain = max(
            0.0,
            development_for(after, mover, self.turn_number)
            - development_for(before, mover, self.turn_number),
        )
        shape_gain = max(
            0.0,
            infantry_shape_score(after, mover) - infantry_shape_score(before, mover),
        )
        formation_gain = max(
            0.0,
            self.formation_quality(after, mover)
            - self.formation_quality(before, mover),
        )
        dispersion_gain = max(
            0.0,
            dispersion_penalty(after, mover) - dispersion_penalty(before, mover),
        )
        optionality_gain = max(
            0.0,
            optionality_score(after, mover) - optionality_score(before, mover),
        )
        contact_gain = max(
            0.0,
            self.approach_distance(before, mover)
            - self.approach_distance(after, mover),
        )
        hq_approach_gain = max(
            0.0,
            self.approach_distance(before, mover, hq_only=True)
            - self.approach_distance(after, mover, hq_only=True),
        )
        frontier_gain = max(
            0.0,
            structure_metrics(after, mover)["frontier_rank"]
            - structure_metrics(before, mover)["frontier_rank"],
        )
        congestion_increase = max(
            0.0,
            congestion_penalty(after, mover) - congestion_penalty(before, mover),
        )
        uncompensated_dispersion = max(
            0.0, dispersion_gain - 0.75 * optionality_gain
        )
        extension_gain = max(
            0.0,
            phase_extension_penalty(after, mover, self.turn_number)
            - phase_extension_penalty(before, mover, self.turn_number),
        )
        forward_infantry_actions = self.forward_infantry_actions(
            before, moves, mover
        )
        coordinated_overpush = max(0, forward_infantry_actions - 1)
        forcing_gain = (
            2.0 * capture_gain
            + 0.65 * deployment_gain
            + 1.5 * threat_gain
            + 1.2 * protection_gain
            + 0.5 * development_gain
            + 0.5 * shape_gain
            + 0.6 * formation_gain
            + 0.75 * optionality_gain
            + 0.8 * contact_gain
            + 0.6 * hq_approach_gain
        )
        # Formation and optionality can oscillate forever. Only material,
        # tactical pressure, safety, and objective-closing movement count as
        # durable progress for the late-game anti-stagnation policy.
        stagnation_progress = (
            2.0 * capture_gain
            + 1.5 * threat_gain
            + 1.0 * protection_gain
            + 2.0 * contact_gain
            + 3.0 * hq_approach_gain
            + 1.5 * frontier_gain
        )
        waste = (
            1.4 * unpurposed_actions
            # Retrospective setup is useful evidence for explanation and
            # training, but it is still dependent on a later action.  Keep
            # its search cost equal to the former no-effect label until a
            # tactical arena proves that relaxing it cannot crowd out mates.
            + 1.4 * setup_actions
            + 1.6 * backfills
            + 1.0 * reversals
            + 0.9 * pure_rotations
            + 1.35 * coordinated_overpush
            + 2.0 * uncompensated_dispersion
            + 1.5 * congestion_increase
            + 2.5 * extension_gain
            + 0.9 * unused_actions
        )
        # Benefits offset wasted motion continuously. This matters for turns
        # containing one mildly useful action and two swaps/reversals: the
        # useful action does not absolve the rest of the turn.
        net_purpose_penalty = max(0.0, waste - 0.70 * forcing_gain)
        mission_penalty = self.paratrooper_mission_penalty(before, after, moves, mover)
        return {
            "capture_gain": capture_gain,
            "deployment_gain": deployment_gain,
            "threat_gain": threat_gain,
            "protection_gain": protection_gain,
            "development_gain": development_gain,
            "formation_gain": formation_gain,
            "dispersion_increase": dispersion_gain,
            "uncompensated_dispersion": uncompensated_dispersion,
            "optionality_gain": optionality_gain,
            "contact_gain": contact_gain,
            "hq_approach_gain": hq_approach_gain,
            "frontier_gain": frontier_gain,
            "congestion_increase": congestion_increase,
            "immobile_units": optionality_metrics(after, mover)["immobile_units"],
            "relocation_options": optionality_metrics(after, mover)["relocation_options"],
            "extension_increase": extension_gain,
            "frontier_rank": structure_metrics(after, mover)["frontier_rank"],
            "frontier_limit": float(early_frontier_rank(self.turn_number)),
            "forward_infantry_actions": float(forward_infantry_actions),
            "coordinated_overpush": float(coordinated_overpush),
            "escape_actions": float(extractions),
            "purposeful_actions": float(purposeful_actions),
            "unpurposed_actions": float(unpurposed_actions),
            "setup_actions": float(setup_actions),
            "development_actions": float(development_actions),
            "formation_actions": float(formation_actions),
            "quiet_actions": float(quiet_actions),
            "counted_actions": float(counted_actions),
            "unused_actions": float(unused_actions),
            "backfills": float(backfills),
            "reversals": float(reversals),
            "pure_rotations": float(pure_rotations),
            "forcing_gain": forcing_gain,
            "stagnation_progress": stagnation_progress,
            "net_purpose_penalty": net_purpose_penalty,
            "paratrooper_mission_penalty": mission_penalty,
            "total_penalty": net_purpose_penalty + mission_penalty,
        }

    def static_score(self, board: engine.BaseBoard) -> float:
        """Blend human heuristics with the trained red win-probability model."""
        heuristic = self.heuristic_score(board)
        if self.value_function is None:
            return heuristic
        key = board.board_fen()
        probability = self.value_cache.get(key)
        if probability is None:
            self.check_time(False)
            probability = float(self.value_function(key, self.turn_number))
            probability = max(0.001, min(0.999, probability))
            self.value_cache[key] = probability
            self.value_model_evaluations += 1
        # A calibrated probability supplies the strategic baseline. Concrete
        # rule/evaluation terms remain visible and personalities can still
        # distinguish equally sound positions.
        model_log_odds = math.log(probability / (1.0 - probability))
        return heuristic + 3.0 * model_log_odds

    @staticmethod
    def board_as_turn(board: engine.BaseBoard, color: bool) -> engine.BaseBoard:
        probe = board.copy()
        probe.turn = color
        probe.turn_moves = 0
        probe.turn_auto_moves = 0
        probe.turn_pieces = engine.BB_EMPTY
        probe.did_offer_draw = False
        probe.did_accept_draw = False
        probe._clear_free_captures()
        list(probe._generate_free_captures(color))
        return probe

    @staticmethod
    def mask_value(board: engine.BaseBoard, mask: int) -> float:
        total = 0.0
        for square in squares(mask):
            piece_type = board.piece_type_at(square)
            total += PIECE_VALUES.get(piece_type, 0.0)
        return total

    def tactical_risk(self, board: engine.BaseBoard, defender: bool) -> Tuple[float, float, float]:
        """Return risk, forced loss, and critical para/artillery exposure.

        Forced start-of-turn captures are resolved far enough to inspect the
        opponent's first voluntary action. This catches a gun left directly
        available to a para and a para left inside a bombardment even when a
        different automatic capture must happen first.
        """
        cache_key = (board.serialize(), defender)
        cached = self.safety_cache.get(cache_key)
        if cached is not None:
            return cached
        self.check_time(False)
        attacker = not defender
        probe = board if board.turn == attacker and board.turn_moves == 0 else self.board_as_turn(board, attacker)
        frontier: List[Tuple[engine.BaseBoard, int]] = [(probe, engine.BB_EMPTY)]
        action_positions: List[Tuple[engine.BaseBoard, int]] = []
        for _ in range(8):
            next_frontier: List[Tuple[engine.BaseBoard, int]] = []
            for position, lost_mask in frontier[:12]:
                self.check_time(False)
                legal = list(position.generate_legal_moves())
                if legal and all(move.name == "AutoCapture" for move in legal):
                    for move in legal[:12]:
                        child_lost = lost_mask
                        if move.capture_preference is not None:
                            child_lost |= engine.BB_SQUARES[move.capture_preference]
                        child = position.copy()
                        child.push(move)
                        next_frontier.append((child, child_lost))
                else:
                    action_positions.append((position, lost_mask))
            if not next_frontier:
                break
            frontier = next_frontier[:12]
        action_positions.extend(frontier if not action_positions else [])

        own = board.occupied_co[defender]
        critical = own & (board.hq | board.airborne_infantry | board.artillery | board.armored_artillery | board.heavy_artillery)
        forced_value = max(
            (self.mask_value(board, lost_mask & own) for _, lost_mask in action_positions),
            default=0.0,
        )
        max_direct_critical = 0.0
        max_other_hanging = 0.0
        same_turn_hq_loss = False
        friendly_non_hq = own & ~board.hq
        for position, _ in action_positions[:12]:
            self.check_time(False)
            if self.has_same_turn_hq_capture(position):
                # A three-action HQ combination is just as forced as a direct
                # capture.  Feeding it into the ordinary critical-risk path
                # makes complete-turn selection retain an HQ escape instead
                # of trusting a superficially quiet position.
                same_turn_hq_loss = True
            for move in position.generate_legal_captures():
                target = move.capture_preference
                if target is None or not (engine.BB_SQUARES[target] & own):
                    continue
                target_mask = engine.BB_SQUARES[target]
                if target_mask & critical:
                    target_type = board.piece_type_at(target)
                    attacker_type = self.move_piece_type(position, move)
                    if (
                        target_type in ARTILLERY_TYPES
                        and attacker_type == engine.AIRBORNE_INFANTRY
                        and move.to_square is not None
                    ):
                        defender_infantry = position.occupied_co[defender] & (
                            position.infantry
                            | position.armored_infantry
                            | position.airborne_infantry
                        )
                        if engine.BB_ADJACENT_SQUARES[move.to_square] & defender_infantry:
                            # A diagonally adjacent infantry covers both para
                            # landing squares beside its artillery. The capture
                            # is possible, but it is not a clean hanging gun.
                            continue
                    max_direct_critical = max(
                        max_direct_critical,
                        PIECE_VALUES.get(target_type, 0.0),
                    )
                    continue
                # Ordinary infantry is treated as hanging only if it has no
                # adjacent friendly unit. Protected trades remain for minimax.
                support = engine.BB_ADJACENT_SQUARES[target] & friendly_non_hq & ~target_mask
                if not support:
                    max_other_hanging = max(
                        max_other_hanging,
                        PIECE_VALUES.get(board.piece_type_at(target), 0.0),
                    )

        if same_turn_hq_loss:
            forced_value = max(forced_value, PIECE_VALUES[engine.HQ])
        risk = forced_value + max_direct_critical + max_other_hanging
        result = (risk, forced_value, forced_value + max_direct_critical)
        self.safety_cache[cache_key] = result
        return result

    def assess_turn_safety(
        self,
        before: engine.BaseBoard,
        after: engine.BaseBoard,
        mover: bool,
    ) -> TacticalSafety:
        if after.is_game_over():
            return TacticalSafety(0.0, 0.0, 100.0, 0.0, 0.0, True)
        baseline, baseline_forced, _ = self.tactical_risk(before, mover)
        risk, forced, critical = self.tactical_risk(after, mover)
        opponent = not mover
        opponent_turn = (
            after
            if after.turn == opponent and after.turn_moves == 0
            else self.board_as_turn(after, opponent)
        )
        loses_hq_this_turn = self.has_same_turn_hq_capture(opponent_turn)
        compensation = max(
            0.0,
            board_material_for(before, opponent) - board_material_for(after, opponent),
        )
        new_risk = max(0.0, risk - baseline)
        uncovered = max(0.0, new_risk - compensation)
        # Tactics are objective. Personalities may choose among safe turns but
        # cannot spend an exposed gun/para without verified material return.
        # An existing mate threat is not a baseline risk the mover may simply
        # preserve. If the completed turn still permits an HQ capture, it is
        # objectively unsafe regardless of material compensation.
        resolved_hq_threat = (
            baseline_forced >= PIECE_VALUES[engine.HQ]
            and not loses_hq_this_turn
            and forced < PIECE_VALUES[engine.HQ]
        )
        safe = not loses_hq_this_turn and (
            resolved_hq_threat
            or (
                uncovered <= 0.75
                and forced <= compensation + 0.75
            )
        )
        forced_loss = max(
            forced,
            PIECE_VALUES[engine.HQ] if loses_hq_this_turn else 0.0,
        )
        critical_loss = max(critical, forced_loss)
        return TacticalSafety(
            risk,
            new_risk,
            compensation,
            forced_loss,
            critical_loss,
            safe,
        )

    @staticmethod
    def points_toward_home(color: bool, orientation: Optional[int]) -> bool:
        if orientation is None:
            return False
        if color == engine.RED:
            return orientation in (engine.ORIENT_SE, engine.ORIENT_S, engine.ORIENT_SW)
        return orientation in (engine.ORIENT_N, engine.ORIENT_NE, engine.ORIENT_NW)

    @staticmethod
    def closest_enemy_distance(board: engine.BaseBoard, square: int, color: bool) -> int:
        enemies = squares(board.occupied_co[not color])
        return min((chebyshev(square, enemy) for enemy in enemies), default=8)

    @staticmethod
    def artillery_lane_masks(
        board: engine.BaseBoard, move: engine.Move
    ) -> Tuple[int, int]:
        if (
            move.from_square is None
            or move.to_square is None
            or move.orientation is None
        ):
            return engine.BB_EMPTY, engine.BB_EMPTY
        piece_type = board.piece_type_at(move.from_square)
        if piece_type not in ARTILLERY_TYPES:
            return engine.BB_EMPTY, engine.BB_EMPTY
        distance = 3 if piece_type == engine.HEAVY_ARTILLERY else 2
        target = board.get_bombardment_target(
            move.to_square, move.orientation, distance
        )
        if target is None:
            return engine.BB_EMPTY, engine.BB_EMPTY
        lane = engine.between_inclusive_end(move.to_square, target)
        friendly_after = (
            board.occupied_co[board.turn]
            & ~engine.BB_SQUARES[move.from_square]
        ) | engine.BB_SQUARES[move.to_square]
        return (
            lane & board.occupied_co[not board.turn],
            lane & friendly_after,
        )

    def artillery_move_allowed(self, board: engine.BaseBoard, move: engine.Move) -> bool:
        """Apply the user's provisional artillery-orientation search rules.

        Pure rotations must create an actual threat. Relocations may choose a
        new useful facing because orientation clones are collapsed later.
        Homeward-facing rotations and lanes aimed only through friendly pieces
        are discarded.
        """
        if move.name != "MoveAndOrient" or move.from_square is None or move.to_square is None:
            return True
        piece_type = board.piece_type_at(move.from_square)
        if piece_type not in ARTILLERY_TYPES:
            return True
        if self.points_toward_home(board.turn, move.orientation):
            return False

        enemy_targets, friendly_blocks = self.artillery_lane_masks(board, move)
        if move.from_square == move.to_square and not enemy_targets:
            return False
        if friendly_blocks and not enemy_targets:
            return False

        distance = self.closest_enemy_distance(board, move.to_square, board.turn)
        if distance > 3 and move.from_square == move.to_square:
            return False
        return True

    @staticmethod
    def move_piece_type(board: engine.BaseBoard, move: engine.Move) -> Optional[int]:
        return move.unit_type if move.name == "Reinforce" else (
            board.piece_type_at(move.from_square) if move.from_square is not None else None
        )

    @staticmethod
    def home_distance(square: int, color: bool) -> int:
        home_rank = 0 if color == engine.RED else 7
        return abs(engine.square_rank(square) - home_rank)

    @staticmethod
    def approach_distance(
        board: engine.BaseBoard, color: bool, *, hq_only: bool = False
    ) -> float:
        """Distance from our mobile force to contact or the opposing HQ.

        Infantry are the preferred contact force; if none survive, any
        non-HQ unit may pursue. Friendly square-swaps cannot improve this
        measure unless the force actually closes on an objective.
        """
        own_non_hq = board.occupied_co[color] & ~board.hq
        own_infantry = own_non_hq & (
            board.infantry | board.armored_infantry | board.airborne_infantry
        )
        pursuers = own_infantry or own_non_hq
        targets = (
            board.occupied_co[not color] & board.hq
            if hq_only
            else board.occupied_co[not color]
        )
        if not pursuers or not targets:
            return 8.0
        return float(
            min(
                chebyshev(source, target)
                for source in squares(pursuers)
                for target in squares(targets)
            )
        )

    def artillery_target_bonus(self, board: engine.BaseBoard, move: engine.Move) -> float:
        piece_type = self.move_piece_type(board, move)
        if piece_type not in ARTILLERY_TYPES or move.name != "MoveAndOrient":
            return 0.0
        child = board.copy()
        child.push(move)
        targets = child.get_bombarded_squares(board.turn) & child.occupied_co[not board.turn]
        value = 0.0
        for square in squares(targets):
            target_type = child.piece_type_at(square)
            value += PIECE_VALUES.get(target_type, 0.0)
        # A two-target windshield-wiper threat is more forcing than two
        # unrelated quiet improvements.
        return 250.0 * value + 250.0 * max(0, engine.popcount(targets) - 1)

    def unlocks_airborne_extraction(self, board: engine.BaseBoard, move: engine.Move) -> bool:
        """Whether this action creates a new homeward move for our paratrooper."""
        if board.turn_moves >= 2 or move.from_square is None:
            return False
        airborne = board.airborne_infantry & board.occupied_co[board.turn]
        if not airborne or not any(chebyshev(move.from_square, square) == 1 for square in squares(airborne)):
            return False

        legal_before = list(board.generate_legal_moves())
        before_by_source: Dict[int, set[str]] = {}
        for candidate in legal_before:
            if candidate.from_square is not None and engine.BB_SQUARES[candidate.from_square] & airborne:
                before_by_source.setdefault(candidate.from_square, set()).add(candidate.uci())

        child = board.copy()
        child.push(move)
        if child.turn != board.turn:
            return False
        for candidate in child.generate_legal_moves():
            source = candidate.from_square
            destination = candidate.to_square
            if source is None or destination is None:
                continue
            if child.piece_type_at(source) != engine.AIRBORNE_INFANTRY:
                continue
            if candidate.uci() in before_by_source.get(source, set()):
                continue
            if self.home_distance(destination, board.turn) < self.home_distance(source, board.turn):
                return True
        return False

    def unlocks_capture_this_turn(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Whether a quiet action unlocks a capture within this turn.

        This bounded two-action probe is active only near the no-progress
        limit. It preserves lane-clearing sequences whose first two actions
        look inert before a third action captures; minimax still decides
        whether the resulting complete turn is sound.
        """
        if (
            self.stagnation_factor() < 0.70
            or move.name in ("AutoCapture", "Skip")
            or move.capture_preference is not None
            or board.turn_moves >= self.max_actions - 1
        ):
            return False
        cache_key = (board.serialize(), move.uci())
        cached = self.capture_setup_move_cache.get(cache_key)
        if cached is not None:
            return cached

        mover = board.turn
        child = board.copy()
        child.push(move)
        if child.is_game_over() or child.turn != mover:
            self.capture_setup_move_cache[cache_key] = False
            return False

        def capture_available(position: engine.BaseBoard) -> bool:
            return any(
                candidate.capture_preference is not None
                for candidate in position.generate_legal_moves()
            )

        if capture_available(child):
            self.capture_setup_move_cache[cache_key] = True
            return True

        remaining_actions = self.max_actions - child.turn_moves
        if remaining_actions >= 2:
            for second in child.generate_legal_moves():
                if second.name in ("AutoCapture", "Skip"):
                    continue
                grandchild = child.copy()
                grandchild.push(second)
                if (
                    not grandchild.is_game_over()
                    and grandchild.turn == mover
                    and capture_available(grandchild)
                ):
                    self.capture_setup_move_cache[cache_key] = True
                    return True
        self.capture_setup_move_cache[cache_key] = False
        return False

    def unlocks_hq_escape(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Whether vacating a neighboring square creates an HQ escape.

        A direct HQ move already receives forcing priority.  This covers the
        preceding action when a friendly blocker must first clear the only
        safe destination, as in ``b7-b6, HQ a8-b7`` from production smoke.
        """
        if (
            move.name in ("AutoCapture", "Skip")
            or move.from_square is None
            or move.to_square is None
            or move.from_square == move.to_square
            or board.turn_moves >= self.max_actions - 1
            or board.piece_type_at(move.from_square) == engine.HQ
        ):
            return False
        own_hqs = board.pieces(engine.HQ, board.turn)
        if not any(
            chebyshev(move.from_square, hq_square) == 1
            for hq_square in own_hqs
        ):
            return False
        _, baseline_forced, _ = self.tactical_risk(board, board.turn)
        if baseline_forced < PIECE_VALUES[engine.HQ]:
            return False

        child = board.copy()
        child.push(move)
        if child.turn != board.turn or child.is_game_over():
            return False
        for hq_move in child.generate_legal_moves():
            if (
                hq_move.from_square is None
                or hq_move.to_square != move.from_square
                or child.piece_type_at(hq_move.from_square) != engine.HQ
            ):
                continue
            escaped = child.copy()
            escaped.push(hq_move)
            _, escaped_forced, _ = self.tactical_risk(escaped, board.turn)
            if escaped_forced < PIECE_VALUES[engine.HQ]:
                return True
        return False

    def resolves_hq_threat(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Whether this action removes an otherwise forced HQ loss.

        A reinforcement can interpose on an artillery line without moving the
        HQ or vacating an escape square. Such actions need check-evasion
        priority while a complete turn is still being assembled.
        """
        if move.name in ("AutoCapture", "Skip"):
            return False
        cache_key = (board.serialize(), move.uci())
        cached = self.hq_defense_move_cache.get(cache_key)
        if cached is not None:
            return cached
        _, baseline_forced, _ = self.tactical_risk(board, board.turn)
        if baseline_forced < PIECE_VALUES[engine.HQ]:
            self.hq_defense_move_cache[cache_key] = False
            return False
        child = board.copy()
        child.push(move)
        _, child_forced, _ = self.tactical_risk(child, board.turn)
        resolved = child_forced < PIECE_VALUES[engine.HQ]
        self.hq_defense_move_cache[cache_key] = resolved
        return resolved

    def has_immediate_hq_capture(self, board: engine.BaseBoard) -> bool:
        """Whether the side to act can take the enemy HQ with one legal action."""
        key = board.serialize()
        cached = self.immediate_hq_capture_cache.get(key)
        if cached is not None:
            return cached
        for candidate in board.generate_legal_moves():
            target = candidate.capture_preference
            if target is not None and board.piece_type_at(target) == engine.HQ:
                self.immediate_hq_capture_cache[key] = True
                return True
        self.immediate_hq_capture_cache[key] = False
        return False

    def has_same_turn_hq_capture(self, board: engine.BaseBoard) -> bool:
        """Whether the side to act can force an HQ capture this turn.

        This is deliberately narrower than complete turn enumeration.  The
        unlock detector examines only setup actions geometrically close to the
        target HQ, including the two quiet setups needed by a paratrooper mate.
        It is therefore cheap enough for turn-safety checks while covering the
        tactical combinations that an atomic-action beam is most likely to
        discard.
        """
        key = board.serialize()
        cached = self.same_turn_hq_capture_cache.get(key)
        if cached is not None:
            return cached
        if self.has_immediate_hq_capture(board):
            self.same_turn_hq_capture_cache[key] = True
            return True
        if board.turn_moves >= self.max_actions - 1:
            self.same_turn_hq_capture_cache[key] = False
            return False
        for move in board.generate_legal_moves():
            if self.unlocks_immediate_hq_capture(board, move):
                self.same_turn_hq_capture_cache[key] = True
                return True
        self.same_turn_hq_capture_cache[key] = False
        return False

    def unlocks_immediate_hq_capture(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Whether a setup action makes a same-turn HQ capture legal.

        GHQ turns are combinations, so a setup move can be forcing even when
        it has no immediate capture. Moving a blocker can open the square a
        second piece needs, while moving an infantry beside the HQ can create
        the engagement needed by that second piece. These actions must survive
        the atomic-action beam.
        """
        if (
            move.name in ("AutoCapture", "Skip")
            or move.from_square is None
            or move.to_square is None
            or move.from_square == move.to_square
            or board.turn_moves >= self.max_actions - 1
        ):
            return False
        enemy_hqs = board.pieces(engine.HQ, not board.turn)
        actions_remaining = self.max_actions - board.turn_moves
        setup_radius = 2 if actions_remaining >= 3 else 1
        if not any(
            chebyshev(move.from_square, hq_square) <= setup_radius
            or chebyshev(move.to_square, hq_square) <= setup_radius
            for hq_square in enemy_hqs
        ):
            # HQ-capture setup happens immediately around the target. This
            # cheap geometric gate keeps unlock detection from regenerating
            # moves for every quiet action in an ordinary position.
            return False
        if self.has_immediate_hq_capture(board):
            # The capture was already legal; this action did not unlock it
            # and must not receive forcing priority merely for preserving it.
            return False

        child = board.copy()
        child.push(move)
        if child.turn != board.turn or child.is_game_over():
            return False
        if self.has_immediate_hq_capture(child):
            return True

        # Some three-action mates need one more setup before the HQ capture
        # becomes legal. That setup is often a capture, but it can also be a
        # quiet move beside the HQ that opens a paratrooper lane or supplies
        # the second engagement. Inspect only captures and moves immediately
        # around the target HQ, and only when two counted actions remain, so
        # this tactical extension stays narrow and deterministic.
        if child.turn_moves >= self.max_actions - 1:
            return False
        for follow_up in child.generate_legal_moves():
            if follow_up.name in ("AutoCapture", "Skip"):
                continue
            near_hq = (
                follow_up.from_square is not None
                and follow_up.to_square is not None
                and any(
                    chebyshev(follow_up.from_square, hq_square) == 1
                    or chebyshev(follow_up.to_square, hq_square) == 1
                    for hq_square in enemy_hqs
                )
            )
            if follow_up.capture_preference is None and not near_hq:
                continue
            grandchild = child.copy()
            grandchild.push(follow_up)
            if (
                grandchild.turn == board.turn
                and not grandchild.is_game_over()
                and self.has_immediate_hq_capture(grandchild)
            ):
                return True
        return False

    def move_priority(self, board: engine.BaseBoard, move: engine.Move) -> float:
        if move.name == "AutoCapture":
            target = board.piece_type_at(move.capture_preference) if move.capture_preference is not None else None
            return 10000.0 + 100.0 * PIECE_VALUES.get(target, 0.0)
        if move.capture_preference is not None:
            target = board.piece_type_at(move.capture_preference)
            priority = 5000.0 + 100.0 * PIECE_VALUES.get(target, 0.0)
            if target != engine.HQ and self.unlocks_immediate_hq_capture(board, move):
                priority += 8000.0
            return priority
        if move.name == "Skip":
            return -10000.0
        priority = 0.0
        piece_type = self.move_piece_type(board, move)
        if (
            piece_type == engine.HQ
            and move.from_square is not None
            and move.to_square is not None
        ):
            # Detect an HQ escape before the atomic-action beam is applied.
            # If mate is noticed only after complete turns are formed, the
            # sole quiet escape may already have been pruned.
            _, baseline_forced, _ = self.tactical_risk(board, board.turn)
            child = board.copy()
            child.push(move)
            _, child_forced, _ = self.tactical_risk(child, board.turn)
            if (
                baseline_forced >= PIECE_VALUES[engine.HQ]
                and child_forced < PIECE_VALUES[engine.HQ]
            ):
                priority += 4950.0
        infantry_types = (
            engine.INFANTRY,
            engine.ARMORED_INFANTRY,
            engine.AIRBORNE_INFANTRY,
        )
        enemy_airborne = board.airborne_infantry & board.occupied_co[not board.turn]
        if (
            piece_type in infantry_types
            and move.to_square is not None
            and engine.BB_ADJACENT_SQUARES[move.to_square] & enemy_airborne
        ):
            # Quiet moves and reinforcements that engage an enemy paratrooper
            # are tactical setup moves. Keep them inside the beam so a later
            # action in the same turn can complete the capture.
            priority += 4500.0
        if (
            piece_type in ARTILLERY_TYPES
            and move.from_square is not None
            and move.to_square is not None
            and move.from_square != move.to_square
            and engine.BB_ADJACENT_SQUARES[move.from_square] & enemy_airborne
        ):
            # Vacating an artillery square can open the lane another infantry
            # needs in order to take an already-engaged paratrooper.
            priority += 4000.0
        if (
            piece_type in ARTILLERY_TYPES
            and move.from_square is not None
            and move.to_square is not None
            and engine.BB_SQUARES[move.from_square] & board.get_bombarded_squares(not board.turn)
            and not (engine.BB_SQUARES[move.to_square] & board.get_bombarded_squares(not board.turn))
        ):
            # Saving a gun that will otherwise be automatically captured is a
            # forcing action, not a generic quiet artillery move.
            priority += 4600.0
        if (
            piece_type == engine.AIRBORNE_INFANTRY
            and move.from_square is not None
            and move.to_square is not None
            and self.home_distance(move.to_square, board.turn) < self.home_distance(move.from_square, board.turn)
        ):
            # Preserve extraction moves once a preceding action has made them
            # legal. This is intentionally comparable with capture ordering.
            priority += 4700.0
        if self.unlocks_airborne_extraction(board, move):
            # Quiet blocker-vacating moves such as g3-f4 must survive long
            # enough for the next action, h2-g3, to be considered.
            priority += 4800.0
        if self.unlocks_capture_this_turn(board, move):
            # Late in a quiet game, keep the lane-clearing setup for a capture
            # even when the setup action itself changes no evaluation feature.
            priority += 3500.0
        if self.unlocks_hq_escape(board, move):
            # Clearing the only HQ flight square is part of the escape, not a
            # generic infantry shuffle. It must survive even the narrow
            # verification beam.
            priority += 8200.0
        if self.resolves_hq_threat(board, move):
            # Interpositions and engagements that turn off a forced HQ loss
            # are check evasions even when they capture nothing.
            priority += 8500.0
        if self.unlocks_immediate_hq_capture(board, move):
            # A quiet action that makes mate legal on the next action is more
            # forcing than ordinary captures and must not be lost to the
            # atomic-action or partial-turn beams.
            priority += 8000.0
        artillery_bonus = self.artillery_target_bonus(board, move)
        if self.turn_number <= EARLY_GAME_LAST_TURN:
            # Early play should build the army, not let one speculative gun
            # threat crowd every reinforcement/formation branch out of the
            # partial-turn beam.
            artillery_bonus *= 0.25
        priority += artillery_bonus
        if piece_type in ARTILLERY_TYPES and move.name == "MoveAndOrient":
            _, friendly_blocks = self.artillery_lane_masks(board, move)
            priority -= 150.0 * engine.popcount(friendly_blocks)
        if piece_type == engine.ARMORED_INFANTRY:
            priority += 20.0
        elif piece_type in ARTILLERY_TYPES:
            priority += 10.0
        if move.name == "MoveAndOrient":
            priority += 5.0
        if self.turn_number <= EARLY_GAME_LAST_TURN and move.capture_preference is None:
            if move.name == "Reinforce":
                priority += 900.0
            child = board.copy()
            child.push(move)
            formation_gain = self.formation_quality(child, board.turn) - self.formation_quality(board, board.turn)
            priority += 250.0 * max(0.0, formation_gain)
            cohesion_gain = dispersion_penalty(board, board.turn) - dispersion_penalty(
                child, board.turn
            )
            priority += 300.0 * cohesion_gain
            if (
                piece_type in (engine.INFANTRY, engine.ARMORED_INFANTRY)
                and move.name != "Reinforce"
                and move.from_square is not None
                and move.to_square is not None
                and relative_rank(move.to_square, board.turn)
                > relative_rank(move.from_square, board.turn)
            ):
                # Advancing is not development by itself. Prefer deployment
                # and connective moves unless the advance creates a concrete
                # threat or protection that later scoring can verify.
                priority -= 140.0 * relative_rank(move.to_square, board.turn)
                priority -= 120.0 * board.turn_moves
            if (
                piece_type in ARTILLERY_TYPES
                and move.from_square is not None
                and move.to_square is not None
                and self.home_distance(move.to_square, board.turn)
                - self.home_distance(move.from_square, board.turn)
                >= 2
            ):
                priority -= 650.0
        return priority

    def diverse_moves(self, board: engine.BaseBoard) -> List[Tuple[float, engine.Move]]:
        """Return strategically filtered actions without applying the beam."""
        legal = list(board.generate_legal_moves())
        if legal and all(move.name == "AutoCapture" for move in legal):
            return [(self.move_priority(board, move), move) for move in legal]

        moves = [
            move
            for move in legal
            if self.artillery_move_allowed(board, move)
            and self.paradrop_allowed(board, move)
            and self.early_extension_allowed(board, move)
        ]
        if len(moves) != len(legal):
            self.exhaustive_within_horizon = False
            self.rule_filtered_actions += len(legal) - len(moves)

        scored = [
            (self.move_priority(board, move), normalized_move_uci(move, board.turn), move)
            for move in moves
        ]
        scored.sort(key=lambda item: (-item[0], item[1]))

        # Collapse orientation clones before applying the beam. The strongest
        # facing for each artillery source/destination survives, so a single
        # gun cannot occupy the entire beam with eight rotations.
        diverse: List[Tuple[float, engine.Move]] = []
        seen = set()
        for priority, _, move in scored:
            piece_type = self.move_piece_type(board, move)
            key = (
                ("artillery", move.from_square, move.to_square)
                if piece_type in ARTILLERY_TYPES and move.name == "MoveAndOrient"
                else (move.name, move.from_square, move.to_square, move.unit_type)
            )
            if key in seen:
                continue
            seen.add(key)
            diverse.append((priority, move))

        if len(diverse) != len(moves):
            self.exhaustive_within_horizon = False
            self.rule_filtered_actions += len(moves) - len(diverse)
        return diverse

    def bounded_diverse_moves(
        self, board: engine.BaseBoard
    ) -> List[Tuple[float, engine.Move]]:
        """Bound one atomic-action layer while preserving forcing actions."""
        diverse_scored = self.diverse_moves(board)
        diverse = [move for _, move in diverse_scored]
        if diverse and all(move.name == "AutoCapture" for move in diverse):
            return diverse_scored

        selected = list(diverse_scored[: self.beam_width])
        priority_by_uci = {move.uci(): priority for priority, move in diverse_scored}
        critical_limit = max(self.beam_width * 2, self.beam_width + 4)
        for priority, move in diverse_scored:
            if len(selected) >= critical_limit:
                break
            if (
                all(selected_move != move for _, selected_move in selected)
                and priority_by_uci.get(move.uci(), 0.0) >= 4000.0
            ):
                selected.append((priority, move))
        if len(selected) != len(diverse):
            self.exhaustive_within_horizon = False
            self.beam_pruned_actions += len(diverse) - len(selected)

        # Ending early is considered only for the final optional action. This
        # prevents a permanently retained Skip from beating quiet two-action
        # setup/extraction combinations after just one move.
        skip = next((move for move in diverse if move.name == "Skip"), None)
        if (
            board.turn_moves >= min(2, self.max_actions)
            and skip is not None
            and all(selected_move != skip for _, selected_move in selected)
        ):
            selected.append((priority_by_uci[skip.uci()], skip))
        return selected

    def ordered_moves(self, board: engine.BaseBoard) -> List[engine.Move]:
        return [move for _, move in self.bounded_diverse_moves(board)]

    @staticmethod
    def _prefer_partial(
        candidate: PartialTurn, incumbent: PartialTurn, color: bool
    ) -> bool:
        if candidate.priority != incumbent.priority:
            return candidate.priority > incumbent.priority
        return normalized_turn_key(candidate.moves, color) < normalized_turn_key(
            incumbent.moves, color
        )

    def _round_robin_partials(
        self,
        partials: Sequence[PartialTurn],
        limit: int,
        color: bool,
        root: engine.BaseBoard,
    ) -> List[PartialTurn]:
        """Preserve multiple first actions instead of one prolific branch."""
        def plan_rank(partial: PartialTurn) -> int:
            if (
                not partial.board.is_game_over()
                and partial.board.turn == color
                and self.has_immediate_hq_capture(partial.board)
            ):
                # A partial combination that has made the HQ capture legal is
                # one action from mate.  It must outrank a higher raw-priority
                # artillery move sharing the same first action; otherwise the
                # narrow reply frontier can discard the mate before minimax
                # sees the completed turn.
                return -1
            keys = self.turn_plan_keys(root, partial.moves)
            return (
                0
                if any(
                    key[0]
                    in (
                        "hq_escape",
                        "hq_escape_unlock",
                        "hq_defense",
                        "capture_setup",
                        "airborne_extraction",
                        "unlock_and_extract",
                        "escape_and_extract",
                        "hq_capture_unlock",
                    )
                    for key in keys
                )
                else 1
            )

        groups: Dict[str, List[PartialTurn]] = {}
        for partial in partials:
            first = normalized_move_uci(partial.moves[0], color) if partial.moves else ""
            groups.setdefault(first, []).append(partial)
        for values in groups.values():
            values.sort(
                key=lambda item: (
                    plan_rank(item),
                    len(item.moves) if item.board.is_game_over() else 99,
                    -item.priority,
                    normalized_turn_key(item.moves, color),
                )
            )

        ordered_groups = sorted(
            groups.values(),
            key=lambda values: (
                plan_rank(values[0]),
                len(values[0].moves) if values[0].board.is_game_over() else 99,
                -values[0].priority,
                normalized_move_uci(values[0].moves[0], color)
                if values[0].moves
                else "",
            ),
        )
        selected: List[PartialTurn] = []
        round_index = 0
        while len(selected) < limit:
            added = False
            for values in ordered_groups:
                if round_index < len(values):
                    selected.append(values[round_index])
                    added = True
                    if len(selected) >= limit:
                        break
            if not added:
                break
            round_index += 1
        return selected

    def turn_plan_keys(self, board: engine.BaseBoard, moves: Sequence[engine.Move]) -> List[Tuple[Any, ...]]:
        """Describe forcing defensive plans independently of action order."""
        working = board.copy()
        artillery_escapes: List[Tuple[int, int]] = []
        airborne_extractions: List[Tuple[int, int]] = []
        hq_escapes: List[Tuple[int, int]] = []
        extraction_unlocks: List[str] = []
        capture_setups: List[str] = []
        hq_escape_unlocks: List[str] = []
        hq_defenses: List[str] = []
        hq_capture_unlocks: List[str] = []
        for move in moves:
            piece_type = self.move_piece_type(working, move)
            if self.unlocks_airborne_extraction(working, move):
                extraction_unlocks.append(move.uci())
            if self.unlocks_capture_this_turn(working, move):
                capture_setups.append(move.uci())
            if self.unlocks_hq_escape(working, move):
                hq_escape_unlocks.append(move.uci())
            if self.resolves_hq_threat(working, move):
                hq_defenses.append(move.uci())
            if self.unlocks_immediate_hq_capture(working, move):
                hq_capture_unlocks.append(move.uci())
            if move.from_square is not None and move.to_square is not None:
                if (
                    piece_type in ARTILLERY_TYPES
                    and engine.BB_SQUARES[move.from_square] & working.get_bombarded_squares(not working.turn)
                    and not engine.BB_SQUARES[move.to_square] & working.get_bombarded_squares(not working.turn)
                ):
                    artillery_escapes.append((move.from_square, move.to_square))
                if (
                    piece_type == engine.AIRBORNE_INFANTRY
                    and self.home_distance(move.to_square, working.turn)
                    < self.home_distance(move.from_square, working.turn)
                ):
                    airborne_extractions.append((move.from_square, move.to_square))
                if piece_type == engine.HQ:
                    _, baseline_forced, _ = self.tactical_risk(
                        working, working.turn
                    )
                    child = working.copy()
                    child.push(move)
                    _, child_forced, _ = self.tactical_risk(child, working.turn)
                    if (
                        baseline_forced >= PIECE_VALUES[engine.HQ]
                        and child_forced < PIECE_VALUES[engine.HQ]
                    ):
                        hq_escapes.append((move.from_square, move.to_square))
            working.push(move)

        keys: List[Tuple[Any, ...]] = []
        keys.extend(("hq_escape", source, destination) for source, destination in hq_escapes)
        keys.extend(("artillery_escape", source, destination) for source, destination in artillery_escapes)
        keys.extend(("airborne_extraction", source, destination) for source, destination in airborne_extractions)
        keys.extend(("airborne_unlock", uci) for uci in extraction_unlocks)
        keys.extend(("capture_setup", uci) for uci in capture_setups)
        keys.extend(("hq_escape_unlock", uci) for uci in hq_escape_unlocks)
        keys.extend(("hq_defense", uci) for uci in hq_defenses)
        keys.extend(("hq_capture_unlock", uci) for uci in hq_capture_unlocks)
        for unlock in extraction_unlocks:
            for airborne_extraction in airborne_extractions:
                keys.append(("unlock_and_extract", unlock) + airborne_extraction)
        for artillery_escape in artillery_escapes:
            for airborne_extraction in airborne_extractions:
                keys.append(("escape_and_extract",) + artillery_escape + airborne_extraction)
        return keys

    def turn_action_classes(
        self, board: engine.BaseBoard, moves: Sequence[engine.Move]
    ) -> Tuple[set[str], bool]:
        """Classify a full turn and flag non-forcing in-place rotations."""
        working = board.copy()
        classes: set[str] = set()
        wasteful_rotation = False
        for move in moves:
            piece_type = self.move_piece_type(working, move)
            if move.capture_preference is not None:
                classes.add("capture")
            if move.name == "Reinforce":
                classes.add("reinforcement")
            if piece_type == engine.AIRBORNE_INFANTRY:
                classes.add("paratrooper")
            elif piece_type in INFANTRY_TYPES:
                classes.add("infantry")
            elif piece_type in ARTILLERY_TYPES and move.name == "MoveAndOrient":
                if move.from_square == move.to_square:
                    classes.add("pure_rotation")
                    child = working.copy()
                    child.push(move)
                    targets = child.get_bombarded_squares(working.turn) & child.occupied_co[not working.turn]
                    if engine.popcount(targets) < 2:
                        wasteful_rotation = True
                else:
                    classes.add("artillery_relocation")
            working.push(move)
        purposes = self.action_purpose_labels(
            board, moves, board.turn, retrospective=False
        )
        if any("develop" in item["roles"] for item in purposes):
            classes.add("development")
        if any("form" in item["roles"] for item in purposes):
            classes.add("formation")
        return classes, wasteful_rotation

    def candidate_sort_key(
        self,
        candidate: TurnCandidate,
        color: bool,
        root_bias: bool = False,
    ) -> Tuple[Any, ...]:
        stagnation_bias = (
            -self.stagnation_factor() * self.stagnation_value(candidate)
            if root_bias
            else 0.0
        )
        return (
            candidate.safety_penalty,
            candidate.paratrooper_mission_penalty,
            stagnation_bias,
            candidate.purpose_penalty,
            -candidate.early_plan_score,
            -(candidate.static_score if color == engine.RED else -candidate.static_score),
            len(candidate.moves) if candidate.board.is_game_over() else 99,
            -candidate.priority,
            normalized_turn_key(candidate.moves, color),
        )

    def select_diverse_turns(
        self,
        board: engine.BaseBoard,
        candidates: Sequence[TurnCandidate],
        turn_width: int,
    ) -> List[TurnCandidate]:
        safe = [candidate for candidate in candidates if candidate.tactically_safe]
        eligible = safe if safe else list(candidates)
        selected: List[TurnCandidate] = []
        rotation_limit = max(1, self.beam_width // 4)
        rotation_count = 0

        def add(candidate: TurnCandidate) -> bool:
            nonlocal rotation_count
            if candidate in selected:
                return False
            _, wasteful = self.turn_action_classes(board, candidate.moves)
            if wasteful and rotation_count >= rotation_limit:
                self.rotation_quota_pruned += 1
                return False
            selected.append(candidate)
            if wasteful:
                rotation_count += 1
            return True

        # Forced wins are not merely another action class. Keep the shortest
        # terminal turn first so an irrelevant setup action cannot precede an
        # already available HQ combination.
        terminal = sorted(
            (candidate for candidate in eligible if candidate.board.is_game_over()),
            key=lambda candidate: (
                len(candidate.moves),
                normalized_turn_key(candidate.moves, board.turn),
            ),
        )
        if terminal:
            add(terminal[0])

        if (
            board.serialize() == self.root_key
            and self.stagnation_factor() >= 0.20
        ):
            # Keep genuinely progressive root alternatives in the minimax
            # beam even when the static evaluator slightly prefers another
            # three-piece conveyor. Search, not a post-hoc rule, then checks
            # them against the opponent reply.
            progressive = sorted(
                eligible,
                key=lambda candidate: (
                    -self.stagnation_value(candidate),
                    self.candidate_sort_key(candidate, board.turn, True),
                ),
            )
            for candidate in progressive[:2]:
                if (
                    self.stagnation_value(candidate) > 0.0
                    or (
                        candidate.conveyor_actions == 0.0
                        and candidate.skip_actions == 0.0
                    )
                ):
                    add(candidate)

        # Guarantee useful alternatives before filling by score. A beam is a
        # count of complete turns retained, not permission for one action type
        # to consume every slot.
        action_classes = (
            ("capture", "development", "formation", "reinforcement", "infantry", "artillery_relocation", "paratrooper")
            if self.turn_number <= EARLY_GAME_LAST_TURN
            else ("capture", "reinforcement", "infantry", "artillery_relocation", "paratrooper")
        )
        for action_class in action_classes:
            quota = 3 if action_class == "development" else 2
            for candidate in eligible:
                classes, wasteful = self.turn_action_classes(board, candidate.moves)
                if action_class == "paratrooper" and candidate.paratrooper_mission_penalty > 0.0:
                    continue
                if action_class in classes and not wasteful and add(candidate):
                    quota -= 1
                    if quota == 0 or len(selected) >= turn_width:
                        break

        for candidate in eligible:
            if len(selected) >= turn_width:
                break
            add(candidate)
        return selected

    def consider_early_root_fallback(
        self,
        root: engine.BaseBoard,
        cache_key: str,
        partial: PartialTurn,
        mover: bool,
    ) -> None:
        """Capture a screened fallback before wide root generation can time out."""
        if self.root_key != cache_key or self.root_fallback is not None:
            return
        counted_actions = sum(
            move.name not in ("AutoCapture", "Skip") for move in partial.moves
        )
        if not partial.board.is_game_over() and counted_actions < self.max_actions:
            return
        _, wasteful_rotation = self.turn_action_classes(root, partial.moves)
        if wasteful_rotation:
            return
        replay = root.copy()
        for move in partial.moves:
            piece_type = self.move_piece_type(replay, move)
            if (
                piece_type == engine.AIRBORNE_INFANTRY
                and move.capture_preference is None
            ):
                if move.name == "Reinforce":
                    return
                if (
                    move.from_square is not None
                    and move.to_square is not None
                    and self.home_distance(move.to_square, replay.turn)
                    >= self.home_distance(move.from_square, replay.turn)
                ):
                    return
            replay.push(move)
        try:
            safety = self.assess_turn_safety(root, partial.board, mover)
        except SearchTimeout:
            return
        if safety.tactically_safe:
            purpose = self.turn_purpose_breakdown(
                root, partial.board, partial.moves, mover, retrospective=False
            )
            action_purposes = self.action_purpose_labels(
                root, partial.moves, mover, retrospective=False
            )
            if any("no_new_effect" in item["roles"] for item in action_purposes):
                return
            if self.turn_number <= EARLY_GAME_LAST_TURN:
                if not self.early_structure_allowed(
                    root,
                    partial.board,
                    partial.moves,
                    mover,
                    action_purposes,
                ):
                    return
            self.root_fallback = TurnCandidate(
                partial.moves,
                partial.board,
                partial.priority,
                self.quick_score(partial.board),
                0.0,
                True,
                purpose["net_purpose_penalty"],
                purpose["paratrooper_mission_penalty"],
                action_purposes,
                self.early_plan_score(action_purposes),
            )

    def trim_filler_action(
        self,
        root: engine.BaseBoard,
        candidate: TurnCandidate,
        mover: bool,
    ) -> Optional[TurnCandidate]:
        """Turn ``useful + useful + filler`` into a legal two-action option.

        This is intentionally derived from a complete retained turn instead
        of keeping arbitrary low-priority Skip partials.  Removing the filler
        must leave both useful actions legal in their original order; if the
        apparent filler secretly unlocked either action, replay rejects it.
        The cleaned line then competes normally with every useful three-action
        line in minimax.
        """
        if self.max_actions != 3 or candidate.board.is_game_over():
            return None
        counted_actions = sum(
            move.name not in ("AutoCapture", "Skip") for move in candidate.moves
        )
        if counted_actions != 3 or any(
            move.name == "Skip" for move in candidate.moves
        ):
            return None
        removable = [
            index
            for index, (move, purpose) in enumerate(
                zip(candidate.moves, candidate.action_purposes)
            )
            if move.name not in ("AutoCapture", "Skip")
            and bool(
                {"no_new_effect", "setup"}.intersection(purpose["roles"])
            )
        ]
        if not removable:
            return None
        if len(removable) > 1:
            viable_removals: List[int] = []
            for omitted in removable:
                probe = root.copy()
                probe_moves: List[engine.Move] = []
                replayable = True
                for index, original_move in enumerate(candidate.moves):
                    if index == omitted:
                        continue
                    replay_move = next(
                        (
                            move
                            for move in probe.generate_legal_moves()
                            if move.uci() == original_move.uci()
                        ),
                        None,
                    )
                    if replay_move is None:
                        replayable = False
                        break
                    probe_moves.append(replay_move)
                    probe.push(replay_move)
                if not replayable or probe.turn != mover or probe.is_game_over():
                    continue
                probe_skip = next(
                    (
                        move
                        for move in probe.generate_legal_moves()
                        if move.name == "Skip"
                    ),
                    None,
                )
                if probe_skip is None:
                    continue
                probe_moves.append(probe_skip)
                probe.push(probe_skip)
                probe_purposes = self.action_purpose_labels(
                    root, probe_moves, mover, retrospective=True
                )
                voluntary = [
                    purpose
                    for move, purpose in zip(probe_moves, probe_purposes)
                    if move.name not in ("AutoCapture", "Skip")
                ]
                if len(voluntary) == 2 and all(
                    "no_new_effect" not in purpose["roles"]
                    for purpose in voluntary
                ):
                    viable_removals.append(omitted)
            if not viable_removals:
                return None
            # Prefer deleting the latest disposable action. Earlier quiet
            # moves are more likely to be setup for what follows.
            removable = [max(viable_removals)]

        working = root.copy()
        cleaned_moves: List[engine.Move] = []
        cleaned_priority = 0.0
        for index, original_move in enumerate(candidate.moves):
            if index == removable[0]:
                continue
            if working.turn != mover or working.is_game_over():
                return None
            replay_move = next(
                (
                    move
                    for move in working.generate_legal_moves()
                    if move.uci() == original_move.uci()
                ),
                None,
            )
            if replay_move is None:
                return None
            cleaned_priority += self.move_priority(working, replay_move)
            cleaned_moves.append(replay_move)
            working.push(replay_move)
        if working.turn != mover or working.is_game_over():
            return None
        skip = next(
            (move for move in working.generate_legal_moves() if move.name == "Skip"),
            None,
        )
        if skip is None:
            return None
        cleaned_priority += self.move_priority(working, skip)
        cleaned_moves.append(skip)
        working.push(skip)

        action_purposes = self.action_purpose_labels(
            root, cleaned_moves, mover, retrospective=True
        )
        voluntary_purposes = [
            purpose
            for move, purpose in zip(cleaned_moves, action_purposes)
            if move.name not in ("AutoCapture", "Skip")
        ]
        if len(voluntary_purposes) != 2 or any(
            "no_new_effect" in purpose["roles"] for purpose in voluntary_purposes
        ):
            return None
        purpose = self.turn_purpose_breakdown(
            root, working, cleaned_moves, mover, retrospective=True
        )
        if purpose["paratrooper_mission_penalty"] > 0.0:
            return None
        if self.turn_number <= EARLY_GAME_LAST_TURN and not self.early_structure_allowed(
            root,
            working,
            cleaned_moves,
            mover,
            action_purposes,
        ):
            return None
        safety = self.assess_turn_safety(root, working, mover)
        terminal = self.terminal_score(working, 0)
        return TurnCandidate(
            cleaned_moves,
            working,
            cleaned_priority,
            self.heuristic_score(working) if terminal is None else terminal,
            max(0.0, safety.new_risk_value - safety.compensation_value),
            safety.tactically_safe,
            purpose["net_purpose_penalty"],
            purpose["paratrooper_mission_penalty"],
            action_purposes,
            self.early_plan_score(action_purposes),
            purpose["stagnation_progress"],
            purpose["backfills"] + purpose["reversals"],
            1.0,
        )

    def generate_turn_candidates(self, board: engine.BaseBoard) -> List[TurnCandidate]:
        """Generate, deduplicate, then prune complete player turns.

        Search no longer truncates to ``beam_width`` after every atomic action.
        A wider, first-action-diverse frontier is carried until the side changes;
        only complete resulting positions become minimax branches.
        """
        cache_key = board.serialize()
        cached = self.turn_cache.get(cache_key)
        if cached is not None:
            self.turn_cache_hits += 1
            return cached

        original_turn = board.turn
        is_root_generation = self.root_key is None or cache_key == self.root_key
        if self.verification_mode and is_root_generation:
            # The verification pass is deliberately small: its job is to
            # finish one complete opponent reply before the clock expires.
            # The later improvement pass is responsible for breadth.
            partial_width = max(8, self.beam_width)
            evaluation_pool_width = max(6, self.beam_width)
            if self.stagnation_factor() >= 0.20:
                partial_width = max(partial_width, self.beam_width * 4)
                evaluation_pool_width = max(
                    evaluation_pool_width, self.beam_width * 3
                )
            if self.stagnation_factor() >= 0.70:
                # Near the no-progress limit, the narrow verification pool is
                # no longer allowed to contain only safe retreats. Multi-step
                # capture constructions often begin with a quiet lane-clearing
                # action and need the same root breadth as the improvement pass.
                partial_width = max(partial_width, self.beam_width * 8)
                evaluation_pool_width = max(
                    evaluation_pool_width, self.beam_width * 6
                )
        elif self.verification_mode:
            # The first reply is a tactical floor, not the final beam. One
            # statically strongest safe reply must finish before breadth is
            # useful; protected forcing/escape plans can still replace the
            # generic slot below.
            partial_width = 4
            evaluation_pool_width = 4
        elif self.time_ms < 1000:
            partial_width = max(12, self.beam_width * 2)
            evaluation_pool_width = max(16, self.beam_width * 3)
        elif is_root_generation:
            # Keep a broad partial frontier at the root so multi-action unlock
            # sequences survive. Cost is controlled later by the much smaller
            # evaluated complete-turn pool and bounded retained beam.
            partial_width = max(48, self.beam_width * 6)
            evaluation_pool_width = max(36, self.beam_width * 6)
        else:
            # Opponent replies receive a narrower beam. This reserves enough
            # of the wall-clock budget to complete an actual reply instead of
            # spending the whole turn enumerating root alternatives.
            partial_width = max(16, self.beam_width * 2)
            evaluation_pool_width = max(24, self.beam_width * 3)
        if self.verification_mode:
            turn_width = 2 if is_root_generation else 1
            turn_capacity = turn_width
            if is_root_generation and self.stagnation_factor() >= 0.20:
                turn_width = max(turn_width, min(5, self.beam_width))
                turn_capacity = turn_width + 2
        else:
            turn_width = (
                max(4, self.beam_width)
                if is_root_generation
                else max(3, (self.beam_width + 1) // 2)
            )
            turn_capacity = turn_width + max(1, self.beam_width // 3)
        frontier = [PartialTurn([], board.copy(), 0.0)]
        completed: List[PartialTurn] = []

        while frontier:
            expanded: Dict[str, PartialTurn] = {}
            for partial in frontier:
                self.check_time()
                if partial.board.is_game_over() or partial.board.turn != original_turn:
                    completed.append(partial)
                    continue

                # The first pass exists to verify several complete root lines,
                # not to enumerate every third-action permutation. Preserve a
                # normal beam plus every forcing/unlock action at each atomic
                # layer; the later improvement pass still receives the broad
                # complete-turn generator.
                actions = (
                    self.bounded_diverse_moves(partial.board)
                    if self.verification_mode
                    else self.diverse_moves(partial.board)
                )
                if not actions:
                    completed.append(partial)
                    continue
                forced = all(move.name == "AutoCapture" for _, move in actions)
                forced_end = all(move.name == "Skip" for _, move in actions)
                for priority, move in actions:
                    self.check_time(False)
                    if (
                        move.name == "Skip"
                        and partial.board.turn_moves < min(2, self.max_actions)
                        and not forced
                        and not forced_end
                    ):
                        continue
                    if (
                        partial.board.turn_moves >= self.max_actions
                        and move.name not in ("Skip", "AutoCapture")
                    ):
                        continue
                    child = partial.board.copy()
                    child.push(move)
                    candidate = PartialTurn(
                        partial.moves + [move],
                        child,
                        partial.priority + priority,
                    )
                    if child.is_game_over() or child.turn != original_turn:
                        self.consider_early_root_fallback(
                            board, cache_key, candidate, original_turn
                        )
                    key = child.serialize()
                    incumbent = expanded.get(key)
                    if incumbent is not None:
                        self.complete_turns_deduplicated += 1
                    if incumbent is None or self._prefer_partial(
                        candidate, incumbent, original_turn
                    ):
                        expanded[key] = candidate

            next_frontier: List[PartialTurn] = []
            for partial in expanded.values():
                if partial.board.is_game_over() or partial.board.turn != original_turn:
                    completed.append(partial)
                    self.consider_early_root_fallback(
                        board, cache_key, partial, original_turn
                    )
                else:
                    next_frontier.append(partial)

            if len(next_frontier) > partial_width:
                self.exhaustive_within_horizon = False
                self.partial_turns_pruned += len(next_frontier) - partial_width
                next_frontier = self._round_robin_partials(
                    next_frontier, partial_width, original_turn, board
                )
            frontier = next_frontier

        self.complete_turns_generated += len(completed)
        unique: Dict[str, PartialTurn] = {}
        for partial in completed:
            key = partial.board.serialize()
            incumbent = unique.get(key)
            if incumbent is None or self._prefer_partial(
                partial, incumbent, original_turn
            ):
                unique[key] = partial
        self.complete_turns_deduplicated += len(completed) - len(unique)

        pool = list(unique.values())
        if len(pool) > evaluation_pool_width:
            self.exhaustive_within_horizon = False
            self.partial_turns_pruned += len(pool) - evaluation_pool_width
            pool = self._round_robin_partials(
                pool, evaluation_pool_width, original_turn, board
            )

        if self.root_key == cache_key and self.root_fallback is None:
            fallback_pool = sorted(
                pool,
                key=lambda partial: (
                    self.turn_action_classes(board, partial.moves)[1],
                    -partial.priority,
                    normalized_turn_key(partial.moves, original_turn),
                ),
            )
            fallback_options: List[TurnCandidate] = []
            for partial in fallback_pool[:12]:
                self.check_time(False)
                safety = self.assess_turn_safety(board, partial.board, original_turn)
                if safety.tactically_safe:
                    purpose = self.turn_purpose_breakdown(
                        board,
                        partial.board,
                        partial.moves,
                        original_turn,
                        retrospective=False,
                    )
                    action_purposes = self.action_purpose_labels(
                        board,
                        partial.moves,
                        original_turn,
                        retrospective=False,
                    )
                    if self.turn_number <= EARLY_GAME_LAST_TURN:
                        if not self.early_structure_allowed(
                            board,
                            partial.board,
                            partial.moves,
                            original_turn,
                            action_purposes,
                        ):
                            continue
                    fallback_options.append(
                        TurnCandidate(
                            partial.moves,
                            partial.board,
                            partial.priority,
                            self.quick_score(partial.board),
                            0.0,
                            True,
                            purpose["net_purpose_penalty"],
                            purpose["paratrooper_mission_penalty"],
                            action_purposes,
                            self.early_plan_score(action_purposes),
                            purpose["stagnation_progress"],
                            purpose["backfills"] + purpose["reversals"],
                            float(sum(move.name == "Skip" for move in partial.moves)),
                        )
                    )
            if fallback_options:
                # Judge the emergency line as a complete turn. A quiet setup
                # action may survive when the resulting turn has a lower net
                # waste cost than its alternatives; it is no longer rejected
                # merely because that one action lacked an immediate label.
                fallback_options.sort(
                    key=lambda candidate: (
                        candidate.purpose_penalty
                        + candidate.paratrooper_mission_penalty,
                        sum(
                            bool(
                                {"no_new_effect", "setup"}.intersection(
                                    item["roles"]
                                )
                            )
                            for item in candidate.action_purposes
                        ),
                        -candidate.early_plan_score,
                        -candidate.priority,
                        normalized_turn_key(candidate.moves, original_turn),
                    )
                )
                self.root_fallback = fallback_options[0]

        candidates = []
        for partial in pool:
            self.check_time(False)
            terminal = self.terminal_score(partial.board, 0)
            safety = self.assess_turn_safety(board, partial.board, original_turn)
            purpose = self.turn_purpose_breakdown(
                board,
                partial.board,
                partial.moves,
                original_turn,
                retrospective=False,
            )
            action_purposes = self.action_purpose_labels(
                board,
                partial.moves,
                original_turn,
                retrospective=False,
            )
            candidates.append(
                TurnCandidate(
                    partial.moves,
                    partial.board,
                    partial.priority,
                    self.heuristic_score(partial.board) if terminal is None else terminal,
                    max(0.0, safety.new_risk_value - safety.compensation_value),
                    safety.tactically_safe,
                    purpose["net_purpose_penalty"],
                    purpose["paratrooper_mission_penalty"],
                    action_purposes,
                    self.early_plan_score(action_purposes),
                    purpose["stagnation_progress"],
                    purpose["backfills"] + purpose["reversals"],
                    float(sum(move.name == "Skip" for move in partial.moves)),
                )
            )
            if not safety.tactically_safe:
                self.tactically_unsafe_turns += 1

        cleaned_by_position: Dict[str, TurnCandidate] = {}
        for candidate in list(candidates):
            self.check_time(False)
            cleaned = self.trim_filler_action(board, candidate, original_turn)
            if cleaned is None:
                continue
            key = cleaned.board.serialize()
            incumbent = cleaned_by_position.get(key)
            if incumbent is None or self.candidate_sort_key(
                cleaned, original_turn, is_root_generation
            ) < self.candidate_sort_key(
                incumbent, original_turn, is_root_generation
            ):
                cleaned_by_position[key] = cleaned
        if cleaned_by_position:
            candidates.extend(cleaned_by_position.values())
            self.purposeful_early_stops_generated += len(cleaned_by_position)
        candidates.sort(
            key=lambda item: self.candidate_sort_key(
                item, original_turn, is_root_generation
            )
        )

        if self.turn_number <= EARLY_GAME_LAST_TURN:
            structured = [
                candidate
                for candidate in candidates
                if self.early_structure_allowed(
                    board,
                    candidate.board,
                    candidate.moves,
                    original_turn,
                    candidate.action_purposes,
                )
            ]
            if structured:
                self.purpose_filtered_turns += len(candidates) - len(structured)
                if len(structured) != len(candidates):
                    self.exhaustive_within_horizon = False
                candidates = structured

        # Purpose is a lexicographic constraint at every phase, not merely a
        # small evaluation term. If a fully purposeful legal turn exists, a
        # square swap/rotation/filler action cannot crowd it out. If every turn
        # has filler, retain only those with the fewest such actions. Safety is
        # the higher-order constraint: a tidy losing turn must never cause a
        # messy HQ escape to be deleted before minimax sees it.
        if candidates:
            safety_pool = [
                candidate for candidate in candidates if candidate.tactically_safe
            ]
            purpose_pool = safety_pool if safety_pool else candidates
            no_effect_counts = [
                sum(
                    bool(
                        {"no_new_effect", "setup"}.intersection(item["roles"])
                    )
                    for item in candidate.action_purposes
                )
                for candidate in purpose_pool
            ]
            minimum_no_effect = min(no_effect_counts)
            permitted_no_effect = minimum_no_effect
            if is_root_generation and self.stagnation_factor() >= 0.30:
                permitted_no_effect += 1
            focused = [
                candidate
                for candidate, count in zip(purpose_pool, no_effect_counts)
                if count <= permitted_no_effect
            ]
            self.purpose_filtered_turns += len(candidates) - len(focused)
            if len(focused) != len(candidates):
                self.exhaustive_within_horizon = False
            candidates = focused

        # Static quality selects most branches; high-priority tactical turns
        # get a separate quota so setup/capture/extraction sequences survive a
        # temporarily inaccurate evaluator.
        selected = self.select_diverse_turns(board, candidates, turn_width)
        tactical = sorted(
            candidates,
            key=lambda item: (
                not item.tactically_safe,
                item.safety_penalty,
                -item.priority,
                normalized_turn_key(item.moves, original_turn),
            ),
        )
        for candidate in tactical:
            if len(selected) >= turn_capacity:
                break
            if (
                candidate not in selected
                and candidate.tactically_safe
                and candidate.priority >= 4000.0
            ):
                selected.append(candidate)

        # A superficially brilliant attack can be refuted while a quieter
        # save-and-extract plan survives. Preserve one representative of each
        # threatened-artillery destination, paratrooper destination, and their
        # combination so minimax—not static ordering—decides between them.
        plan_representatives: Dict[Tuple[Any, ...], TurnCandidate] = {}
        for candidate in candidates:
            for plan_key in self.turn_plan_keys(board, candidate.moves):
                if plan_key not in plan_representatives:
                    plan_representatives[plan_key] = candidate
        plan_items = sorted(
            plan_representatives.items(),
            key=lambda item: (
                0
                if item[0][0]
                in (
                    "hq_escape",
                    "hq_escape_unlock",
                    "hq_defense",
                    "capture_setup",
                    "unlock_and_extract",
                    "escape_and_extract",
                )
                else 1,
                self.candidate_sort_key(item[1], original_turn),
            ),
        )
        preserved_plans: List[TurnCandidate] = []
        plan_limit = max(1, min(3, self.beam_width // 2))
        for _, candidate in plan_items:
            if (
                candidate in preserved_plans
                or not candidate.tactically_safe
                or len(preserved_plans) >= plan_limit
            ):
                continue
            preserved_plans.append(candidate)
            if candidate in selected:
                continue
            if len(selected) < turn_capacity:
                selected.append(candidate)
                continue
            # The cap controls cost, but it may not erase a multi-action save
            # or extraction sequence. Replace the weakest generic slot.
            replace_index = next(
                (
                    index
                    for index in range(len(selected) - 1, -1, -1)
                    if selected[index] not in preserved_plans
                ),
                None,
            )
            if replace_index is not None:
                selected[replace_index] = candidate

        if len(selected) != len(candidates):
            self.exhaustive_within_horizon = False
            self.complete_turns_pruned += len(candidates) - len(selected)
        self.turn_cache[cache_key] = selected
        if self.root_key == cache_key and selected:
            self.root_fallback = selected[0]
        return selected

    def alphabeta(self, board: engine.BaseBoard, turns_left: int, alpha: float, beta: float) -> SearchResult:
        self.check_time()
        terminal = self.terminal_score(board, turns_left)
        if terminal is not None:
            return SearchResult(terminal, [])

        legal = list(board.generate_legal_moves())
        if not legal:
            return SearchResult(self.static_score(board), [])

        # Forced automatic captures are resolved even at the nominal horizon,
        # so static evaluation never counts a unit that is already certain to
        # disappear before another player action.
        if all(move.name == "AutoCapture" for move in legal):
            maximizing = board.turn == engine.RED
            best = SearchResult(-math.inf if maximizing else math.inf, [])
            for move in legal:
                child = board.copy()
                child.push(move)
                result = self.alphabeta(child, turns_left, alpha, beta)
                if (maximizing and result.score > best.score) or (not maximizing and result.score < best.score):
                    best = SearchResult(result.score, [move] + result.pv)
                if maximizing:
                    alpha = max(alpha, best.score)
                else:
                    beta = min(beta, best.score)
                if beta <= alpha:
                    break
            return best

        if turns_left <= 0:
            # GHQ has check-like threats even though the rules do not name a
            # check state.  A battery or infantry combination can finish a
            # turn with an HQ capture already legal for its next turn while
            # the defender receives exactly one chance to disrupt it.  A
            # plain depth-two leaf used to stop here and miss the forced mate.
            if self.has_same_turn_hq_capture(board):
                return SearchResult(
                    MATE_SCORE if board.turn == engine.RED else -MATE_SCORE,
                    [],
                )

            latent_attacker = not board.turn
            latent_probe = self.board_as_turn(board, latent_attacker)
            if (
                self.hq_leaf_extension_enabled
                and self.has_same_turn_hq_capture(latent_probe)
            ):
                evasions = self.generate_turn_candidates(board)
                if evasions:
                    maximizing = board.turn == engine.RED
                    best = SearchResult(
                        -math.inf if maximizing else math.inf,
                        [],
                    )
                    for turn in evasions:
                        self.check_time()
                        if self.has_same_turn_hq_capture(turn.board):
                            leaf_score = (
                                MATE_SCORE
                                if turn.board.turn == engine.RED
                                else -MATE_SCORE
                            )
                        else:
                            leaf_score = self.static_score(turn.board)
                        transition_penalty = (
                            turn.purpose_penalty
                            + turn.paratrooper_mission_penalty
                        )
                        candidate = (
                            leaf_score - transition_penalty
                            if maximizing
                            else leaf_score + transition_penalty
                        )
                        if (
                            maximizing and candidate > best.score
                        ) or (
                            not maximizing and candidate < best.score
                        ):
                            best = SearchResult(candidate, list(turn.moves))
                        if maximizing:
                            alpha = max(alpha, best.score)
                        else:
                            beta = min(beta, best.score)
                        if beta <= alpha:
                            break
                    return best
            return SearchResult(self.static_score(board), [])

        key = (board.serialize(), turns_left)
        cached = self.table.get(key)
        if cached is not None:
            self.transposition_hits += 1
            return SearchResult(cached.score, list(cached.pv))

        turns = self.generate_turn_candidates(board)
        if not turns:
            return SearchResult(self.static_score(board), [])

        maximizing = board.turn == engine.RED
        best = SearchResult(-math.inf if maximizing else math.inf, [])
        complete = True
        is_root = board.serialize() == self.root_key
        root_scores: List[Tuple[float, TurnCandidate]] = []
        for turn in turns:
            result = self.alphabeta(turn.board, turns_left - 1, alpha, beta)
            transition_penalty = turn.purpose_penalty + turn.paratrooper_mission_penalty
            early_bonus = (
                0.40 * turn.early_plan_score
                if self.turn_number <= EARLY_GAME_LAST_TURN
                else 0.0
            )
            turn_quality = early_bonus - transition_penalty
            if is_root and self.stagnation_factor() > 0.0:
                turn_quality += (
                    self.stagnation_factor()
                    * self.stagnation_value(turn)
                )
            candidate = (
                result.score + turn_quality
                if maximizing
                else result.score - turn_quality
            )
            if is_root:
                root_scores.append((candidate, turn))
                self.root_verified_lines.append(
                    (candidate, turn, list(result.pv))
                )
                self.root_ranked_turns = sorted(
                    root_scores,
                    key=lambda item: (
                        -item[0] if maximizing else item[0],
                        normalized_turn_key(item[1].moves, board.turn),
                    ),
                )
            if (maximizing and candidate > best.score) or (not maximizing and candidate < best.score):
                best = SearchResult(candidate, list(turn.moves) + result.pv)
            if maximizing:
                alpha = max(alpha, best.score)
            else:
                beta = min(beta, best.score)
            if beta <= alpha:
                complete = False
                break
        if complete:
            self.table[key] = SearchResult(best.score, list(best.pv))
            if is_root:
                self.root_ranked_turns = sorted(
                    root_scores,
                    key=lambda item: (
                        -item[0] if maximizing else item[0],
                        normalized_turn_key(item[1].moves, board.turn),
                    ),
                )
        return best


def purposeful_complete_turn_seed(
    board: engine.BaseBoard,
    personality: str,
    turn_number: int = 1,
    max_actions: int = 3,
) -> SearchResult:
    """Fast full-turn seed used before iterative search.

    This fallback must remain able to assemble a useful multi-action turn.
    A quiet first action can unlock pressure, protection, or mobility on the
    second action, so ``no_new_effect`` is a cost rather than an atomic-action
    veto. Hard tactical rules still apply.
    """
    working = board.copy()
    original_turn = working.turn
    moves: List[engine.Move] = []
    vacated: set[int] = set()
    fallback_rules = Searcher(
        personality,
        time_ms=60_000,
        beam_width=4,
        turn_number=turn_number,
        max_actions=max_actions,
    )
    while working.turn == original_turn and not working.is_game_over():
        legal = list(working.generate_legal_moves())
        if not legal:
            break
        candidates: List[Tuple[float, float, str, engine.Move]] = []
        for move in legal:
            piece_type = Searcher.move_piece_type(working, move)
            if move.name == "Skip" and working.turn_moves < max_actions:
                continue
            if (
                working.turn_moves >= max_actions
                and move.name not in ("Skip", "AutoCapture")
            ):
                continue
            if (
                move.name == "Reinforce"
                and piece_type == engine.AIRBORNE_INFANTRY
                and move.capture_preference is None
            ):
                # A deadline fallback may not commit the para merely to spend
                # an action. Preserve it until search proves a capture plan.
                continue
            if piece_type in ARTILLERY_TYPES and not fallback_rules.artillery_move_allowed(
                working, move
            ):
                continue
            if not fallback_rules.early_extension_allowed(working, move):
                continue
            if (
                piece_type == engine.AIRBORNE_INFANTRY
                and move.name != "Reinforce"
                and move.capture_preference is None
                and move.from_square is not None
                and move.to_square is not None
                and fallback_rules.home_distance(move.to_square, working.turn)
                >= fallback_rules.home_distance(move.from_square, working.turn)
            ):
                continue

            child = working.copy()
            child.push(move)
            candidate_moves = moves + [move]
            candidate_purposes = fallback_rules.action_purpose_labels(
                board,
                candidate_moves,
                original_turn,
                retrospective=False,
            )
            if not fallback_rules.early_structure_allowed(
                board,
                child,
                candidate_moves,
                original_turn,
                candidate_purposes,
            ):
                continue
            purpose_penalty = (
                1.4
                if candidate_purposes
                and "no_new_effect" in candidate_purposes[-1]["roles"]
                else 0.0
            )
            if (
                move.capture_preference is None
                and move.from_square is not None
                and move.to_square is not None
                and move.from_square != move.to_square
            ):
                pressure_gain = (
                    artillery_pressure(child, original_turn)
                    + fallback_rules.infantry_pressure(child, original_turn)
                    - artillery_pressure(working, original_turn)
                    - fallback_rules.infantry_pressure(working, original_turn)
                )
                protection_gain = (
                    fallback_rules.positional_risk(working, original_turn)
                    - fallback_rules.positional_risk(child, original_turn)
                )
                shape_gain = (
                    infantry_shape_score(child, original_turn)
                    - infantry_shape_score(working, original_turn)
                )
                concrete_purpose = (
                    pressure_gain > 0.05
                    or protection_gain > 0.10
                    or shape_gain > 0.10
                    or fallback_rules.unlocks_airborne_extraction(working, move)
                )
                lateral_or_backward = (
                    fallback_rules.home_distance(move.to_square, original_turn)
                    <= fallback_rules.home_distance(move.from_square, original_turn)
                )
                if (
                    (move.to_square in vacated or lateral_or_backward)
                    and piece_type != engine.AIRBORNE_INFANTRY
                    and not concrete_purpose
                ):
                    # Backfills, reversals, and quiet lateral actions are bad
                    # when they accomplish nothing, but making them illegal
                    # here caused the deadline fallback to reject every legal
                    # move and double-skip live self-play games. Preserve them
                    # as costly setup actions so a later action can supply the
                    # turn's net purpose.
                    purpose_penalty += 1.6 if move.to_square in vacated else 1.0
            red_score = quick_evaluation(child, turn_number)
            utility = red_score if original_turn == engine.RED else -red_score
            if turn_number <= EARLY_GAME_LAST_TURN:
                utility += 2.5 * fallback_rules.early_plan_score(
                    candidate_purposes
                )
            utility -= purpose_penalty
            candidates.append(
                (
                    utility,
                    -purpose_penalty,
                    normalized_move_uci(move, original_turn),
                    move,
                )
            )
        if not candidates:
            # Positional rules are preferences, not permission to return a
            # corrupt half-turn.  If the strict fallback filters every action,
            # retry ordinary moves with a large penalty while retaining the
            # hard paratrooper and opening-frontier prohibitions.  This is
            # especially important when search times out after one action:
            # the serialized position is mid-turn and must still be completed.
            for move in legal:
                piece_type = Searcher.move_piece_type(working, move)
                if move.name in ("Skip", "AutoCapture"):
                    continue
                if (
                    move.name == "Reinforce"
                    and piece_type == engine.AIRBORNE_INFANTRY
                    and move.capture_preference is None
                ):
                    continue
                if not fallback_rules.early_extension_allowed(working, move):
                    continue
                if (
                    piece_type == engine.AIRBORNE_INFANTRY
                    and move.name != "Reinforce"
                    and move.capture_preference is None
                    and move.from_square is not None
                    and move.to_square is not None
                    and fallback_rules.home_distance(move.to_square, working.turn)
                    >= fallback_rules.home_distance(move.from_square, working.turn)
                ):
                    continue
                child = working.copy()
                child.push(move)
                red_score = quick_evaluation(child, turn_number)
                utility = red_score if original_turn == engine.RED else -red_score
                candidate_purposes = fallback_rules.action_purpose_labels(
                    board,
                    moves + [move],
                    original_turn,
                    retrospective=False,
                )
                if turn_number <= EARLY_GAME_LAST_TURN:
                    utility += 2.5 * fallback_rules.early_plan_score(
                        candidate_purposes
                    )
                candidates.append(
                    (
                        utility - 8.0,
                        -8.0,
                        normalized_move_uci(move, original_turn),
                        move,
                    )
                )

        if not candidates:
            # Ending early is preferable to emitting an invalid partial turn.
            # The production engine decides whether Skip is legal here.
            skip = next((move for move in legal if move.name == "Skip"), None)
            if skip is None:
                break
            candidates.append((
                quick_evaluation(working, turn_number)
                if original_turn == engine.RED
                else -quick_evaluation(working, turn_number),
                -12.0,
                skip.uci(),
                skip,
            ))
        chosen = max(candidates, key=lambda item: (item[0], item[1], item[2]))[3]
        if (
            chosen.from_square is not None
            and chosen.to_square is not None
            and chosen.from_square != chosen.to_square
        ):
            vacated.add(chosen.from_square)
        moves.append(chosen)
        working.push(chosen)

    # The greedy seed is allowed to use a quiet action to unlock a later one,
    # but once the full turn exists we can distinguish that setup from a truly
    # disposable third action. Apply the same replay-and-skip cleanup used by
    # searched candidates so a deadline fallback cannot reintroduce filler.
    if (
        max_actions == 3
        and not working.is_game_over()
        and working.turn != original_turn
    ):
        action_purposes = fallback_rules.action_purpose_labels(
            board, moves, original_turn, retrospective=False
        )
        seed_candidate = TurnCandidate(
            moves=list(moves),
            board=working,
            priority=0.0,
            static_score=quick_evaluation(working, turn_number),
            action_purposes=action_purposes,
        )
        trimmed = fallback_rules.trim_filler_action(
            board, seed_candidate, original_turn
        )
        if trimmed is not None:
            moves = list(trimmed.moves)
            working = trimmed.board
    return SearchResult(quick_evaluation(working, turn_number), moves)


def first_turn_from_pv(board: engine.BaseBoard, pv: Sequence[engine.Move]) -> Tuple[List[engine.Move], engine.BaseBoard]:
    working = board.copy()
    original_turn = working.turn
    selected: List[engine.Move] = []
    for move in pv:
        if working.turn != original_turn:
            break
        selected.append(move)
        working.push(move)
    return selected, working


def rotate_opening_uci(uci: str) -> str:
    move = engine.Move.from_uci(uci)
    rotated = engine.Move(
        name=move.name,
        from_square=None if move.from_square is None else 63 - move.from_square,
        to_square=None if move.to_square is None else 63 - move.to_square,
        unit_type=move.unit_type,
        orientation=None if move.orientation is None else (move.orientation + 4) % 8,
        capture_preference=(
            None if move.capture_preference is None else 63 - move.capture_preference
        ),
        auto_capture_type=move.auto_capture_type,
    )
    return rotated.uci()


def normalized_opening_signature(
    board: engine.BaseBoard, color: bool
) -> Tuple[Tuple[Any, ...], Tuple[int, ...]]:
    pieces: List[Tuple[Any, ...]] = []
    for square in engine.scan_forward(board.occupied_co[color]):
        piece = board.piece_at(square)
        if piece is None:
            continue
        normalized_square = square if color == engine.RED else 63 - square
        orientation = piece.orientation
        if orientation is not None and color == engine.BLUE:
            orientation = (orientation + 4) % 8
        pieces.append((piece.piece_type, normalized_square, orientation))
    return tuple(sorted(pieces)), tuple(count for _, count in board.reserves[color])


def weighted_opening_choice(
    choices: Sequence[Tuple[TurnCandidate, int]], seed: int, position_key: str
) -> Optional[TurnCandidate]:
    if not choices:
        return None
    picker = random.Random(f"ghq-opening-v1:{seed}:{position_key}")
    weights = [math.sqrt(count) for _, count in choices]
    return picker.choices([candidate for candidate, _ in choices], weights=weights, k=1)[0]


def opening_book_turn(
    board: engine.BaseBoard,
    turn_number: int,
    searcher: Searcher,
    opening_seed: int = 0,
) -> Optional[TurnCandidate]:
    """Sample a recent human opening only when it remains legal and safe."""
    key = (board.turn, turn_number)
    if turn_number not in (1, 2, 3, 4):
        return None
    anchors = OPENING_BOOK_ANCHORS.get(key)
    if anchors is not None:
        for square_name, piece_type in anchors:
            square = engine.parse_square(square_name)
            if (
                board.piece_type_at(square) != piece_type
                or not (engine.BB_SQUARES[square] & board.occupied_co[board.turn])
            ):
                return None

    if turn_number in (1, 2):
        raw_choices = OPENING_FIRST_TURNS
    else:
        signature = normalized_opening_signature(board, board.turn)
        signature_key = next(
            (name for name, expected in OPENING_SIGNATURE_KEYS.items() if signature == expected),
            None,
        )
        if signature_key is None:
            return None
        raw_choices = OPENING_CONTINUATIONS[signature_key]

    valid: List[Tuple[TurnCandidate, int]] = []
    mover = board.turn
    for normalized_ucis, count in raw_choices:
        ucis = (
            normalized_ucis
            if mover == engine.RED
            else tuple(rotate_opening_uci(uci) for uci in normalized_ucis)
        )
        planned_ucis = ucis[: searcher.max_actions]
        working = board.copy()
        moves: List[engine.Move] = []
        for uci in planned_ucis:
            legal = {move.uci(): move for move in working.generate_legal_moves()}
            move = legal.get(uci)
            if move is None:
                break
            moves.append(move)
            working.push(move)
        if len(moves) != len(planned_ucis):
            continue
        if not working.is_game_over() and working.turn == mover:
            skip = next(
                (move for move in working.generate_legal_moves() if move.name == "Skip"),
                None,
            )
            if skip is None:
                continue
            moves.append(skip)
            working.push(skip)
        if not working.is_game_over() and working.turn == mover:
            continue
        try:
            safety = searcher.assess_turn_safety(board, working, mover)
        except SearchTimeout:
            continue
        if not safety.tactically_safe:
            continue
        purpose = searcher.turn_purpose_breakdown(
            board, working, moves, mover, retrospective=False
        )
        action_purposes = searcher.action_purpose_labels(
            board, moves, mover, retrospective=False
        )
        valid.append((TurnCandidate(
            moves=moves,
            board=working,
            priority=0.0,
            static_score=searcher.quick_score(working),
            safety_penalty=0.0,
            tactically_safe=True,
            purpose_penalty=purpose["net_purpose_penalty"],
            paratrooper_mission_penalty=purpose["paratrooper_mission_penalty"],
            action_purposes=action_purposes,
            early_plan_score=searcher.early_plan_score(action_purposes),
        ), count))
    return weighted_opening_choice(valid, opening_seed, board.serialize())


def search(
    board: engine.BaseBoard,
    personality: str,
    time_ms: int,
    max_depth: int,
    beam_width: int,
    turn_number: int = 1,
    value_function: Optional[Any] = None,
    opening_seed: int = 0,
    max_actions: int = 3,
    stagnation_turns: int = 0,
) -> Dict[str, Any]:
    started = time.monotonic()
    searcher = Searcher(
        personality,
        time_ms,
        beam_width,
        turn_number=turn_number,
        value_function=value_function,
        max_actions=max_actions,
        stagnation_turns=stagnation_turns,
    )
    searcher.root_key = board.serialize()
    best: Optional[SearchResult] = None
    completed_depth = 0
    timed_out = False
    fallback_kind = "none"
    emergency_seed: Optional[SearchResult] = None
    verified_seed: Optional[SearchResult] = None
    book_turn = opening_book_turn(board, turn_number, searcher, opening_seed)
    opening_book_used = book_turn is not None
    if book_turn is not None:
        transition_penalty = book_turn.purpose_penalty + book_turn.paratrooper_mission_penalty
        early_bonus = (
            0.40 * book_turn.early_plan_score
            if turn_number <= EARLY_GAME_LAST_TURN
            else 0.0
        )
        turn_quality = early_bonus - transition_penalty
        score = (
            book_turn.static_score + turn_quality
            if board.turn == engine.RED
            else book_turn.static_score - turn_quality
        )
        best = SearchResult(score, list(book_turn.moves))
    else:
        # Establish a legal, purposeful full turn before spending the budget on
        # minimax. If iterative search expires, we still return three counted
        # actions (unless the game ended) rather than inventing a late greedy
        # line or stopping after two actions.
        emergency_seed = purposeful_complete_turn_seed(
            board, personality, turn_number, max_actions=max_actions
        )
        seed_moves, seed_board = first_turn_from_pv(board, emergency_seed.pv)
        seed_purpose = searcher.turn_purpose_breakdown(
            board,
            seed_board,
            seed_moves,
            board.turn,
            retrospective=False,
        )
        seed_penalty = seed_purpose["total_penalty"]
        emergency_seed.score += (
            -seed_penalty if board.turn == engine.RED else seed_penalty
        )
        seed_counted_actions = sum(
            move.name not in ("AutoCapture", "Skip") for move in seed_moves
        )
        if seed_board.is_game_over() or seed_counted_actions >= searcher.max_actions:
            try:
                seed_safety = searcher.assess_turn_safety(
                    board, seed_board, board.turn
                )
            except SearchTimeout:
                seed_safety = None
            if seed_safety is not None and seed_safety.tactically_safe:
                seed_action_purposes = searcher.action_purpose_labels(
                    board, seed_moves, board.turn, retrospective=False
                )
                searcher.root_fallback = TurnCandidate(
                    seed_moves,
                    seed_board,
                    0.0,
                    searcher.quick_score(seed_board),
                    max(
                        0.0,
                        seed_safety.new_risk_value
                        - seed_safety.compensation_value,
                    ),
                    True,
                    seed_purpose["net_purpose_penalty"],
                    seed_purpose["paratrooper_mission_penalty"],
                    seed_action_purposes,
                    searcher.early_plan_score(seed_action_purposes),
                )
        requested_depth = max(1, max_depth)
        final_deadline = searcher.deadline
        if requested_depth >= 2:
            searcher.hq_leaf_extension_enabled = True
            # First verify the purposeful emergency turn against one complete
            # opponent reply. This produces a tactically checked floor even if
            # enumerating alternative root turns later exhausts the budget.
            # It is still labelled a fallback because root alternatives were
            # not all compared at the same horizon.
            if seed_board.is_game_over():
                verified_seed = SearchResult(emergency_seed.score, list(seed_moves))
            elif seed_board.turn != board.turn:
                searcher.verification_mode = True
                searcher.deadline = min(
                    final_deadline,
                    started + max(0.05, time_ms / 1000.0 * 0.12),
                )
                try:
                    reply = searcher.alphabeta(
                        seed_board, 1, -math.inf, math.inf
                    )
                    early_bonus = (
                        0.40 * searcher.early_plan_score(
                            searcher.action_purpose_labels(
                                board,
                                seed_moves,
                                board.turn,
                                retrospective=False,
                            )
                        )
                        if turn_number <= EARLY_GAME_LAST_TURN
                        else 0.0
                    )
                    seed_quality = early_bonus - seed_penalty
                    verified_seed = SearchResult(
                        reply.score
                        + (seed_quality if board.turn == engine.RED else -seed_quality),
                        list(seed_moves) + list(reply.pv),
                    )
                except SearchTimeout:
                    timed_out = True
                finally:
                    searcher.deadline = final_deadline
                    # A completed seed reply is a valid depth-one
                    # transposition. Reuse it when the narrow root pass reaches
                    # that same child instead of paying for the reply twice.

            # Reserve the first 90% of the budget for a narrow but complete
            # depth-two pass. Depth two means our complete turn plus one full
            # opponent reply. A later broad pass may improve it, but may never
            # erase this tactically verified result merely by timing out.
            searcher.verification_mode = True
            searcher.root_verified_lines = []
            searcher.root_ranked_turns = []
            searcher.deadline = min(
                final_deadline,
                started + max(0.05, time_ms / 1000.0 * 0.90),
            )
            try:
                best = searcher.alphabeta(board, 2, -math.inf, math.inf)
                completed_depth = 2
            except SearchTimeout:
                timed_out = True
                partial_verified = searcher.best_verified_root_result(board.turn)
                if partial_verified is not None:
                    best = partial_verified
                    completed_depth = 2
                    fallback_kind = "safe"
            finally:
                searcher.deadline = final_deadline

            if completed_depth >= 2 and abs(best.score if best else 0.0) < MATE_SCORE:
                # Widen only after reply verification. Keep the verified PV and
                # ranked candidates if this improvement pass expires.
                searcher.verification_mode = False
                searcher.table.clear()
                searcher.turn_cache.clear()
                for depth in range(2, requested_depth + 1):
                    try:
                        iteration = searcher.alphabeta(
                            board, depth, -math.inf, math.inf
                        )
                    except SearchTimeout:
                        timed_out = True
                        break
                    best = iteration
                    completed_depth = depth
                    if abs(iteration.score) >= MATE_SCORE:
                        break
            elif completed_depth < 2:
                if verified_seed is not None:
                    best = verified_seed
                    completed_depth = 2
                    fallback_kind = "safe"
                else:
                    # Root generation often completed even when a reply did
                    # not. Finish a stable depth-one result only when no
                    # reply-verified floor exists.
                    searcher.verification_mode = False
                    try:
                        best = searcher.alphabeta(board, 1, -math.inf, math.inf)
                        completed_depth = 1
                    except SearchTimeout:
                        timed_out = True
        else:
            try:
                best = searcher.alphabeta(board, 1, -math.inf, math.inf)
                completed_depth = 1
            except SearchTimeout:
                timed_out = True

    if best is None:
        if verified_seed is not None:
            best = verified_seed
            completed_depth = 2
            fallback_kind = "safe"
        elif searcher.root_fallback is not None:
            fallback_penalty = (
                searcher.root_fallback.purpose_penalty
                + searcher.root_fallback.paratrooper_mission_penalty
            )
            fallback_bonus = (
                0.40 * searcher.root_fallback.early_plan_score
                if turn_number <= EARLY_GAME_LAST_TURN
                else 0.0
            )
            fallback_quality = fallback_bonus - fallback_penalty
            fallback_score = searcher.root_fallback.static_score
            fallback_score += (
                fallback_quality if board.turn == engine.RED else -fallback_quality
            )
            best = SearchResult(
                fallback_score,
                list(searcher.root_fallback.moves),
            )
            fallback_kind = "safe"
        else:
            best = emergency_seed or purposeful_complete_turn_seed(
                board, personality, turn_number, max_actions=max_actions
            )
            fallback_kind = "seeded"
    first_turn, resulting_board = first_turn_from_pv(board, best.pv)
    if not first_turn and not board.is_game_over():
        fallback = purposeful_complete_turn_seed(
            board, personality, turn_number, max_actions=max_actions
        )
        first_turn, resulting_board = first_turn_from_pv(board, fallback.pv)
        best = fallback
        seed_purpose = searcher.turn_purpose_breakdown(
            board,
            resulting_board,
            first_turn,
            board.turn,
            retrospective=False,
        )
        seed_penalty = seed_purpose["total_penalty"]
        best.score += -seed_penalty if board.turn == engine.RED else seed_penalty
        fallback_kind = "seeded"

    elapsed_ms = (time.monotonic() - started) * 1000.0
    automatic = [move.uci() for move in first_turn if move.name == "AutoCapture"]
    actions = [move.uci() for move in first_turn if move.name != "AutoCapture"]
    root_eval = evaluation_breakdown(board, personality, turn_number)
    resulting_eval = evaluation_breakdown(resulting_board, personality, turn_number + 1)
    current_player_score = best.score if board.turn == engine.RED else -best.score
    ranked_root_turns = list(searcher.root_ranked_turns)
    if not ranked_root_turns:
        shallow_turns = searcher.turn_cache.get(searcher.root_key or "", [])
        for turn in shallow_turns:
            transition_penalty = (
                turn.purpose_penalty + turn.paratrooper_mission_penalty
            )
            early_bonus = (
                0.40 * turn.early_plan_score
                if turn_number <= EARLY_GAME_LAST_TURN
                else 0.0
            )
            quality = early_bonus - transition_penalty
            red_score = (
                turn.static_score + quality
                if board.turn == engine.RED
                else turn.static_score - quality
            )
            ranked_root_turns.append((red_score, turn))
        ranked_root_turns.sort(
            key=lambda item: (
                -item[0] if board.turn == engine.RED else item[0],
                normalized_turn_key(item[1].moves, board.turn),
            )
        )

    selected_move_key = tuple(move.uci() for move in first_turn)
    if not any(
        tuple(move.uci() for move in turn.moves) == selected_move_key
        for _, turn in ranked_root_turns
    ):
        # A later improvement pass may time out after the reply-verified pass
        # has already supplied the move we return.  In that case
        # ``root_ranked_turns`` can describe the interrupted pass and omit the
        # actual recommendation.  Preserve the invariant that candidate
        # telemetry always contains the selected complete turn.
        selected_candidate = next(
            (
                turn
                for _, turn, _ in searcher.root_verified_lines
                if tuple(move.uci() for move in turn.moves) == selected_move_key
            ),
            None,
        )
        if selected_candidate is None:
            selected_candidate = next(
                (
                    turn
                    for turn in searcher.turn_cache.get(searcher.root_key or "", [])
                    if tuple(move.uci() for move in turn.moves) == selected_move_key
                ),
                None,
            )
        if selected_candidate is None:
            action_purposes = searcher.action_purpose_labels(
                board, first_turn, board.turn
            )
            purpose = searcher.turn_purpose_breakdown(
                board, resulting_board, first_turn, board.turn
            )
            selected_candidate = TurnCandidate(
                list(first_turn),
                resulting_board,
                0.0,
                best.score,
                action_purposes=action_purposes,
                purpose_penalty=purpose["net_purpose_penalty"],
                paratrooper_mission_penalty=purpose[
                    "paratrooper_mission_penalty"
                ],
                early_plan_score=searcher.early_plan_score(action_purposes),
                progress_score=purpose["stagnation_progress"],
                conveyor_actions=purpose["backfills"] + purpose["reversals"],
                skip_actions=float(sum(move.name == "Skip" for move in first_turn)),
            )
        ranked_root_turns.insert(0, (best.score, selected_candidate))

    candidate_turns = []
    seen_candidate_moves = set()
    for red_score, candidate_turn in ranked_root_turns[:8]:
        all_moves = tuple(move.uci() for move in candidate_turn.moves)
        if all_moves in seen_candidate_moves:
            continue
        seen_candidate_moves.add(all_moves)
        candidate_turns.append(
            {
                "rank": len(candidate_turns) + 1,
                "automatic_captures": [
                    move.uci()
                    for move in candidate_turn.moves
                    if move.name == "AutoCapture"
                ],
                "actions": [
                    move.uci()
                    for move in candidate_turn.moves
                    if move.name != "AutoCapture"
                ],
                "all_moves": list(all_moves),
                "resulting_fen": candidate_turn.board.board_fen(),
                "score": round(
                    red_score if board.turn == engine.RED else -red_score, 4
                ),
                "action_purposes": candidate_turn.action_purposes,
                "purpose": {
                    key: round(value, 4)
                    for key, value in searcher.turn_purpose_breakdown(
                        board,
                        candidate_turn.board,
                        candidate_turn.moves,
                        board.turn,
                    ).items()
                },
            }
        )
    exhaustive = (
        searcher.exhaustive_within_horizon
        and not timed_out
        and completed_depth == max(1, max_depth)
    )
    recommendation_label = (
        "opening book"
        if opening_book_used
        else "best move"
        if exhaustive
        else "safe fallback"
        if fallback_kind == "safe"
        else "complete-turn seed"
        if fallback_kind == "seeded"
        else "best found"
    )
    return {
        "recommendation_label": recommendation_label,
        "input_fen": board.board_fen(),
        "side_to_move": color_name(board.turn),
        "best_turn": {
            "automatic_captures": automatic,
            "actions": actions,
            "all_moves": [move.uci() for move in first_turn],
            "resulting_fen": resulting_board.board_fen(),
            "action_purposes": searcher.action_purpose_labels(
                board, first_turn, board.turn
            ),
            "purpose": {
                key: round(value, 4)
                for key, value in searcher.turn_purpose_breakdown(
                    board, resulting_board, first_turn, board.turn
                ).items()
            },
        },
        "principal_variation": [move.uci() for move in best.pv],
        "candidate_turns": candidate_turns,
        "score": {
            "current_player": round(current_player_score, 4),
            "red": round(best.score, 4),
        },
        "search": {
            "completed_depth_in_turns": completed_depth,
            "requested_depth_in_turns": max_depth,
            "base_complete_turn_width": beam_width,
            "max_actions": searcher.max_actions,
            "nodes": searcher.nodes,
            "elapsed_ms": round(elapsed_ms, 2),
            "timed_out": timed_out,
            "fallback_used": fallback_kind,
            "opening_book_used": opening_book_used,
            "early_game_focus": turn_number <= EARLY_GAME_LAST_TURN,
            "approximate": True,
            "exhaustive_within_requested_horizon": exhaustive,
            "rule_filtered_actions": searcher.rule_filtered_actions,
            "beam_pruned_actions": searcher.beam_pruned_actions,
            "partial_turns_pruned": searcher.partial_turns_pruned,
            "complete_turns_generated": searcher.complete_turns_generated,
            "complete_turns_deduplicated": searcher.complete_turns_deduplicated,
            "complete_turns_pruned": searcher.complete_turns_pruned,
            "purposeful_early_stops_generated": searcher.purposeful_early_stops_generated,
            "tactically_unsafe_turns": searcher.tactically_unsafe_turns,
            "rotation_quota_pruned": searcher.rotation_quota_pruned,
            "purpose_filtered_turns": searcher.purpose_filtered_turns,
            "value_model_evaluations": searcher.value_model_evaluations,
            "turn_cache_hits": searcher.turn_cache_hits,
            "transposition_hits": searcher.transposition_hits,
        },
        "evaluation": {
            "before": root_eval,
            "after_best_turn": resulting_eval,
        },
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fen", default=engine.STARTING_FEN, help="GHQ FEN (defaults to the starting position)")
    parser.add_argument("--time-ms", type=int, default=2000, help="search budget in milliseconds")
    parser.add_argument("--max-depth", type=int, default=2, help="maximum search depth in complete player turns")
    parser.add_argument(
        "--beam-width",
        type=int,
        default=12,
        help="base width used for complete-turn candidate generation",
    )
    parser.add_argument(
        "--max-actions",
        type=int,
        choices=(2, 3),
        default=3,
        help="maximum voluntary actions before ending the turn",
    )
    parser.add_argument("--personality", choices=sorted(PERSONALITIES), default="balanced")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        board = engine.BaseBoard(args.fen)
        result = search(
            board,
            args.personality,
            args.time_ms,
            args.max_depth,
            args.beam_width,
            max_actions=args.max_actions,
        )
    except (ValueError, AssertionError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
