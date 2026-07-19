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
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

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
POLICY_SCORE_WEIGHT = 3.0
MISSIONLESS_PARATROOPER_PENALTY = 9.0
PARATROOPER_VALUABLE_TARGET_VALUE = PIECE_VALUES[engine.ARTILLERY]
PARATROOPER_MULTI_TARGET_COUNT = 2
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


BoardKey = Tuple[int, ...]


def board_key(board: engine.BaseBoard) -> BoardKey:
    """Exact uncompressed state key for hot in-search caches."""
    return (
        board.occupied,
        board.infantry,
        board.armored_infantry,
        board.airborne_infantry,
        board.artillery,
        board.armored_artillery,
        board.heavy_artillery,
        board.hq,
        *board.occupied_co,
        *board.bombarded_co,
        *board.adjacent_infantry_squares_co,
        board.orientation_bit0,
        board.orientation_bit1,
        board.orientation_bit2,
        board.turn_pieces,
        board.free_capture_clusters,
        board.free_capture_enemies,
        board.free_capture_num_allowed,
        int(board.turn),
        board.turn_moves,
        board.turn_auto_moves,
        *board.reserves[0].to_ints(),
        *board.reserves[1].to_ints(),
        int(board.did_offer_draw),
        int(board.did_accept_draw),
    )


CHEBYSHEV_DISTANCES = tuple(
    tuple(
        max(abs((left & 7) - (right & 7)), abs((left >> 3) - (right >> 3)))
        for right in range(64)
    )
    for left in range(64)
)


def chebyshev(a: int, b: int) -> int:
    # This sits on the hottest tactical-safety path (millions of calls per
    # search). Precomputing the engine's ordinary 0..63 rank-major distance
    # avoids repeated coordinate extraction while preserving exactly the
    # engine.square_distance result.
    return CHEBYSHEV_DISTANCES[a][b]


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


def artillery_forced_response_burden(
    board: engine.BaseBoard, color: bool
) -> float:
    """Value a lane that makes the opponent answer multiple valuable threats.

    Minimax normally sees the opponent move both targets and then evaluates a
    quiet leaf, losing the tempo value of the fork. Preserve a bounded bonus
    for the turn that created the fork. One attacked piece is ordinary
    pressure; two or more distinct non-HQ targets consume scarce actions and
    can herd valuable material into a shrinking pocket.
    """
    targets = (
        board.get_bombarded_squares(color)
        & board.occupied_co[not color]
        & ~board.hq
    )
    values = [
        PIECE_VALUES.get(board.piece_type_at(square), 0.0)
        for square in squares(targets)
    ]
    if len(values) < 2:
        return 0.0
    return 0.25 * sum(values) + 0.50 * (len(values) - 1)


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
        policy_function: Optional[Any] = None,
        max_actions: int = 3,
        stagnation_turns: int = 0,
    ) -> None:
        self.personality = personality
        self.time_ms = max(1, time_ms)
        self.turn_number = max(1, turn_number)
        self.value_function = value_function
        self.policy_function = policy_function
        self.max_actions = max(2, min(3, max_actions))
        self.stagnation_turns = max(0, stagnation_turns)
        self.deadline = time.monotonic() + max(1, time_ms) / 1000.0
        self.beam_width = max(1, beam_width)
        self.nodes = 0
        self.table: Dict[Tuple[BoardKey, int], SearchResult] = {}
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
        self.policy_model_evaluations = 0
        self.turn_cache: Dict[BoardKey, List[TurnCandidate]] = {}
        self.value_cache: Dict[str, float] = {}
        self.policy_cache: Dict[Tuple[str, bool], float] = {}
        self.safety_cache: Dict[Tuple[BoardKey, bool], Tuple[float, float, float]] = {}
        self.immediate_hq_capture_cache: Dict[BoardKey, bool] = {}
        self.exact_same_turn_hq_capture_cache: Dict[Tuple[BoardKey, bool], bool] = {}
        self.hq_survival_probe_nodes = 0
        self.hq_survival_reply_nodes = 0
        self.same_turn_hq_capture_cache: Dict[BoardKey, bool] = {}
        self.hq_capture_unlock_move_cache: Dict[Tuple[BoardKey, str], bool] = {}
        self.hq_defense_move_cache: Dict[Tuple[BoardKey, str], bool] = {}
        self.hq_defense_unlock_move_cache: Dict[Tuple[BoardKey, str], bool] = {}
        self.atomic_hq_defense_cache: Dict[BoardKey, bool] = {}
        self.hq_escape_unlock_move_cache: Dict[Tuple[BoardKey, str], bool] = {}
        self.capture_setup_move_cache: Dict[Tuple[BoardKey, str], bool] = {}
        self.followup_capture_value_cache: Dict[Tuple[BoardKey, str], float] = {}
        self.root_key: Optional[BoardKey] = None
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
        """Return new captures unlocked for the remainder of this turn.

        Merely threatening a piece from the landing square on a future turn is
        not a concrete mission. Count only targets that another remaining
        action can legally capture now and that were not already capturable
        before committing the paratrooper.
        """
        if not self.is_paradrop(board, move) or move.to_square is None:
            return {}
        mover = board.turn
        before_targets = {
            candidate.capture_preference
            for candidate in board.generate_legal_moves()
            if candidate.capture_preference is not None
        }
        child = board.copy()
        child.push(move)
        if child.is_game_over() or child.turn != mover:
            return {}
        targets: Dict[int, float] = {}
        for candidate in child.generate_legal_moves():
            target = candidate.capture_preference
            if target is not None and target not in before_targets:
                targets[target] = PIECE_VALUES.get(
                    child.piece_type_at(target), 0.0
                )
        return targets

    def paradrop_mission_targets(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> Dict[int, float]:
        """Return the captured target plus concrete same-turn follow-up captures."""
        targets: Dict[int, float] = {}
        if move.capture_preference is not None:
            target = move.capture_preference
            targets[target] = PIECE_VALUES.get(board.piece_type_at(target), 0.0)
        for target, value in self.paradrop_capture_targets(board, move).items():
            targets[target] = max(value, targets.get(target, 0.0))
        return targets

    def paradrop_allowed(self, board: engine.BaseBoard, move: engine.Move) -> bool:
        """Require a concrete mission before committing the unique paratrooper.

        An even trade for one ordinary gun is strategically poor because it
        gives up the continuing deterrent.  A commitment must instead expose
        multiple valuable targets or participate in an HQ combination. A
        plausible first step home does not justify a single-target mission.
        """
        if not self.is_paradrop(board, move):
            return True
        targets = self.paradrop_mission_targets(board, move)
        if not targets:
            return False
        if any(
            board.piece_type_at(target) == engine.HQ for target in targets
        ):
            return True
        if self.unlocks_immediate_hq_capture(board, move):
            return True
        valuable = [
            value
            for value in targets.values()
            if value >= PARATROOPER_VALUABLE_TARGET_VALUE
        ]
        if len(valuable) >= PARATROOPER_MULTI_TARGET_COUNT:
            return True
        return False

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
        """Flag a paradrop unless its concrete combination is executed.

        Making another capture legal is only a promise. The completed turn
        must actually take one of the targets unlocked by the landing (or
        capture the HQ); otherwise the para was consumed for a single trade.
        The narrow non-capturing exception is an HQ bodyguard mission: the
        same turn relocates the HQ, lands the para beside it, and removes an
        immediate enemy HQ-capture combination.
        """
        working = before.copy()
        penalty = 0.0
        hq_defense_mission = self.paratrooper_hq_defense_mission(
            before, after, moves, mover
        )
        for index, move in enumerate(moves):
            if self.is_paradrop(working, move):
                direct_hq_capture = (
                    move.capture_preference is not None
                    and working.piece_type_at(move.capture_preference)
                    == engine.HQ
                )
                unlocked_targets = set(
                    self.paradrop_capture_targets(working, move)
                )
                converted_target = any(
                    later.capture_preference in unlocked_targets
                    for later in moves[index + 1 :]
                    if later.capture_preference is not None
                )
                if not (
                    hq_defense_mission
                    or (
                        self.paradrop_allowed(working, move)
                        and (direct_hq_capture or converted_target)
                    )
                ):
                    penalty += MISSIONLESS_PARATROOPER_PENALTY
            working.push(move)
        return penalty

    def paratrooper_hq_defense_mission(
        self,
        before: engine.BaseBoard,
        after: engine.BaseBoard,
        moves: Sequence[engine.Move],
        mover: bool,
    ) -> bool:
        """Whether a non-capturing paradrop is necessary to evade HQ capture.

        Geometry alone is not enough. A live game required the para to land
        two squares from a relocated HQ, where it interdicted the final capture
        lane.  Certify the mission causally: the complete turn must remove an
        existing same-turn HQ capture, and omitting the paradrop must restore
        that capture (or make the remaining defense sequence illegal).
        """
        if after.turn == mover or after.is_game_over():
            return False
        if not before.pieces(engine.HQ, mover) or not after.pieces(engine.HQ, mover):
            return False

        working = before.copy()
        para_indexes: List[int] = []
        for index, move in enumerate(moves):
            if (
                self.is_paradrop(working, move)
                and move.capture_preference is None
            ):
                para_indexes.append(index)
            working.push(move)
        if not para_indexes:
            return False

        latent_attack = self.board_as_turn(before, not mover)
        if (
            not self.has_same_turn_hq_capture(latent_attack)
            or self.has_same_turn_hq_capture(after)
        ):
            return False

        for omitted_index in para_indexes:
            without_para = before.copy()
            legal_without_para = True
            for index, move in enumerate(moves):
                if index == omitted_index:
                    continue
                if not without_para.is_legal(move):
                    legal_without_para = False
                    break
                without_para.push(move)
            if not legal_without_para:
                return True
            if without_para.turn == mover and not without_para.is_game_over():
                skip = engine.Move.from_uci("skip")
                if not without_para.is_legal(skip):
                    return True
                without_para.push(skip)
            if (
                not without_para.is_game_over()
                and self.has_same_turn_hq_capture(
                    self.board_as_turn(without_para, not mover)
                )
            ):
                return True
        return False

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
            resolves_hq = (
                include_tactical_roles and self.resolves_hq_threat(working, move)
            )
            if resolves_hq:
                roles.append("hq_defense")
            elif include_tactical_roles and self.unlocks_hq_defense(working, move):
                roles.append("hq_defense_unlock")
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
            hq_encirclement_gain = (
                self.hq_encirclement_pressure(child, mover)
                - self.hq_encirclement_pressure(working, mover)
            )
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
                and max(contact_gain, hq_approach_gain, hq_encirclement_gain) > 0.0
            ):
                roles.append("advance")
            if move.name == "Skip":
                roles.append("end_turn")
            if not roles:
                roles.append("no_new_effect")
            result.append({"move": move.uci(), "roles": list(dict.fromkeys(roles))})
            working = child

        if self.paratrooper_hq_defense_mission(before, working, moves, mover):
            replay = before.copy()
            for index, move in enumerate(moves):
                if self.is_paradrop(replay, move):
                    result[index]["roles"] = list(
                        dict.fromkeys([*result[index]["roles"], "hq_defense"])
                    )
                replay.push(move)

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
        include_tactical_roles: bool = True,
    ) -> Dict[str, float]:
        """Measure what a full turn changed, beyond merely spending actions."""
        action_purposes = self.action_purpose_labels(
            before,
            moves,
            mover,
            retrospective=retrospective,
            include_tactical_roles=include_tactical_roles,
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
        hq_encirclement_gain = max(
            0.0,
            self.hq_encirclement_pressure(after, mover)
            - self.hq_encirclement_pressure(before, mover),
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
            + 0.6 * hq_encirclement_gain
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
            + 1.5 * hq_encirclement_gain
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
            "hq_encirclement_gain": hq_encirclement_gain,
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

    def deadline_safe_action_purpose_labels(
        self,
        before: engine.BaseBoard,
        moves: Sequence[engine.Move],
        mover: bool,
        retrospective: bool = True,
    ) -> List[Dict[str, Any]]:
        """Keep response telemetry from invalidating a completed search."""
        try:
            return self.action_purpose_labels(
                before, moves, mover, retrospective=retrospective
            )
        except SearchTimeout:
            return self.action_purpose_labels(
                before,
                moves,
                mover,
                retrospective=retrospective,
                include_tactical_roles=False,
            )

    def deadline_safe_turn_purpose_breakdown(
        self,
        before: engine.BaseBoard,
        after: engine.BaseBoard,
        moves: Sequence[engine.Move],
        mover: bool,
        retrospective: bool = True,
    ) -> Dict[str, float]:
        """Return bounded purpose telemetry even after the search deadline."""
        try:
            return self.turn_purpose_breakdown(
                before,
                after,
                moves,
                mover,
                retrospective=retrospective,
            )
        except SearchTimeout:
            return self.turn_purpose_breakdown(
                before,
                after,
                moves,
                mover,
                retrospective=retrospective,
                include_tactical_roles=False,
            )

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

    def transition_policy_score(
        self, board: engine.BaseBoard, mover: bool
    ) -> float:
        """Return mover-positive strategic guidance for one complete turn.

        Multi-target artillery pressure is retained across the opponent reply
        because forcing two escapes has tempo value even when minimax reaches
        a quiet leaf. The learned policy, when present, supplies the remaining
        bounded adjustment.
        """
        forcing_bonus = artillery_forced_response_burden(board, mover)
        if self.policy_function is None:
            return forcing_bonus
        key = (board.board_fen(), mover)
        cached = self.policy_cache.get(key)
        if cached is not None:
            return cached
        self.check_time(False)
        adjustment = float(
            self.policy_function(key[0], self.turn_number, mover)
        )
        if not math.isfinite(adjustment):
            raise ValueError("policy function returned a non-finite score")
        # Policy fits use calibrated-logit units. Bound their influence below
        # concrete tactical and mate terms while preserving strategic ordering.
        score = forcing_bonus + POLICY_SCORE_WEIGHT * max(
            -3.0, min(3.0, adjustment)
        )
        self.policy_cache[key] = score
        self.policy_model_evaluations += 1
        return score

    def deadline_safe_transition_policy_score(
        self, board: engine.BaseBoard, mover: bool
    ) -> float:
        """Use available policy guidance without invalidating a fallback.

        Search-time policy evaluation deliberately raises ``SearchTimeout`` so
        iterative deepening stops promptly. Seed, fallback, and response
        telemetry can run after that deadline; a missing uncached score must
        not turn an otherwise legal result into an API failure.
        """
        try:
            return self.transition_policy_score(board, mover)
        except SearchTimeout:
            return 0.0

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

    def tactical_risk(
        self,
        board: engine.BaseBoard,
        defender: bool,
        check_hq_combinations: bool = True,
    ) -> Tuple[float, float, float]:
        """Return risk, forced loss, and critical para/artillery exposure.

        Forced start-of-turn captures are resolved far enough to inspect the
        opponent's first voluntary action. This catches a gun left directly
        available to a para and a para left inside a bombardment even when a
        different automatic capture must happen first.
        """
        cache_key = (board_key(board), defender, check_hq_combinations)
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
            if check_hq_combinations and self.has_same_turn_hq_capture(position):
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
        check_hq_combinations: bool = True,
    ) -> TacticalSafety:
        if after.is_game_over():
            return TacticalSafety(0.0, 0.0, 100.0, 0.0, 0.0, True)
        baseline, baseline_forced, _ = self.tactical_risk(
            before, mover, check_hq_combinations
        )
        risk, forced, critical = self.tactical_risk(
            after, mover, check_hq_combinations
        )
        opponent = not mover
        opponent_turn = (
            after
            if after.turn == opponent and after.turn_moves == 0
            else self.board_as_turn(after, opponent)
        )
        loses_hq_this_turn = (
            self.has_same_turn_hq_capture(opponent_turn)
            if check_hq_combinations
            else False
        )
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
        reduced_existing_forced_loss = (
            forced + 0.75 < baseline_forced
            and uncovered <= 0.75
        )
        safe = not loses_hq_this_turn and (
            resolved_hq_threat
            or reduced_existing_forced_loss
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
    def forward_orientation(color: bool) -> int:
        """The stable straight-ahead facing for a relocated artillery piece."""
        return engine.ORIENT_N if color == engine.RED else engine.ORIENT_S

    @staticmethod
    def is_diagonal_orientation(orientation: Optional[int]) -> bool:
        return orientation is not None and orientation % 2 == 1

    @staticmethod
    def artillery_para_cover_points(
        board: engine.BaseBoard, square: int, color: bool
    ) -> int:
        """Count landing-square coverage: diagonal infantry cover two, cardinal one."""
        friendly_infantry = board.occupied_co[color] & (
            board.infantry | board.armored_infantry | board.airborne_infantry
        )
        file_index = engine.square_file(square)
        rank_index = engine.square_rank(square)
        coverage = 0
        for infantry in squares(
            engine.BB_REGULAR_MOVES[square] & friendly_infantry
        ):
            diagonal = (
                engine.square_file(infantry) != file_index
                and engine.square_rank(infantry) != rank_index
            )
            coverage += 2 if diagonal else 1
        return coverage

    def diagonal_artillery_move_is_safe(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Apply the cheap safety floor before minimax verifies the whole turn."""
        if move.to_square is None:
            return False
        mover = board.turn
        child = board.copy()
        child.push(move)
        destination = move.to_square
        if engine.BB_SQUARES[destination] & child.get_bombarded_squares(not mover):
            return False

        enemy = not mover
        enemy_home = engine.BB_RANK_1 if enemy == engine.RED else engine.BB_RANK_8
        enemy_para_ready = bool(
            child.airborne_infantry & child.occupied_co[enemy] & enemy_home
        ) or child.get_reserve_count(engine.AIRBORNE_INFANTRY, enemy) > 0
        if (
            enemy_para_ready
            and self.artillery_para_cover_points(child, destination, mover) < 3
        ):
            return False
        return artillery_exposure_penalty(
            child, mover
        ) <= artillery_exposure_penalty(board, mover) + 1e-9

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

        Pure rotations must create an actual threat. A relocation defaults to
        the straight-ahead facing. A diagonal facing survives only for the
        protected, multi-target "windshield wiper" exception; minimax then
        performs the full tactical safety check on the completed turn.
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
        if (
            move.from_square == move.to_square
            and friendly_blocks
            and not enemy_targets
        ):
            return False

        if (
            move.from_square != move.to_square
            and move.orientation != self.forward_orientation(board.turn)
            and not enemy_targets
        ):
            # A relocated gun should keep the stable forward facing unless a
            # different cardinal direction creates real pressure. This rules
            # out edge-facing moves such as b8-b7← (blocked by the infantry
            # on a7) without suppressing a sideways shot at an actual target.
            return False

        if (
            piece_type in (engine.ARMORED_ARTILLERY, engine.HEAVY_ARTILLERY)
            and move.from_square != move.to_square
            and self.home_distance(move.to_square, board.turn)
            >= self.home_distance(move.from_square, board.turn)
            and self.home_distance(move.to_square, board.turn) >= 2
        ):
            enemy = not board.turn
            enemy_home = (
                engine.BB_RANK_1 if enemy == engine.RED else engine.BB_RANK_8
            )
            enemy_para_ready = bool(
                board.airborne_infantry
                & board.occupied_co[enemy]
                & enemy_home
            ) or board.get_reserve_count(engine.AIRBORNE_INFANTRY, enemy) > 0
            if (
                enemy_para_ready
                and self.artillery_para_cover_points(
                    board, move.to_square, board.turn
                )
                < 3
            ):
                # Do not advance the valuable guns beyond the protected rear
                # ranks while the enemy para remains available. A nominal
                # multi-target lane is not compensation if the gun can simply
                # be taken first. Homeward retreats and adequately protected
                # escapes remain searchable.
                return False

        if self.is_diagonal_orientation(move.orientation):
            if engine.popcount(enemy_targets) < 2:
                return False
            if not self.diagonal_artillery_move_is_safe(board, move):
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

    @staticmethod
    def hq_encirclement_pressure(board: engine.BaseBoard, color: bool) -> float:
        """Reward several infantry closing on the opposing HQ, not only the nearest.

        Nearest-piece distance saturates as soon as one infantry reaches the HQ.
        A winning pursuit still needs the rest of the formation to close the net,
        especially while a lone HQ is retreating.  Only pieces within four squares
        contribute, so remote shuffles cannot masquerade as objective progress.
        """
        targets = board.occupied_co[not color] & board.hq
        if not targets:
            return 0.0
        own_non_hq = board.occupied_co[color] & ~board.hq
        pursuers = own_non_hq & (
            board.infantry | board.armored_infantry | board.airborne_infantry
        )
        if not pursuers:
            pursuers = own_non_hq
        return float(
            sum(
                max(
                    0,
                    5
                    - min(
                        chebyshev(source, target)
                        for target in squares(targets)
                    ),
                )
                for source in squares(pursuers)
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

        A one-action setup that exposes a valuable capture is forcing in every
        phase. The broader two-quiet-action probe remains reserved for late
        no-progress positions so ordinary opening development stays cheap.
        """
        late_probe = self.stagnation_factor() >= 0.70
        if (
            move.name in ("AutoCapture", "Skip")
            or move.capture_preference is not None
            or board.turn_moves >= self.max_actions - 1
        ):
            return False
        cache_key = (board_key(board), move.uci())
        cached = self.capture_setup_move_cache.get(cache_key)
        if cached is not None:
            return cached

        # If a capture is already legal, spending a quiet action first is not
        # a capture setup. This prevents the anti-stall priority from rewarding
        # gratuitous motion before taking available material.
        if any(
            candidate.capture_preference is not None
            for candidate in board.generate_legal_moves()
        ):
            self.capture_setup_move_cache[cache_key] = False
            return False

        if not late_probe:
            valuable_targets = board.occupied_co[not board.turn] & (
                board.hq
                | board.armored_infantry
                | board.airborne_infantry
                | board.artillery
                | board.armored_artillery
                | board.heavy_artillery
            )
            relevant_squares = tuple(
                square
                for square in (move.from_square, move.to_square)
                if square is not None
            )
            if not any(
                chebyshev(square, target) <= 2
                for square in relevant_squares
                for target in squares(valuable_targets)
            ):
                self.capture_setup_move_cache[cache_key] = False
                return False

        mover = board.turn
        child = board.copy()
        child.push(move)
        if child.is_game_over() or child.turn != mover:
            self.capture_setup_move_cache[cache_key] = False
            return False

        def capture_values(position: engine.BaseBoard) -> List[float]:
            return [
                PIECE_VALUES.get(
                    position.piece_type_at(candidate.capture_preference), 0.0
                )
                for candidate in position.generate_legal_moves()
                if candidate.capture_preference is not None
            ]

        unlocked_values = capture_values(child)
        if unlocked_values and (late_probe or max(unlocked_values) >= 3.0):
            self.capture_setup_move_cache[cache_key] = True
            return True

        remaining_actions = self.max_actions - child.turn_moves
        if late_probe and remaining_actions >= 2:
            for second in child.generate_legal_moves():
                if second.name in ("AutoCapture", "Skip"):
                    continue
                grandchild = child.copy()
                grandchild.push(second)
                if (
                    not grandchild.is_game_over()
                    and grandchild.turn == mover
                    and capture_values(grandchild)
                ):
                    self.capture_setup_move_cache[cache_key] = True
                    return True
        self.capture_setup_move_cache[cache_key] = False
        return False

    def followup_capture_value(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> float:
        """Value the best additional capture made legal by this capture."""
        if move.capture_preference is None:
            return 0.0
        cache_key = (board_key(board), move.uci())
        cached = self.followup_capture_value_cache.get(cache_key)
        if cached is not None:
            return cached
        mover = board.turn
        child = board.copy()
        child.push(move)
        if child.is_game_over() or child.turn != mover:
            self.followup_capture_value_cache[cache_key] = 0.0
            return 0.0
        value = max(
            (
                PIECE_VALUES.get(
                    child.piece_type_at(candidate.capture_preference), 0.0
                )
                for candidate in child.generate_legal_moves()
                if candidate.capture_preference is not None
            ),
            default=0.0,
        )
        self.followup_capture_value_cache[cache_key] = value
        return value

    def unlocks_hq_escape(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Whether a quiet action makes a later HQ evacuation safe.

        The setup may vacate the destination, interpose on an attack lane, or
        remove a defender that made an already-legal HQ move tactically lose.
        A direct HQ move receives defense priority separately once safe.
        """
        if (
            move.name in ("AutoCapture", "Skip")
            or board.turn_moves >= self.max_actions - 1
            or self.move_piece_type(board, move) == engine.HQ
            or (
                move.from_square is not None
                and move.to_square is not None
                and move.from_square == move.to_square
            )
        ):
            return False
        cache_key = (board_key(board), move.uci())
        cached = self.hq_escape_unlock_move_cache.get(cache_key)
        if cached is not None:
            return cached
        own_hqs = board.pieces(engine.HQ, board.turn)
        vacates_adjacent_square = bool(
            move.from_square is not None
            and any(
                chebyshev(move.from_square, hq_square) == 1
                for hq_square in own_hqs
            )
        )
        supports_evacuation = bool(
            board.turn_moves == 0
            and move.to_square is not None
            and any(
                chebyshev(move.to_square, hq_square) <= 2
                for hq_square in own_hqs
            )
        )
        if not vacates_adjacent_square and not supports_evacuation:
            self.hq_escape_unlock_move_cache[cache_key] = False
            return False
        _, baseline_forced, _ = self.tactical_risk(board, board.turn)
        if baseline_forced < PIECE_VALUES[engine.HQ]:
            self.hq_escape_unlock_move_cache[cache_key] = False
            return False

        child = board.copy()
        child.push(move)
        if child.turn != board.turn or child.is_game_over():
            self.hq_escape_unlock_move_cache[cache_key] = False
            return False
        for hq_move in child.generate_legal_moves():
            if (
                hq_move.from_square is None
                or child.piece_type_at(hq_move.from_square) != engine.HQ
                or (
                    not supports_evacuation
                    and hq_move.to_square != move.from_square
                )
            ):
                continue
            escaped = child.copy()
            escaped.push(hq_move)
            _, escaped_forced, _ = self.tactical_risk(escaped, board.turn)
            if escaped_forced < PIECE_VALUES[engine.HQ]:
                self.hq_escape_unlock_move_cache[cache_key] = True
                return True
        self.hq_escape_unlock_move_cache[cache_key] = False
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
        cache_key = (board_key(board), move.uci())
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

    def unlocks_hq_defense(
        self, board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Whether this setup makes a one-action HQ defense available.

        Some GHQ checks cannot be answered by a single atomic action. One
        defender may first have to interpose, engage an attacker, or open a
        home-rank reinforcement square before the second action actually
        removes the forced HQ loss. Detect that pair before the atomic beam
        discards its quiet first half.
        """
        if (
            move.name in ("AutoCapture", "Skip")
            or board.turn_moves != 0
        ):
            return False
        cache_key = (board_key(board), move.uci())
        cached = self.hq_defense_unlock_move_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.has_atomic_hq_defense(board):
            # The ordinary check-evasion and HQ-escape paths are both cheaper
            # and stronger. Do not spend deadline budget looking for a
            # two-action substitute when an atomic answer already exists.
            self.hq_defense_unlock_move_cache[cache_key] = False
            return False
        own_hqs = board.pieces(engine.HQ, board.turn)
        if not own_hqs or not any(
            square is not None
            and any(chebyshev(square, hq_square) <= 3 for hq_square in own_hqs)
            for square in (move.from_square, move.to_square)
        ):
            # A two-action HQ defense is assembled around the threatened HQ.
            # This gate avoids an O(actions^2) tactical probe for every remote
            # shuffle in an otherwise large late-game position.
            self.hq_defense_unlock_move_cache[cache_key] = False
            return False
        _, baseline_forced, _ = self.tactical_risk(board, board.turn)
        if baseline_forced < PIECE_VALUES[engine.HQ]:
            self.hq_defense_unlock_move_cache[cache_key] = False
            return False

        child = board.copy()
        child.push(move)
        if child.turn != board.turn or child.is_game_over():
            self.hq_defense_unlock_move_cache[cache_key] = False
            return False
        _, child_forced, _ = self.tactical_risk(child, board.turn)
        if child_forced < PIECE_VALUES[engine.HQ]:
            # The action is a direct defense, not merely its setup.
            self.hq_defense_unlock_move_cache[cache_key] = False
            return False
        follow_ups = [
            follow_up
            for follow_up in child.generate_legal_moves()
            if follow_up.name not in ("AutoCapture", "Skip")
            and any(
                square is not None
                and any(
                    chebyshev(square, hq_square) <= 3
                    for hq_square in own_hqs
                )
                for square in (follow_up.from_square, follow_up.to_square)
            )
        ]
        follow_ups.sort(
            key=lambda follow_up: (
                0 if follow_up.capture_preference is not None else 1,
                0 if follow_up.name == "Reinforce" else 1,
                min(
                    chebyshev(square, hq_square)
                    for square in (follow_up.from_square, follow_up.to_square)
                    if square is not None
                    for hq_square in own_hqs
                ),
                follow_up.uci(),
            )
        )
        unlocked = any(
            self.resolves_hq_threat(child, follow_up)
            for follow_up in follow_ups[:16]
        )
        self.hq_defense_unlock_move_cache[cache_key] = unlocked
        return unlocked

    def has_atomic_hq_defense(self, board: engine.BaseBoard) -> bool:
        """Whether one legal action resolves the HQ threat or unlocks escape."""
        key = board_key(board)
        cached = self.atomic_hq_defense_cache.get(key)
        if cached is not None:
            return cached
        available = any(
            self.resolves_hq_threat(board, candidate)
            or self.unlocks_hq_escape(board, candidate)
            for candidate in board.generate_legal_moves()
            if candidate.name not in ("AutoCapture", "Skip")
        )
        self.atomic_hq_defense_cache[key] = available
        return available

    def has_immediate_hq_capture(self, board: engine.BaseBoard) -> bool:
        """Whether the side to act can take the enemy HQ with one legal action."""
        key = board_key(board)
        cached = self.immediate_hq_capture_cache.get(key)
        if cached is not None:
            return cached
        enemy_hqs = board.pieces(engine.HQ, not board.turn)
        capture_destinations = engine.BB_EMPTY
        for hq_square in enemy_hqs:
            capture_destinations |= engine.BB_ADJACENT_SQUARES[hq_square]
        # Every voluntary HQ capture is generated by an infantry destination
        # adjacent to the HQ. Restricting the destination mask avoids scanning
        # remote infantry moves and all artillery orientations. Forced
        # bombardments/free captures are still yielded before this mask is
        # consulted by the production engine.
        for candidate in board.generate_legal_captures(
            to_mask=capture_destinations
        ):
            target = candidate.capture_preference
            if target is not None and board.piece_type_at(target) == engine.HQ:
                self.immediate_hq_capture_cache[key] = True
                return True
        self.immediate_hq_capture_cache[key] = False
        return False

    def has_same_turn_hq_capture(self, board: engine.BaseBoard) -> bool:
        """Whether the side to act can force an HQ capture this turn.

        This is deliberately narrower than complete turn enumeration. It uses
        the equivalence-collapsed HQ action set, which retains every capture
        plus local quiet infantry setups. Keeping remote captures matters:
        GHQ capture obligations can consume an action before a local HQ attack
        becomes legal.
        """
        key = board_key(board)
        cached = self.same_turn_hq_capture_cache.get(key)
        if cached is not None:
            return cached
        if self.has_immediate_hq_capture(board):
            self.same_turn_hq_capture_cache[key] = True
            return True
        if board.turn_moves >= self.max_actions - 1:
            self.same_turn_hq_capture_cache[key] = False
            return False

        actions_remaining = self.max_actions - board.turn_moves
        first_positions: Dict[BoardKey, engine.BaseBoard] = {}
        for move in self.exact_hq_capture_moves(board):
            child = board.copy()
            child.push(move)
            if child.turn != board.turn or child.is_game_over():
                continue
            if self.has_immediate_hq_capture(child):
                self.same_turn_hq_capture_cache[key] = True
                return True
            first_positions.setdefault(board_key(child), child)

        if actions_remaining >= 3:
            # Search the two-setup case once per unique intermediate board.
            # Calling ``unlocks_immediate_hq_capture`` for every first move
            # regenerated the same move-order permutations hundreds of times
            # inside each safety check, which could consume the entire Vercel
            # turn budget before minimax completed one opponent reply.
            for child in first_positions.values():
                for follow_up in self.exact_hq_capture_moves(child):
                    grandchild = child.copy()
                    grandchild.push(follow_up)
                    if (
                        grandchild.turn == board.turn
                        and not grandchild.is_game_over()
                        and self.has_immediate_hq_capture(grandchild)
                    ):
                        self.same_turn_hq_capture_cache[key] = True
                        return True
        self.same_turn_hq_capture_cache[key] = False
        return False

    def exact_same_turn_hq_capture(
        self,
        board: engine.BaseBoard,
        attacker: bool,
        remaining_nodes: List[int],
    ) -> Optional[bool]:
        """Prove whether the side to act can capture the HQ this turn.

        The ordinary detector is intentionally geometric and cheap enough for
        every candidate.  A last-chance survival override needs the opposite
        tradeoff: it may call this bounded complete-turn enumeration, but it
        must never certify safety from an incomplete probe.
        """
        cache_key = (board_key(board), attacker)
        cached = self.exact_same_turn_hq_capture_cache.get(cache_key)
        if cached is not None:
            return cached
        mover = board.turn
        frontier = [board.copy()]
        seen: set[str] = set()
        while frontier:
            if time.monotonic() >= self.deadline:
                return None
            if remaining_nodes[0] <= 0:
                return None
            remaining_nodes[0] -= 1
            self.hq_survival_reply_nodes += 1
            current = frontier.pop()
            key = board_key(current)
            if key in seen:
                continue
            seen.add(key)
            outcome = current.outcome()
            if outcome is not None:
                if outcome.winner == attacker:
                    self.exact_same_turn_hq_capture_cache[cache_key] = True
                    return True
                continue
            if current.turn != mover:
                continue
            moves = list(self.exact_hq_capture_moves(current))
            moves.sort(
                key=lambda move: (
                    self.exact_hq_capture_move_priority(current, move),
                    normalized_move_uci(move, mover),
                )
            )
            for move in moves:
                target = move.capture_preference
                if (
                    target is not None
                    and current.piece_type_at(target) == engine.HQ
                ):
                    self.exact_same_turn_hq_capture_cache[cache_key] = True
                    return True
                # A third voluntary action ends the attacker's turn.  It can
                # still win immediately by leaving the defending HQ under a
                # terminal bombardment, so push it and inspect the outcome.
                # Do not enqueue a non-terminal child whose turn has already
                # passed: the old proof popped hundreds of thousands of those
                # dead leaves only to discard them below. Forced AutoCaptures
                # do not consume a voluntary action and must still be followed.
                if (
                    current.turn_moves >= self.max_actions - 1
                    and move.name != "AutoCapture"
                ):
                    if not self.terminal_action_may_capture_hq(current, move):
                        continue
                    child = current.copy()
                    child.push(move)
                    outcome = child.outcome()
                    if outcome is not None and outcome.winner == attacker:
                        self.exact_same_turn_hq_capture_cache[cache_key] = True
                        return True
                    continue
                child = current.copy()
                child.push(move)
                frontier.append(child)
        self.exact_same_turn_hq_capture_cache[cache_key] = False
        return False

    @staticmethod
    def terminal_action_may_capture_hq(
        board: engine.BaseBoard, move: engine.Move
    ) -> bool:
        """Whether a turn-ending action can produce an HQ-capture outcome.

        A legal action that neither directly captures the HQ nor leaves it in
        the attacker's artillery field cannot end in an HQ capture.  If the
        HQ is already bombarded, keep every action because moving or engaging
        the defender's last mobile piece can make that bombardment terminal.
        Otherwise only an artillery move newly aimed through the HQ needs the
        comparatively expensive child/outcome construction.
        """
        enemy_hq = board.hq & board.occupied_co[not board.turn]
        if enemy_hq & board.bombarded_co[board.turn]:
            return True
        piece_type = Searcher.move_piece_type(board, move)
        if (
            piece_type is None
            or not engine.is_artillery(piece_type)
            or move.to_square is None
        ):
            return False
        orientation = move.orientation
        if orientation is None and move.name == "Reinforce":
            orientation = (
                engine.ORIENT_N
                if board.turn == engine.RED
                else engine.ORIENT_S
            )
        if orientation is None:
            return False
        distance = 3 if piece_type == engine.HEAVY_ARTILLERY else 2
        target = board.get_bombardment_target(
            move.to_square, orientation, distance
        )
        return bool(
            target is not None
            and engine.between_inclusive_end(move.to_square, target) & enemy_hq
        )

    @staticmethod
    def exact_hq_capture_move_priority(
        board: engine.BaseBoard, move: engine.Move
    ) -> Tuple[int, int, int]:
        """Order exact attack branches without changing their membership."""
        enemy_hqs = list(board.pieces(engine.HQ, not board.turn))
        target = move.capture_preference
        captures_hq = bool(
            target is not None and board.piece_type_at(target) == engine.HQ
        )
        captures_piece = target is not None
        piece_type = Searcher.move_piece_type(board, move)
        is_infantry_action = bool(
            piece_type is not None and engine.is_infantry(piece_type)
        )
        destination_distance = min(
            (
                chebyshev(move.to_square, hq_square)
                for hq_square in enemy_hqs
                if move.to_square is not None
            ),
            default=9,
        )
        # The DFS pops the greatest key first.  Immediate captures lead, then
        # other captures and infantry actions closest to the target HQ.
        return (
            3 if captures_hq else 2 if captures_piece else 1 if is_infantry_action else 0,
            -destination_distance,
            1 if move.name in ("Move", "Reinforce", "AutoCapture") else 0,
        )

    @staticmethod
    def exact_hq_capture_moves(
        board: engine.BaseBoard,
    ) -> Iterator[engine.Move]:
        """Yield an exact, equivalence-collapsed HQ-capture action set.

        This is not a tactical heuristic.  It removes only actions that cannot
        help the mover capture an HQ during the current turn:

        * Skip ends the turn without taking a piece.
        * A stationary artillery rotation is retained when its new lane
          directly bombards the enemy HQ. GHQ resolves that bombardment when
          the turn ends, so the rotation can itself be the mating action.
        * Reinforcing non-infantry only occupies a square; it cannot join an
          infantry capture cluster or remove an enemy.
        * In crowded positions, artillery relocation orientations are
          equivalent unless an orientation directly bombards the HQ. Once
          the board is sparse, retain every orientation: a lane can engage or
          bombard an intermediate piece and thereby enable a later HQ shot.

        Every capture remains because a mandatory remote capture can consume
        an action and change which local HQ capture becomes legal next. Quiet
        infantry actions remain within three squares of the HQ: a first-step
        engagement at distance three can unlock a two-action capture chain.
        Other quiet relocations retain the tighter two-square radius. Global
        paratrooper moves remain whenever their destination enters that local
        radius. Infantry capture variants also remain distinct.
        The resulting search is complete for same-turn HQ capture while
        avoiding remote moves and hundreds of irrelevant orientation branches.
        """
        enemy_hqs = list(board.pieces(engine.HQ, not board.turn))
        enemy_hq_mask = board.hq & board.occupied_co[not board.turn]
        preserve_artillery_orientations = surviving_non_hq_count(board) <= 12
        artillery_relocations: set[Tuple[int, int]] = set()
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
            if move.capture_preference is not None:
                yield move
                continue
            piece_type = Searcher.move_piece_type(board, move)
            local_radius = (
                3
                if piece_type is not None and engine.is_infantry(piece_type)
                else 2
            )
            if move.name != "AutoCapture" and not any(
                chebyshev(square, hq_square) <= local_radius
                for square in (
                    move.from_square,
                    move.to_square,
                )
                if square is not None
                for hq_square in enemy_hqs
            ):
                continue
            yield move

    @staticmethod
    def artillery_hq_interdiction_squares(
        board: engine.BaseBoard,
        move: engine.Move,
        hq_square: int,
        mover: bool,
    ) -> int:
        """Return newly bombarded setup squares near the friendly HQ.

        A gun can defend the HQ from well outside the local survival radius.
        In one exact audit, moving the heavy artillery b2-c1 and facing east
        denied f1, the transit square required by an otherwise forced enemy
        paratrooper combination.  Key these actions by where their lane takes
        effect rather than by the gun's distance from the HQ.
        """
        if (
            move.name != "MoveAndOrient"
            or move.from_square is None
            or move.to_square is None
            or move.orientation is None
        ):
            return engine.BB_EMPTY
        piece_type = board.piece_type_at(move.from_square)
        if piece_type is None or not engine.is_artillery(piece_type):
            return engine.BB_EMPTY
        distance = 3 if piece_type == engine.HEAVY_ARTILLERY else 2
        target = board.get_bombardment_target(
            move.to_square, move.orientation, distance
        )
        if target is None or target == move.to_square:
            return engine.BB_EMPTY

        hq_file = engine.square_file(hq_square)
        hq_rank = engine.square_rank(hq_square)
        defense_zone = engine.BB_EMPTY
        for file_index in range(max(0, hq_file - 2), min(7, hq_file + 2) + 1):
            for rank_index in range(
                max(0, hq_rank - 2), min(7, hq_rank + 2) + 1
            ):
                defense_zone |= engine.BB_SQUARES[
                    engine.square(file_index, rank_index)
                ]
        lane = engine.between_inclusive_end(move.to_square, target)
        return (
            lane
            & defense_zone
            & ~board.bombarded_co[mover]
        )

    def find_hq_survival_turn(
        self,
        root: engine.BaseBoard,
        max_probe_nodes: int = 20_000,
        max_reply_nodes: int = 100_000,
    ) -> Optional[Tuple[List[engine.Move], engine.BaseBoard]]:
        """Find and exactly verify a nearby turn that avoids immediate HQ loss.

        This is a bounded mate-delay floor, not another positional search. It
        runs only after minimax selected an immediately losing turn. Candidate
        actions stay near the HQ (plus captures and Skip), which covers
        interpositions, engagements, coordinated evacuations, and the vital
        option of declining to walk the HQ onto a losing square.
        """
        mover = root.turn
        frontier: List[Tuple[engine.BaseBoard, List[engine.Move]]] = [
            (root.copy(), [])
        ]
        seen: set[BoardKey] = set()
        completed: Dict[BoardKey, Tuple[engine.BaseBoard, List[engine.Move]]] = {}
        probe_nodes = 0
        while frontier and probe_nodes < max_probe_nodes:
            if time.monotonic() >= self.deadline:
                return None
            board, line = frontier.pop()
            probe_nodes += 1
            self.hq_survival_probe_nodes += 1
            key = board_key(board)
            if key in seen:
                continue
            seen.add(key)
            outcome = board.outcome()
            if outcome is not None or board.turn != mover:
                if outcome is not None and outcome.winner == mover:
                    return line, board
                incumbent = completed.get(key)
                if incumbent is None or normalized_turn_key(
                    line, mover
                ) < normalized_turn_key(incumbent[1], mover):
                    completed[key] = (board, line)
                continue

            hq_squares = list(board.pieces(engine.HQ, mover))
            if not hq_squares:
                return None
            hq_square = hq_squares[0]
            actions: List[Tuple[float, str, engine.Move]] = []
            for move in board.generate_legal_moves():
                piece_type = self.move_piece_type(board, move)
                hq_interdiction = self.artillery_hq_interdiction_squares(
                    board, move, hq_square, mover
                )
                distances = [
                    chebyshev(square, hq_square)
                    for square in (move.from_square, move.to_square)
                    if square is not None
                ]
                if move.name == "AutoCapture":
                    priority = 20_000.0
                elif move.name == "Skip":
                    priority = 10_000.0
                elif piece_type == engine.HQ:
                    priority = 5_000.0 - 100.0 * min(distances or [9])
                elif hq_interdiction:
                    priority = 4_000.0 + 25.0 * engine.popcount(hq_interdiction)
                elif move.capture_preference is not None:
                    priority = 2_500.0 - 100.0 * min(distances or [9])
                elif distances and min(distances) <= 3:
                    priority = -100.0 * min(distances)
                else:
                    continue
                actions.append(
                    (priority, normalized_move_uci(move, mover), move)
                )
            actions.sort(key=lambda item: (item[0], item[1]))
            for _, _, move in actions:
                child = board.copy()
                child.push(move)
                frontier.append((child, [*line, move]))

        # Do not let the first complete but obviously losing defense consume
        # the entire exact-reply budget.  Generate the nearby defensive turns
        # first, then verify the positions that leave the defender with the
        # strongest cheap evaluation.  Exact verification remains the only
        # authority that may certify a line as safe.
        ranked = sorted(
            completed.values(),
            key=lambda item: (
                -(
                    self.quick_score(item[0])
                    if mover == engine.RED
                    else -self.quick_score(item[0])
                ),
                normalized_turn_key(item[1], mover),
            ),
        )

        # Preserve a small, separately verified survival floor before a
        # complicated top-ranked reply can consume the whole exact budget.
        # This mattered in a live position where Skip was provably safe in
        # 15k nodes, while the strongest quick-score candidate remained
        # inconclusive after 100k. We still search the ranked defenses first;
        # the minimal line is returned only when those proofs run out of room.
        certified_minimal: Optional[
            Tuple[List[engine.Move], engine.BaseBoard]
        ] = None
        minimal_moves, minimal_board = deterministic_skip_turn(root)
        minimal_outcome = minimal_board.outcome()
        reserved_nodes = min(25_000, max(0, max_reply_nodes // 4))
        minimal_used = 0
        if minimal_outcome is not None:
            if minimal_outcome.winner != (not mover):
                certified_minimal = (minimal_moves, minimal_board)
        elif not self.has_same_turn_hq_capture(minimal_board) and reserved_nodes:
            minimal_budget = [reserved_nodes]
            minimal_capture = self.exact_same_turn_hq_capture(
                minimal_board, not mover, minimal_budget
            )
            minimal_used = reserved_nodes - minimal_budget[0]
            if minimal_capture is False:
                certified_minimal = (minimal_moves, minimal_board)

        reply_budget = [max(0, max_reply_nodes - minimal_used)]
        certified_minimal_key = (
            board_key(certified_minimal[1])
            if certified_minimal is not None
            else None
        )
        certified_minimal_score = (
            (
                self.quick_score(certified_minimal[1])
                if mover == engine.RED
                else -self.quick_score(certified_minimal[1])
            )
            if certified_minimal is not None
            else -math.inf
        )
        for board, line in ranked:
            if time.monotonic() >= self.deadline:
                return certified_minimal
            if (
                certified_minimal_key is not None
                and board_key(board) == certified_minimal_key
            ):
                continue
            candidate_score = (
                self.quick_score(board)
                if mover == engine.RED
                else -self.quick_score(board)
            )
            if (
                certified_minimal is not None
                and candidate_score < certified_minimal_score - 1e-9
            ):
                continue
            purpose = self.deadline_safe_turn_purpose_breakdown(
                root,
                board,
                line,
                mover,
                retrospective=False,
            )
            if purpose["paratrooper_mission_penalty"] > 0.0:
                # This exact-survival pass runs after approximate minimax has
                # selected an immediately losing line. It may delay mate, but
                # it is not permission to bypass the global para doctrine for
                # a speculative one-piece trade.
                continue
            if self.has_same_turn_hq_capture(board):
                continue
            capture = self.exact_same_turn_hq_capture(
                board, not mover, reply_budget
            )
            if capture is False:
                return line, board
            if capture is None:
                return certified_minimal
        return certified_minimal

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
        cache_key = (board_key(board), move.uci())
        cached = self.hq_capture_unlock_move_cache.get(cache_key)
        if cached is not None:
            return cached
        if self.has_immediate_hq_capture(board):
            # The capture was already legal; this action did not unlock it
            # and must not receive forcing priority merely for preserving it.
            self.hq_capture_unlock_move_cache[cache_key] = False
            return False

        child = board.copy()
        child.push(move)
        if child.turn != board.turn or child.is_game_over():
            self.hq_capture_unlock_move_cache[cache_key] = False
            return False
        if self.has_immediate_hq_capture(child):
            self.hq_capture_unlock_move_cache[cache_key] = True
            return True

        # Some three-action mates need one more setup before the HQ capture
        # becomes legal. That setup is often a capture, but it can also be a
        # quiet move beside the HQ that opens a paratrooper lane or supplies
        # the second engagement. Inspect only captures and moves immediately
        # around the target HQ, and only when two counted actions remain, so
        # this tactical extension stays narrow and deterministic.
        if child.turn_moves >= self.max_actions - 1:
            self.hq_capture_unlock_move_cache[cache_key] = False
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
                self.hq_capture_unlock_move_cache[cache_key] = True
                return True
        self.hq_capture_unlock_move_cache[cache_key] = False
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
            followup_value = self.followup_capture_value(board, move)
            if followup_value > 0.0:
                # Prefer an equal capture that keeps a second conversion
                # legal over one that terminates the combination.
                priority += 4500.0 + 100.0 * followup_value
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
            # Keep a quiet lane/engagement setup for a valuable capture even
            # when the action itself changes no evaluation feature.
            priority += 4800.0
        if self.unlocks_hq_escape(board, move):
            # Clearing the only HQ flight square is part of the escape, not a
            # generic infantry shuffle. It must survive even the narrow
            # verification beam.
            priority += 8200.0
        resolves_hq = self.resolves_hq_threat(board, move)
        if resolves_hq:
            # Interpositions and engagements that turn off a forced HQ loss
            # are check evasions even when they capture nothing.
            priority += 8500.0
        elif self.unlocks_hq_defense(board, move):
            # Preserve the quiet first half of a two-action check evasion.
            # It is almost as forcing as the direct interposition that follows.
            priority += 8300.0
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
            if (
                move.from_square is not None
                and move.to_square is not None
                and move.from_square != move.to_square
                and move.orientation == self.forward_orientation(board.turn)
            ):
                # Orientation clones are collapsed before the beam. Preserve
                # the stable default unless another facing proves real value.
                priority += 350.0
        if piece_type == engine.ARMORED_INFANTRY:
            priority += 20.0
        elif piece_type in ARTILLERY_TYPES:
            priority += 10.0
        if (
            self.stagnation_factor() > 0.0
            and piece_type in infantry_types
            and move.from_square is not None
            and move.to_square is not None
        ):
            enemy_hqs = board.occupied_co[not board.turn] & board.hq
            if enemy_hqs:
                before_distance = min(
                    chebyshev(move.from_square, target)
                    for target in squares(enemy_hqs)
                )
                after_distance = min(
                    chebyshev(move.to_square, target)
                    for target in squares(enemy_hqs)
                )
                # Preserve objective-closing infantry from each source in the
                # narrow verification beam.  This avoids six equivalent moves
                # by a remote armored infantry crowding out an HQ encirclement.
                priority += (
                    120.0
                    * self.stagnation_factor()
                    * max(0, before_distance - after_distance)
                )
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

        if (
            self.stagnation_factor() >= 0.20
            and board_key(board) == self.root_key
        ):
            # A narrow verification beam used to contain only home-rank and
            # artillery shuffles, even after self-play had proven a cycle.
            # Preserve two forward infantry first actions; complete-turn
            # purpose and tactical safety filters still decide whether the
            # resulting plans reach minimax.
            progress_slots = 2
            for priority, move in diverse_scored:
                piece_type = self.move_piece_type(board, move)
                if (
                    piece_type in INFANTRY_TYPES
                    and move.name != "Reinforce"
                    and move.from_square is not None
                    and move.to_square is not None
                    and relative_rank(move.to_square, board.turn)
                    > relative_rank(move.from_square, board.turn)
                    and all(
                        selected_move != move
                        for _, selected_move in selected
                    )
                ):
                    selected.append((priority, move))
                    progress_slots -= 1
                    if progress_slots == 0:
                        break
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
                        "hq_defense_unlock",
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
        hq_defense_unlocks: List[str] = []
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
            resolves_hq = self.resolves_hq_threat(working, move)
            if resolves_hq:
                hq_defenses.append(move.uci())
            elif self.unlocks_hq_defense(working, move):
                hq_defense_unlocks.append(move.uci())
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
        keys.extend(("hq_defense_unlock", uci) for uci in hq_defense_unlocks)
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
            board_key(board) == self.root_key
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
            if purpose["paratrooper_mission_penalty"] > 0.0:
                return
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
        cleaned_options: List[TurnCandidate] = []
        # There are at most three voluntary actions, so enumerate every
        # non-empty subset of disposable actions. This covers both
        # ``useful + useful + filler`` and ``useful + filler + filler`` while
        # replay proving that no omitted action unlocked what remains.
        for mask in range(1, 1 << len(removable)):
            omitted = {
                removable[offset]
                for offset in range(len(removable))
                if mask & (1 << offset)
            }
            working = root.copy()
            cleaned_moves: List[engine.Move] = []
            cleaned_priority = 0.0
            replayable = True
            for index, original_move in enumerate(candidate.moves):
                if index in omitted:
                    continue
                if working.turn != mover or working.is_game_over():
                    replayable = False
                    break
                replay_move = next(
                    (
                        move
                        for move in working.generate_legal_moves()
                        if move.uci() == original_move.uci()
                    ),
                    None,
                )
                if replay_move is None:
                    replayable = False
                    break
                cleaned_priority += self.move_priority(working, replay_move)
                cleaned_moves.append(replay_move)
                working.push(replay_move)
            if not replayable or working.turn != mover or working.is_game_over():
                continue
            skip = next(
                (
                    move
                    for move in working.generate_legal_moves()
                    if move.name == "Skip"
                ),
                None,
            )
            if skip is None:
                continue
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
            if not voluntary_purposes or len(voluntary_purposes) > 2 or any(
                "no_new_effect" in purpose["roles"]
                for purpose in voluntary_purposes
            ):
                continue
            purpose = self.turn_purpose_breakdown(
                root, working, cleaned_moves, mover, retrospective=True
            )
            if purpose["paratrooper_mission_penalty"] > 0.0:
                continue
            if (
                self.turn_number <= EARLY_GAME_LAST_TURN
                and not self.early_structure_allowed(
                    root,
                    working,
                    cleaned_moves,
                    mover,
                    action_purposes,
                )
            ):
                continue
            safety = self.assess_turn_safety(root, working, mover)
            terminal = self.terminal_score(working, 0)
            cleaned_options.append(
                TurnCandidate(
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
            )
        if not cleaned_options:
            return None
        cleaned_options.sort(
            key=lambda item: self.candidate_sort_key(item, mover)
        )
        return cleaned_options[0]

    def generate_turn_candidates(self, board: engine.BaseBoard) -> List[TurnCandidate]:
        """Generate, deduplicate, then prune complete player turns.

        Search no longer truncates to ``beam_width`` after every atomic action.
        A wider, first-action-diverse frontier is carried until the side changes;
        only complete resulting positions become minimax branches.
        """
        cache_key = board_key(board)
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
            if self.stagnation_factor() >= 0.70:
                # Do not spend the tactical-floor budget on extra root lines
                # merely because a game has had a few quiet turns.  Early
                # widening caused the pass to time out before *any* opponent
                # reply completed.  Breadth is justified only as the actual
                # no-progress limit approaches.
                partial_width = max(partial_width, self.beam_width * 4)
                evaluation_pool_width = max(
                    evaluation_pool_width, self.beam_width * 3
                )
            if self.stagnation_factor() >= 0.90:
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
            if is_root_generation and self.stagnation_factor() >= 0.70:
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
                    key = board_key(child)
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
            key = board_key(partial.board)
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
                    if purpose["paratrooper_mission_penalty"] > 0.0:
                        continue
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
            key = board_key(cleaned.board)
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
        if is_root_generation and self.root_fallback is not None:
            fallback_key = board_key(self.root_fallback.board)
            if not any(
                board_key(candidate.board) == fallback_key
                for candidate in candidates
            ):
                # The deadline seed is generated before minimax precisely so
                # a coordinated legal defense survives narrow root pruning.
                # Make that safety floor an actual minimax branch instead of
                # using it only when root generation returns nothing at all.
                candidates.append(self.root_fallback)
        mission_complete_candidates = [
            candidate
            for candidate in candidates
            if candidate.paratrooper_mission_penalty <= 0.0
        ]
        if len(mission_complete_candidates) != len(candidates):
            self.purpose_filtered_turns += (
                len(candidates) - len(mission_complete_candidates)
            )
            self.exhaustive_within_horizon = False
            candidates = mission_complete_candidates
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
            if (
                is_root_generation
                and self.stagnation_factor() >= 0.20
            ):
                # A cosmetically purposeful conveyor seed must not delete a
                # genuinely progressive root plan merely because that plan
                # contains one quiet setup action. The action still pays its
                # purpose penalty; this only lets minimax verify the reply.
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
                    "hq_defense_unlock",
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
                        transition_quality = (
                            0.0
                            if abs(leaf_score) >= MATE_SCORE
                            else self.transition_policy_score(
                                turn.board, board.turn
                            )
                            - transition_penalty
                        )
                        candidate = (
                            leaf_score + transition_quality
                            if maximizing
                            else leaf_score - transition_quality
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

        key = (board_key(board), turns_left)
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
        is_root = board_key(board) == self.root_key
        root_scores: List[Tuple[float, TurnCandidate]] = []
        for turn in turns:
            result = self.alphabeta(turn.board, turns_left - 1, alpha, beta)
            transition_penalty = turn.purpose_penalty + turn.paratrooper_mission_penalty
            early_bonus = (
                0.40 * turn.early_plan_score
                if self.turn_number <= EARLY_GAME_LAST_TURN
                else 0.0
            )
            turn_quality = (
                0.0
                if abs(result.score) >= MATE_SCORE
                else early_bonus
                - transition_penalty
                + self.transition_policy_score(turn.board, board.turn)
            )
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
    time_ms: int = 2_000,
    stagnation_turns: int = 0,
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
    pending_para_targets: set[int] = set()
    fallback_rules = Searcher(
        personality,
        # This is a deadline floor, not a second full search.  The historical
        # 60-second budget could make a nominal 30-second native request run
        # past Vercel's function limit when a tactical-risk probe was costly.
        time_ms=max(50, min(2_000, time_ms)),
        beam_width=4,
        turn_number=turn_number,
        max_actions=max_actions,
        stagnation_turns=stagnation_turns,
    )
    stagnation_factor = max(
        0.0, min(1.0, (stagnation_turns - 4) / 20.0)
    )
    try:
        _, starting_forced_loss, _ = fallback_rules.tactical_risk(
            board, original_turn
        )
    except SearchTimeout:
        starting_forced_loss = 0.0
    started_under_hq_threat = (
        starting_forced_loss >= PIECE_VALUES[engine.HQ]
    )
    while working.turn == original_turn and not working.is_game_over():
        legal = list(working.generate_legal_moves())
        if not legal:
            break
        candidates: List[Tuple[float, float, str, engine.Move]] = []
        for move in legal:
            piece_type = Searcher.move_piece_type(working, move)
            if (
                pending_para_targets
                and move.name != "AutoCapture"
                and move.capture_preference not in pending_para_targets
            ):
                # A para commitment is not a completed mission until the
                # immediately unlocked second capture is actually converted.
                continue
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
            if not fallback_rules.paradrop_allowed(working, move):
                continue
            if not fallback_rules.early_extension_allowed(working, move):
                continue

            child = working.copy()
            child.push(move)
            candidate_moves = moves + [move]
            candidate_purposes = fallback_rules.deadline_safe_action_purpose_labels(
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
            latest_roles = set(candidate_purposes[-1]["roles"])
            if "hq_defense" in latest_roles:
                # A deadline seed is the last safety net when minimax has not
                # completed even one root. Never let ordinary positional gain
                # displace a move already proven to remove a forced HQ loss.
                utility += MATE_SCORE
            elif "hq_defense_unlock" in latest_roles:
                # A coordinated defense may require a quiet setup before the
                # interposition or engagement becomes legal. Keep that first
                # half ahead of every ordinary positional action.
                utility += 0.95 * MATE_SCORE
            elif "hq_escape_unlock" in latest_roles:
                # Some checks require vacating/interposing before the HQ can
                # evacuate later in the same turn. Preserve that setup ahead
                # of all non-terminal positional preferences.
                utility += 0.90 * MATE_SCORE
            if started_under_hq_threat:
                try:
                    _, candidate_forced_loss, _ = fallback_rules.tactical_risk(
                        child, original_turn
                    )
                except SearchTimeout:
                    candidate_forced_loss = PIECE_VALUES[engine.HQ]
                if candidate_forced_loss < PIECE_VALUES[engine.HQ]:
                    # Once the sequence has answered the check, every later
                    # action must preserve the answer. This prevents a greedy
                    # third move from reopening an HQ line that the first two
                    # actions just closed.
                    utility += MATE_SCORE
            if turn_number <= EARLY_GAME_LAST_TURN:
                utility += 2.5 * fallback_rules.early_plan_score(
                    candidate_purposes
                )
            if (
                stagnation_factor >= 0.20
                and move.capture_preference is None
                and piece_type in INFANTRY_TYPES
                and move.from_square is not None
                and move.to_square is not None
            ):
                # The bounded deadline seed is often the only root line a
                # cold native worker can reply-verify.  During an established
                # repetition cycle, do not let two cosmetically purposeful
                # backward infantry shuffles become that seed.  Prefer a
                # forward step; the completed turn still has to pass the same
                # tactical-safety gate as every other fallback.
                forward_delta = (
                    fallback_rules.home_distance(move.to_square, original_turn)
                    - fallback_rules.home_distance(move.from_square, original_turn)
                )
                utility += 6.0 * stagnation_factor * forward_delta
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
                if (
                    pending_para_targets
                    and move.name != "AutoCapture"
                    and move.capture_preference not in pending_para_targets
                ):
                    continue
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
                if not fallback_rules.paradrop_allowed(working, move):
                    continue
                child = working.copy()
                child.push(move)
                red_score = quick_evaluation(child, turn_number)
                utility = red_score if original_turn == engine.RED else -red_score
                candidate_purposes = fallback_rules.deadline_safe_action_purpose_labels(
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
        if chosen.capture_preference in pending_para_targets:
            pending_para_targets.clear()
        if fallback_rules.is_paradrop(working, chosen):
            pending_para_targets = set(
                fallback_rules.paradrop_capture_targets(working, chosen)
            )
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
        action_purposes = fallback_rules.deadline_safe_action_purpose_labels(
            board, moves, original_turn, retrospective=False
        )
        seed_candidate = TurnCandidate(
            moves=list(moves),
            board=working,
            priority=0.0,
            static_score=quick_evaluation(working, turn_number),
            action_purposes=action_purposes,
        )
        try:
            trimmed = fallback_rules.trim_filler_action(
                board, seed_candidate, original_turn
            )
        except SearchTimeout:
            trimmed = None
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


def deterministic_skip_turn(
    board: engine.BaseBoard,
) -> Tuple[List[engine.Move], engine.BaseBoard]:
    """End a turn without inventing a voluntary tactical mission.

    Auto-captures are mandatory GHQ bookkeeping and must be resolved before a
    Skip can end the turn.  This is the last-resort policy floor used only when
    every scored/fallback line somehow violates an objective return invariant.
    It deliberately cannot deploy or move a paratrooper.
    """
    working = board.copy()
    mover = working.turn
    moves: List[engine.Move] = []
    while working.turn == mover and not working.is_game_over():
        legal = list(working.generate_legal_moves())
        forced = sorted(
            (move for move in legal if move.name == "AutoCapture"),
            key=lambda move: normalized_move_uci(move, mover),
        )
        if forced:
            move = forced[0]
        else:
            skips = [move for move in legal if move.name == "Skip"]
            if not skips:
                raise RuntimeError("GHQ engine exposed no legal Skip action")
            move = skips[0]
        moves.append(move)
        working.push(move)
    return moves, working


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
        purpose = searcher.deadline_safe_turn_purpose_breakdown(
            board,
            working,
            moves,
            mover,
            retrospective=False,
        )
        if purpose["paratrooper_mission_penalty"] > 0.0:
            continue
        action_purposes = searcher.deadline_safe_action_purpose_labels(
            board,
            moves,
            mover,
            retrospective=False,
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


def bounded_seed_safety(
    board: engine.BaseBoard,
    seed_board: engine.BaseBoard,
    personality: str,
    turn_number: int,
    beam_width: int,
    max_actions: int,
    time_ms: int,
    check_hq_combinations: bool = True,
) -> Optional[TacticalSafety]:
    """Assess the emergency seed without consuming the minimax deadline."""
    probe = Searcher(
        personality,
        time_ms=time_ms,
        beam_width=max(4, beam_width),
        turn_number=turn_number,
        max_actions=max_actions,
    )
    try:
        if check_hq_combinations:
            return probe.assess_turn_safety(board, seed_board, board.turn)
        return probe.assess_turn_safety(
            board,
            seed_board,
            board.turn,
            check_hq_combinations=False,
        )
    except SearchTimeout:
        return None


def material_safe_recovery_turn(
    board: engine.BaseBoard,
    personality: str,
    turn_number: int,
    beam_width: int,
    max_actions: int,
    time_ms: int,
) -> Optional[TurnCandidate]:
    """Find a safe turn after the ordinary root beam retains only losses.

    Forced bombardments can form a chain: moving the eventual victim is not
    enough if an earlier automatic capture changes the bombardment lanes.  A
    timed-out search therefore gets one small, independent recovery pass.  It
    tries moves by pieces in the opponent's first forced-capture layer first,
    greedily completes each turn, and applies both material and HQ safety
    checks before returning anything.
    """
    deadline = time.monotonic() + max(50, time_ms) / 1000.0
    mover = board.turn
    ordering_probe = Searcher(
        personality,
        time_ms=max(50, time_ms),
        beam_width=max(4, beam_width),
        turn_number=turn_number,
        max_actions=max_actions,
    )
    forced_targets = engine.BB_EMPTY
    try:
        opponent_turn = ordering_probe.board_as_turn(board, not mover)
        opponent_moves = list(opponent_turn.generate_legal_moves())
        if opponent_moves and all(move.name == "AutoCapture" for move in opponent_moves):
            for move in opponent_moves:
                if move.capture_preference is not None:
                    forced_targets |= engine.BB_SQUARES[move.capture_preference]
    except (SearchTimeout, ValueError, AssertionError):
        forced_targets = engine.BB_EMPTY

    legal = list(board.generate_legal_moves())
    voluntary_exists = any(move.name not in ("AutoCapture", "Skip") for move in legal)

    def recovery_order(move: engine.Move) -> Tuple[int, int, int, str]:
        rescues_forced_target = bool(
            move.from_square is not None
            and engine.BB_SQUARES[move.from_square] & forced_targets
        )
        capture = move.capture_preference is not None
        filler_skip = move.name == "Skip" and voluntary_exists
        return (
            0 if rescues_forced_target else 1,
            0 if capture else 1,
            1 if filler_skip else 0,
            move.uci(),
        )

    legal.sort(key=recovery_order)
    candidates: List[Tuple[Tuple[float, float, float, str], TurnCandidate]] = []
    for first_move in legal[:32]:
        if time.monotonic() >= deadline:
            break
        child = board.copy()
        try:
            child.push(first_move)
        except (ValueError, AssertionError):
            continue
        moves = [first_move]
        resulting = child
        if not child.is_game_over() and child.turn == mover:
            remaining_ms = max(50, int((deadline - time.monotonic()) * 1000.0))
            tail_seed = purposeful_complete_turn_seed(
                child,
                personality,
                turn_number,
                max_actions=max_actions,
                time_ms=min(250, remaining_ms),
            )
            tail_moves, resulting = first_turn_from_pv(child, tail_seed.pv)
            moves.extend(tail_moves)
        if not resulting.is_game_over() and resulting.turn == mover:
            continue

        remaining_ms = max(50, int((deadline - time.monotonic()) * 1000.0))
        safety = bounded_seed_safety(
            board,
            resulting,
            personality,
            turn_number,
            beam_width,
            max_actions,
            min(500, remaining_ms),
            check_hq_combinations=False,
        )
        if safety is None or not safety.tactically_safe:
            continue

        hq_probe = Searcher(
            personality,
            time_ms=min(500, remaining_ms),
            beam_width=4,
            turn_number=turn_number,
            max_actions=max_actions,
        )
        try:
            if hq_probe.has_same_turn_hq_capture(resulting):
                continue
        except SearchTimeout:
            continue

        purpose = ordering_probe.deadline_safe_turn_purpose_breakdown(
            board, resulting, moves, mover, retrospective=False
        )
        if purpose["paratrooper_mission_penalty"] > 0.0:
            continue
        action_purposes = ordering_probe.deadline_safe_action_purpose_labels(
            board, moves, mover, retrospective=False
        )
        candidate = TurnCandidate(
            moves=moves,
            board=resulting,
            priority=0.0,
            static_score=ordering_probe.quick_score(resulting),
            safety_penalty=max(
                0.0, safety.new_risk_value - safety.compensation_value
            ),
            tactically_safe=True,
            purpose_penalty=purpose["net_purpose_penalty"],
            paratrooper_mission_penalty=purpose["paratrooper_mission_penalty"],
            action_purposes=action_purposes,
            early_plan_score=ordering_probe.early_plan_score(action_purposes),
        )
        mover_score = (
            candidate.static_score if mover == engine.RED else -candidate.static_score
        )
        candidates.append((
            (
                safety.forced_loss_value,
                candidate.safety_penalty,
                -mover_score,
                normalized_turn_key(moves, mover),
            ),
            candidate,
        ))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


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
    policy_function: Optional[Any] = None,
) -> Dict[str, Any]:
    started = time.monotonic()
    # The Vercel Python function has a 60-second outer limit. Search used to
    # reset its main budget after seed construction and then start fresh
    # post-search verifiers, allowing a nominal 20-second request to run past
    # the platform ceiling and return an HTML 504. Preserve a generous
    # tactical reserve while leaving time for response construction and the
    # caller's network overhead.
    hard_budget_ms = min(
        50_000,
        max(time_ms + 3_000, int(time_ms * 2.25)),
    )
    overall_deadline = started + hard_budget_ms / 1000.0
    # Exact HQ proofs are valuable, but they must not consume the small slice
    # needed to prove that the final serialized turn does not simply hang a
    # piece.  The latter check is deliberately cheap (HQ combinations are
    # handled separately) and is the last line of defence against a timed-out
    # seed such as the historical f5-h3 armored-infantry suicide.
    final_safety_reserve_ms = min(
        5_000,
        max(1_000, int(time_ms * 0.20)),
    )
    post_search_probe_deadline = (
        overall_deadline - final_safety_reserve_ms / 1000.0
    )

    def remaining_overall_ms(maximum: int) -> int:
        return max(
            0,
            min(maximum, int((overall_deadline - time.monotonic()) * 1000.0)),
        )

    searcher = Searcher(
        personality,
        time_ms,
        beam_width,
        turn_number=turn_number,
        value_function=value_function,
        policy_function=policy_function,
        max_actions=max_actions,
        stagnation_turns=stagnation_turns,
    )
    searcher.root_key = board_key(board)
    best: Optional[SearchResult] = None
    completed_depth = 0
    timed_out = False
    fallback_kind = "none"
    hq_survival_override_used = False
    hq_survival_reply_verified = False
    hq_exact_return_probe_used = False
    policy_return_guard_used = False
    tactical_return_guard_used = False
    safe_fallback_reply_verified = False
    safe_fallback_reply_nodes = 0
    emergency_seed: Optional[SearchResult] = None
    emergency_seed_safe = False
    seed_reply_retry_used = False
    seed_moves: List[engine.Move] = []
    seed_board = board
    verified_seed: Optional[SearchResult] = None
    book_turn = opening_book_turn(board, turn_number, searcher, opening_seed)
    opening_book_used = book_turn is not None
    seed_time_ms = max(50, min(2_000, int(time_ms * 0.08)))
    if book_turn is not None:
        transition_penalty = book_turn.purpose_penalty + book_turn.paratrooper_mission_penalty
        early_bonus = (
            0.40 * book_turn.early_plan_score
            if turn_number <= EARLY_GAME_LAST_TURN
            else 0.0
        )
        turn_quality = (
            early_bonus
            - transition_penalty
            + searcher.deadline_safe_transition_policy_score(
                book_turn.board, board.turn
            )
        )
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
            board,
            personality,
            turn_number,
            max_actions=max_actions,
            time_ms=seed_time_ms,
            stagnation_turns=stagnation_turns,
        )
        seed_moves, seed_board = first_turn_from_pv(board, emergency_seed.pv)
        seed_purpose = searcher.deadline_safe_turn_purpose_breakdown(
            board,
            seed_board,
            seed_moves,
            board.turn,
            retrospective=False,
        )
        seed_penalty = seed_purpose["total_penalty"]
        seed_policy_score = searcher.deadline_safe_transition_policy_score(
            seed_board, board.turn
        )
        emergency_seed.score += (
            seed_policy_score - seed_penalty
            if board.turn == engine.RED
            else seed_penalty - seed_policy_score
        )
        seed_counted_actions = sum(
            move.name not in ("AutoCapture", "Skip") for move in seed_moves
        )

        def scored_verified_seed(reply: SearchResult) -> SearchResult:
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
            seed_quality = early_bonus - seed_penalty + seed_policy_score
            return SearchResult(
                reply.score
                + (seed_quality if board.turn == engine.RED else -seed_quality),
                list(seed_moves) + list(reply.pv),
            )

        if seed_board.is_game_over() or seed_counted_actions >= searcher.max_actions:
            # Seed construction and safety are a bounded pre-search floor.
            # Running this check on the main Searcher let a complicated
            # safety probe consume the entire minimax deadline before the
            # reserved opponent-reply verification could even begin. Keep it
            # isolated so a timeout can only forfeit seed certification, not
            # the actual search budget.
            seed_safety = bounded_seed_safety(
                board,
                seed_board,
                personality,
                turn_number,
                beam_width,
                max_actions,
                seed_time_ms,
            )
            if (
                seed_safety is not None
                and seed_safety.tactically_safe
                and seed_purpose["paratrooper_mission_penalty"] <= 0.0
            ):
                emergency_seed_safe = True
                seed_action_purposes = searcher.deadline_safe_action_purpose_labels(
                    board,
                    seed_moves,
                    board.turn,
                    retrospective=False,
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
        # Seed construction and its isolated safety probe are a prerequisite,
        # not part of minimax. Reset the main deadline here so a slow Vercel
        # worker still receives the requested tactical-search budget. Before
        # this reset the separate seed searcher could consume most of the
        # absolute deadline, leaving zero time to generate root alternatives.
        main_search_started = time.monotonic()
        final_deadline = min(
            overall_deadline,
            main_search_started + max(1, time_ms) / 1000.0,
        )
        searcher.deadline = final_deadline
        if requested_depth >= 2:
            # First verify the purposeful emergency turn against one complete
            # opponent reply. This produces a tactically checked floor even if
            # enumerating alternative root turns later exhausts the budget.
            # It is still labelled a fallback because root alternatives were
            # not all compared at the same horizon. Keep the optional latent-HQ
            # extension out of this bounded floor: immediate HQ captures at the
            # leaf are still recognized, while a second defensive turn cannot
            # consume the reserve before one opponent reply is certified.
            if seed_board.is_game_over():
                verified_seed = SearchResult(emergency_seed.score, list(seed_moves))
            elif seed_board.turn != board.turn:
                searcher.verification_mode = True
                searcher.deadline = min(
                    final_deadline,
                    main_search_started
                    + max(0.05, time_ms / 1000.0 * 0.40),
                )
                try:
                    reply = searcher.alphabeta(
                        seed_board, 1, -math.inf, math.inf
                    )
                    verified_seed = scored_verified_seed(reply)
                except SearchTimeout:
                    timed_out = True
                finally:
                    searcher.deadline = final_deadline
                    # A completed seed reply is a valid depth-one
                    # transposition. Reuse it when the narrow root pass reaches
                    # that same child instead of paying for the reply twice.

            searcher.hq_leaf_extension_enabled = True
            # If the emergency line is already reply-verified, alternatives
            # may consume the entire remaining budget: timing out can fall
            # back to that certified floor. Otherwise stop at 80% and retain
            # the final slice for one last seed-reply attempt. The initial
            # seed probe receives a contiguous 40% because interrupted turn
            # generation is not cacheable; two 20% probes repeated the same
            # work and caused rare depth-zero Vercel fallbacks.
            searcher.verification_mode = True
            searcher.root_verified_lines = []
            searcher.root_ranked_turns = []
            searcher.deadline = min(
                final_deadline,
                main_search_started
                + max(
                    0.05,
                    time_ms
                    / 1000.0
                    * (1.0 if verified_seed is not None else 0.80),
                ),
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
                if (
                    verified_seed is None
                    and not seed_board.is_game_over()
                    and seed_board.turn != board.turn
                ):
                    # The narrow root pass owns the first 80% of the absolute
                    # deadline. If it cannot finish one reply, spend the
                    # reserved final slice completing the already-started
                    # emergency-seed reply instead of restarting a shallow
                    # root search. Partial transposition entries from the
                    # first seed probe are reusable here, so difficult Vercel
                    # positions can still return a reply-verified floor.
                    seed_reply_retry_used = True
                    searcher.verification_mode = True
                    searcher.hq_leaf_extension_enabled = False
                    try:
                        reply = searcher.alphabeta(
                            seed_board, 1, -math.inf, math.inf
                        )
                        verified_seed = scored_verified_seed(reply)
                    except SearchTimeout:
                        timed_out = True
                    finally:
                        searcher.hq_leaf_extension_enabled = True
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

    # Every remaining exact probe, recovery, and fallback verifier shares one
    # absolute request deadline. Individual helpers may finish earlier, but no
    # later replacement path receives a fresh clock that can cross Vercel's
    # hard function limit.
    searcher.deadline = post_search_probe_deadline

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
            fallback_quality = (
                fallback_bonus
                - fallback_penalty
                + searcher.deadline_safe_transition_policy_score(
                    searcher.root_fallback.board, board.turn
                )
            )
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
                board,
                personality,
                turn_number,
                max_actions=max_actions,
                time_ms=seed_time_ms,
                stagnation_turns=stagnation_turns,
            )
            fallback_kind = "seeded"
    first_turn, resulting_board = first_turn_from_pv(board, best.pv)
    selected_current_player_score = (
        best.score if board.turn == engine.RED else -best.score
    )
    if (
        emergency_seed is not None
        and emergency_seed_safe
        and completed_depth < 2
        and not resulting_board.is_game_over()
        and resulting_board.turn != board.turn
        and (
            selected_current_player_score <= -MATE_SCORE + 100.0
            or searcher.has_same_turn_hq_capture(resulting_board)
        )
    ):
        # A partial shallow search can finish with only losing root branches
        # even though the precomputed safety seed found a coordinated check
        # evasion. Safety is lexicographic here: never replace a known legal
        # escape with an immediately losing line merely because the latter
        # came from an incomplete minimax pass.
        best = emergency_seed
        first_turn = list(seed_moves)
        resulting_board = seed_board
        completed_depth = 0
        fallback_kind = "safe"
    selected_current_player_score = (
        best.score if board.turn == engine.RED else -best.score
    )
    exact_return_hq_threat = False
    if (
        not resulting_board.is_game_over()
        and resulting_board.turn != board.turn
        and surviving_non_hq_count(resulting_board) <= 12
        and not searcher.has_same_turn_hq_capture(resulting_board)
    ):
        # Sparse late games can hide artillery-orientation combinations that
        # the deliberately collapsed reply beam does not retain.  Exhaust the
        # complete same-turn HQ action set before returning such a position.
        # The sparse-board gate keeps this exact proof out of ordinary crowded
        # middlegames; an incomplete proof never certifies safety.
        hq_exact_return_probe_used = True
        exact_return_hq_threat = (
            searcher.exact_same_turn_hq_capture(
                resulting_board,
                resulting_board.turn,
                [100_000],
            )
            is True
        )
    if (
        not resulting_board.is_game_over()
        and resulting_board.turn != board.turn
        and (
            selected_current_player_score <= -MATE_SCORE + 100.0
            or searcher.has_same_turn_hq_capture(resulting_board)
            or exact_return_hq_threat
        )
    ):
        survival = searcher.find_hq_survival_turn(
            board,
            max_reply_nodes=(
                1_000_000
                if surviving_non_hq_count(board) <= 12
                else 100_000
            ),
        )
        if survival is not None:
            # Minimax can correctly decide that every line is eventually
            # losing yet still use purpose terms to choose a line that loses
            # the HQ immediately. Exact one-turn verification is the final
            # lexicographic floor: survive now and force the opponent to prove
            # the longer win. The result remains a labelled fallback because
            # it deliberately overrides the approximate horizon ordering.
            first_turn, resulting_board = survival
            hq_survival_override_used = True
            survival_purpose = searcher.deadline_safe_turn_purpose_breakdown(
                board,
                resulting_board,
                first_turn,
                board.turn,
                retrospective=False,
            )
            survival_penalty = survival_purpose["total_penalty"]
            survival_early_bonus = (
                0.40
                * searcher.early_plan_score(
                    searcher.deadline_safe_action_purpose_labels(
                        board,
                        first_turn,
                        board.turn,
                        retrospective=False,
                    )
                )
                if turn_number <= EARLY_GAME_LAST_TURN
                else 0.0
            )
            survival_quality = (
                survival_early_bonus
                - survival_penalty
                + searcher.deadline_safe_transition_policy_score(
                    resulting_board, board.turn
                )
            )
            previous_verification_mode = searcher.verification_mode
            searcher.verification_mode = True
            try:
                reply = searcher.alphabeta(
                    resulting_board, 1, -math.inf, math.inf
                )
                verified_score = reply.score + (
                    survival_quality
                    if board.turn == engine.RED
                    else -survival_quality
                )
                best = SearchResult(
                    verified_score, list(first_turn) + list(reply.pv)
                )
                completed_depth = 2
                hq_survival_reply_verified = True
            except SearchTimeout:
                timed_out = True
                best = SearchResult(
                    searcher.quick_score(resulting_board), first_turn
                )
                completed_depth = 0
            finally:
                searcher.verification_mode = previous_verification_mode
            fallback_kind = "safe"

    # Release the protected slice only after every potentially expensive HQ
    # proof has finished. Material safety and its recovery pass can now fail
    # closed without letting the request cross the absolute Vercel deadline.
    searcher.deadline = overall_deadline

    # Approximate minimax is allowed to rank positions imperfectly, but it may
    # not return a turn that the objective safety probe already knows loses a
    # para, artillery, or unsupported infantry by force.  A timed-out native
    # search once selected an armored-infantry move to h3 even though its own
    # PV contained the immediate ``sfh3`` capture.  Recheck the selected turn
    # from the completed root assessments, then restore the strongest checked
    # safe root alternative. Unknown return paths receive a fresh bounded
    # assessment. This guard is deliberately after every ordinary search or
    # fallback path and before the policy guards below.
    selected_move_key = tuple(move.uci() for move in first_turn)
    known_root_candidates = [
        candidate for _, candidate in searcher.root_ranked_turns
    ]
    known_root_candidates.extend(
        candidate for _, candidate, _ in searcher.root_verified_lines
    )
    if searcher.root_fallback is not None:
        known_root_candidates.append(searcher.root_fallback)
    known_root_candidates.extend(
        searcher.turn_cache.get(searcher.root_key or "", [])
    )
    selected_candidate = next(
        (
            candidate
            for candidate in known_root_candidates
            if tuple(move.uci() for move in candidate.moves)
            == selected_move_key
        ),
        None,
    )
    # TurnCandidate objects are created only after assess_turn_safety returns;
    # a timed-out assessment never becomes a candidate. Reuse that completed
    # proof instead of spending the post-search reserve proving the same root
    # position again. Unknown return paths still fail closed through a fresh
    # bounded assessment.
    if selected_candidate is not None:
        selected_is_tactically_safe = selected_candidate.tactically_safe
    else:
        selected_safety_budget = remaining_overall_ms(seed_time_ms)
        selected_safety = (
            bounded_seed_safety(
                board,
                resulting_board,
                personality,
                turn_number,
                beam_width,
                max_actions,
                selected_safety_budget,
                check_hq_combinations=False,
            )
            if selected_safety_budget >= 50
            else None
        )
        selected_is_tactically_safe = bool(
            selected_safety is not None and selected_safety.tactically_safe
        )
    if (
        not hq_survival_override_used
        and not selected_is_tactically_safe
    ):
        verified_seed_move_key = tuple(move.uci() for move in seed_moves)
        verified_seed_is_known_safe = emergency_seed_safe or any(
            candidate.tactically_safe
            and tuple(move.uci() for move in candidate.moves)
            == verified_seed_move_key
            for candidate in known_root_candidates
        )
        # The reply-first floor may already have proved the emergency seed
        # against one complete opponent turn while either the bounded seed
        # probe or ordinary root generation proved its objective safety.
        # Prefer that exact doubly-proven line over a merely safety-screened
        # replacement. Otherwise a late tactical guard can throw away depth
        # two, time out in the fresh verifier, and report a depth-zero fallback.
        if verified_seed is not None and verified_seed_is_known_safe:
            first_turn = list(seed_moves)
            resulting_board = seed_board
            best = verified_seed
            completed_depth = 2
            fallback_kind = "safe"
            tactical_return_guard_used = True

        replacement_options: List[TurnCandidate] = []
        seen_replacements: set[Tuple[str, ...]] = set()

        def consider_replacement(candidate: Optional[TurnCandidate]) -> None:
            if candidate is None:
                return
            # Candidate construction already completed the objective tactical
            # assessment. Discard known losses before the recovery pass and
            # preserve the post-search budget for verifying the opponent's
            # full reply to the first known-safe replacement.
            if not candidate.tactically_safe:
                return
            move_key = tuple(move.uci() for move in candidate.moves)
            if move_key == tuple(move.uci() for move in first_turn):
                return
            if move_key in seen_replacements:
                return
            seen_replacements.add(move_key)
            replacement_options.append(candidate)

        if not tactical_return_guard_used:
            for _, candidate in searcher.root_ranked_turns:
                consider_replacement(candidate)
            consider_replacement(searcher.root_fallback)
            for candidate in searcher.turn_cache.get(searcher.root_key or "", []):
                consider_replacement(candidate)

            for replacement in replacement_options[:8]:
                first_turn = list(replacement.moves)
                resulting_board = replacement.board
                best = SearchResult(replacement.static_score, list(first_turn))
                completed_depth = 0
                fallback_kind = "safe"
                tactical_return_guard_used = True
                break

        if not tactical_return_guard_used:
            recovery_budget = remaining_overall_ms(
                min(5_000, max(500, int(time_ms * 0.25)))
            )
            recovery = (
                material_safe_recovery_turn(
                    board,
                    personality,
                    turn_number,
                    beam_width,
                    max_actions,
                    recovery_budget,
                )
                if recovery_budget >= 50
                else None
            )
            if recovery is not None:
                first_turn = list(recovery.moves)
                resulting_board = recovery.board
                best = SearchResult(recovery.static_score, list(first_turn))
                completed_depth = 0
                fallback_kind = "safe"
                tactical_return_guard_used = True

    selected_policy = searcher.deadline_safe_turn_purpose_breakdown(
        board,
        resulting_board,
        first_turn,
        board.turn,
        retrospective=False,
    )
    if selected_policy["paratrooper_mission_penalty"] > 0.0:
        policy_return_guard_used = True
        # HQ-survival and deadline recovery are deliberately allowed to
        # override approximate score ordering, but never the objective para
        # mission policy. Restore the already screened root fallback before
        # response serialization so no return path can leak a violating turn.
        replacement = searcher.root_fallback
        if (
            replacement is not None
            and replacement.tactically_safe
            and replacement.paratrooper_mission_penalty <= 0.0
        ):
            first_turn = list(replacement.moves)
            resulting_board = replacement.board
            best = SearchResult(replacement.static_score, list(first_turn))
            completed_depth = 0
            fallback_kind = "safe"
        else:
            clean_seed_budget = remaining_overall_ms(seed_time_ms)
            if clean_seed_budget >= 50:
                clean_seed = purposeful_complete_turn_seed(
                    board,
                    personality,
                    turn_number,
                    max_actions=max_actions,
                    time_ms=clean_seed_budget,
                    stagnation_turns=stagnation_turns,
                )
                clean_moves, clean_board = first_turn_from_pv(
                    board, clean_seed.pv
                )
                clean_purpose = searcher.deadline_safe_turn_purpose_breakdown(
                    board,
                    clean_board,
                    clean_moves,
                    board.turn,
                    retrospective=False,
                )
                clean_safety_budget = remaining_overall_ms(seed_time_ms)
                clean_safety = (
                    bounded_seed_safety(
                        board,
                        clean_board,
                        personality,
                        turn_number,
                        beam_width,
                        max_actions,
                        clean_safety_budget,
                        check_hq_combinations=False,
                    )
                    if clean_safety_budget >= 50
                    else None
                )
                if (
                    clean_purpose["paratrooper_mission_penalty"] <= 0.0
                    and clean_safety is not None
                    and clean_safety.tactically_safe
                ):
                    first_turn = clean_moves
                    resulting_board = clean_board
                    best = clean_seed
                    completed_depth = 0
                    fallback_kind = "seeded"
    if not first_turn and not board.is_game_over():
        empty_pv_seed_budget = remaining_overall_ms(seed_time_ms)
        fallback = purposeful_complete_turn_seed(
            board,
            personality,
            turn_number,
            max_actions=max_actions,
            time_ms=max(1, empty_pv_seed_budget),
            stagnation_turns=stagnation_turns,
        )
        first_turn, resulting_board = first_turn_from_pv(board, fallback.pv)
        best = fallback
        seed_purpose = searcher.deadline_safe_turn_purpose_breakdown(
            board,
            resulting_board,
            first_turn,
            board.turn,
            retrospective=False,
        )
        seed_penalty = seed_purpose["total_penalty"]
        seed_quality = (
            searcher.deadline_safe_transition_policy_score(
                resulting_board, board.turn
            )
            - seed_penalty
        )
        best.score += seed_quality if board.turn == engine.RED else -seed_quality
        fallback_kind = "seeded"

    # This check intentionally lives after every ordinary fallback.  Candidate
    # filters, opening books, deadline seeds, and HQ-survival overrides may be
    # refactored independently, but none may return a missionless paratrooper
    # action to the caller.  The deterministic floor resolves only mandatory
    # captures and then ends the turn, so the invariant is unconditional.
    final_policy = searcher.deadline_safe_turn_purpose_breakdown(
        board,
        resulting_board,
        first_turn,
        board.turn,
        retrospective=False,
    )
    if final_policy["paratrooper_mission_penalty"] > 0.0:
        policy_return_guard_used = True
        first_turn, resulting_board = deterministic_skip_turn(board)
        best = SearchResult(
            searcher.quick_score(resulting_board), list(first_turn)
        )
        completed_depth = 0
        fallback_kind = "safe"

    # Safety and policy guards run after the main minimax deadline. Returning
    # their replacement at depth zero made otherwise useful self-play games
    # ineligible for training even when Vercel still had enough function time
    # for one complete opponent reply. Verify the final safe replacement with
    # a fresh, isolated searcher; never claim depth two on a partial reply.
    if (
        fallback_kind == "safe"
        and completed_depth < 2
        and not resulting_board.is_game_over()
        and resulting_board.turn != board.turn
    ):
        verifier_budget = remaining_overall_ms(
            min(30_000, max(2_000, int(time_ms * 1.50)))
        )
        verifier = Searcher(
            personality,
            time_ms=max(1, verifier_budget),
            beam_width=max(4, min(6, beam_width)),
            turn_number=turn_number,
            value_function=value_function,
            policy_function=policy_function,
            max_actions=max_actions,
            stagnation_turns=stagnation_turns,
        )
        # This isolated search begins at an opponent child of the real root.
        # Preserve that identity so generate_turn_candidates uses the narrow
        # reply-verification path (with the same atomic tactical beam) instead
        # of spending the reserve on a second broad root frontier.
        verifier.root_key = board_key(board)
        verifier.verification_mode = True
        verifier.hq_leaf_extension_enabled = False
        try:
            if verifier_budget < 50:
                raise SearchTimeout
            reply = verifier.alphabeta(
                resulting_board, 1, -math.inf, math.inf
            )
            selected_purpose = searcher.deadline_safe_turn_purpose_breakdown(
                board,
                resulting_board,
                first_turn,
                board.turn,
                retrospective=False,
            )
            selected_quality = (
                (
                    0.40
                    * searcher.early_plan_score(
                        searcher.deadline_safe_action_purpose_labels(
                            board,
                            first_turn,
                            board.turn,
                            retrospective=False,
                        )
                    )
                    if turn_number <= EARLY_GAME_LAST_TURN
                    else 0.0
                )
                - selected_purpose["total_penalty"]
                + searcher.deadline_safe_transition_policy_score(
                    resulting_board, board.turn
                )
            )
            verified_score = reply.score + (
                selected_quality if board.turn == engine.RED else -selected_quality
            )
            best = SearchResult(
                verified_score, list(first_turn) + list(reply.pv)
            )
            completed_depth = 2
            safe_fallback_reply_verified = True
            if hq_survival_override_used:
                hq_survival_reply_verified = True
        except SearchTimeout:
            timed_out = True
        finally:
            safe_fallback_reply_nodes = verifier.nodes

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
            quality = (
                early_bonus
                - transition_penalty
                + searcher.deadline_safe_transition_policy_score(
                    turn.board, board.turn
                )
            )
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
            action_purposes = searcher.deadline_safe_action_purpose_labels(
                board,
                first_turn,
                board.turn,
            )
            purpose = searcher.deadline_safe_turn_purpose_breakdown(
                board,
                resulting_board,
                first_turn,
                board.turn,
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
                    for key, value in searcher.deadline_safe_turn_purpose_breakdown(
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
    # Include fallback construction and response telemetry. Measuring before
    # these bounded post-search steps hid the real wall-clock cost precisely
    # when a deadline fallback was most expensive.
    elapsed_ms = (time.monotonic() - started) * 1000.0
    return {
        "recommendation_label": recommendation_label,
        "input_fen": board.board_fen(),
        "side_to_move": color_name(board.turn),
        "best_turn": {
            "automatic_captures": automatic,
            "actions": actions,
            "all_moves": [move.uci() for move in first_turn],
            "resulting_fen": resulting_board.board_fen(),
            "action_purposes": searcher.deadline_safe_action_purpose_labels(
                board,
                first_turn,
                board.turn,
            ),
            "purpose": {
                key: round(value, 4)
                for key, value in searcher.deadline_safe_turn_purpose_breakdown(
                    board,
                    resulting_board,
                    first_turn,
                    board.turn,
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
            "hard_deadline_ms": hard_budget_ms,
            "hard_deadline_reached": time.monotonic() >= overall_deadline,
            "final_safety_reserve_ms": final_safety_reserve_ms,
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
            "policy_model_evaluations": searcher.policy_model_evaluations,
            "turn_cache_hits": searcher.turn_cache_hits,
            "transposition_hits": searcher.transposition_hits,
            "hq_survival_probe_nodes": searcher.hq_survival_probe_nodes,
            "hq_survival_reply_nodes": searcher.hq_survival_reply_nodes,
            "hq_survival_override_used": hq_survival_override_used,
            "hq_survival_reply_verified": hq_survival_reply_verified,
            "hq_exact_return_probe_used": hq_exact_return_probe_used,
            "policy_return_guard_used": policy_return_guard_used,
            "tactical_return_guard_used": tactical_return_guard_used,
            "safe_fallback_reply_verified": safe_fallback_reply_verified,
            "safe_fallback_reply_nodes": safe_fallback_reply_nodes,
            "seed_reply_verified": verified_seed is not None,
            "seed_reply_retry_used": seed_reply_retry_used,
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
