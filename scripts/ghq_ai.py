#!/usr/bin/env python3
"""Explainable, headless GHQ search engine.

Search depth is measured in complete player turns. A turn can contain automatic
captures followed by up to three player actions, matching public/engine.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
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
    },
    "mobile": {
        "material": 0.90,
        "support": 0.85,
        "mobility": 1.60,
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
        "open_board_armored_infantry": 1.55,
        "artillery_formation": 0.70,
    },
    "battery_commander": {
        "artillery_formation": 1.55,
        "artillery_pressure": 1.45,
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
    "artillery_formation": 1.0,
    "artillery_pressure": 1.0,
    "artillery_protection": 1.0,
    "mobility": 1.0,
    "open_board_armored_infantry": 1.0,
    "airborne_survival": 1.0,
    "hq_safety": 1.0,
    "development": 1.0,
}

MATE_SCORE = 1_000_000.0


def squares(mask: int) -> List[int]:
    return list(engine.scan_reversed(mask))


def chebyshev(a: int, b: int) -> int:
    return max(abs(engine.square_file(a) - engine.square_file(b)), abs(engine.square_rank(a) - engine.square_rank(b)))


def color_name(color: bool) -> str:
    return "blue" if color == engine.BLUE else "red"


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
    anchor = max(range(8), key=lambda rank: rank_power[rank])
    penalty = 0.0
    for square in units:
        advance = engine.square_rank(square) - anchor if color == engine.RED else anchor - engine.square_rank(square)
        if advance >= 2:
            piece_type = board.piece_type_at(square)
            if piece_type is not None:
                penalty += 0.35 * PIECE_VALUES[piece_type] * (advance - 1)
    return penalty


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
        "artillery_formation": artillery_formation(board, engine.RED) - artillery_formation(board, engine.BLUE),
        "artillery_pressure": artillery_pressure(board, engine.RED) - artillery_pressure(board, engine.BLUE),
        "artillery_protection": artillery_exposure_penalty(board, engine.BLUE)
        - artillery_exposure_penalty(board, engine.RED),
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
    ) -> None:
        self.personality = personality
        self.time_ms = max(1, time_ms)
        self.turn_number = max(1, turn_number)
        self.value_function = value_function
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
        self.tactically_unsafe_turns = 0
        self.rotation_quota_pruned = 0
        self.value_model_evaluations = 0
        self.turn_cache: Dict[str, List[TurnCandidate]] = {}
        self.value_cache: Dict[str, float] = {}
        self.safety_cache: Dict[Tuple[str, bool], Tuple[float, float, float]] = {}
        self.root_key: Optional[str] = None
        self.root_fallback: Optional[TurnCandidate] = None

    def check_time(self, count_node: bool = True) -> None:
        if count_node:
            self.nodes += 1
        if time.monotonic() >= self.deadline:
            raise SearchTimeout

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
        friendly_non_hq = own & ~board.hq
        for position, _ in action_positions[:12]:
            self.check_time(False)
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
        baseline, _, _ = self.tactical_risk(before, mover)
        risk, forced, critical = self.tactical_risk(after, mover)
        opponent = not mover
        compensation = max(
            0.0,
            board_material_for(before, opponent) - board_material_for(after, opponent),
        )
        new_risk = max(0.0, risk - baseline)
        uncovered = max(0.0, new_risk - compensation)
        # Tactics are objective. Personalities may choose among safe turns but
        # cannot spend an exposed gun/para without verified material return.
        safe = uncovered <= 0.75 and forced <= compensation + 0.75
        return TacticalSafety(risk, new_risk, compensation, forced, critical, safe)

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

    def artillery_move_allowed(self, board: engine.BaseBoard, move: engine.Move) -> bool:
        """Apply the user's provisional artillery-orientation search rules.

        Artillery farther than three squares from its nearest enemy may still
        relocate, but it keeps its current useful facing rather than creating
        eight equivalent orientation branches. Pure distant rotations are
        discarded. Homeward-facing rotations are currently discarded at all
        distances.
        """
        if move.name != "MoveAndOrient" or move.from_square is None or move.to_square is None:
            return True
        piece_type = board.piece_type_at(move.from_square)
        if piece_type not in ARTILLERY_TYPES:
            return True
        if self.points_toward_home(board.turn, move.orientation):
            return False

        distance = self.closest_enemy_distance(board, move.to_square, board.turn)
        if distance <= 3:
            return True
        if move.from_square == move.to_square:
            return False

        current = board.get_orientation(move.from_square)
        if current is None or self.points_toward_home(board.turn, current):
            current = engine.ORIENT_N if board.turn == engine.RED else engine.ORIENT_S
        return move.orientation == current

    @staticmethod
    def move_piece_type(board: engine.BaseBoard, move: engine.Move) -> Optional[int]:
        return move.unit_type if move.name == "Reinforce" else (
            board.piece_type_at(move.from_square) if move.from_square is not None else None
        )

    @staticmethod
    def home_distance(square: int, color: bool) -> int:
        home_rank = 0 if color == engine.RED else 7
        return abs(engine.square_rank(square) - home_rank)

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

    def move_priority(self, board: engine.BaseBoard, move: engine.Move) -> float:
        if move.name == "AutoCapture":
            target = board.piece_type_at(move.capture_preference) if move.capture_preference is not None else None
            return 10000.0 + 100.0 * PIECE_VALUES.get(target, 0.0)
        if move.capture_preference is not None:
            target = board.piece_type_at(move.capture_preference)
            return 5000.0 + 100.0 * PIECE_VALUES.get(target, 0.0)
        if move.name == "Skip":
            return -10000.0
        priority = 0.0
        piece_type = self.move_piece_type(board, move)
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
        priority += self.artillery_target_bonus(board, move)
        if piece_type == engine.ARMORED_INFANTRY:
            priority += 20.0
        elif piece_type in ARTILLERY_TYPES:
            priority += 10.0
        if move.name == "MoveAndOrient":
            priority += 5.0
        return priority

    def diverse_moves(self, board: engine.BaseBoard) -> List[Tuple[float, engine.Move]]:
        """Return strategically filtered actions without applying the beam."""
        legal = list(board.generate_legal_moves())
        if legal and all(move.name == "AutoCapture" for move in legal):
            return [(self.move_priority(board, move), move) for move in legal]

        moves = [move for move in legal if self.artillery_move_allowed(board, move)]
        if len(moves) != len(legal):
            self.exhaustive_within_horizon = False
            self.rule_filtered_actions += len(legal) - len(moves)

        scored = [(self.move_priority(board, move), move.uci(), move) for move in moves]
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

    def ordered_moves(self, board: engine.BaseBoard) -> List[engine.Move]:
        diverse_scored = self.diverse_moves(board)
        diverse = [move for _, move in diverse_scored]
        if diverse and all(move.name == "AutoCapture" for move in diverse):
            return diverse

        selected = diverse[: self.beam_width]
        priority_by_uci = {move.uci(): priority for priority, move in diverse_scored}
        critical_limit = max(self.beam_width * 2, self.beam_width + 4)
        for move in diverse:
            if len(selected) >= critical_limit:
                break
            if move not in selected and priority_by_uci.get(move.uci(), 0.0) >= 4000.0:
                selected.append(move)
        if len(selected) != len(diverse):
            self.exhaustive_within_horizon = False
            self.beam_pruned_actions += len(diverse) - len(selected)

        # Ending early is considered only for the final optional action. This
        # prevents a permanently retained Skip from beating quiet two-action
        # setup/extraction combinations after just one move.
        skip = next((move for move in diverse if move.name == "Skip"), None)
        if board.turn_moves >= 2 and skip is not None and skip not in selected:
            selected.append(skip)
        return selected

    @staticmethod
    def _prefer_partial(candidate: PartialTurn, incumbent: PartialTurn) -> bool:
        if candidate.priority != incumbent.priority:
            return candidate.priority > incumbent.priority
        return tuple(move.uci() for move in candidate.moves) < tuple(move.uci() for move in incumbent.moves)

    @staticmethod
    def _round_robin_partials(partials: Sequence[PartialTurn], limit: int) -> List[PartialTurn]:
        """Preserve multiple first actions instead of one prolific branch."""
        groups: Dict[str, List[PartialTurn]] = {}
        for partial in partials:
            first = partial.moves[0].uci() if partial.moves else ""
            groups.setdefault(first, []).append(partial)
        for values in groups.values():
            values.sort(key=lambda item: (-item.priority, tuple(move.uci() for move in item.moves)))

        ordered_groups = sorted(
            groups.values(),
            key=lambda values: (-values[0].priority, values[0].moves[0].uci() if values[0].moves else ""),
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
        extraction_unlocks: List[str] = []
        for move in moves:
            piece_type = self.move_piece_type(working, move)
            if self.unlocks_airborne_extraction(working, move):
                extraction_unlocks.append(move.uci())
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
            working.push(move)

        keys: List[Tuple[Any, ...]] = []
        keys.extend(("artillery_escape", source, destination) for source, destination in artillery_escapes)
        keys.extend(("airborne_extraction", source, destination) for source, destination in airborne_extractions)
        keys.extend(("airborne_unlock", uci) for uci in extraction_unlocks)
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
        return classes, wasteful_rotation

    @staticmethod
    def candidate_sort_key(candidate: TurnCandidate, color: bool) -> Tuple[Any, ...]:
        return (
            candidate.safety_penalty,
            -(candidate.static_score if color == engine.RED else -candidate.static_score),
            len(candidate.moves) if candidate.board.is_game_over() else 99,
            -candidate.priority,
            tuple(move.uci() for move in candidate.moves),
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

        # Guarantee useful alternatives before filling by score. A beam is a
        # count of complete turns retained, not permission for one action type
        # to consume every slot.
        for action_class in (
            "reinforcement",
            "infantry",
            "artillery_relocation",
            "paratrooper",
        ):
            quota = 2
            for candidate in eligible:
                classes, wasteful = self.turn_action_classes(board, candidate.moves)
                if action_class in classes and not wasteful and add(candidate):
                    quota -= 1
                    if quota == 0 or len(selected) >= turn_width:
                        break

        for candidate in eligible:
            if len(selected) >= turn_width:
                break
            add(candidate)
        return selected

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
        partial_width = max(48, self.beam_width * 6)
        evaluation_pool_width = max(72, self.beam_width * 8)
        turn_width = max(8, self.beam_width * 2)
        frontier = [PartialTurn([], board.copy(), 0.0)]
        completed: List[PartialTurn] = []

        while frontier:
            expanded: Dict[str, PartialTurn] = {}
            for partial in frontier:
                self.check_time()
                if partial.board.is_game_over() or partial.board.turn != original_turn:
                    completed.append(partial)
                    continue

                actions = self.diverse_moves(partial.board)
                if not actions:
                    completed.append(partial)
                    continue
                forced = all(move.name == "AutoCapture" for _, move in actions)
                for priority, move in actions:
                    self.check_time(False)
                    if move.name == "Skip" and partial.board.turn_moves < 2 and not forced:
                        continue
                    child = partial.board.copy()
                    child.push(move)
                    candidate = PartialTurn(
                        partial.moves + [move],
                        child,
                        partial.priority + priority,
                    )
                    key = child.serialize()
                    incumbent = expanded.get(key)
                    if incumbent is not None:
                        self.complete_turns_deduplicated += 1
                    if incumbent is None or self._prefer_partial(candidate, incumbent):
                        expanded[key] = candidate

            next_frontier: List[PartialTurn] = []
            for partial in expanded.values():
                if partial.board.is_game_over() or partial.board.turn != original_turn:
                    completed.append(partial)
                else:
                    next_frontier.append(partial)

            if len(next_frontier) > partial_width:
                self.exhaustive_within_horizon = False
                self.partial_turns_pruned += len(next_frontier) - partial_width
                next_frontier = self._round_robin_partials(next_frontier, partial_width)
            frontier = next_frontier

        self.complete_turns_generated += len(completed)
        unique: Dict[str, PartialTurn] = {}
        for partial in completed:
            key = partial.board.serialize()
            incumbent = unique.get(key)
            if incumbent is None or self._prefer_partial(partial, incumbent):
                unique[key] = partial
        self.complete_turns_deduplicated += len(completed) - len(unique)

        pool = list(unique.values())
        if len(pool) > evaluation_pool_width:
            self.exhaustive_within_horizon = False
            self.partial_turns_pruned += len(pool) - evaluation_pool_width
            pool = self._round_robin_partials(pool, evaluation_pool_width)

        if self.root_key == cache_key and self.root_fallback is None:
            fallback_pool = sorted(
                pool,
                key=lambda partial: (
                    self.turn_action_classes(board, partial.moves)[1],
                    -partial.priority,
                    tuple(move.uci() for move in partial.moves),
                ),
            )
            for partial in fallback_pool[:12]:
                self.check_time(False)
                safety = self.assess_turn_safety(board, partial.board, original_turn)
                if safety.tactically_safe:
                    self.root_fallback = TurnCandidate(
                        partial.moves,
                        partial.board,
                        partial.priority,
                        self.quick_score(partial.board),
                        0.0,
                        True,
                    )
                    break

        candidates = []
        for partial in pool:
            self.check_time(False)
            terminal = self.terminal_score(partial.board, 0)
            safety = self.assess_turn_safety(board, partial.board, original_turn)
            candidates.append(
                TurnCandidate(
                    partial.moves,
                    partial.board,
                    partial.priority,
                    self.heuristic_score(partial.board) if terminal is None else terminal,
                    max(0.0, safety.new_risk_value - safety.compensation_value),
                    safety.tactically_safe,
                )
            )
            if not safety.tactically_safe:
                self.tactically_unsafe_turns += 1
        candidates.sort(key=lambda item: self.candidate_sort_key(item, original_turn))

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
                tuple(move.uci() for move in item.moves),
            ),
        )
        for candidate in tactical:
            if len(selected) >= turn_width + self.beam_width:
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
        for candidate in plan_representatives.values():
            if candidate not in selected and candidate.tactically_safe:
                selected.append(candidate)

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
        for turn in turns:
            result = self.alphabeta(turn.board, turns_left - 1, alpha, beta)
            candidate = result.score
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
        return best


def greedy_complete_turn(
    board: engine.BaseBoard, personality: str, turn_number: int = 1
) -> SearchResult:
    """Fast emergency completion used only when no search depth finishes."""
    working = board.copy()
    original_turn = working.turn
    moves: List[engine.Move] = []
    while working.turn == original_turn and not working.is_game_over():
        legal = list(working.generate_legal_moves())
        if not legal:
            break
        candidates: List[Tuple[float, str, engine.Move]] = []
        for move in legal:
            target = working.piece_type_at(move.capture_preference) if move.capture_preference is not None else None
            piece_type = Searcher.move_piece_type(working, move)
            score = 0.0
            if move.name == "AutoCapture":
                score += 10000.0
            if target is not None:
                score += 5000.0 + 100.0 * PIECE_VALUES.get(target, 0.0)
            if move.name == "Reinforce":
                score += 300.0 + (30.0 if piece_type in INFANTRY_TYPES else 0.0)
            elif piece_type in INFANTRY_TYPES:
                score += 120.0
            elif piece_type in ARTILLERY_TYPES and move.from_square != move.to_square:
                score += 60.0
            elif piece_type in ARTILLERY_TYPES:
                score -= 200.0
            if move.name == "Skip":
                score -= 1000.0 if working.turn_moves < 2 else 20.0
            candidates.append((score, move.uci(), move))
        chosen = max(candidates, key=lambda item: (item[0], item[1]))[2]
        moves.append(chosen)
        working.push(chosen)
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


def search(
    board: engine.BaseBoard,
    personality: str,
    time_ms: int,
    max_depth: int,
    beam_width: int,
    turn_number: int = 1,
    value_function: Optional[Any] = None,
) -> Dict[str, Any]:
    started = time.monotonic()
    searcher = Searcher(
        personality,
        time_ms,
        beam_width,
        turn_number=turn_number,
        value_function=value_function,
    )
    searcher.root_key = board.serialize()
    best: Optional[SearchResult] = None
    completed_depth = 0
    timed_out = False
    fallback_kind = "none"
    for depth in range(1, max(1, max_depth) + 1):
        try:
            iteration = searcher.alphabeta(board, depth, -math.inf, math.inf)
        except SearchTimeout:
            timed_out = True
            break
        best = iteration
        completed_depth = depth
        if abs(iteration.score) >= MATE_SCORE:
            break

    if best is None:
        if searcher.root_fallback is not None:
            best = SearchResult(
                searcher.root_fallback.static_score,
                list(searcher.root_fallback.moves),
            )
            fallback_kind = "safe"
        else:
            best = greedy_complete_turn(board, personality, turn_number)
            fallback_kind = "greedy"
    first_turn, resulting_board = first_turn_from_pv(board, best.pv)
    if not first_turn and not board.is_game_over():
        fallback = greedy_complete_turn(board, personality, turn_number)
        first_turn, resulting_board = first_turn_from_pv(board, fallback.pv)
        best = fallback
        fallback_kind = "greedy"

    elapsed_ms = (time.monotonic() - started) * 1000.0
    automatic = [move.uci() for move in first_turn if move.name == "AutoCapture"]
    actions = [move.uci() for move in first_turn if move.name != "AutoCapture"]
    root_eval = evaluation_breakdown(board, personality, turn_number)
    resulting_eval = evaluation_breakdown(resulting_board, personality, turn_number + 1)
    current_player_score = best.score if board.turn == engine.RED else -best.score
    exhaustive = (
        searcher.exhaustive_within_horizon
        and not timed_out
        and completed_depth == max(1, max_depth)
    )
    recommendation_label = (
        "best move"
        if exhaustive
        else "safe fallback"
        if fallback_kind == "safe"
        else "greedy fallback"
        if fallback_kind == "greedy"
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
        },
        "principal_variation": [move.uci() for move in best.pv],
        "score": {
            "current_player": round(current_player_score, 4),
            "red": round(best.score, 4),
        },
        "search": {
            "completed_depth_in_turns": completed_depth,
            "requested_depth_in_turns": max_depth,
            "base_complete_turn_width": beam_width,
            "nodes": searcher.nodes,
            "elapsed_ms": round(elapsed_ms, 2),
            "timed_out": timed_out,
            "fallback_used": fallback_kind,
            "approximate": True,
            "exhaustive_within_requested_horizon": exhaustive,
            "rule_filtered_actions": searcher.rule_filtered_actions,
            "beam_pruned_actions": searcher.beam_pruned_actions,
            "partial_turns_pruned": searcher.partial_turns_pruned,
            "complete_turns_generated": searcher.complete_turns_generated,
            "complete_turns_deduplicated": searcher.complete_turns_deduplicated,
            "complete_turns_pruned": searcher.complete_turns_pruned,
            "tactically_unsafe_turns": searcher.tactically_unsafe_turns,
            "rotation_quota_pruned": searcher.rotation_quota_pruned,
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
    parser.add_argument("--personality", choices=sorted(PERSONALITIES), default="balanced")
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        board = engine.BaseBoard(args.fen)
        result = search(board, args.personality, args.time_ms, args.max_depth, args.beam_width)
    except (ValueError, AssertionError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=None if args.compact else 2, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
