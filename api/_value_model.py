"""Native inference for GHQ gradient-boosted value checkpoints.

The feature definitions mirror ``src/game/value-model/features.ts``.  Keeping
inference in CPython removes the Pyodide Python/JavaScript callback at every
leaf while preserving the exact exported tree artifacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import _engine as engine


PIECE_TYPES = (
    engine.HQ,
    engine.INFANTRY,
    engine.ARMORED_INFANTRY,
    engine.AIRBORNE_INFANTRY,
    engine.ARTILLERY,
    engine.ARMORED_ARTILLERY,
    engine.HEAVY_ARTILLERY,
)
RESERVE_TYPES = PIECE_TYPES[1:]
INFANTRY_TYPES = {
    engine.INFANTRY,
    engine.ARMORED_INFANTRY,
    engine.AIRBORNE_INFANTRY,
}
ARTILLERY_TYPES = {
    engine.ARTILLERY,
    engine.ARMORED_ARTILLERY,
    engine.HEAVY_ARTILLERY,
}
PIECE_NAMES = {
    piece_type: engine.piece_name(piece_type).lower() for piece_type in PIECE_TYPES
}
PIECE_VALUES = {
    engine.HQ: 100.0,
    engine.INFANTRY: 1.0,
    engine.ARMORED_INFANTRY: 3.0,
    engine.AIRBORNE_INFANTRY: 5.0,
    engine.ARTILLERY: 3.0,
    engine.ARMORED_ARTILLERY: 5.0,
    engine.HEAVY_ARTILLERY: 6.0,
}
MOBILITY = {
    engine.HQ: 1,
    engine.INFANTRY: 1,
    engine.ARMORED_INFANTRY: 2,
    engine.AIRBORNE_INFANTRY: 1,
    engine.ARTILLERY: 1,
    engine.ARMORED_ARTILLERY: 2,
    engine.HEAVY_ARTILLERY: 1,
}


def _load(name: str) -> Dict[str, Any]:
    path = Path(__file__).resolve().with_name(name)
    return json.loads(path.read_text(encoding="utf-8"))


ARTIFACTS = {
    "incumbent": _load("_model_incumbent.json"),
    "challenger": _load("_model_challenger.json"),
}


def _coordinate(square: int) -> Tuple[int, int]:
    return 7 - square // 8, square % 8


def _square(at: Tuple[int, int]) -> int:
    return (7 - at[0]) * 8 + at[1]


def _chebyshev(left: Tuple[int, int], right: Tuple[int, int]) -> int:
    return max(abs(left[0] - right[0]), abs(left[1] - right[1]))


def _manhattan(left: Tuple[int, int], right: Tuple[int, int]) -> int:
    return abs(left[0] - right[0]) + abs(left[1] - right[1])


def _neighbors(at: Tuple[int, int], diagonal: bool) -> Iterable[Tuple[int, int]]:
    for row_delta in (-1, 0, 1):
        for column_delta in (-1, 0, 1):
            if row_delta == 0 and column_delta == 0:
                continue
            if not diagonal and abs(row_delta) + abs(column_delta) != 1:
                continue
            row = at[0] + row_delta
            column = at[1] + column_delta
            if 0 <= row < 8 and 0 <= column < 8:
                yield row, column


def _pieces(board: engine.BaseBoard, color: bool) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for square in engine.SQUARES:
        piece = board.piece_at(square)
        if piece is not None and piece.color == color:
            result.append(
                {
                    "at": _coordinate(square),
                    "square": square,
                    "type": piece.piece_type,
                    "orientation": piece.orientation,
                }
            )
    return result


def _nearest_support(candidate: Dict[str, Any], friendly: Sequence[Dict[str, Any]]) -> int:
    distances = [
        _chebyshev(candidate["at"], other["at"])
        for other in friendly
        if other is not candidate
        and other["type"] not in (engine.HQ, engine.AIRBORNE_INFANTRY)
    ]
    return min(distances) if distances else 8


def _heavy_centered(heavy: Dict[str, Any], artillery: Sequence[Dict[str, Any]]) -> bool:
    row, column = heavy["at"]
    same_rank = [gun for gun in artillery if gun is not heavy and gun["at"][0] == row]
    same_file = [gun for gun in artillery if gun is not heavy and gun["at"][1] == column]
    return (
        any(gun["at"][1] < column for gun in same_rank)
        and any(gun["at"][1] > column for gun in same_rank)
    ) or (
        any(gun["at"][0] < row for gun in same_file)
        and any(gun["at"][0] > row for gun in same_file)
    )


def _pseudo_mobility(board: engine.BaseBoard, pieces: Sequence[Dict[str, Any]]) -> float:
    moves = 0
    for candidate in pieces:
        if candidate["type"] == engine.HQ:
            continue
        speed = MOBILITY[candidate["type"]]
        row, column = candidate["at"]
        for target_row in range(max(0, row - speed), min(7, row + speed) + 1):
            for target_column in range(
                max(0, column - speed), min(7, column + speed) + 1
            ):
                target = (target_row, target_column)
                if target == candidate["at"]:
                    continue
                if _chebyshev(candidate["at"], target) <= speed and board.piece_at(
                    _square(target)
                ) is None:
                    moves += 1
    return math.log1p(moves)


def _structure_metrics(
    board: engine.BaseBoard, pieces: Sequence[Dict[str, Any]], home_rank: int
) -> Dict[str, float]:
    material = [
        piece
        for piece in pieces
        if piece["type"] not in (engine.HQ, engine.AIRBORNE_INFANTRY)
    ]
    relocation_options = 0
    immobile_count = 0
    home_rank_immobile_count = 0
    for candidate in material:
        speed = MOBILITY[candidate["type"]]
        row, column = candidate["at"]
        options = 0
        for target_row in range(max(0, row - speed), min(7, row + speed) + 1):
            for target_column in range(
                max(0, column - speed), min(7, column + speed) + 1
            ):
                target = (target_row, target_column)
                if target != candidate["at"] and _chebyshev(
                    candidate["at"], target
                ) <= speed and board.piece_at(_square(target)) is None:
                    options += 1
        relocation_options += options
        if options == 0:
            immobile_count += 1
            if row == home_rank:
                home_rank_immobile_count += 1

    remaining = set(range(len(material)))
    component_sizes: List[int] = []
    while remaining:
        first = remaining.pop()
        stack = [first]
        size = 0
        while stack:
            current = stack.pop()
            size += 1
            connected = [
                index
                for index in remaining
                if _chebyshev(material[current]["at"], material[index]["at"]) <= 1
            ]
            for index in connected:
                remaining.remove(index)
                stack.append(index)
        component_sizes.append(size)
    count = len(material)
    return {
        "relocation_options": float(relocation_options),
        "mean_relocation_options": relocation_options / count if count else 0.0,
        "immobile_count": float(immobile_count),
        "home_rank_immobile_count": float(home_rank_immobile_count),
        "connected_components": float(len(component_sizes)),
        "largest_component_ratio": max(component_sizes) / count if count else 1.0,
    }


def _structure_v2(
    board: engine.BaseBoard, color: bool, friendly: Sequence[Dict[str, Any]]
) -> Dict[str, float]:
    home_rank = 7 if color == engine.RED else 0
    infantry = [
        piece
        for piece in friendly
        if piece["type"] in INFANTRY_TYPES and piece["type"] != engine.AIRBORNE_INFANTRY
    ]
    material = [
        piece
        for piece in friendly
        if piece["type"] not in (engine.HQ, engine.AIRBORNE_INFANTRY)
    ]
    result = {
        "infantry_vertical_adjacent_pairs": 0.0,
        "infantry_diagonal_adjacent_pairs": 0.0,
        "infantry_same_file_run_excess": 0.0,
        "infantry_isolated_count": 0.0,
        "infantry_distinct_files": 0.0,
        "infantry_file_span": 0.0,
        "infantry_rank_span": 0.0,
        "infantry_frontier_count": 0.0,
        "material_pair_distance_mean": 0.0,
        "material_file_span": 0.0,
        "material_rank_span": 0.0,
    }
    for first in range(len(infantry)):
        for second in range(first + 1, len(infantry)):
            row_distance = abs(infantry[first]["at"][0] - infantry[second]["at"][0])
            column_distance = abs(
                infantry[first]["at"][1] - infantry[second]["at"][1]
            )
            if row_distance == 1 and column_distance == 0:
                result["infantry_vertical_adjacent_pairs"] += 1
            if row_distance == 1 and column_distance == 1:
                result["infantry_diagonal_adjacent_pairs"] += 1
    for column in range(8):
        rows = sorted(piece["at"][0] for piece in infantry if piece["at"][1] == column)
        run = 1
        for index in range(1, len(rows)):
            if rows[index] == rows[index - 1] + 1:
                run += 1
                if run >= 3:
                    result["infantry_same_file_run_excess"] += 1
            else:
                run = 1
    result["infantry_isolated_count"] = float(
        sum(
            all(other is candidate or _chebyshev(candidate["at"], other["at"]) > 1 for other in material)
            for candidate in infantry
        )
    )
    infantry_files = [piece["at"][1] for piece in infantry]
    infantry_ranks = [piece["at"][0] for piece in infantry]
    result["infantry_distinct_files"] = float(len(set(infantry_files)))
    if infantry_files:
        result["infantry_file_span"] = float(max(infantry_files) - min(infantry_files))
        result["infantry_rank_span"] = float(max(infantry_ranks) - min(infantry_ranks))
        frontier = max(abs(piece["at"][0] - home_rank) for piece in infantry)
        result["infantry_frontier_count"] = float(
            sum(abs(piece["at"][0] - home_rank) == frontier for piece in infantry)
        )
    material_files = [piece["at"][1] for piece in material]
    material_ranks = [piece["at"][0] for piece in material]
    if material_files:
        result["material_file_span"] = float(max(material_files) - min(material_files))
        result["material_rank_span"] = float(max(material_ranks) - min(material_ranks))
    distances = [
        _chebyshev(material[first]["at"], material[second]["at"])
        for first in range(len(material))
        for second in range(first + 1, len(material))
    ]
    result["material_pair_distance_mean"] = (
        sum(distances) / len(distances) if distances else 0.0
    )
    return result


def _tactical_v3(
    board: engine.BaseBoard, color: bool, friendly: Sequence[Dict[str, Any]]
) -> Dict[str, float]:
    result = {
        "hq_enemy_infantry_distance_min": 0.0,
        "hq_enemy_armored_infantry_distance_min": 0.0,
        "hq_enemy_airborne_infantry_distance_min": 0.0,
        "hq_enemy_infantry_within_two": 0.0,
        "hq_enemy_infantry_within_three": 0.0,
        "hq_friendly_infantry_within_two": 0.0,
        "hq_friendly_infantry_within_three": 0.0,
        "hq_attack_pressure": 0.0,
        "hq_defense_density": 0.0,
    }
    hq = next((piece for piece in friendly if piece["type"] == engine.HQ), None)
    if hq is None:
        return result
    enemy_infantry = [
        piece for piece in _pieces(board, not color) if piece["type"] in INFANTRY_TYPES
    ]
    friendly_infantry = [
        piece for piece in friendly if piece["type"] in INFANTRY_TYPES
    ]

    def distance(piece: Dict[str, Any]) -> int:
        return _manhattan(hq["at"], piece["at"])

    def minimum_distance(piece_type: int) -> float:
        distances = [
            distance(piece)
            for piece in enemy_infantry
            if piece["type"] == piece_type
        ]
        return float(min(distances) if distances else 15)

    result["hq_enemy_infantry_distance_min"] = float(
        min((distance(piece) for piece in enemy_infantry), default=15)
    )
    result["hq_enemy_armored_infantry_distance_min"] = minimum_distance(
        engine.ARMORED_INFANTRY
    )
    result["hq_enemy_airborne_infantry_distance_min"] = minimum_distance(
        engine.AIRBORNE_INFANTRY
    )
    result["hq_enemy_infantry_within_two"] = float(
        sum(distance(piece) <= 2 for piece in enemy_infantry)
    )
    result["hq_enemy_infantry_within_three"] = float(
        sum(distance(piece) <= 3 for piece in enemy_infantry)
    )
    result["hq_friendly_infantry_within_two"] = float(
        sum(distance(piece) <= 2 for piece in friendly_infantry)
    )
    result["hq_friendly_infantry_within_three"] = float(
        sum(distance(piece) <= 3 for piece in friendly_infantry)
    )
    for attacker in enemy_infantry:
        proximity = max(0, 4 - distance(attacker))
        weight = (
            1.5
            if attacker["type"] == engine.ARMORED_INFANTRY
            else 1.25
            if attacker["type"] == engine.AIRBORNE_INFANTRY
            else 1.0
        )
        result["hq_attack_pressure"] += proximity * weight
    result["hq_defense_density"] = float(
        sum(max(0, 4 - distance(defender)) for defender in friendly_infantry)
    )
    return result


def _side_features(board: engine.BaseBoard, color: bool) -> Dict[str, float]:
    opponent = not color
    friendly = _pieces(board, color)
    artillery = [piece for piece in friendly if piece["type"] in ARTILLERY_TYPES]
    infantry = [piece for piece in friendly if piece["type"] in INFANTRY_TYPES]
    home_rank = 7 if color == engine.RED else 0
    result: Dict[str, float] = {}
    counts = {piece_type: 0 for piece_type in PIECE_TYPES}
    for piece in friendly:
        counts[piece["type"]] += 1
    for piece_type in PIECE_TYPES:
        result[f"board_{PIECE_NAMES[piece_type]}"] = float(counts[piece_type])
    for piece_type in RESERVE_TYPES:
        result[f"reserve_{PIECE_NAMES[piece_type]}"] = float(
            board.get_reserve_count(piece_type, color)
        )

    result["material_board"] = sum(PIECE_VALUES[piece["type"]] for piece in friendly)
    result["material_total"] = result["material_board"] + sum(
        board.get_reserve_count(piece_type, color) * PIECE_VALUES[piece_type]
        for piece_type in RESERVE_TYPES
    )
    result["pieces_board"] = float(len(friendly))
    result["pieces_reserve"] = float(
        sum(board.get_reserve_count(piece_type, color) for piece_type in RESERVE_TYPES)
    )
    result["infantry_board"] = float(len(infantry))
    result["artillery_board"] = float(len(artillery))
    result["home_rank_occupancy"] = float(
        sum(piece["at"][0] == home_rank for piece in friendly)
    )
    result.update(_structure_metrics(board, friendly, home_rank))
    rank_counts = [sum(piece["at"][0] == rank for piece in friendly) for rank in range(8)]
    result["max_rank_occupancy"] = float(max(rank_counts, default=0))
    advances = [
        abs(piece["at"][0] - home_rank)
        for piece in friendly
        if piece["type"] != engine.HQ
    ]
    result["advancement_mean"] = sum(advances) / len(advances) if advances else 0.0
    result["advancement_max"] = float(max(advances, default=0))

    support_material = [
        piece
        for piece in friendly
        if piece["type"] not in (engine.HQ, engine.AIRBORNE_INFANTRY)
    ]
    rank_power = [0.0] * 8
    for piece in support_material:
        rank_power[piece["at"][0]] += PIECE_VALUES[piece["type"]]
    anchor_rank = rank_power.index(max(rank_power))
    result["unsupported_count"] = 0.0
    result["unsupported_value"] = 0.0
    result["overextended_count"] = 0.0
    result["overextended_value"] = 0.0
    support_distance_total = 0
    for piece in support_material:
        distance = _nearest_support(piece, friendly)
        support_distance_total += distance
        if distance > 1:
            result["unsupported_count"] += 1
            result["unsupported_value"] += PIECE_VALUES[piece["type"]]
        ranks_past_anchor = (
            anchor_rank - piece["at"][0]
            if color == engine.RED
            else piece["at"][0] - anchor_rank
        )
        if ranks_past_anchor >= 2:
            result["overextended_count"] += 1
            result["overextended_value"] += PIECE_VALUES[piece["type"]] * (
                ranks_past_anchor - 1
            )
    result["support_distance_mean"] = (
        support_distance_total / len(support_material) if support_material else 0.0
    )

    result["artillery_adjacent_pairs"] = float(
        sum(
            _chebyshev(artillery[first]["at"], artillery[second]["at"]) == 1
            for first in range(len(artillery))
            for second in range(first + 1, len(artillery))
        )
    )
    result["heavy_artillery_centered"] = float(
        any(
            gun["type"] == engine.HEAVY_ARTILLERY
            and _heavy_centered(gun, artillery)
            for gun in artillery
        )
    )
    result["artillery_cardinal_count"] = 0.0
    result["artillery_diagonal_count"] = 0.0
    result["artillery_diagonal_infantry_cover"] = 0.0
    result["artillery_cardinal_infantry_cover"] = 0.0
    result["artillery_protected_count"] = 0.0
    for gun in artillery:
        orientation = gun["orientation"]
        if orientation is None:
            orientation = engine.ORIENT_N if color == engine.RED else engine.ORIENT_S
        if orientation % 2 == 0:
            result["artillery_cardinal_count"] += 1
        else:
            result["artillery_diagonal_count"] += 1
        protected = False
        for at in _neighbors(gun["at"], True):
            piece = board.piece_at(_square(at))
            if piece is None or piece.color != color or piece.piece_type not in INFANTRY_TYPES:
                continue
            protected = True
            if at[0] != gun["at"][0] and at[1] != gun["at"][1]:
                result["artillery_diagonal_infantry_cover"] += 1
            else:
                result["artillery_cardinal_infantry_cover"] += 1
        if protected:
            result["artillery_protected_count"] += 1

    result["bombarded_squares"] = 0.0
    result["bombarded_enemy_count"] = 0.0
    result["bombarded_enemy_value"] = 0.0
    bombardment = board.bombarded_co[color]
    for square in engine.SQUARES:
        if not bombardment & engine.BB_SQUARES[square]:
            continue
        result["bombarded_squares"] += 1
        target = board.piece_at(square)
        if target is not None and target.color == opponent:
            result["bombarded_enemy_count"] += 1
            result["bombarded_enemy_value"] += PIECE_VALUES[target.piece_type]

    engagements = set()
    for piece in infantry:
        for at in _neighbors(piece["at"], False):
            target = board.piece_at(_square(at))
            if target is not None and target.color == opponent and target.piece_type in INFANTRY_TYPES:
                engagements.add(tuple(sorted((piece["at"], at))))
    result["infantry_engagements"] = float(len(engagements))

    paratroopers = [piece for piece in friendly if piece["type"] == engine.AIRBORNE_INFANTRY]
    result["paratrooper_ready"] = float(
        board.get_reserve_count(engine.AIRBORNE_INFANTRY, color) > 0
        or any(piece["at"][0] == home_rank for piece in paratroopers)
    )
    deployed = [piece for piece in paratroopers if piece["at"][0] != home_rank]
    result["paratrooper_deployed"] = float(len(deployed))
    result["paratrooper_distance_home"] = 0.0
    result["paratrooper_supported"] = 0.0
    result["paratrooper_engaged"] = 0.0
    for paratrooper in deployed:
        result["paratrooper_distance_home"] += abs(paratrooper["at"][0] - home_rank)
        result["paratrooper_supported"] += float(
            any(
                (piece := board.piece_at(_square(at))) is not None
                and piece.color == color
                and piece.piece_type != engine.HQ
                for at in _neighbors(paratrooper["at"], True)
            )
        )
        result["paratrooper_engaged"] += float(
            any(
                (piece := board.piece_at(_square(at))) is not None
                and piece.color == opponent
                and piece.piece_type in INFANTRY_TYPES
                for at in _neighbors(paratrooper["at"], False)
            )
        )

    hq = next((piece for piece in friendly if piece["type"] == engine.HQ), None)
    result["hq_adjacent_enemy_infantry"] = 0.0
    result["hq_adjacent_friendly"] = 0.0
    result["hq_escape_squares"] = 0.0
    if hq is None:
        result["hq_bombarded"] = 1.0
    else:
        result["hq_bombarded"] = float(
            bool(board.bombarded_co[opponent] & engine.BB_SQUARES[hq["square"]])
        )
        for at in _neighbors(hq["at"], True):
            piece = board.piece_at(_square(at))
            if piece is not None and piece.color == color:
                result["hq_adjacent_friendly"] += 1
            if (
                piece is not None
                and piece.color == opponent
                and piece.piece_type in INFANTRY_TYPES
                and (at[0] == hq["at"][0] or at[1] == hq["at"][1])
            ):
                result["hq_adjacent_enemy_infantry"] += 1
            if piece is None and not (
                board.bombarded_co[opponent] & engine.BB_SQUARES[_square(at)]
            ):
                enemy_adjacent = any(
                    (neighbor := board.piece_at(_square(neighbor_at))) is not None
                    and neighbor.color == opponent
                    and neighbor.piece_type in INFANTRY_TYPES
                    for neighbor_at in _neighbors(at, False)
                )
                if not enemy_adjacent:
                    result["hq_escape_squares"] += 1
    result["pseudo_mobility"] = _pseudo_mobility(board, friendly)
    result.update(_structure_v2(board, color, friendly))
    result.update(_tactical_v3(board, color, friendly))
    return result


def extract_features(
    board: engine.BaseBoard, turn_number: int, perspective: bool, artifact: Dict[str, Any]
) -> List[float]:
    own = _side_features(board, perspective)
    opponent = _side_features(board, not perspective)
    reserve_total = sum(
        board.get_reserve_count(piece_type, color)
        for color in (engine.RED, engine.BLUE)
        for piece_type in RESERVE_TYPES
    )
    occupied = board.occupied.bit_count()
    non_hq_on_board = (board.occupied & ~board.hq).bit_count()
    fixed = {
        "turn_progress": min(max(turn_number, 0), 100) / 100.0,
        "board_fill": occupied / 64.0,
        "surviving_unit_fraction": (non_hq_on_board + reserve_total) / 26.0,
        "own_to_move": float(board.turn == perspective),
    }
    values: List[float] = []
    for name in artifact["feature_names"]:
        if name in fixed:
            value = fixed[name]
        elif name.startswith("own_"):
            value = own[name[4:]]
        elif name.startswith("opp_"):
            value = opponent[name[4:]]
        elif name.startswith("diff_"):
            key = name[5:]
            value = own[key] - opponent[key]
        else:
            raise ValueError(f"Unknown native GHQ value feature: {name}")
        if not math.isfinite(value):
            raise ValueError(f"Invalid native GHQ value feature: {name}")
        values.append(float(value))
    return values


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def predict_from_features(features: Sequence[float], artifact: Dict[str, Any]) -> float:
    if len(features) != len(artifact["feature_names"]):
        raise ValueError("Native GHQ value feature schema mismatch")
    raw = float(artifact["base_raw_score"])
    learning_rate = float(artifact["learning_rate"])
    for tree in artifact["trees"]:
        node = 0
        while tree["children_left"][node] != -1:
            feature = tree["feature"][node]
            node = (
                tree["children_left"][node]
                if features[feature] <= tree["threshold"][node]
                else tree["children_right"][node]
            )
        raw += learning_rate * tree["value"][node]
    calibration = artifact["calibration"]
    calibrated = calibration["scale"] * raw + calibration["intercept"]
    correction = artifact.get("linear_correction")
    if correction:
        calibrated += sum(
            coefficient * features[index]
            for index, coefficient in zip(
                correction["feature_indices"], correction["coefficients"]
            )
        )
    tree_correction = artifact.get("tree_correction")
    if tree_correction:
        correction_rate = float(tree_correction["learning_rate"])
        for tree in tree_correction["trees"]:
            node = 0
            while tree["children_left"][node] != -1:
                feature = tree["feature"][node]
                node = (
                    tree["children_left"][node]
                    if features[feature] <= tree["threshold"][node]
                    else tree["children_right"][node]
                )
            calibrated += correction_rate * tree["value"][node]
    return _sigmoid(calibrated)


def policy_adjustment_from_features(
    features: Sequence[float], artifact: Dict[str, Any]
) -> float:
    """Score a resulting position for move ordering, not win calibration."""
    if len(features) != len(artifact["feature_names"]):
        raise ValueError("Native GHQ value feature schema mismatch")
    correction = artifact.get("policy_correction")
    if not correction:
        return 0.0
    indices = correction["feature_indices"]
    coefficients = correction["coefficients"]
    scale = float(correction.get("scale", 1.0))
    if len(indices) != len(coefficients):
        raise ValueError("Native GHQ policy correction schema mismatch")
    if any(
        not isinstance(index, int) or index < 0 or index >= len(features)
        for index in indices
    ):
        raise ValueError("Native GHQ policy correction feature is out of range")
    if not math.isfinite(scale) or not 0.0 <= scale <= 1.0:
        raise ValueError("Native GHQ policy correction scale is invalid")
    return scale * float(
        sum(
            coefficient * features[index]
            for index, coefficient in zip(indices, coefficients)
        )
    )


def predict_zero_sum(
    fen: str, turn_number: int, perspective: bool, version: str = "incumbent"
) -> float:
    artifact = ARTIFACTS[version]
    board = engine.BaseBoard(fen)
    own_hq = bool(board.pieces_mask(engine.HQ, perspective))
    opponent_hq = bool(board.pieces_mask(engine.HQ, not perspective))
    if not own_hq:
        return 0.0
    if not opponent_hq:
        return 1.0
    own = predict_from_features(
        extract_features(board, turn_number, perspective, artifact), artifact
    )
    other = predict_from_features(
        extract_features(board, turn_number, not perspective, artifact), artifact
    )
    total = own + other
    return own / total if total > 0 else 0.5


def red_value_function(version: str):
    if version not in ARTIFACTS:
        raise ValueError(f"Unknown value model: {version}")

    def evaluate(fen: str, turn_number: int) -> float:
        return predict_zero_sum(fen, turn_number, engine.RED, version)

    return evaluate


def policy_function(version: str):
    if version not in ARTIFACTS:
        raise ValueError(f"Unknown value model: {version}")
    artifact = ARTIFACTS[version]

    def evaluate(fen: str, turn_number: int, perspective: bool) -> float:
        board = engine.BaseBoard(fen)
        return policy_adjustment_from_features(
            extract_features(board, turn_number, perspective, artifact), artifact
        )

    return evaluate
