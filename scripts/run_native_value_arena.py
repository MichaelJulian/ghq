#!/usr/bin/env python3
"""Screen two GHQ value artifacts with paired native-engine games.

This is deliberately a screening arena, not a promotion decision.  It uses
the production Python rules and search, gives both artifacts the same opening
seed and personality in a color-swapped pair, and reports whether a challenger
changes play in a promising direction before consuming a durable Vercel arena.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

import _engine as engine  # noqa: E402
import _ghq_ai as ghq_ai  # noqa: E402
import _value_model as value_model  # noqa: E402


@dataclass(frozen=True)
class GameConfig:
    index: int
    seed: int
    baseline_path: str
    challenger_path: str
    time_ms: int
    max_depth: int
    beam_width: int
    max_turns: int
    repetition_limit: int
    no_progress_turns: int


@dataclass(frozen=True)
class StrategicProgress:
    frontier_rank: int
    enemy_hq_distance: int
    enemy_hq_pressure: int


@dataclass
class GameResult:
    index: int
    pair: int
    seed: int
    personality: str
    challenger_color: str
    winner: Optional[str]
    termination: str
    turns: int
    challenger_score: float
    decisions: int
    opening_book_decisions: int
    verified_decisions: int
    fallback_counts: Dict[str, int]
    completed_depth_counts: Dict[str, int]
    move_turns: list[list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--challenger", required=True, type=Path)
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--time-ms", type=int, default=2_000)
    parser.add_argument("--max-depth", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--repetition-limit", type=int, default=3)
    parser.add_argument("--no-progress-turns", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0x474851)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.games < 2 or args.games % 2:
        parser.error("--games must be a positive even number")
    if not 50 <= args.time_ms <= 30_000:
        parser.error("--time-ms must be from 50 through 30000")
    if not 2 <= args.beam_width <= 16:
        parser.error("--beam-width must be from 2 through 16")
    if args.max_turns < 4:
        parser.error("--max-turns must be at least 4")
    if not 2 <= args.repetition_limit <= 10:
        parser.error("--repetition-limit must be from 2 through 10")
    if not 4 <= args.no_progress_turns <= 100:
        parser.error("--no-progress-turns must be from 4 through 100")
    if args.workers < 1:
        parser.error("--workers must be positive")
    return args


def load_artifact(path: str) -> Dict[str, Any]:
    artifact = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "feature_names",
        "base_raw_score",
        "learning_rate",
        "calibration",
        "trees",
    }
    missing = sorted(required - artifact.keys())
    if missing:
        raise ValueError(f"{path} is missing model fields: {missing}")
    return artifact


def artifact_fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def red_value_function(artifact: Dict[str, Any]):
    def evaluate(fen: str, turn_number: int) -> float:
        board = engine.BaseBoard(fen)
        red_hq = bool(board.pieces_mask(engine.HQ, engine.RED))
        blue_hq = bool(board.pieces_mask(engine.HQ, engine.BLUE))
        if not red_hq:
            return 0.0
        if not blue_hq:
            return 1.0
        red = value_model.predict_from_features(
            value_model.extract_features(
                board, turn_number, engine.RED, artifact
            ),
            artifact,
        )
        blue = value_model.predict_from_features(
            value_model.extract_features(
                board, turn_number, engine.BLUE, artifact
            ),
            artifact,
        )
        total = red + blue
        return red / total if total > 0 else 0.5

    return evaluate


def policy_function(artifact: Dict[str, Any]):
    """Return the artifact's move-ranking score without changing value."""

    def evaluate(fen: str, turn_number: int, perspective: bool) -> float:
        board = engine.BaseBoard(fen)
        return value_model.policy_adjustment_from_features(
            value_model.extract_features(
                board, turn_number, perspective, artifact
            ),
            artifact,
        )

    return evaluate


def winner_name(outcome: Any) -> Optional[str]:
    if outcome is None or outcome.winner is None:
        return None
    return "RED" if outcome.winner == engine.RED else "BLUE"


def strategic_progress(
    board: engine.BaseBoard, color: bool
) -> StrategicProgress:
    """Mirror the durable TypeScript no-progress landmarks exactly."""
    own = board.occupied_co[color] & ~board.hq
    own_squares = list(engine.scan_reversed(own))
    enemy_hqs = list(engine.scan_reversed(board.hq & board.occupied_co[not color]))
    frontier_rank = max(
        (
            engine.square_rank(square) + 1
            if color == engine.RED
            else 8 - engine.square_rank(square)
            for square in own_squares
        ),
        default=0,
    )
    if own_squares and enemy_hqs:
        enemy_hq_distance = min(
            ghq_ai.chebyshev(square, hq)
            for square in own_squares
            for hq in enemy_hqs
        )
    else:
        enemy_hq_distance = 8
    infantry = own & (
        board.infantry | board.armored_infantry | board.airborne_infantry
    )
    pursuers = list(engine.scan_reversed(infantry or own))
    enemy_hq_pressure = (
        sum(
            max(0, 5 - min(ghq_ai.chebyshev(square, hq) for hq in enemy_hqs))
            for square in pursuers
        )
        if enemy_hqs
        else 0
    )
    return StrategicProgress(
        frontier_rank=frontier_rank,
        enemy_hq_distance=enemy_hq_distance,
        enemy_hq_pressure=enemy_hq_pressure,
    )


def extends_strategic_best(
    best: StrategicProgress, current: StrategicProgress
) -> bool:
    return (
        current.frontier_rank > best.frontier_rank
        or current.enemy_hq_distance < best.enemy_hq_distance
        or current.enemy_hq_pressure > best.enemy_hq_pressure
    )


def merge_strategic_best(
    best: StrategicProgress, current: StrategicProgress
) -> StrategicProgress:
    return StrategicProgress(
        frontier_rank=max(best.frontier_rank, current.frontier_rank),
        enemy_hq_distance=min(
            best.enemy_hq_distance, current.enemy_hq_distance
        ),
        enemy_hq_pressure=max(
            best.enemy_hq_pressure, current.enemy_hq_pressure
        ),
    )


def action_made_progress(uci: str) -> bool:
    return (
        uci.startswith("r")
        or (uci.startswith("s") and uci != "skip")
        or "x" in uci
    )


def play_game(config: GameConfig) -> GameResult:
    baseline = load_artifact(config.baseline_path)
    challenger = load_artifact(config.challenger_path)
    pair = config.index // 2
    challenger_color = engine.RED if config.index % 2 == 0 else engine.BLUE
    personalities = sorted(ghq_ai.PERSONALITIES)
    personality = personalities[pair % len(personalities)]
    pair_seed = (config.seed + (pair + 1) * 0x85EBCA6B) & 0xFFFF_FFFF
    values = {
        challenger_color: red_value_function(challenger),
        not challenger_color: red_value_function(baseline),
    }
    policies = {
        challenger_color: policy_function(challenger),
        not challenger_color: policy_function(baseline),
    }
    board = engine.BaseBoard(engine.STARTING_FEN)
    occurrences = Counter({board.board_fen(): 1})
    strategic_best = {
        engine.RED: strategic_progress(board, engine.RED),
        engine.BLUE: strategic_progress(board, engine.BLUE),
    }
    turns_without_progress = 0
    fallbacks: Counter[str] = Counter()
    depths: Counter[str] = Counter()
    decisions = 0
    opening_book_decisions = 0
    verified_decisions = 0
    move_turns: list[list[str]] = []
    termination = "max-turns"
    winner: Optional[str] = None
    game_over = False
    turn_number = 1

    while turn_number <= config.max_turns:
        mover = board.turn
        result = ghq_ai.search(
            board,
            personality,
            config.time_ms,
            config.max_depth,
            config.beam_width,
            turn_number,
            value_function=values[mover],
            opening_seed=pair_seed,
            max_actions=3,
            stagnation_turns=turns_without_progress,
            policy_function=policies[mover],
        )
        telemetry = result["search"]
        fallbacks[str(telemetry["fallback_used"])] += 1
        depths[str(telemetry["completed_depth_in_turns"])] += 1
        if telemetry["opening_book_used"]:
            opening_book_decisions += 1
        if (
            telemetry["opening_book_used"]
            or telemetry["completed_depth_in_turns"] >= 2
        ):
            verified_decisions += 1
        decisions += 1
        selected_moves = list(result["best_turn"]["all_moves"])
        # Both members start from the exact same board and opening seed. If
        # the artifacts are behaviorally identical, swapping which artifact
        # controls RED leaves the raw turn sequence identical. The first raw
        # mismatch therefore measures an actual policy effect without trying
        # to manufacture spatial symmetry around RED's first-move advantage.
        move_turns.append(selected_moves)
        for uci in selected_moves:
            move = engine.Move.from_uci(uci)
            if not board.is_legal(move):
                raise RuntimeError(
                    f"game {config.index + 1} search returned illegal move {uci}"
                )
            board.push(move)
            outcome = board.outcome()
            if outcome is not None:
                winner = winner_name(outcome)
                termination = str(outcome.termination)
                game_over = True
                break
        if game_over:
            break
        fen = board.board_fen()
        occurrences[fen] += 1
        if occurrences[fen] >= config.repetition_limit:
            termination = "repetition"
            break
        current_progress = strategic_progress(board, mover)
        made_strategic_progress = extends_strategic_best(
            strategic_best[mover], current_progress
        )
        strategic_best[mover] = merge_strategic_best(
            strategic_best[mover], current_progress
        )
        turns_without_progress = (
            0
            if made_strategic_progress
            or any(action_made_progress(uci) for uci in selected_moves)
            else turns_without_progress + 1
        )
        if turns_without_progress >= config.no_progress_turns:
            termination = "no-progress"
            break
        turn_number += 1

    challenger_name = "RED" if challenger_color == engine.RED else "BLUE"
    challenger_score = (
        0.5 if winner is None else 1.0 if winner == challenger_name else 0.0
    )
    return GameResult(
        index=config.index,
        pair=pair,
        seed=pair_seed,
        personality=personality,
        challenger_color=challenger_name,
        winner=winner,
        termination=termination,
        turns=decisions,
        challenger_score=challenger_score,
        decisions=decisions,
        opening_book_decisions=opening_book_decisions,
        verified_decisions=verified_decisions,
        fallback_counts=dict(fallbacks),
        completed_depth_counts=dict(depths),
        move_turns=move_turns,
    )


def summarize(args: argparse.Namespace, games: list[GameResult]) -> Dict[str, Any]:
    games.sort(key=lambda game: game.index)
    points = sum(game.challenger_score for game in games)
    score_rate = points / len(games)
    clamped = min(0.999, max(0.001, score_rate))
    pairs = []
    policy_pairs = []
    for index in range(0, len(games), 2):
        left, right = games[index : index + 2]
        if left.pair != right.pair or left.seed != right.seed:
            raise RuntimeError("native arena produced a broken color pair")
        pairs.append((left.challenger_score + right.challenger_score) / 2)
        common_turns = 0
        for left_turn, right_turn in zip(
            left.move_turns, right.move_turns
        ):
            if left_turn != right_turn:
                break
            common_turns += 1
        compared_turns = min(
            len(left.move_turns), len(right.move_turns)
        )
        policy_pairs.append(
            {
                "pair": left.pair,
                "commonPrefixTurns": common_turns,
                "comparedTurns": compared_turns,
                "diverged": common_turns < compared_turns
                or len(left.move_turns) != len(right.move_turns),
            }
        )
    fallback_counts: Counter[str] = Counter()
    depth_counts: Counter[str] = Counter()
    for game in games:
        fallback_counts.update(game.fallback_counts)
        depth_counts.update(game.completed_depth_counts)
    decisions = sum(game.decisions for game in games)
    opening_book_decisions = sum(game.opening_book_decisions for game in games)
    searched_decisions = decisions - opening_book_decisions
    verified_decisions = sum(game.verified_decisions for game in games)
    verified_rate = verified_decisions / decisions if decisions else 0.0
    quality_reasons = []
    if fallback_counts["seeded"]:
        quality_reasons.append("seeded-fallback-decision")
    if searched_decisions <= 0:
        quality_reasons.append("no-searched-decisions")
    if verified_rate < 0.8:
        quality_reasons.append("fewer-than-80-percent-verified-decisions")
    rendered_games = []
    for game in games:
        record = asdict(game)
        move_turns = record.pop("move_turns")
        record["moveDigestSha256"] = hashlib.sha256(
            json.dumps(move_turns, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        rendered_games.append(record)
    return {
        "format": "ghq-native-value-screen-v1",
        "screeningOnly": True,
        "config": {
            "games": args.games,
            "pairs": args.games // 2,
            "timeMs": args.time_ms,
            "maxDepth": args.max_depth,
            "beamWidth": args.beam_width,
            "maxTurns": args.max_turns,
            "repetitionLimit": args.repetition_limit,
            "noProgressTurns": args.no_progress_turns,
            "seed": args.seed & 0xFFFF_FFFF,
            "workers": args.workers,
        },
        "provenance": {
            "engine": "public/engine.py",
            "baselinePath": str(args.baseline),
            "baselineSha256": artifact_fingerprint(args.baseline),
            "challengerPath": str(args.challenger),
            "challengerSha256": artifact_fingerprint(args.challenger),
        },
        "challenger": {
            "points": points,
            "scoreRate": round(score_rate, 4),
            "eloDifference": round(400 * math.log10(clamped / (1 - clamped)), 1),
            "pairWins": sum(score > 0.5 for score in pairs),
            "pairTies": sum(score == 0.5 for score in pairs),
            "pairLosses": sum(score < 0.5 for score in pairs),
        },
        "policyDivergence": {
            "divergedPairs": sum(pair["diverged"] for pair in policy_pairs),
            "totalPairs": len(policy_pairs),
            "pairs": policy_pairs,
        },
        "outcomes": dict(Counter(game.winner or "DRAW" for game in games)),
        "terminations": dict(Counter(game.termination for game in games)),
        "search": {
            "decisions": decisions,
            "openingBookDecisions": opening_book_decisions,
            "searchedDecisions": searched_decisions,
            "verifiedDecisions": verified_decisions,
            "verifiedRate": round(verified_rate, 4),
            "fallbackCounts": dict(fallback_counts),
            "completedDepthCounts": dict(depth_counts),
        },
        "qualityGate": {
            "passed": not quality_reasons,
            "reasons": quality_reasons,
        },
        "games": rendered_games,
    }


def main() -> None:
    args = parse_args()
    configs = [
        GameConfig(
            index=index,
            seed=args.seed & 0xFFFF_FFFF,
            baseline_path=str(args.baseline),
            challenger_path=str(args.challenger),
            time_ms=args.time_ms,
            max_depth=args.max_depth,
            beam_width=args.beam_width,
            max_turns=args.max_turns,
            repetition_limit=args.repetition_limit,
            no_progress_turns=args.no_progress_turns,
        )
        for index in range(args.games)
    ]
    if args.workers == 1:
        games = [play_game(config) for config in configs]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            games = list(pool.map(play_game, configs))
    report = summarize(args, games)
    rendered = f"{json.dumps(report, indent=2)}\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
