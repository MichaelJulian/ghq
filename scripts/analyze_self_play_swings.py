#!/usr/bin/env python3
"""Find material-collapse decision windows in downloaded durable self-play."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "public"))

import engine  # noqa: E402
import ghq_ai  # noqa: E402


COLORS = {"RED": engine.RED, "BLUE": engine.BLUE}
PLAYERS = ("RED", "BLUE")
PIECE_TYPES = tuple(ghq_ai.NON_HQ_TYPES)
PIECE_VALUE_BY_NAME = {
    str(engine.PIECE_NAMES[piece_type]): float(
        ghq_ai.PIECE_VALUES[piece_type]
    )
    for piece_type in PIECE_TYPES
}
DEFAULT_WINDOWS = (2, 4, 6, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        action="append",
        type=Path,
        help="Downloaded durable game JSONL; repeat for multiple files.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--windows",
        default=",".join(str(value) for value in DEFAULT_WINDOWS),
        help="Comma-separated inclusive 1-10 windows/ranges, e.g. 2,4,6-10.",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Largest unfavorable and favorable windows retained per side.",
    )
    return parser.parse_args()


def parse_windows(value: str) -> tuple[int, ...]:
    windows: set[int] = set()
    for raw_token in value.split(","):
        token = raw_token.strip()
        if not token:
            raise ValueError("--windows contains an empty token")
        if "-" in token:
            pieces = token.split("-")
            if len(pieces) != 2:
                raise ValueError(f"invalid window range: {token}")
            start, stop = (int(piece) for piece in pieces)
            if start > stop:
                raise ValueError(f"window range must be ascending: {token}")
            windows.update(range(start, stop + 1))
        else:
            windows.add(int(token))
    if not windows:
        raise ValueError("at least one window is required")
    invalid = sorted(window for window in windows if not 1 <= window <= 10)
    if invalid:
        raise ValueError(
            "window sizes must be from 1 through 10: "
            + ", ".join(str(window) for window in invalid)
        )
    return tuple(sorted(windows))


def other_player(player: str) -> str:
    return "BLUE" if player == "RED" else "RED"


def piece_inventory(board: engine.BaseBoard, color: bool) -> Dict[str, int]:
    return {
        str(engine.PIECE_NAMES[piece_type]): (
            engine.popcount(board.pieces_mask(piece_type, color))
            + board.get_reserve_count(piece_type, color)
        )
        for piece_type in PIECE_TYPES
    }


def inventory_value(inventory: Dict[str, Any]) -> float:
    return round(
        sum(
            float(count) * PIECE_VALUE_BY_NAME.get(str(piece_type), 0.0)
            for piece_type, count in inventory.items()
        ),
        4,
    )


class PositionInspector:
    """Cache material and bounded tactical metrics for every recorded FEN."""

    def __init__(self) -> None:
        self._cache: Dict[str, Dict[str, Any]] = {}

    def inspect(self, fen: str, turn_number: int) -> Dict[str, Any]:
        cached = self._cache.get(fen)
        if cached is not None:
            return cached
        board = engine.BaseBoard(fen)
        players: Dict[str, Dict[str, Any]] = {}
        for player in PLAYERS:
            color = COLORS[player]
            inventory = piece_inventory(board, color)
            # check_hq_combinations=False keeps this an immediate material
            # probe rather than an expensive same-turn mate search. A fresh,
            # generous probe per color avoids sharing a search deadline.
            probe = ghq_ai.Searcher(
                "balanced",
                time_ms=1_000_000,
                beam_width=1,
                turn_number=turn_number,
            )
            risk, forced_loss, critical_exposure = probe.tactical_risk(
                board,
                color,
                check_hq_combinations=False,
            )
            players[player] = {
                "inventory": inventory,
                "materialValue": inventory_value(inventory),
                "tacticalRiskValue": round(float(risk), 4),
                "forcedLossValue": round(float(forced_loss), 4),
                "criticalExposureValue": round(
                    float(critical_exposure), 4
                ),
            }
        inspected = {"fen": fen, "players": players}
        self._cache[fen] = inspected
        return inspected


PositionMetrics = Callable[[str, int], Dict[str, Any]]


def perspective_position(
    inspected: Dict[str, Any], player: str
) -> Dict[str, Any]:
    opponent = other_player(player)
    own = inspected["players"][player]
    enemy = inspected["players"][opponent]
    return {
        "fen": inspected["fen"],
        "own": own,
        "opponent": enemy,
        "materialBalance": round(
            float(own["materialValue"]) - float(enemy["materialValue"]),
            4,
        ),
    }


def exchange_metrics(
    before: Dict[str, Any], after: Dict[str, Any]
) -> Dict[str, Any]:
    own_change = round(
        float(after["own"]["materialValue"])
        - float(before["own"]["materialValue"]),
        4,
    )
    opponent_change = round(
        float(after["opponent"]["materialValue"])
        - float(before["opponent"]["materialValue"]),
        4,
    )
    own_lost = round(max(0.0, -own_change), 4)
    opponent_lost = round(max(0.0, -opponent_change), 4)
    net = round(opponent_lost - own_lost, 4)
    if net > 0.0:
        assessment = "favorable"
    elif net < 0.0:
        assessment = "unfavorable"
    elif own_lost > 0.0 or opponent_lost > 0.0:
        assessment = "even"
    else:
        assessment = "quiet"
    return {
        "ownMaterialChange": own_change,
        "opponentMaterialChange": opponent_change,
        "ownMaterialLost": own_lost,
        "opponentMaterialLost": opponent_lost,
        "netMaterialExchange": net,
        "materialBalanceDelta": round(
            float(after["materialBalance"])
            - float(before["materialBalance"]),
            4,
        ),
        "assessment": assessment,
    }


def tactical_metric_deltas(
    before: Dict[str, Any], after: Dict[str, Any]
) -> Dict[str, float]:
    deltas: Dict[str, float] = {}
    for perspective in ("own", "opponent"):
        for metric in (
            "tacticalRiskValue",
            "forcedLossValue",
            "criticalExposureValue",
        ):
            key = perspective + metric[0].upper() + metric[1:] + "Delta"
            deltas[key] = round(
                float(after[perspective][metric])
                - float(before[perspective][metric]),
                4,
            )
    return deltas


def causal_collapse_metrics(
    before: Dict[str, Any],
    after_selected_turn: Dict[str, Any],
    exchange: Dict[str, Any],
    window_turns: int,
) -> Dict[str, Any]:
    """Separate newly caused exposure from an already-forced material loss.

    This is deliberately conservative: a loss covered by the forced-loss value
    that existed before the selected turn is not called avoidable.  Likewise,
    material spent while the selected turn has a clean forced HQ win is not a
    collapse: training on that window would teach the model to preserve pieces
    instead of converting the game.  Both remain visible for audit, but are
    excluded from actionable collapse examples.
    """
    preexisting_forced_loss = float(before["own"]["forcedLossValue"])
    own_material_lost = float(exchange["ownMaterialLost"])
    covered_by_preexisting = min(
        own_material_lost, preexisting_forced_loss
    )
    loss_beyond_preexisting = max(
        0.0, own_material_lost - preexisting_forced_loss
    )
    selected_deltas = tactical_metric_deltas(before, after_selected_turn)
    new_risk = max(0.0, selected_deltas["ownTacticalRiskValueDelta"])
    new_forced = max(0.0, selected_deltas["ownForcedLossValueDelta"])
    new_critical = max(
        0.0, selected_deltas["ownCriticalExposureValueDelta"]
    )
    unfavorable = exchange["assessment"] == "unfavorable"
    newly_exposed = new_forced > 0.0 or new_critical > 0.0
    own_forced_after = float(
        after_selected_turn["own"]["forcedLossValue"]
    )
    opponent_forced_after = float(
        after_selected_turn["opponent"]["forcedLossValue"]
    )
    forced_hq_win = (
        opponent_forced_after >= ghq_ai.PIECE_VALUES[engine.HQ]
        and own_forced_after < ghq_ai.PIECE_VALUES[engine.HQ]
    )

    if not unfavorable:
        category = "not-unfavorable"
        actionable = False
    elif forced_hq_win:
        category = "forced-hq-win-tradeoff"
        actionable = False
    elif loss_beyond_preexisting <= 0.0 and not newly_exposed:
        category = "within-preexisting-forced-loss"
        actionable = False
    elif newly_exposed:
        category = "new-exposure-collapse"
        actionable = True
    elif window_turns == 2:
        # A two-turn window is exactly the selected turn plus the opponent's
        # reply.  Material lost here is causally close enough to train from;
        # the zero detector delta identifies a tactical-metric blind spot.
        category = "latent-reply-collapse"
        actionable = True
    else:
        category = "unresolved-additional-loss"
        actionable = False

    return {
        "category": category,
        "actionable": actionable,
        "preexistingForcedLossValue": round(preexisting_forced_loss, 4),
        "ownMaterialLostCoveredByPreexistingForcedLoss": round(
            covered_by_preexisting, 4
        ),
        "ownMaterialLostBeyondPreexistingForcedLoss": round(
            loss_beyond_preexisting, 4
        ),
        "selectedTurnNewTacticalRiskValue": round(new_risk, 4),
        "selectedTurnNewForcedLossValue": round(new_forced, 4),
        "selectedTurnNewCriticalExposureValue": round(new_critical, 4),
        "selectedTurnForcedHqWin": forced_hq_win,
        "attributionWindowTurns": int(window_turns),
    }


def decision_search_quality(decision: Dict[str, Any]) -> Dict[str, Any]:
    telemetry = decision.get("searchTelemetry") or {}
    depth = int(decision.get("completedDepth") or 0)
    fallback = str(decision.get("fallback") or "unknown")
    final_safety_certified = telemetry.get("finalSafetyCertified")
    return {
        "completedDepth": depth,
        "fallback": fallback,
        "timedOut": bool(decision.get("timedOut")),
        "completedTurn": bool(decision.get("completedTurn")),
        "selectedRank": int(decision.get("selectedRank") or 1),
        "hasCompleteOpponentReply": depth >= 2,
        "unverifiedFallback": final_safety_certified is False
        or fallback == "seeded"
        or (fallback not in ("none", "unknown") and depth < 2),
        "finalSafetyCertified": final_safety_certified,
        "finalSafetyGuardUsed": bool(
            telemetry.get("finalSafetyGuardUsed")
        ),
        "finalSafetyProbeUsed": bool(
            telemetry.get("finalSafetyProbeUsed")
        ),
        "finalSafetyFloorRestored": bool(
            telemetry.get("finalSafetyFloorRestored")
        ),
        "finalSafetyFloorSource": telemetry.get("finalSafetyFloorSource"),
        "recommendationLabel": decision.get("recommendationLabel"),
        "searchBackend": decision.get("searchBackend"),
        "searchValueModelBackend": decision.get(
            "searchValueModelBackend"
        ),
        "searchCodeVersion": decision.get("searchCodeVersion"),
        "nodes": int(telemetry.get("nodes") or 0),
        "elapsedMs": round(float(telemetry.get("elapsedMs") or 0.0), 4),
    }


def aggregate_search_quality(
    decisions: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    qualities = [decision_search_quality(decision) for decision in decisions]
    return {
        "decisions": len(qualities),
        "minimumCompletedDepth": min(
            (quality["completedDepth"] for quality in qualities), default=0
        ),
        "depthAtLeastTwoDecisions": sum(
            quality["hasCompleteOpponentReply"] for quality in qualities
        ),
        "fallbackDecisions": sum(
            quality["fallback"] != "none" for quality in qualities
        ),
        "unverifiedFallbackDecisions": sum(
            quality["unverifiedFallback"] for quality in qualities
        ),
        "finalSafetyCertifiedDecisions": sum(
            quality["finalSafetyCertified"] is True for quality in qualities
        ),
        "uncertifiedFinalSafetyDecisions": sum(
            quality["finalSafetyCertified"] is False for quality in qualities
        ),
        "missingFinalSafetyTelemetryDecisions": sum(
            quality["finalSafetyCertified"] is None for quality in qualities
        ),
        "finalSafetyGuardDecisions": sum(
            quality["finalSafetyGuardUsed"] for quality in qualities
        ),
        "finalSafetyFloorRestoredDecisions": sum(
            quality["finalSafetyFloorRestored"] for quality in qualities
        ),
        "timedOutDecisions": sum(
            quality["timedOut"] for quality in qualities
        ),
        "allDecisionsHaveCompleteOpponentReply": bool(qualities)
        and all(
            quality["hasCompleteOpponentReply"] for quality in qualities
        ),
    }


@dataclass(frozen=True)
class VerifiedGame:
    payload: Dict[str, Any]
    decisions: tuple[Dict[str, Any], ...]


def replay_game(game: Dict[str, Any]) -> VerifiedGame:
    game_id = str(game.get("gameId") or "<missing-game-id>")
    termination = str((game.get("outcome") or {}).get("termination") or "")
    # Durable max-turn results deliberately set completed=false even though
    # the workflow has ended and every recorded turn is final. They are still
    # valid collapse evidence; an actually running/partial payload is not.
    if game.get("completed") is not True and termination != "max-turns":
        raise ValueError(f"game {game_id} is not a completed result")
    raw_decisions = game.get("decisions")
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise ValueError(f"game {game_id} has no decisions")
    decisions: List[Dict[str, Any]] = []
    prior_resulting_fen: str | None = None
    prior_turn_number = 0
    for index, raw_decision in enumerate(raw_decisions):
        if not isinstance(raw_decision, dict):
            raise ValueError(
                f"game {game_id} decision {index} is not an object"
            )
        decision = raw_decision
        turn_number = int(decision["turnNumber"])
        if index > 0 and turn_number <= prior_turn_number:
            raise ValueError(
                f"game {game_id} decision turns are not strictly increasing at "
                f"index {index}"
            )
        if decision.get("completedTurn") is not True:
            raise ValueError(
                f"game {game_id} turn {turn_number} is not a completed turn"
            )
        fen = str(decision["fen"])
        resulting_fen = str(decision["resultingFen"])
        if prior_resulting_fen is not None and fen != prior_resulting_fen:
            raise ValueError(
                f"game {game_id} decision {index} breaks FEN continuity"
            )
        player = str(decision["player"])
        if player not in COLORS:
            raise ValueError(
                f"game {game_id} decision {index} has invalid player {player}"
            )
        board = engine.BaseBoard(fen)
        if board.turn != COLORS[player]:
            raise ValueError(
                f"game {game_id} decision {index} player disagrees with FEN"
            )
        moves = decision.get("selectedMoves")
        if not isinstance(moves, list):
            raise ValueError(
                f"game {game_id} decision {index} has no selectedMoves list"
            )
        for uci_value in moves:
            uci = str(uci_value)
            legal_moves = {
                move.uci(): move for move in board.generate_legal_moves()
            }
            move = legal_moves.get(uci)
            if move is None:
                raise ValueError(
                    f"game {game_id} turn {turn_number} has illegal move {uci}"
                )
            board.push(move)
        if board.board_fen() != resulting_fen:
            raise ValueError(
                f"game {game_id} turn {turn_number} replay does not match "
                "resultingFen"
            )
        decisions.append(decision)
        prior_resulting_fen = resulting_fen
        prior_turn_number = turn_number

    if game.get("initialFen") != decisions[0]["fen"]:
        raise ValueError(f"game {game_id} initialFen does not match decisions")
    if game.get("finalFen") != decisions[-1]["resultingFen"]:
        raise ValueError(f"game {game_id} finalFen does not match decisions")
    return VerifiedGame(game, tuple(decisions))


def material_event(
    decision: Dict[str, Any],
    decision_index: int,
    perspective: str,
    metrics: PositionMetrics,
) -> Dict[str, Any] | None:
    before = perspective_position(
        metrics(str(decision["fen"]), int(decision["turnNumber"])),
        perspective,
    )
    after = perspective_position(
        metrics(str(decision["resultingFen"]), int(decision["turnNumber"])),
        perspective,
    )
    exchange = exchange_metrics(before, after)
    if exchange["assessment"] == "quiet":
        return None
    return {
        "decisionIndex": decision_index,
        "turnNumber": int(decision["turnNumber"]),
        "player": str(decision["player"]),
        "fen": str(decision["fen"]),
        "resultingFen": str(decision["resultingFen"]),
        "moves": [str(move) for move in decision["selectedMoves"]],
        "searchQuality": decision_search_quality(decision),
        **exchange,
    }


def build_window_record(
    game: VerifiedGame,
    start_index: int,
    window_turns: int,
    metrics: PositionMetrics,
) -> Dict[str, Any]:
    window = game.decisions[start_index : start_index + window_turns]
    start = window[0]
    end = window[-1]
    player = str(start["player"])
    start_turn = int(start["turnNumber"])
    end_turn = int(end["turnNumber"])
    before = perspective_position(
        metrics(str(start["fen"]), start_turn), player
    )
    after_selected_turn = perspective_position(
        metrics(str(start["resultingFen"]), start_turn), player
    )
    after = perspective_position(
        metrics(str(end["resultingFen"]), end_turn), player
    )
    events = [
        event
        for offset, decision in enumerate(window)
        if (
            event := material_event(
                decision,
                start_index + offset,
                player,
                metrics,
            )
        )
        is not None
    ]
    game_id = str(game.payload["gameId"])
    selected_turn_exchange = exchange_metrics(before, after_selected_turn)
    selected_turn_tactical_deltas = tactical_metric_deltas(
        before, after_selected_turn
    )
    window_exchange = exchange_metrics(before, after)
    return {
        "windowId": (
            f"{game_id}:{player}:turn-{start_turn}:index-{start_index}:"
            f"window-{window_turns}"
        ),
        "generationId": game.payload.get("generationId"),
        "gameId": game_id,
        "seed": game.payload.get("seed"),
        "windowTurns": window_turns,
        "startDecisionIndex": start_index,
        "endDecisionIndex": start_index + window_turns - 1,
        "startTurnNumber": start_turn,
        "endTurnNumber": end_turn,
        "player": player,
        "agentId": start.get("agentId"),
        "opponentId": start.get("opponentId"),
        "personality": start.get("personality"),
        "fen": str(start["fen"]),
        "resultingFen": str(start["resultingFen"]),
        "windowEndFen": str(end["resultingFen"]),
        "moves": [str(move) for move in start["selectedMoves"]],
        "searchQuality": decision_search_quality(start),
        "windowSearchQuality": aggregate_search_quality(window),
        "before": before,
        "afterSelectedTurn": after_selected_turn,
        "selectedTurnExchange": selected_turn_exchange,
        "selectedTurnTacticalDeltas": selected_turn_tactical_deltas,
        "after": after,
        "exchange": window_exchange,
        "windowTacticalDeltas": tactical_metric_deltas(before, after),
        "causalCollapse": causal_collapse_metrics(
            before, after_selected_turn, window_exchange, window_turns
        ),
        "materialEvents": events,
    }


def unfavorable_sort_key(record: Dict[str, Any]) -> tuple[Any, ...]:
    exchange = record["exchange"]
    return (
        float(exchange["netMaterialExchange"]),
        -float(exchange["ownMaterialLost"]),
        float(exchange["opponentMaterialLost"]),
        int(record["windowTurns"]),
        str(record["gameId"]),
        int(record["startTurnNumber"]),
        int(record["startDecisionIndex"]),
    )


def favorable_sort_key(record: Dict[str, Any]) -> tuple[Any, ...]:
    exchange = record["exchange"]
    return (
        -float(exchange["netMaterialExchange"]),
        -float(exchange["opponentMaterialLost"]),
        float(exchange["ownMaterialLost"]),
        int(record["windowTurns"]),
        str(record["gameId"]),
        int(record["startTurnNumber"]),
        int(record["startDecisionIndex"]),
    )


def summarize_records(
    records: Sequence[Dict[str, Any]], top: int
) -> Dict[str, Any]:
    unfavorable = [
        record
        for record in records
        if record["exchange"]["assessment"] == "unfavorable"
    ]
    favorable = [
        record
        for record in records
        if record["exchange"]["assessment"] == "favorable"
    ]
    even = [
        record
        for record in records
        if record["exchange"]["assessment"] == "even"
    ]
    quiet = [
        record
        for record in records
        if record["exchange"]["assessment"] == "quiet"
    ]
    actionable_collapses = [
        record
        for record in unfavorable
        if record["causalCollapse"]["actionable"]
    ]
    preexisting_forced_loss = [
        record
        for record in unfavorable
        if record["causalCollapse"]["category"]
        == "within-preexisting-forced-loss"
    ]
    category_counts = {
        category: sum(
            1
            for record in unfavorable
            if record["causalCollapse"]["category"] == category
        )
        for category in sorted(
            {
                record["causalCollapse"]["category"]
                for record in unfavorable
            }
        )
    }
    new_exposure_collapses = [
        record
        for record in unfavorable
        if record["causalCollapse"]["category"]
        == "new-exposure-collapse"
    ]
    latent_reply_collapses = [
        record
        for record in unfavorable
        if record["causalCollapse"]["category"]
        == "latent-reply-collapse"
    ]
    return {
        "evaluatedWindows": len(records),
        "unfavorableWindows": len(unfavorable),
        "favorableTradeWindows": len(favorable),
        "evenTradeWindows": len(even),
        "quietWindows": len(quiet),
        "actionableCollapseWindows": len(actionable_collapses),
        "preexistingForcedLossWindows": len(preexisting_forced_loss),
        "causalCategoryCounts": category_counts,
        "largestUnfavorable": sorted(
            unfavorable, key=unfavorable_sort_key
        )[:top],
        "largestActionableCollapses": sorted(
            actionable_collapses, key=unfavorable_sort_key
        )[:top],
        "largestNewExposureCollapses": sorted(
            new_exposure_collapses, key=unfavorable_sort_key
        )[:top],
        "largestLatentReplyCollapses": sorted(
            latent_reply_collapses, key=unfavorable_sort_key
        )[:top],
        "largestPreexistingForcedLossRealizations": sorted(
            preexisting_forced_loss, key=unfavorable_sort_key
        )[:top],
        "largestFavorableTrades": sorted(
            favorable, key=favorable_sort_key
        )[:top],
    }


def analyze_games(
    games: Iterable[Dict[str, Any]],
    windows: Sequence[int],
    top: int,
    position_metrics: PositionMetrics | None = None,
) -> Dict[str, Any]:
    if top < 1:
        raise ValueError("--top must be at least one")
    normalized_windows = tuple(sorted(set(int(window) for window in windows)))
    if not normalized_windows or any(
        window < 1 or window > 10 for window in normalized_windows
    ):
        raise ValueError("window sizes must be from 1 through 10")
    verified_games = [replay_game(game) for game in games]
    verified_games.sort(key=lambda game: str(game.payload["gameId"]))
    game_ids = [str(game.payload["gameId"]) for game in verified_games]
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("input contains duplicate game IDs")

    inspector = PositionInspector()
    metrics = position_metrics or inspector.inspect
    all_records: List[Dict[str, Any]] = []
    game_results: List[Dict[str, Any]] = []
    for game in verified_games:
        game_records: List[Dict[str, Any]] = []
        for window_turns in normalized_windows:
            for start_index in range(
                0, len(game.decisions) - window_turns + 1
            ):
                record = build_window_record(
                    game,
                    start_index,
                    window_turns,
                    metrics,
                )
                game_records.append(record)
                all_records.append(record)
        by_window: List[Dict[str, Any]] = []
        for window_turns in normalized_windows:
            by_player = {
                player: summarize_records(
                    [
                        record
                        for record in game_records
                        if record["windowTurns"] == window_turns
                        and record["player"] == player
                    ],
                    1,
                )
                for player in PLAYERS
            }
            by_window.append(
                {"windowTurns": window_turns, "byPlayer": by_player}
            )
        game_results.append(
            {
                "generationId": game.payload.get("generationId"),
                "gameId": game.payload["gameId"],
                "seed": game.payload.get("seed"),
                "decisions": len(game.decisions),
                "outcome": game.payload.get("outcome"),
                "replay": {
                    "verified": True,
                    "initialFen": game.payload.get("initialFen"),
                    "finalFen": game.payload.get("finalFen"),
                },
                "overallByPlayer": {
                    player: summarize_records(
                        [
                            record
                            for record in game_records
                            if record["player"] == player
                        ],
                        1,
                    )
                    for player in PLAYERS
                },
                "windows": by_window,
            }
        )

    global_windows = []
    for window_turns in normalized_windows:
        by_player = {
            player: summarize_records(
                [
                    record
                    for record in all_records
                    if record["windowTurns"] == window_turns
                    and record["player"] == player
                ],
                top,
            )
            for player in PLAYERS
        }
        global_windows.append(
            {"windowTurns": window_turns, "byPlayer": by_player}
        )

    return {
        "format": "ghq-self-play-decision-swings-v1",
        "config": {"windows": list(normalized_windows), "top": top},
        "completedGames": len(verified_games),
        "verifiedDecisions": sum(
            len(game.decisions) for game in verified_games
        ),
        "gameResults": game_results,
        "overallByPlayer": {
            player: summarize_records(
                [
                    record
                    for record in all_records
                    if record["player"] == player
                ],
                top,
            )
            for player in PLAYERS
        },
        "windows": global_windows,
    }


def load_jsonl(
    paths: Sequence[Path],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    games: List[Dict[str, Any]] = []
    inputs: List[Dict[str, Any]] = []
    for path in sorted(paths, key=lambda value: str(value)):
        data = path.read_bytes()
        lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
        file_games: List[Dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{path}:{line_number} is not valid JSON: {error.msg}"
                ) from error
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            file_games.append(payload)
        games.extend(file_games)
        inputs.append(
            {
                "path": str(path),
                "sha256": hashlib.sha256(data).hexdigest(),
                "games": len(file_games),
            }
        )
    return games, inputs


def main() -> None:
    args = parse_args()
    windows = parse_windows(args.windows)
    games, inputs = load_jsonl(args.input)
    report = analyze_games(games, windows, args.top)
    report["inputs"] = inputs
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "completedGames": report["completedGames"],
                "verifiedDecisions": report["verifiedDecisions"],
                "windows": list(windows),
                "output": str(args.output),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
