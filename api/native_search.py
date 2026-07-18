"""Native CPython search endpoint for the production GHQ engine.

This endpoint intentionally returns the raw Python search result.  The Next.js
analysis layer remains responsible for exploration, repetition avoidance,
model presentation, and persistent caching so those semantics stay identical
across the Pyodide and native backends.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, Optional


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _engine as engine  # noqa: E402
import _ghq_ai as ghq_ai  # noqa: E402
import _value_model as value_model  # noqa: E402


PERSONALITIES = frozenset(ghq_ai.PERSONALITIES)
CODE_VERSION = os.environ.get("VERCEL_GIT_COMMIT_SHA", "local-unversioned-search")


class NativeSearchInputError(ValueError):
    """The request cannot be searched as supplied."""


def _integer(
    payload: Dict[str, Any],
    name: str,
    fallback: int,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(name, fallback)
    if isinstance(value, bool) or not isinstance(value, int):
        raise NativeSearchInputError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )
    if value < minimum or value > maximum:
        raise NativeSearchInputError(
            f"{name} must be an integer from {minimum} through {maximum}"
        )
    return value


def _board(payload: Dict[str, Any]) -> engine.BaseBoard:
    serialized = payload.get("serializedState")
    fen = payload.get("fen")
    if isinstance(serialized, str) and serialized:
        return engine.BaseBoard.deserialize(serialized)
    if isinstance(fen, str) and fen:
        return engine.BaseBoard(fen)
    raise NativeSearchInputError("fen or serializedState is required")


def _outcome(board: engine.BaseBoard) -> Optional[Dict[str, Any]]:
    result = board.outcome()
    if result is None:
        return None
    winner = None
    if result.winner is engine.RED:
        winner = "RED"
    elif result.winner is engine.BLUE:
        winner = "BLUE"
    return {"winner": winner, "termination": result.termination}


def describe_native_position(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Return canonical state metadata without spending a search budget."""

    personality = payload.get("personality", "balanced")
    if not isinstance(personality, str) or personality not in PERSONALITIES:
        raise NativeSearchInputError(f"Unknown personality: {personality}")
    turn_number = _integer(payload, "turnNumber", 1, 1, 2_000)
    try:
        board = _board(payload)
    except NativeSearchInputError:
        raise
    except Exception as error:
        raise NativeSearchInputError(str(error)) from error
    return {
        "codeVersion": CODE_VERSION,
        "fen": board.board_fen(),
        "sideToMove": "RED" if board.is_red_turn() else "BLUE",
        "serializedState": board.serialize(),
        "outcome": _outcome(board),
        "evaluation": ghq_ai.evaluation_breakdown(board, personality, turn_number),
    }


def run_native_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a request, run the canonical engine, and replay its best turn."""

    personality = payload.get("personality", "balanced")
    if not isinstance(personality, str) or personality not in PERSONALITIES:
        raise NativeSearchInputError(f"Unknown personality: {personality}")
    time_ms = _integer(payload, "timeMs", 30_000, 50, 30_000)
    max_depth = _integer(payload, "maxDepth", 3, 1, 3)
    beam_width = _integer(payload, "beamWidth", 8, 2, 16)
    turn_number = _integer(payload, "turnNumber", 1, 1, 2_000)
    opening_seed = _integer(payload, "openingSeed", 0, 0, 0xFFFF_FFFF)
    max_actions = _integer(payload, "maxActions", 3, 2, 3)
    if max_actions != 3:
        raise NativeSearchInputError(
            "Native search currently supports the production three-action ruleset"
        )
    stagnation_turns = _integer(payload, "stagnationTurns", 0, 0, 400)
    value_model_version = payload.get("valueModel", "incumbent")
    if value_model_version not in value_model.ARTIFACTS:
        raise NativeSearchInputError(f"Unknown value model: {value_model_version}")

    try:
        board = _board(payload)
    except NativeSearchInputError:
        raise
    except Exception as error:
        raise NativeSearchInputError(str(error)) from error

    input_fen = board.board_fen()
    side_to_move = "RED" if board.is_red_turn() else "BLUE"
    raw_search = ghq_ai.search(
        board,
        personality,
        time_ms,
        max_depth,
        beam_width,
        turn_number,
        value_function=value_model.red_value_function(value_model_version),
        opening_seed=opening_seed,
        max_actions=max_actions,
        stagnation_turns=stagnation_turns,
    )
    raw_search["search"]["backend"] = "native-python"
    raw_search["search"]["value_model_backend"] = "native-gbdt"
    raw_search["search"]["value_model_version"] = value_model_version
    raw_search["search"]["code_version"] = CODE_VERSION

    for uci in raw_search["best_turn"]["all_moves"]:
        move = engine.Move.from_uci(uci)
        if not board.is_legal(move):
            raise RuntimeError(f"Search returned illegal production move: {uci}")
        board.push(move)

    return {
        "codeVersion": CODE_VERSION,
        "fen": input_fen,
        "sideToMove": side_to_move,
        "resultingFen": board.board_fen(),
        "serializedState": board.serialize(),
        "outcome": _outcome(board),
        "afterEvaluation": ghq_ai.evaluation_breakdown(
            board, personality, turn_number + 1
        ),
        "search": raw_search,
    }


class handler(BaseHTTPRequestHandler):
    """Vercel Python runtime entry point."""

    def _json(self, status: int, value: Dict[str, Any]) -> None:
        body = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        self._json(
            200,
            {
                "ok": True,
                "backend": "native-python",
                "engine": "public/engine.py",
                "codeVersion": CODE_VERSION,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler contract
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                raise NativeSearchInputError("A JSON request body is required")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise NativeSearchInputError("The JSON request must be an object")
            if payload.get("mode") == "describe":
                self._json(200, describe_native_position(payload))
            else:
                self._json(200, run_native_search(payload))
        except (NativeSearchInputError, json.JSONDecodeError) as error:
            self._json(400, {"error": str(error)})
        except Exception as error:  # Vercel logs preserve the full exception.
            self._json(500, {"error": str(error)})
