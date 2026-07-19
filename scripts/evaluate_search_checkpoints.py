#!/usr/bin/env python3
"""Compare two GHQ search-code checkpoints on tactical positions.

This is a black-box promotion gate. Each checkpoint runs in an isolated Python
process with its own production API bundle and the same value artifact. The
selected turns are then replayed and scored with objective tactical metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "run_search_checkpoint.py"
sys.path.insert(0, str(ROOT / "public"))
sys.path.insert(0, str(ROOT / "scripts"))

import engine  # noqa: E402
import ghq_ai  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-api-dir", required=True, type=Path)
    parser.add_argument("--candidate-api-dir", required=True, type=Path)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--worker-timeout-seconds", type=int, default=90)
    return parser.parse_args()


def read_corpus(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "ghq-tactical-search-corpus-v1":
        raise ValueError("unsupported tactical corpus format")
    positions = payload.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("tactical corpus has no positions")
    ids = [str(position.get("id")) for position in positions]
    if any(not identifier or identifier == "None" for identifier in ids):
        raise ValueError("every tactical corpus position needs an id")
    if len(ids) != len(set(ids)):
        raise ValueError("tactical corpus contains duplicate ids")
    return payload


def search_config(position: Dict[str, Any]) -> Dict[str, int]:
    raw = position.get("search") or {}
    config = {
        "timeMs": int(raw.get("timeMs", 20_000)),
        "maxDepth": int(raw.get("maxDepth", 2)),
        "beamWidth": int(raw.get("beamWidth", 6)),
        "maxActions": int(raw.get("maxActions", 3)),
        "stagnationTurns": int(raw.get("stagnationTurns", 0)),
    }
    if config["timeMs"] < 1:
        raise ValueError("timeMs must be positive")
    if config["maxDepth"] < 1:
        raise ValueError("maxDepth must be positive")
    if config["beamWidth"] < 1:
        raise ValueError("beamWidth must be positive")
    if config["maxActions"] not in (2, 3):
        raise ValueError("maxActions must be two or three")
    if config["stagnationTurns"] < 0:
        raise ValueError("stagnationTurns cannot be negative")
    return config


def run_checkpoint(
    python: str,
    api_dir: Path,
    position: Dict[str, Any],
    timeout_seconds: int,
) -> Dict[str, Any]:
    config = search_config(position)
    command = [
        python,
        str(WORKER),
        "--api-dir",
        str(api_dir),
        "--fen",
        str(position["fen"]),
        "--turn-number",
        str(int(position["turnNumber"])),
        "--personality",
        str(position["personality"]),
        "--time-ms",
        str(config["timeMs"]),
        "--max-depth",
        str(config["maxDepth"]),
        "--beam-width",
        str(config["beamWidth"]),
        "--max-actions",
        str(config["maxActions"]),
        "--opening-seed",
        str(int(position.get("openingSeed", 0))),
        "--stagnation-turns",
        str(config["stagnationTurns"]),
    ]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=max(timeout_seconds, int(config["timeMs"] / 1000) + 10),
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"checkpoint failed for {position['id']}: "
            f"{completed.stderr.strip() or completed.stdout.strip()}"
        )
    return json.loads(completed.stdout)


def replay_selected_turn(
    position: Dict[str, Any], isolated: Dict[str, Any]
) -> tuple[engine.BaseBoard, engine.BaseBoard, List[str]]:
    result = isolated["result"]
    if result.get("input_fen") != position["fen"]:
        raise ValueError(f"{position['id']} checkpoint input FEN drifted")
    before = engine.BaseBoard(str(position["fen"]))
    after = before.copy()
    selected = [str(move) for move in result["best_turn"]["all_moves"]]
    for uci in selected:
        legal = {move.uci(): move for move in after.generate_legal_moves()}
        move = legal.get(uci)
        if move is None:
            raise ValueError(f"{position['id']} returned illegal move {uci}")
        after.push(move)
    if after.board_fen() != result["best_turn"]["resulting_fen"]:
        raise ValueError(f"{position['id']} resulting FEN failed replay")
    return before, after, selected


def total_material(board: engine.BaseBoard, color: bool) -> float:
    return round(
        sum(
            ghq_ai.PIECE_VALUES[piece_type]
            * (
                engine.popcount(board.pieces_mask(piece_type, color))
                + board.get_reserve_count(piece_type, color)
            )
            for piece_type in ghq_ai.NON_HQ_TYPES
        ),
        4,
    )


def material_exchange(
    before: engine.BaseBoard, after: engine.BaseBoard, mover: bool
) -> Dict[str, float]:
    own_change = round(
        total_material(after, mover) - total_material(before, mover), 4
    )
    opponent_change = round(
        total_material(after, not mover)
        - total_material(before, not mover),
        4,
    )
    own_lost = round(max(0.0, -own_change), 4)
    opponent_lost = round(max(0.0, -opponent_change), 4)
    return {
        "ownMaterialChange": own_change,
        "opponentMaterialChange": opponent_change,
        "ownMaterialLost": own_lost,
        "opponentMaterialLost": opponent_lost,
        "netMaterialExchange": round(opponent_lost - own_lost, 4),
    }


def replay_opponent_reply(
    before: engine.BaseBoard,
    after_selected: engine.BaseBoard,
    selected: Sequence[str],
    isolated: Dict[str, Any],
) -> Dict[str, Any]:
    pv = [str(move) for move in isolated["result"]["principal_variation"]]
    if pv[: len(selected)] != list(selected):
        return {
            "complete": False,
            "moves": [],
            "resultingFen": after_selected.board_fen(),
            "twoTurnExchange": material_exchange(
                before, after_selected, before.turn
            ),
        }
    reply_board = after_selected.copy()
    reply_moves: List[str] = []
    for uci in pv[len(selected) :]:
        if reply_board.is_game_over() or reply_board.turn == before.turn:
            break
        legal = {
            move.uci(): move for move in reply_board.generate_legal_moves()
        }
        move = legal.get(uci)
        if move is None:
            raise ValueError(f"principal variation contains illegal move {uci}")
        reply_board.push(move)
        reply_moves.append(uci)
    complete = reply_board.is_game_over() or reply_board.turn == before.turn
    return {
        "complete": complete,
        "moves": reply_moves,
        "resultingFen": reply_board.board_fen(),
        "twoTurnExchange": material_exchange(
            before, reply_board, before.turn
        ),
    }


def assess_result(
    position: Dict[str, Any], isolated: Dict[str, Any]
) -> Dict[str, Any]:
    before, after, selected = replay_selected_turn(position, isolated)
    mover = before.turn
    probe = ghq_ai.Searcher(
        "balanced",
        time_ms=1_000_000,
        beam_width=1,
        turn_number=int(position["turnNumber"]),
    )
    input_risk, input_forced, input_critical = probe.tactical_risk(
        before, mover, check_hq_combinations=True
    )
    risk, forced, critical = probe.tactical_risk(
        after, mover, check_hq_combinations=True
    )
    opponent_risk, opponent_forced, opponent_critical = probe.tactical_risk(
        after, not mover, check_hq_combinations=True
    )
    outcome = after.outcome()
    search = isolated["result"]["search"]
    opponent_reply = replay_opponent_reply(
        before, after, selected, isolated
    )
    return {
        "checkpoint": isolated["checkpoint"],
        "selectedMoves": selected,
        "resultingFen": after.board_fen(),
        "score": isolated["result"]["score"],
        "selectedTurnExchange": material_exchange(before, after, mover),
        "opponentReply": opponent_reply,
        "search": {
            "completedDepth": int(search["completed_depth_in_turns"]),
            "fallback": str(search["fallback_used"]),
            "timedOut": bool(search["timed_out"]),
            "elapsedMs": float(search["elapsed_ms"]),
            "nodes": int(search["nodes"]),
            "finalSafetyCertified": search.get("final_safety_certified"),
            "finalSafetyGuardUsed": search.get("final_safety_guard_used"),
            "finalSafetyFloorRestored": search.get(
                "final_safety_floor_restored"
            ),
            "finalSafetyFloorSource": search.get(
                "final_safety_floor_source"
            ),
        },
        "inputSafety": {
            "tacticalRisk": round(float(input_risk), 4),
            "forcedLoss": round(float(input_forced), 4),
            "criticalExposure": round(float(input_critical), 4),
        },
        "resultingSafety": {
            "tacticalRisk": round(float(risk), 4),
            "forcedLoss": round(float(forced), 4),
            "criticalExposure": round(float(critical), 4),
            "opponentTacticalRisk": round(float(opponent_risk), 4),
            "opponentForcedLoss": round(float(opponent_forced), 4),
            "opponentCriticalExposure": round(
                float(opponent_critical), 4
            ),
            "immediateHqLoss": bool(
                forced >= ghq_ai.PIECE_VALUES[engine.HQ]
                or (
                    outcome is not None
                    and outcome.winner is not None
                    and outcome.winner != mover
                )
            ),
        },
    }


def check(
    name: str, passed: bool, actual: Any, expected: Any
) -> Dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "expected": expected,
    }


def candidate_checks(
    position: Dict[str, Any],
    baseline: Dict[str, Any],
    candidate: Dict[str, Any],
) -> List[Dict[str, Any]]:
    acceptance = position.get("acceptance") or {}
    checks: List[Dict[str, Any]] = []
    candidate_safety = candidate["resultingSafety"]
    baseline_safety = baseline["resultingSafety"]
    candidate_search = candidate["search"]
    for field, label in (
        ("engineSha256", "same-engine"),
        ("valueRuntimeSha256", "same-value-runtime"),
        ("modelSha256", "same-value-model"),
    ):
        baseline_fingerprint = baseline["checkpoint"].get(field)
        candidate_fingerprint = candidate["checkpoint"].get(field)
        checks.append(
            check(
                label,
                candidate_fingerprint == baseline_fingerprint,
                candidate_fingerprint,
                baseline_fingerprint,
            )
        )
    forbidden = set(str(move) for move in acceptance.get("forbiddenMoves", []))
    if forbidden:
        selected_forbidden = sorted(
            forbidden.intersection(candidate["selectedMoves"])
        )
        checks.append(
            check("forbidden-moves", not selected_forbidden, selected_forbidden, [])
        )
    if acceptance.get("requireFinalSafetyCertified", True):
        checks.append(
            check(
                "final-safety-certified",
                candidate_search["finalSafetyCertified"] is True,
                candidate_search["finalSafetyCertified"],
                True,
            )
        )
    minimum_depth = int(acceptance.get("minimumCompletedDepth", 0))
    if minimum_depth:
        checks.append(
            check(
                "minimum-completed-depth",
                candidate_search["completedDepth"] >= minimum_depth,
                candidate_search["completedDepth"],
                f">={minimum_depth}",
            )
        )
        if minimum_depth >= 2:
            checks.append(
                check(
                    "complete-opponent-reply",
                    candidate["opponentReply"]["complete"],
                    candidate["opponentReply"]["complete"],
                    True,
                )
            )
    if acceptance.get("avoidImmediateHqLoss", True):
        checks.append(
            check(
                "avoid-immediate-hq-loss",
                not candidate_safety["immediateHqLoss"],
                candidate_safety["immediateHqLoss"],
                False,
            )
        )
    if "maxCandidateForcedLoss" in acceptance:
        maximum = float(acceptance["maxCandidateForcedLoss"])
        checks.append(
            check(
                "maximum-forced-loss",
                candidate_safety["forcedLoss"] <= maximum,
                candidate_safety["forcedLoss"],
                f"<={maximum}",
            )
        )
    if "maxCandidateCriticalExposure" in acceptance:
        maximum = float(acceptance["maxCandidateCriticalExposure"])
        checks.append(
            check(
                "maximum-critical-exposure",
                candidate_safety["criticalExposure"] <= maximum,
                candidate_safety["criticalExposure"],
                f"<={maximum}",
            )
        )
    if acceptance.get("noWorseThanBaselineForcedLoss", True):
        checks.append(
            check(
                "no-worse-than-baseline-forced-loss",
                candidate_safety["forcedLoss"]
                <= baseline_safety["forcedLoss"],
                candidate_safety["forcedLoss"],
                f"<={baseline_safety['forcedLoss']}",
            )
        )
    if acceptance.get("noWorseThanBaselineCriticalExposure", True):
        checks.append(
            check(
                "no-worse-than-baseline-critical-exposure",
                candidate_safety["criticalExposure"]
                <= baseline_safety["criticalExposure"],
                candidate_safety["criticalExposure"],
                f"<={baseline_safety['criticalExposure']}",
            )
        )
    if "minimumTwoTurnNetExchange" in acceptance:
        minimum = float(acceptance["minimumTwoTurnNetExchange"])
        actual = candidate["opponentReply"]["twoTurnExchange"][
            "netMaterialExchange"
        ]
        checks.append(
            check(
                "minimum-two-turn-net-exchange",
                candidate["opponentReply"]["complete"]
                and actual >= minimum,
                actual,
                f">={minimum} with a complete reply",
            )
        )
    if acceptance.get("noWorseThanBaselineTwoTurnNetExchange", True):
        actual = candidate["opponentReply"]["twoTurnExchange"][
            "netMaterialExchange"
        ]
        baseline_exchange = baseline["opponentReply"]["twoTurnExchange"][
            "netMaterialExchange"
        ]
        checks.append(
            check(
                "no-worse-than-baseline-two-turn-net-exchange",
                (
                    not baseline["opponentReply"]["complete"]
                    or (
                        candidate["opponentReply"]["complete"]
                        and actual >= baseline_exchange
                    )
                ),
                actual,
                f">={baseline_exchange}",
            )
        )
    return checks


def evaluate(
    corpus: Dict[str, Any],
    baseline_api_dir: Path,
    candidate_api_dir: Path,
    python: str,
    workers: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    cases: List[Dict[str, Any]] = []
    for position in corpus["positions"]:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            baseline_future = executor.submit(
                run_checkpoint,
                python,
                baseline_api_dir,
                position,
                timeout_seconds,
            )
            candidate_future = executor.submit(
                run_checkpoint,
                python,
                candidate_api_dir,
                position,
                timeout_seconds,
            )
            baseline = assess_result(position, baseline_future.result())
            candidate = assess_result(position, candidate_future.result())
        checks = candidate_checks(position, baseline, candidate)
        objective_safety_improvement = (
            candidate["resultingSafety"]["forcedLoss"]
            < baseline["resultingSafety"]["forcedLoss"]
            or candidate["resultingSafety"]["criticalExposure"]
            < baseline["resultingSafety"]["criticalExposure"]
        )
        certification_improvement = (
            candidate["search"]["finalSafetyCertified"] is True
            and baseline["search"]["finalSafetyCertified"] is not True
        )
        cases.append(
            {
                "id": position["id"],
                "source": position.get("source"),
                "fen": position["fen"],
                "turnNumber": position["turnNumber"],
                "personality": position["personality"],
                "search": search_config(position),
                "baseline": baseline,
                "candidate": candidate,
                "candidateChecks": checks,
                "candidatePassed": all(item["passed"] for item in checks),
                "objectiveSafetyImprovement": objective_safety_improvement,
                "certificationImprovement": certification_improvement,
                "behaviorChanged": (
                    baseline["selectedMoves"] != candidate["selectedMoves"]
                ),
            }
        )
    candidate_failures = sum(not case["candidatePassed"] for case in cases)
    safety_regressions = sum(
        case["candidate"]["resultingSafety"]["forcedLoss"]
        > case["baseline"]["resultingSafety"]["forcedLoss"]
        or case["candidate"]["resultingSafety"]["criticalExposure"]
        > case["baseline"]["resultingSafety"]["criticalExposure"]
        for case in cases
    )
    reply_exchange_regressions = sum(
        case["baseline"]["opponentReply"]["complete"]
        and (
            not case["candidate"]["opponentReply"]["complete"]
            or case["candidate"]["opponentReply"]["twoTurnExchange"][
                "netMaterialExchange"
            ]
            < case["baseline"]["opponentReply"]["twoTurnExchange"][
                "netMaterialExchange"
            ]
        )
        for case in cases
    )
    objective_improvements = sum(
        case["objectiveSafetyImprovement"] for case in cases
    )
    certification_improvements = sum(
        case["certificationImprovement"] for case in cases
    )
    return {
        "format": "ghq-search-checkpoint-evaluation-v1",
        "corpus": {
            "name": corpus.get("name"),
            "format": corpus["format"],
            "positions": len(cases),
        },
        "summary": {
            "cases": len(cases),
            "candidatePassedCases": len(cases) - candidate_failures,
            "candidateFailedCases": candidate_failures,
            "safetyRegressions": safety_regressions,
            "replyExchangeRegressions": reply_exchange_regressions,
            "objectiveSafetyImprovements": objective_improvements,
            "certificationImprovements": certification_improvements,
            "behaviorChanges": sum(case["behaviorChanged"] for case in cases),
            "approvedForPairedArena": bool(cases)
            and candidate_failures == 0
            and safety_regressions == 0
            and reply_exchange_regressions == 0,
        },
        "cases": cases,
    }


def main() -> int:
    args = parse_args()
    corpus = read_corpus(args.corpus)
    report = evaluate(
        corpus,
        args.baseline_api_dir.resolve(),
        args.candidate_api_dir.resolve(),
        args.python,
        args.workers,
        args.worker_timeout_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": report["summary"], "output": str(args.output)}))
    return 0 if report["summary"]["approvedForPairedArena"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
