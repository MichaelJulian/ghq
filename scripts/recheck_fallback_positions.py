#!/usr/bin/env python3
"""Replay previously unverified self-play turns against production search."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="https://ghq-one.vercel.app")
    parser.add_argument("--time-ms", type=int, default=30_000)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=6)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def is_unverified(decision: Dict[str, Any]) -> bool:
    fallback = decision.get("fallback", "none")
    depth = int(decision.get("completedDepth", 0))
    return fallback == "seeded" or (fallback != "none" and depth < 2)


def load_cases(path: Path) -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        game = json.loads(line)
        for decision in game.get("decisions", []):
            if is_unverified(decision):
                cases.append(
                    {
                        "gameId": game["gameId"],
                        "decision": decision,
                    }
                )
    return cases


def search_verified(metadata: Dict[str, Any]) -> bool:
    fallback = metadata.get("fallback_used", "none")
    depth = int(metadata.get("completed_depth_in_turns", 0))
    return fallback != "seeded" and depth >= 2


def replay_case(
    case: Dict[str, Any],
    *,
    endpoint: str,
    time_ms: int,
    max_depth: int,
    beam_width: int,
) -> Dict[str, Any]:
    decision = case["decision"]
    payload = {
        "fen": decision["fen"],
        "personality": decision["personality"],
        "timeMs": time_ms,
        "maxDepth": max_depth,
        "beamWidth": beam_width,
        "turnNumber": decision["turnNumber"],
        "openingSeed": int(decision.get("explorationSeed", 0)) & 0xFFFF_FFFF,
        "maxActions": int(decision.get("selfActionLimit", 3)),
        "stagnationTurns": 0,
        "valueModel": decision.get("valueModel", "incumbent"),
    }
    request = Request(
        endpoint,
        data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=90) as response:  # noqa: S310 - explicit API URL
        result = json.load(response)
    search = result["search"]
    metadata = search["search"]
    return {
        "gameId": case["gameId"],
        "turnNumber": decision["turnNumber"],
        "player": decision["player"],
        "personality": decision["personality"],
        "fen": decision["fen"],
        "original": {
            "moves": decision["selectedMoves"],
            "fallback": decision["fallback"],
            "completedDepth": decision["completedDepth"],
        },
        "replay": {
            "codeVersion": result.get("codeVersion"),
            "moves": search["best_turn"]["all_moves"],
            "fallback": metadata.get("fallback_used"),
            "completedDepth": metadata.get("completed_depth_in_turns"),
            "elapsedMs": metadata.get("elapsed_ms"),
            "timedOut": metadata.get("timed_out"),
            "seedReplyVerified": metadata.get("seed_reply_verified", False),
            "seedSafetyRetryUsed": metadata.get("seed_safety_retry_used", False),
            "seedSafetyRetryVerified": metadata.get(
                "seed_safety_retry_verified", False
            ),
            "safeFallbackReplyVerified": metadata.get(
                "safe_fallback_reply_verified", False
            ),
            "tacticalReturnGuardUsed": metadata.get(
                "tactical_return_guard_used", False
            ),
            "verified": search_verified(metadata),
        },
    }


def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(records)
    verified = sum(bool(row["replay"]["verified"]) for row in rows)
    retries = sum(bool(row["replay"]["seedSafetyRetryUsed"]) for row in rows)
    retries_verified = sum(
        bool(row["replay"]["seedSafetyRetryVerified"]) for row in rows
    )
    return {
        "cases": len(rows),
        "verified": verified,
        "stillUnverified": len(rows) - verified,
        "verificationRate": verified / len(rows) if rows else 0.0,
        "seedSafetyRetries": retries,
        "seedSafetyRetriesVerified": retries_verified,
        "records": rows,
    }


def main() -> None:
    args = parse_args()
    if not 50 <= args.time_ms <= 30_000:
        raise ValueError("--time-ms must be from 50 through 30000")
    if not 1 <= args.max_depth <= 3:
        raise ValueError("--max-depth must be from 1 through 3")
    if not 2 <= args.beam_width <= 16:
        raise ValueError("--beam-width must be from 2 through 16")
    if not 1 <= args.workers <= 16:
        raise ValueError("--workers must be from 1 through 16")
    cases = load_cases(args.input)
    endpoint = f"{args.base_url.rstrip('/')}/api/native_search"

    def replay(case: Dict[str, Any]) -> Dict[str, Any]:
        return replay_case(
            case,
            endpoint=endpoint,
            time_ms=args.time_ms,
            max_depth=args.max_depth,
            beam_width=args.beam_width,
        )

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        report = summarize(pool.map(replay, cases))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(report, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
