#!/usr/bin/env python3
"""Run one GHQ search against an isolated API checkpoint.

The evaluator launches this file in a fresh interpreter for each checkpoint.
That isolation is important because both bundles expose the same absolute
module names (``_engine``, ``_ghq_ai``, and ``_value_model``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-dir", required=True, type=Path)
    parser.add_argument("--fen", required=True)
    parser.add_argument("--turn-number", required=True, type=int)
    parser.add_argument("--personality", required=True)
    parser.add_argument("--time-ms", required=True, type=int)
    parser.add_argument("--max-depth", required=True, type=int)
    parser.add_argument("--beam-width", required=True, type=int)
    parser.add_argument("--max-actions", type=int, default=3)
    parser.add_argument("--opening-seed", type=int, default=0)
    parser.add_argument("--stagnation-turns", type=int, default=0)
    parser.add_argument("--model", default="_model_incumbent.json")
    return parser.parse_args()


def fingerprint(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_artifact(path: Path) -> Dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
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


def main() -> int:
    args = parse_args()
    api_dir = args.api_dir.resolve()
    for name in ("_engine.py", "_ghq_ai.py", "_value_model.py", args.model):
        path = api_dir / name
        if not path.is_file():
            raise ValueError(f"checkpoint is missing {path}")

    sys.path.insert(0, str(api_dir))
    import _engine as engine  # type: ignore  # noqa: E402
    import _ghq_ai as ghq_ai  # type: ignore  # noqa: E402
    import _value_model as value_model  # type: ignore  # noqa: E402

    artifact_path = api_dir / args.model
    artifact = load_artifact(artifact_path)

    def red_value(fen: str, turn_number: int) -> float:
        board = engine.BaseBoard(fen)
        if not board.pieces_mask(engine.HQ, engine.RED):
            return 0.0
        if not board.pieces_mask(engine.HQ, engine.BLUE):
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

    def policy(fen: str, turn_number: int, perspective: bool) -> float:
        board = engine.BaseBoard(fen)
        return value_model.policy_adjustment_from_features(
            value_model.extract_features(
                board, turn_number, perspective, artifact
            ),
            artifact,
        )

    board = engine.BaseBoard(args.fen)
    result = ghq_ai.search(
        board,
        args.personality,
        args.time_ms,
        args.max_depth,
        args.beam_width,
        args.turn_number,
        value_function=red_value,
        policy_function=policy,
        opening_seed=args.opening_seed,
        max_actions=args.max_actions,
        stagnation_turns=args.stagnation_turns,
    )
    print(
        json.dumps(
            {
                "format": "ghq-isolated-search-result-v1",
                "checkpoint": {
                    "apiDir": str(api_dir),
                    "searchCodeSha256": fingerprint(api_dir / "_ghq_ai.py"),
                    "engineSha256": fingerprint(api_dir / "_engine.py"),
                    "valueRuntimeSha256": fingerprint(
                        api_dir / "_value_model.py"
                    ),
                    "modelSha256": fingerprint(artifact_path),
                },
                "input": {
                    "fen": args.fen,
                    "turnNumber": args.turn_number,
                    "personality": args.personality,
                    "timeMs": args.time_ms,
                    "maxDepth": args.max_depth,
                    "beamWidth": args.beam_width,
                    "maxActions": args.max_actions,
                    "openingSeed": args.opening_seed,
                    "stagnationTurns": args.stagnation_turns,
                },
                "result": result,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
