#!/usr/bin/env python3
"""Approve a value-model challenger only when holdout evidence is sufficient."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def promotion_decision(report: Dict[str, Any]) -> Dict[str, Any]:
    bootstrap = report.get("paired_bootstrap_test")
    if not bootstrap:
        return {
            "approved": False,
            "gates": [],
            "reason": "paired game-bootstrap evidence is missing",
        }

    sources = bootstrap.get("by_source", {})
    human = sources.get("human")
    self_play = sources.get("vercel_self_play")
    perspectives = bootstrap.get("by_perspective", {})
    source_perspectives = bootstrap.get("by_source_and_perspective", {})
    if any(color not in perspectives for color in ("RED", "BLUE")) or any(
        color not in source_perspectives.get(source, {})
        for source in ("human", "vercel_self_play")
        for color in ("RED", "BLUE")
    ):
        return {
            "approved": False,
            "gates": [],
            "reason": "paired color-stratified evidence is missing",
        }
    gates: List[Dict[str, Any]] = []

    def gate(name: str, value: float, maximum: float, rationale: str) -> None:
        gates.append(
            {
                "name": name,
                "passed": value <= maximum,
                "value": value,
                "maximum": maximum,
                "rationale": rationale,
            }
        )

    gate(
        "overall-log-loss-mean",
        bootstrap["log_loss"]["candidate_minus_baseline"],
        0.0,
        "The challenger must improve mean game-balanced test log loss.",
    )
    gate(
        "overall-log-loss-upper-ci",
        bootstrap["log_loss"]["ci95_high"],
        0.01,
        "The 95% upper bound may not allow material aggregate regression.",
    )
    gate(
        "overall-brier-mean",
        bootstrap["brier"]["candidate_minus_baseline"],
        0.0,
        "The challenger must improve mean game-balanced Brier score.",
    )
    if human:
        gate(
            "human-log-loss-mean",
            human["log_loss"]["candidate_minus_baseline"],
            0.01,
            "Self-play learning may not materially regress human-game prediction.",
        )
        gate(
            "human-log-loss-upper-ci",
            human["log_loss"]["ci95_high"],
            0.03,
            "Human holdout uncertainty must remain inside a bounded regression.",
        )
    if self_play:
        gate(
            "self-play-log-loss-mean",
            self_play["log_loss"]["candidate_minus_baseline"],
            0.0,
            "A self-play challenger must actually improve self-play prediction.",
        )
    for color in ("RED", "BLUE"):
        gate(
            f"overall-{color.lower()}-log-loss-mean",
            perspectives[color]["log_loss"]["candidate_minus_baseline"],
            0.01,
            f"Aggregate {color} prediction may not materially regress.",
        )
        gate(
            f"human-{color.lower()}-log-loss-mean",
            source_perspectives["human"][color]["log_loss"][
                "candidate_minus_baseline"
            ],
            0.015,
            f"Human-game {color} prediction must stay inside the bounded tradeoff.",
        )
        gate(
            f"self-play-{color.lower()}-log-loss-mean",
            source_perspectives["vercel_self_play"][color]["log_loss"][
                "candidate_minus_baseline"
            ],
            0.0,
            f"Self-play learning must improve {color} prediction, not only the opposite color.",
        )

    return {
        "approved": bool(gates) and all(item["passed"] for item in gates),
        "gates": gates,
    }


def main() -> None:
    args = parse_args()
    decision = promotion_decision(
        json.loads(args.report.read_text(encoding="utf-8"))
    )
    rendered = json.dumps(decision, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    if args.require_pass and not decision["approved"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
