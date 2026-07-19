#!/usr/bin/env python3
"""Safely combine the canonical human and exact-revision self-play datasets."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human", required=True, type=Path)
    parser.add_argument("--self-play", required=True, type=Path)
    parser.add_argument("--code-version", required=True)
    parser.add_argument("--value-model-checkpoint", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    schemas: List[Dict[str, Any]] = []
    samples: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if item.get("type") == "schema":
                schemas.append(item)
            elif item.get("type") == "sample":
                samples.append(item)
            else:
                raise ValueError(
                    f"{path}:{line_number}: unsupported record type {item.get('type')!r}"
                )
    if len(schemas) != 1:
        raise ValueError(f"{path} must contain exactly one schema record")
    if not samples:
        raise ValueError(f"{path} contains no samples")
    return schemas[0], samples


def validate_schema(
    human_schema: Dict[str, Any], self_play_schema: Dict[str, Any]
) -> List[str]:
    for name, schema in (("human", human_schema), ("self-play", self_play_schema)):
        if schema.get("format") != "ghq-value-features-v1":
            raise ValueError(f"{name} dataset has an unsupported feature format")
        feature_names = schema.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError(f"{name} dataset is missing feature_names")
    if human_schema["feature_names"] != self_play_schema["feature_names"]:
        raise ValueError("human and self-play feature schemas do not match")
    for field in ("self_play_search_backend", "self_play_value_model_backend"):
        if not str(self_play_schema.get(field) or "").strip():
            raise ValueError(f"self-play dataset is missing {field}")
    for field in (
        "paired_complete_only",
        "exact_hq_audit_required",
        "paratrooper_policy_audit_required",
        "zero_unverified_fallbacks_required",
        "color_swap_integrity_verified",
        "behavior_quality_telemetry_required",
    ):
        if self_play_schema.get(field) is not True:
            raise ValueError(f"self-play dataset has not verified {field}")
    if not str(self_play_schema.get("exact_hq_audit_sha256") or "").strip():
        raise ValueError("self-play dataset is missing exact_hq_audit_sha256")
    if int(self_play_schema.get("exact_hq_audit_max_nodes") or 0) < 100_000:
        raise ValueError("self-play dataset has an insufficient exact HQ audit")
    return list(human_schema["feature_names"])


def require_common_sample_fields(
    sample: Dict[str, Any], feature_count: int, dataset_name: str
) -> None:
    for field in (
        "game_id",
        "created_at",
        "outcome_reason",
        "turn",
        "perspective",
        "label",
        "features",
    ):
        if field not in sample:
            raise ValueError(f"{dataset_name} sample is missing {field}")
    if sample["perspective"] not in ("RED", "BLUE"):
        raise ValueError(f"{dataset_name} sample has an invalid perspective")
    if float(sample["label"]) not in (0.0, 0.5, 1.0):
        raise ValueError(f"{dataset_name} sample has an invalid label")
    if not isinstance(sample["features"], list) or len(sample["features"]) != feature_count:
        raise ValueError(f"{dataset_name} sample has a feature schema mismatch")


def normalize_human_samples(
    samples: Iterable[Dict[str, Any]], feature_count: int
) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for original in samples:
        sample = dict(original)
        require_common_sample_fields(sample, feature_count, "human")
        source = sample.get("source", "human")
        if source != "human" or sample.get("generation_id"):
            raise ValueError("human dataset contains a self-play or mislabeled sample")
        sample["source"] = "human"
        normalized.append(sample)
    return normalized


def validate_self_play_samples(
    samples: Iterable[Dict[str, Any]],
    feature_count: int,
    code_version: str,
    value_model_checkpoint: str,
    search_backend: str,
    value_model_backend: str,
) -> List[Dict[str, Any]]:
    validated: List[Dict[str, Any]] = []
    games_by_pair: Dict[str, set[str]] = defaultdict(set)
    pair_by_game: Dict[str, str] = {}
    generation_by_pair: Dict[str, set[str]] = defaultdict(set)
    for original in samples:
        sample = dict(original)
        require_common_sample_fields(sample, feature_count, "self-play")
        if sample.get("source") != "vercel_self_play":
            raise ValueError("self-play dataset contains a non-self-play sample")
        if sample.get("code_version") != code_version:
            raise ValueError(
                "self-play code version mismatch: "
                f"expected {code_version}, received {sample.get('code_version') or 'missing'}"
            )
        if sample.get("behavior_value_model_checkpoint") != value_model_checkpoint:
            raise ValueError(
                "self-play behavior checkpoint mismatch: "
                f"expected {value_model_checkpoint}, received "
                f"{sample.get('behavior_value_model_checkpoint') or 'missing'}"
            )
        if sample.get("behavior_search_backend") != search_backend:
            raise ValueError(
                "self-play search backend mismatch: "
                f"expected {search_backend}, received "
                f"{sample.get('behavior_search_backend') or 'missing'}"
            )
        if sample.get("behavior_value_model_backend") != value_model_backend:
            raise ValueError(
                "self-play value backend mismatch: "
                f"expected {value_model_backend}, received "
                f"{sample.get('behavior_value_model_backend') or 'missing'}"
            )
        for field in (
            "behavior_agent_id",
            "behavior_opponent_id",
            "behavior_personality",
        ):
            if not str(sample.get(field) or "").strip():
                raise ValueError(f"self-play sample is missing {field}")
        if not isinstance(sample.get("behavior_selected_moves"), list):
            raise ValueError("self-play sample is missing behavior_selected_moves")
        completed_depth = sample.get("behavior_completed_depth")
        if not isinstance(completed_depth, int) or completed_depth < 2:
            raise ValueError("self-play sample lacks a complete opponent reply")
        if sample.get("behavior_fallback") not in ("none", "safe"):
            raise ValueError("self-play sample has an unverified behavior fallback")
        if not isinstance(sample.get("behavior_timed_out"), bool):
            raise ValueError("self-play sample is missing behavior_timed_out")
        game_id = str(sample.get("game_id") or "").strip()
        pair_id = str(sample.get("pair_id") or "").strip()
        generation_id = str(sample.get("generation_id") or "").strip()
        if not pair_id or not generation_id:
            raise ValueError("self-play sample is missing pair or generation provenance")
        prior_pair = pair_by_game.setdefault(game_id, pair_id)
        if prior_pair != pair_id:
            raise ValueError(f"self-play game {game_id} belongs to multiple pairs")
        games_by_pair[pair_id].add(game_id)
        generation_by_pair[pair_id].add(generation_id)
        validated.append(sample)
    incomplete = sorted(pair_id for pair_id, games in games_by_pair.items() if len(games) != 2)
    if incomplete:
        raise ValueError(
            "self-play dataset contains incomplete color-swapped pairs: "
            + ", ".join(incomplete[:5])
        )
    mixed_generations = sorted(
        pair_id for pair_id, generations in generation_by_pair.items() if len(generations) != 1
    )
    if mixed_generations:
        raise ValueError(
            "self-play pairs cross generation boundaries: "
            + ", ".join(mixed_generations[:5])
        )
    return validated


def reject_duplicate_samples(samples: Iterable[Dict[str, Any]]) -> None:
    seen: set[Tuple[str, str, int, str]] = set()
    for sample in samples:
        key = (
            str(sample["source"]),
            str(sample["game_id"]),
            int(sample["turn"]),
            str(sample["perspective"]),
        )
        if key in seen:
            raise ValueError(f"duplicate value sample {key}")
        seen.add(key)


def merge_datasets(
    human_path: Path,
    self_play_path: Path,
    output_path: Path,
    code_version: str,
    value_model_checkpoint: str,
) -> Dict[str, int]:
    human_schema, raw_human = read_jsonl(human_path)
    self_play_schema, raw_self_play = read_jsonl(self_play_path)
    feature_names = validate_schema(human_schema, self_play_schema)
    human = normalize_human_samples(raw_human, len(feature_names))
    self_play = validate_self_play_samples(
        raw_self_play,
        len(feature_names),
        code_version,
        value_model_checkpoint,
        str(self_play_schema["self_play_search_backend"]),
        str(self_play_schema["self_play_value_model_backend"]),
    )
    combined = [*human, *self_play]
    reject_duplicate_samples(combined)

    self_play_games = {sample["game_id"] for sample in self_play}
    pair_ids = {sample["pair_id"] for sample in self_play}
    schema = {
        "type": "schema",
        "format": "ghq-value-features-v1",
        "feature_names": feature_names,
        "ruleset": "three-actions",
        "source": "human+vercel-self-play",
        "self_play_code_version": code_version,
        "self_play_behavior_value_model_checkpoint": value_model_checkpoint,
        "self_play_search_backend": self_play_schema[
            "self_play_search_backend"
        ],
        "self_play_value_model_backend": self_play_schema[
            "self_play_value_model_backend"
        ],
        "paired_complete_only": True,
        "exact_hq_audit_required": True,
        "paratrooper_policy_audit_required": True,
        "zero_unverified_fallbacks_required": True,
        "color_swap_integrity_verified": True,
        "behavior_quality_telemetry_required": True,
        "exact_hq_audit_sha256": self_play_schema[
            "exact_hq_audit_sha256"
        ],
        "exact_hq_audit_max_nodes": self_play_schema[
            "exact_hq_audit_max_nodes"
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(schema, separators=(",", ":")) + "\n")
        for sample in combined:
            handle.write(json.dumps(sample, separators=(",", ":")) + "\n")
    return {
        "human_samples": len(human),
        "self_play_samples": len(self_play),
        "self_play_games": len(self_play_games),
        "self_play_pairs": len(pair_ids),
        "total_samples": len(combined),
    }


def main() -> None:
    args = parse_args()
    stats = merge_datasets(
        args.human,
        args.self_play,
        args.output,
        args.code_version,
        args.value_model_checkpoint,
    )
    print(json.dumps({**stats, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
