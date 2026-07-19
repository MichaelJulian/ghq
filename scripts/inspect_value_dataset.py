#!/usr/bin/env python3
"""Report value-dataset readiness without starting model training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_value_model import dataset_readiness_summary, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_names, rows, _, _ = load_dataset(args.dataset)
    report = dataset_readiness_summary(feature_names, rows)
    payload = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
