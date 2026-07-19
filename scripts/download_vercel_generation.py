#!/usr/bin/env python3
"""Download completed GHQ self-play games through the bounded Vercel API."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--base-url", default="https://ghq-one.vercel.app")
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args()


def read_json(url: str) -> Dict[str, Any]:
    request = Request(url, headers={"Accept": "application/json"})
    with urlopen(request, timeout=90) as response:  # noqa: S310 - explicit API URL
        return json.load(response)


def main() -> None:
    args = parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError("--workers must be from 1 through 16")
    base_url = args.base_url.rstrip("/")
    generation = quote(args.generation, safe="")
    summary = read_json(
        f"{base_url}/api/self-play/generations/{generation}/summary"
    )
    expected = int(summary.get("expectedGames") or summary.get("games") or 0)
    if expected <= 0:
        raise RuntimeError("Generation has no manifest or completed games")

    def download(number: int) -> Optional[Dict[str, Any]]:
        game_id = f"{args.generation}-{number:04d}"
        url = (
            f"{base_url}/api/self-play/generations/{generation}/games/"
            f"{quote(game_id, safe='')}"
        )
        try:
            return read_json(url)
        except HTTPError as error:
            if error.code == 404:
                return None
            raise

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        games = [game for game in pool.map(download, range(1, expected + 1)) if game]
    games.sort(key=lambda game: str(game.get("gameId", "")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(game, separators=(",", ":")) + "\n" for game in games),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "generationId": args.generation,
                "expectedGames": expected,
                "downloadedGames": len(games),
                "output": str(args.output),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
