"""Prevent the Vercel Python bundle from drifting from canonical sources."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NativeRuntimeMirrorTest(unittest.TestCase):
    def test_engine_is_the_exact_production_engine(self) -> None:
        self.assertEqual(
            (ROOT / "api" / "_engine.py").read_bytes(),
            (ROOT / "public" / "engine.py").read_bytes(),
        )

    def test_search_differs_only_by_its_bundled_engine_import(self) -> None:
        bundled = (ROOT / "api" / "_ghq_ai.py").read_text(encoding="utf-8")
        canonical = (ROOT / "scripts" / "ghq_ai.py").read_text(encoding="utf-8")
        self.assertEqual(
            bundled.replace("import _engine as engine  # noqa: E402", "import engine  # noqa: E402"),
            canonical,
        )

    def test_value_artifacts_are_exact_checkpoint_copies(self) -> None:
        self.assertEqual(
            (ROOT / "api" / "_model_incumbent.json").read_bytes(),
            (ROOT / "src" / "game" / "value-model" / "model.generated.json").read_bytes(),
        )
        self.assertEqual(
            (ROOT / "api" / "_model_challenger.json").read_bytes(),
            (
                ROOT
                / "src"
                / "game"
                / "value-model"
                / "model.challenger.generated.json"
            ).read_bytes(),
        )


if __name__ == "__main__":
    unittest.main()
