"""Regression tests for the self-contained Vercel native-search bundle."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class NativeApiBundleTest(unittest.TestCase):
    def test_native_search_bundle_imports_without_public_engine(self) -> None:
        """The deployed Python function cannot import files from ``public/``."""

        with tempfile.TemporaryDirectory() as directory:
            bundle = Path(directory)
            for name in (
                "_engine.py",
                "_ghq_ai.py",
                "_value_model.py",
                "_model_incumbent.json",
                "_model_challenger.json",
            ):
                shutil.copy2(ROOT / "api" / name, bundle / name)

            result = subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    (
                        "import sys; "
                        f"sys.path.insert(0, {str(bundle)!r}); "
                        "import _engine, _ghq_ai, _value_model; "
                        "assert _ghq_ai.engine is _engine; "
                        "assert _value_model.engine is _engine"
                    ),
                ],
                cwd=bundle,
                capture_output=True,
                text=True,
                check=False,
            )

        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
