import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_value_datasets import merge_datasets


FEATURE_NAMES = ["one", "two"]


def write_dataset(path: Path, samples, **schema_overrides):
    schema = {
        "type": "schema",
        "format": "ghq-value-features-v1",
        "feature_names": FEATURE_NAMES,
        **schema_overrides,
    }
    path.write_text(
        "\n".join(json.dumps(item) for item in (schema, *samples)) + "\n",
        encoding="utf-8",
    )


def sample(game_id: str, **overrides):
    return {
        "type": "sample",
        "game_id": game_id,
        "created_at": "2026-07-17T00:00:00Z",
        "outcome_reason": "hq-capture",
        "turn": 5,
        "perspective": "RED",
        "label": 1,
        "features": [1.0, 2.0],
        **overrides,
    }


class MergeValueDatasetsTests(unittest.TestCase):
    def paths(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        return temporary, root / "human.jsonl", root / "self.jsonl", root / "out.jsonl"

    def test_merges_and_normalizes_a_complete_exact_revision_pair(self):
        temporary, human, self_play, output = self.paths()
        self.addCleanup(temporary.cleanup)
        write_dataset(human, [sample("human-game")])
        common = {
            "source": "vercel_self_play",
            "generation_id": "generation",
            "pair_id": "generation-pair-0001",
            "code_version": "commit-a",
            "behavior_value_model_checkpoint": "checkpoint-a",
        }
        write_dataset(
            self_play,
            [sample("generation-0001", **common), sample("generation-0002", **common)],
        )

        stats = merge_datasets(
            human, self_play, output, "commit-a", "checkpoint-a"
        )

        records = [json.loads(line) for line in output.read_text().splitlines()]
        self.assertEqual(stats["self_play_pairs"], 1)
        self.assertEqual(stats["total_samples"], 3)
        self.assertEqual(records[0]["self_play_code_version"], "commit-a")
        self.assertEqual(
            records[0]["self_play_behavior_value_model_checkpoint"],
            "checkpoint-a",
        )
        self.assertEqual(records[1]["source"], "human")

    def test_rejects_orphaned_color_swapped_game(self):
        temporary, human, self_play, output = self.paths()
        self.addCleanup(temporary.cleanup)
        write_dataset(human, [sample("human-game")])
        write_dataset(
            self_play,
            [
                sample(
                    "generation-0001",
                    source="vercel_self_play",
                    generation_id="generation",
                    pair_id="generation-pair-0001",
                    code_version="commit-a",
                    behavior_value_model_checkpoint="checkpoint-a",
                )
            ],
        )
        with self.assertRaisesRegex(ValueError, "incomplete color-swapped pairs"):
            merge_datasets(human, self_play, output, "commit-a", "checkpoint-a")

    def test_rejects_schema_or_revision_mismatch(self):
        temporary, human, self_play, output = self.paths()
        self.addCleanup(temporary.cleanup)
        write_dataset(human, [sample("human-game")])
        common = {
            "source": "vercel_self_play",
            "generation_id": "generation",
            "pair_id": "generation-pair-0001",
            "code_version": "commit-b",
            "behavior_value_model_checkpoint": "checkpoint-a",
        }
        write_dataset(
            self_play,
            [sample("generation-0001", **common), sample("generation-0002", **common)],
        )
        with self.assertRaisesRegex(ValueError, "code version mismatch"):
            merge_datasets(human, self_play, output, "commit-a", "checkpoint-a")

        write_dataset(
            self_play,
            [sample("generation-0001", **common), sample("generation-0002", **common)],
            feature_names=["different"],
        )
        with self.assertRaisesRegex(ValueError, "feature schemas do not match"):
            merge_datasets(human, self_play, output, "commit-b", "checkpoint-a")

    def test_rejects_duplicate_samples(self):
        temporary, human, self_play, output = self.paths()
        self.addCleanup(temporary.cleanup)
        duplicate = sample("human-game")
        write_dataset(human, [duplicate, duplicate])
        common = {
            "source": "vercel_self_play",
            "generation_id": "generation",
            "pair_id": "generation-pair-0001",
            "code_version": "commit-a",
            "behavior_value_model_checkpoint": "checkpoint-a",
        }
        write_dataset(
            self_play,
            [sample("generation-0001", **common), sample("generation-0002", **common)],
        )
        with self.assertRaisesRegex(ValueError, "duplicate value sample"):
            merge_datasets(human, self_play, output, "commit-a", "checkpoint-a")

    def test_rejects_mixed_behavior_checkpoints(self):
        temporary, human, self_play, output = self.paths()
        self.addCleanup(temporary.cleanup)
        write_dataset(human, [sample("human-game")])
        common = {
            "source": "vercel_self_play",
            "generation_id": "generation",
            "pair_id": "generation-pair-0001",
            "code_version": "commit-a",
        }
        write_dataset(
            self_play,
            [
                sample(
                    "generation-0001",
                    **common,
                    behavior_value_model_checkpoint="checkpoint-a",
                ),
                sample(
                    "generation-0002",
                    **common,
                    behavior_value_model_checkpoint="checkpoint-b",
                ),
            ],
        )
        with self.assertRaisesRegex(ValueError, "behavior checkpoint mismatch"):
            merge_datasets(
                human, self_play, output, "commit-a", "checkpoint-a"
            )


if __name__ == "__main__":
    unittest.main()
