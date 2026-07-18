import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import audit_hq_loss  # noqa: E402


class ExactHqLossAuditTests(unittest.TestCase):
    def test_classifies_an_unthreatened_hq_position_as_survivable(self):
        result = audit_hq_loss.audit_hq_loss(
            "q7/8/8/8/8/8/8/7Q - - b", example_limit=2
        )

        self.assertFalse(result["forced_hq_loss"])
        self.assertGreater(result["safe_turns"], 0)
        self.assertLessEqual(len(result["safe_examples"]), 2)

    def test_classifies_a_production_forced_mate_as_forced(self):
        result = audit_hq_loss.audit_hq_loss(
            "q7/7i/i7/1i2i3/i7/5i2/i7/6Qf - - r", example_limit=2
        )

        self.assertTrue(result["forced_hq_loss"])
        self.assertEqual(result["safe_turns"], 0)
        self.assertEqual(result["complete_turn_states"], 5)


if __name__ == "__main__":
    unittest.main()
