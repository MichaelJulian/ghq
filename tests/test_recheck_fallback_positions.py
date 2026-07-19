import unittest

from scripts.recheck_fallback_positions import is_unverified, search_verified, summarize


class RecheckFallbackPositionsTest(unittest.TestCase):
    def test_identifies_only_blind_fallbacks(self):
        self.assertTrue(is_unverified({"fallback": "seeded", "completedDepth": 2}))
        self.assertTrue(is_unverified({"fallback": "safe", "completedDepth": 0}))
        self.assertFalse(is_unverified({"fallback": "safe", "completedDepth": 2}))
        self.assertFalse(is_unverified({"fallback": "none", "completedDepth": 0}))

    def test_requires_a_full_reply_and_nonseeded_result(self):
        self.assertTrue(
            search_verified({"fallback_used": "none", "completed_depth_in_turns": 2})
        )
        self.assertTrue(
            search_verified({"fallback_used": "safe", "completed_depth_in_turns": 2})
        )
        self.assertFalse(
            search_verified({"fallback_used": "seeded", "completed_depth_in_turns": 2})
        )
        self.assertFalse(
            search_verified({"fallback_used": "safe", "completed_depth_in_turns": 0})
        )

    def test_summarizes_verification_and_retry_coverage(self):
        report = summarize(
            [
                {
                    "replay": {
                        "verified": True,
                        "seedSafetyRetryUsed": True,
                        "seedSafetyRetryVerified": True,
                    }
                },
                {
                    "replay": {
                        "verified": False,
                        "seedSafetyRetryUsed": True,
                        "seedSafetyRetryVerified": False,
                    }
                },
            ]
        )
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["verified"], 1)
        self.assertEqual(report["stillUnverified"], 1)
        self.assertEqual(report["verificationRate"], 0.5)
        self.assertEqual(report["seedSafetyRetries"], 2)
        self.assertEqual(report["seedSafetyRetriesVerified"], 1)


if __name__ == "__main__":
    unittest.main()
