import unittest

from scripts.check_value_model_promotion import promotion_decision


def evidence(overall=-0.02, upper=-0.001, human=0.005, human_upper=0.02, self_play=-0.04):
    return {
        "paired_bootstrap_test": {
            "log_loss": {
                "candidate_minus_baseline": overall,
                "ci95_high": upper,
            },
            "brier": {"candidate_minus_baseline": -0.01},
            "by_source": {
                "human": {
                    "log_loss": {
                        "candidate_minus_baseline": human,
                        "ci95_high": human_upper,
                    }
                },
                "vercel_self_play": {
                    "log_loss": {"candidate_minus_baseline": self_play}
                },
            },
            "by_perspective": {
                color: {"log_loss": {"candidate_minus_baseline": -0.01}}
                for color in ("RED", "BLUE")
            },
            "by_source_and_perspective": {
                "human": {
                    color: {"log_loss": {"candidate_minus_baseline": 0.005}}
                    for color in ("RED", "BLUE")
                },
                "vercel_self_play": {
                    color: {"log_loss": {"candidate_minus_baseline": -0.03}}
                    for color in ("RED", "BLUE")
                },
            },
        }
    }


class PromotionDecisionTests(unittest.TestCase):
    def test_accepts_bounded_human_tradeoff_and_clear_overall_gain(self):
        result = promotion_decision(evidence())
        self.assertTrue(result["approved"])

    def test_rejects_human_regression_even_when_self_play_improves(self):
        result = promotion_decision(evidence(human=0.02, human_upper=0.05))
        self.assertFalse(result["approved"])
        failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
        self.assertIn("human-log-loss-mean", failed)
        self.assertIn("human-log-loss-upper-ci", failed)

    def test_rejects_missing_paired_game_evidence(self):
        result = promotion_decision({})
        self.assertFalse(result["approved"])
        self.assertIn("missing", result["reason"])

    def test_rejects_when_no_candidate_passed_validation_constraints(self):
        report = evidence()
        report["validation_constraints_passed"] = False
        result = promotion_decision(report)
        self.assertFalse(result["approved"])
        self.assertIn("validation-stage", result["reason"])

    def test_rejects_color_specific_self_play_regression(self):
        report = evidence()
        report["paired_bootstrap_test"]["by_source_and_perspective"][
            "vercel_self_play"
        ]["RED"]["log_loss"]["candidate_minus_baseline"] = 0.02
        result = promotion_decision(report)
        self.assertFalse(result["approved"])
        failed = {gate["name"] for gate in result["gates"] if not gate["passed"]}
        self.assertIn("self-play-red-log-loss-mean", failed)

    def test_rejects_legacy_report_without_color_evidence(self):
        report = evidence()
        del report["paired_bootstrap_test"]["by_perspective"]
        result = promotion_decision(report)
        self.assertFalse(result["approved"])
        self.assertIn("color-stratified", result["reason"])


if __name__ == "__main__":
    unittest.main()
