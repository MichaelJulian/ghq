import copy
import json
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_self_play_swings as swings  # noqa: E402


TURN_34_FEN = (
    "2ir↓1i2/qi1r↓it↓ir↓/5fh↙f/1R↑2R↑3/5R↑2/1T↑H↑I1I1I/"
    "1FI1I1I1/F2I1F1Q IP p b"
)
TURN_34_RESULT = (
    "2ir↓1i2/qi1r↓it↓ir↓/5f1f/1R↑2R↑1h←1/5R↑2/1T↑H↑2I1I/"
    "1FI1I1I1/F2I1F1Q IP p r"
)
TURN_35_RESULT = (
    "2ir↓1i2/q2r↓1t↓ir↓/2R↑4f/4R↑1R↑1/6I1/1T↑H↑4I/"
    "1FI1I1I1/F2I1F1Q IP p b"
)


def collapse_game(game_id="collapse-game"):
    return {
        "format": "ghq-self-play-game-v1",
        "generationId": "test-generation",
        "gameId": game_id,
        "seed": 123,
        "initialFen": TURN_34_FEN,
        "initialTurnNumber": 34,
        "finalFen": TURN_35_RESULT,
        "decisions": [
            {
                "turnNumber": 34,
                "player": "BLUE",
                "fen": TURN_34_FEN,
                "resultingFen": TURN_34_RESULT,
                "personality": "fortress",
                "agentId": "blue-fortress",
                "opponentId": "red-balanced",
                "selectedMoves": ["sbd3", "g6g5←", "skip"],
                "selectedRank": 1,
                "completedDepth": 2,
                "timedOut": True,
                "fallback": "none",
                "completedTurn": True,
                "recommendationLabel": "best found",
                "searchBackend": "native-python",
                "searchValueModelBackend": "native-gbdt",
                "searchCodeVersion": "test-revision",
                "searchTelemetry": {"nodes": 168, "elapsedMs": 21752.63},
            },
            {
                "turnNumber": 35,
                "player": "RED",
                "fen": TURN_34_RESULT,
                "resultingFen": TURN_35_RESULT,
                "personality": "balanced",
                "agentId": "red-balanced",
                "opponentId": "blue-fortress",
                "selectedMoves": [
                    "sbe7",
                    "sbb7",
                    "sbf6",
                    "f3g4xg5",
                    "b5c6↑",
                    "f4g5↑",
                ],
                "selectedRank": 1,
                "completedDepth": 2,
                "timedOut": True,
                "fallback": "none",
                "completedTurn": True,
                "recommendationLabel": "best found",
                "searchBackend": "native-python",
                "searchValueModelBackend": "native-gbdt",
                "searchCodeVersion": "test-revision",
                "searchTelemetry": {"nodes": 186, "elapsedMs": 24000.13},
            },
        ],
        "outcome": {"termination": "max-turns"},
        "completed": True,
    }


class SelfPlaySwingAnalysisTests(unittest.TestCase):
    def test_parses_deduplicated_windows_and_ranges(self):
        self.assertEqual(swings.parse_windows("2,4,6-8,8"), (2, 4, 6, 7, 8))
        for value in ("0", "11", "3-1", "1,,2"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                swings.parse_windows(value)

    def test_replays_and_finds_reply_window_collapse_and_favorable_trade(self):
        report = swings.analyze_games([collapse_game()], (1, 2), 3)

        self.assertEqual(report["completedGames"], 1)
        self.assertEqual(report["verifiedDecisions"], 2)
        self.assertTrue(report["gameResults"][0]["replay"]["verified"])

        one_turn = report["windows"][0]["byPlayer"]
        red_trade = one_turn["RED"]["largestFavorableTrades"][0]
        self.assertEqual(red_trade["startTurnNumber"], 35)
        self.assertEqual(red_trade["exchange"]["opponentMaterialLost"], 11.0)
        self.assertEqual(red_trade["exchange"]["netMaterialExchange"], 11.0)
        self.assertEqual(red_trade["exchange"]["assessment"], "favorable")

        two_turn = report["windows"][1]["byPlayer"]["BLUE"]
        collapse = two_turn["largestUnfavorable"][0]
        self.assertEqual(collapse["startTurnNumber"], 34)
        self.assertEqual(collapse["endTurnNumber"], 35)
        self.assertEqual(collapse["moves"], ["sbd3", "g6g5←", "skip"])
        self.assertEqual(collapse["before"]["own"]["materialValue"], 36.0)
        self.assertEqual(collapse["before"]["opponent"]["materialValue"], 42.0)
        self.assertEqual(
            collapse["afterSelectedTurn"]["own"]["tacticalRiskValue"], 11.0
        )
        self.assertEqual(
            collapse["afterSelectedTurn"]["own"]["forcedLossValue"], 5.0
        )
        self.assertEqual(
            collapse["afterSelectedTurn"]["own"][
                "criticalExposureValue"
            ],
            11.0,
        )
        self.assertEqual(
            collapse["selectedTurnExchange"]["netMaterialExchange"], 1.0
        )
        self.assertEqual(
            collapse["selectedTurnTacticalDeltas"][
                "ownTacticalRiskValueDelta"
            ],
            6.0,
        )
        self.assertEqual(collapse["after"]["own"]["materialValue"], 25.0)
        self.assertEqual(collapse["after"]["opponent"]["materialValue"], 41.0)
        self.assertEqual(collapse["exchange"]["ownMaterialLost"], 11.0)
        self.assertEqual(collapse["exchange"]["opponentMaterialLost"], 1.0)
        self.assertEqual(collapse["exchange"]["netMaterialExchange"], -10.0)
        self.assertEqual(collapse["exchange"]["materialBalanceDelta"], -10.0)
        self.assertEqual(collapse["exchange"]["assessment"], "unfavorable")
        self.assertEqual(
            collapse["causalCollapse"]["category"],
            "new-exposure-collapse",
        )
        self.assertTrue(collapse["causalCollapse"]["actionable"])
        self.assertEqual(
            collapse["causalCollapse"][
                "ownMaterialLostBeyondPreexistingForcedLoss"
            ],
            6.0,
        )
        self.assertEqual(two_turn["actionableCollapseWindows"], 1)
        self.assertEqual(two_turn["preexistingForcedLossWindows"], 0)
        self.assertEqual(
            two_turn["causalCategoryCounts"],
            {"new-exposure-collapse": 1},
        )
        self.assertEqual(
            two_turn["largestActionableCollapses"][0]["windowId"],
            collapse["windowId"],
        )
        self.assertEqual(
            two_turn["largestNewExposureCollapses"][0]["windowId"],
            collapse["windowId"],
        )
        self.assertEqual(two_turn["largestLatentReplyCollapses"], [])
        self.assertEqual(collapse["searchQuality"]["completedDepth"], 2)
        self.assertTrue(
            collapse["windowSearchQuality"][
                "allDecisionsHaveCompleteOpponentReply"
            ]
        )
        self.assertEqual(
            [event["turnNumber"] for event in collapse["materialEvents"]],
            [34, 35],
        )
        self.assertEqual(
            collapse["materialEvents"][1]["searchQuality"]["nodes"], 186
        )
        self.assertIn("tacticalRiskValue", collapse["before"]["own"])
        self.assertIn("forcedLossValue", collapse["after"]["own"])
        self.assertIn("criticalExposureValue", collapse["after"]["opponent"])

    def test_preexisting_forced_loss_is_not_called_actionable(self):
        before = {
            "own": {
                "materialValue": 30.0,
                "tacticalRiskValue": 8.0,
                "forcedLossValue": 8.0,
                "criticalExposureValue": 8.0,
            },
            "opponent": {
                "materialValue": 30.0,
                "tacticalRiskValue": 0.0,
                "forcedLossValue": 0.0,
                "criticalExposureValue": 0.0,
            },
            "materialBalance": 0.0,
        }
        after_selected = {
            "own": {
                "materialValue": 30.0,
                "tacticalRiskValue": 5.0,
                "forcedLossValue": 5.0,
                "criticalExposureValue": 5.0,
            },
            "opponent": {
                "materialValue": 30.0,
                "tacticalRiskValue": 0.0,
                "forcedLossValue": 0.0,
                "criticalExposureValue": 0.0,
            },
            "materialBalance": 0.0,
        }
        exchange = {
            "ownMaterialLost": 8.0,
            "opponentMaterialLost": 0.0,
            "netMaterialExchange": -8.0,
            "assessment": "unfavorable",
        }

        causal = swings.causal_collapse_metrics(
            before, after_selected, exchange, 2
        )

        self.assertEqual(
            causal["category"], "within-preexisting-forced-loss"
        )
        self.assertFalse(causal["actionable"])
        self.assertEqual(
            causal["ownMaterialLostCoveredByPreexistingForcedLoss"], 8.0
        )
        self.assertEqual(
            causal["ownMaterialLostBeyondPreexistingForcedLoss"], 0.0
        )
        self.assertEqual(causal["selectedTurnNewForcedLossValue"], 0.0)
        self.assertEqual(
            causal["selectedTurnNewCriticalExposureValue"], 0.0
        )

    def test_immediate_undetected_reply_loss_is_actionable_but_later_loss_is_not(self):
        before = {
            "own": {
                "materialValue": 30.0,
                "tacticalRiskValue": 0.0,
                "forcedLossValue": 0.0,
                "criticalExposureValue": 0.0,
            },
            "opponent": {
                "materialValue": 30.0,
                "tacticalRiskValue": 0.0,
                "forcedLossValue": 0.0,
                "criticalExposureValue": 0.0,
            },
            "materialBalance": 0.0,
        }
        after_selected = copy.deepcopy(before)
        exchange = {
            "ownMaterialLost": 6.0,
            "opponentMaterialLost": 0.0,
            "netMaterialExchange": -6.0,
            "assessment": "unfavorable",
        }

        immediate = swings.causal_collapse_metrics(
            before, after_selected, exchange, 2
        )
        later = swings.causal_collapse_metrics(
            before, after_selected, exchange, 4
        )

        self.assertEqual(immediate["category"], "latent-reply-collapse")
        self.assertTrue(immediate["actionable"])
        self.assertEqual(immediate["attributionWindowTurns"], 2)
        self.assertEqual(later["category"], "unresolved-additional-loss")
        self.assertFalse(later["actionable"])

    def test_material_spent_during_clean_forced_hq_win_is_not_actionable(self):
        before = {
            "own": {
                "materialValue": 30.0,
                "tacticalRiskValue": 3.0,
                "forcedLossValue": 3.0,
                "criticalExposureValue": 3.0,
            },
            "opponent": {
                "materialValue": 30.0,
                "tacticalRiskValue": 0.0,
                "forcedLossValue": 0.0,
                "criticalExposureValue": 0.0,
            },
            "materialBalance": 0.0,
        }
        after_selected = copy.deepcopy(before)
        after_selected["own"].update(
            tacticalRiskValue=3.0,
            forcedLossValue=0.0,
            criticalExposureValue=3.0,
        )
        after_selected["opponent"].update(
            tacticalRiskValue=105.0,
            forcedLossValue=100.0,
            criticalExposureValue=105.0,
        )
        exchange = {
            "ownMaterialLost": 3.0,
            "opponentMaterialLost": 0.0,
            "netMaterialExchange": -3.0,
            "assessment": "unfavorable",
        }

        causal = swings.causal_collapse_metrics(
            before, after_selected, exchange, 2
        )

        self.assertEqual(causal["category"], "forced-hq-win-tradeoff")
        self.assertFalse(causal["actionable"])
        self.assertTrue(causal["selectedTurnForcedHqWin"])

    def test_replay_fails_closed_on_a_resulting_fen_mismatch(self):
        game = collapse_game()
        game["decisions"][0]["resultingFen"] = TURN_34_FEN
        with self.assertRaisesRegex(ValueError, "replay does not match"):
            swings.replay_game(game)

    def test_output_order_is_independent_of_jsonl_game_order(self):
        first = collapse_game("z-game")
        second = collapse_game("a-game")
        forward = swings.analyze_games([first, second], (2,), 2)
        reverse = swings.analyze_games([second, first], (2,), 2)

        self.assertEqual(
            json.dumps(forward, sort_keys=True),
            json.dumps(reverse, sort_keys=True),
        )
        self.assertEqual(
            [game["gameId"] for game in forward["gameResults"]],
            ["a-game", "z-game"],
        )
        self.assertEqual(
            [
                record["gameId"]
                for record in forward["windows"][0]["byPlayer"]["BLUE"][
                    "largestUnfavorable"
                ]
            ],
            ["a-game", "z-game"],
        )

    def test_uncertified_final_turn_is_unverified_even_at_depth_two(self):
        game = collapse_game()
        game["decisions"][0]["searchTelemetry"].update(
            {
                "finalSafetyCertified": False,
                "finalSafetyGuardUsed": True,
                "finalSafetyProbeUsed": True,
                "finalSafetyFloorRestored": True,
                "finalSafetyFloorSource": "verified_emergency_seed",
            }
        )

        report = swings.analyze_games([game], (2,), 1)
        collapse = report["windows"][0]["byPlayer"]["BLUE"][
            "largestUnfavorable"
        ][0]
        self.assertTrue(collapse["searchQuality"]["unverifiedFallback"])
        self.assertFalse(collapse["searchQuality"]["finalSafetyCertified"])
        self.assertTrue(collapse["searchQuality"]["finalSafetyGuardUsed"])
        self.assertTrue(
            collapse["searchQuality"]["finalSafetyFloorRestored"]
        )
        self.assertEqual(
            collapse["searchQuality"]["finalSafetyFloorSource"],
            "verified_emergency_seed",
        )
        self.assertEqual(
            collapse["windowSearchQuality"]["unverifiedFallbackDecisions"],
            2,
        )
        self.assertEqual(
            collapse["windowSearchQuality"][
                "uncertifiedFinalSafetyDecisions"
            ],
            1,
        )
        self.assertEqual(
            collapse["windowSearchQuality"][
                "missingFinalSafetyTelemetryDecisions"
            ],
            1,
        )
        self.assertEqual(
            collapse["windowSearchQuality"][
                "finalSafetyFloorRestoredDecisions"
            ],
            1,
        )

    def test_missing_final_safety_proof_is_unverified_even_at_depth_two(self):
        game = collapse_game()
        quality = swings.decision_search_quality(game["decisions"][0])

        self.assertEqual(quality["completedDepth"], 2)
        self.assertIsNone(quality["finalSafetyCertified"])
        self.assertTrue(quality["unverifiedFallback"])

    def test_rejects_incomplete_and_duplicate_games(self):
        max_turns = collapse_game("max-turns-game")
        max_turns["completed"] = False
        self.assertEqual(
            swings.analyze_games([max_turns], (2,), 1)["completedGames"], 1
        )

        incomplete = collapse_game()
        incomplete["completed"] = False
        incomplete["outcome"] = {"termination": "running"}
        with self.assertRaisesRegex(ValueError, "is not a completed result"):
            swings.analyze_games([incomplete], (2,), 1)
        with self.assertRaisesRegex(ValueError, "duplicate game IDs"):
            swings.analyze_games(
                [collapse_game(), copy.deepcopy(collapse_game())], (2,), 1
            )


if __name__ == "__main__":
    unittest.main()
