import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("ghq_ai", ROOT / "scripts" / "ghq_ai.py")
assert SPEC and SPEC.loader
ghq_ai = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ghq_ai
SPEC.loader.exec_module(ghq_ai)
engine = ghq_ai.engine
PARATROOPER_TRAP_FEN = (
    "q3i3/1iif2f1/ir↓3r↓2/8/t↓r↓6/2f2R↑1I/3T↑H↑1I1/"
    "FR↑p1P1Q1 IIIII iiii r"
)
PARATROOPER_EXTRACTION_FEN = (
    "q2r↓4/ir↓i2r↘2/8/3h↘4/fi5P/6f1/3H↑I2p/"
    "1FR↑2T↗FQ IIIIIF iiiii b"
)
POST_EXTRACTION_PRESSURE_FEN = (
    "q2r↓4/ir↓i2r↘2/4h↘3/8/fi3f1P/5Ip1/3H↑2F1/"
    "1FR↑2T↗1Q IIIIIF iiiii r"
)
TURN_28_FORCED_MATE_FEN = (
    "q2r↓4/ir↓i2r↘2/8/3h↘4/fi5P/3FH↑1f1/4IF1p/"
    "2R↑2T↗1Q IIIIIF iiiii b"
)
VERTICAL_INFANTRY_FEN = (
    "qr↓h↓r↓iiii/iiiffr↓t←p/5f2/7I/7I/7I/1R↑1R↑4/"
    "FIFPT↑H↑R↑Q - - r"
)
STAGGERED_INFANTRY_FEN = (
    "qr↓h↓r↓iiii/iiiffr↓t←p/5f2/8/6I1/5I1I/1R↑1R↑4/"
    "FIFPT↑H↑R↑Q - - r"
)


class EvaluationTests(unittest.TestCase):
    def test_starting_position_is_symmetric(self):
        breakdown = ghq_ai.evaluation_breakdown(engine.BaseBoard())
        self.assertAlmostEqual(breakdown["total_red"], 0.0)
        self.assertAlmostEqual(
            sum(breakdown["weighted_components"].values()),
            breakdown["total_red"],
            places=3,
        )

    def test_personalities_change_weights_not_features(self):
        board = engine.BaseBoard()
        balanced = ghq_ai.evaluation_breakdown(board, "balanced")
        fortress = ghq_ai.evaluation_breakdown(board, "fortress")
        self.assertEqual(balanced["components"], fortress["components"])
        self.assertGreater(fortress["weights"]["support"], balanced["weights"]["support"])

    def test_starting_position_is_not_treated_as_open_endgame(self):
        board = engine.BaseBoard()
        board.remove_from_reserve(engine.ARMORED_INFANTRY, engine.BLUE)
        breakdown = ghq_ai.evaluation_breakdown(board)
        self.assertEqual(breakdown["components"]["open_board_armored_infantry"], 0.0)

    def test_engagement_increases_deployed_paratrooper_risk(self):
        board = engine.BaseBoard(PARATROOPER_TRAP_FEN)
        before = ghq_ai.airborne_survival_penalty(board, engine.BLUE)
        board.push(engine.Move.from_uci("rid1"))
        after = ghq_ai.airborne_survival_penalty(board, engine.BLUE)
        self.assertGreater(after, before)

    def test_paratrooper_in_bombardment_has_catastrophic_penalty(self):
        safe = engine.BaseBoard("7q/8/8/8/p7/8/8/R↑6Q - - r")
        exposed = engine.BaseBoard("7q/8/8/8/8/8/p7/R↑6Q - - r")
        self.assertGreater(
            ghq_ai.airborne_survival_penalty(exposed, engine.BLUE),
            ghq_ai.airborne_survival_penalty(safe, engine.BLUE) + 6.0,
        )

    def test_early_development_rewards_material_on_the_board(self):
        undeveloped = engine.BaseBoard("7q/8/8/8/8/8/8/7Q IIIII - r")
        developed = engine.BaseBoard("7q/8/8/8/8/8/8/IIIII2Q - - r")
        self.assertGreater(
            ghq_ai.development_for(developed, engine.RED, 3),
            ghq_ai.development_for(undeveloped, engine.RED, 3),
        )

    def test_diagonal_infantry_cover_is_better_for_artillery(self):
        diagonal = engine.BaseBoard(
            "2p4q/8/8/8/8/2R↑5/1I6/7Q - - r"
        )
        cardinal = engine.BaseBoard(
            "2p4q/8/8/8/8/2R↑5/2I5/7Q - - r"
        )
        self.assertLess(
            ghq_ai.artillery_exposure_penalty(diagonal, engine.RED),
            ghq_ai.artillery_exposure_penalty(cardinal, engine.RED),
        )

    def test_staggered_infantry_front_beats_same_file_column(self):
        vertical = engine.BaseBoard(VERTICAL_INFANTRY_FEN)
        staggered = engine.BaseBoard(STAGGERED_INFANTRY_FEN)
        self.assertGreater(
            ghq_ai.infantry_shape_score(staggered, engine.RED),
            ghq_ai.infantry_shape_score(vertical, engine.RED),
        )
        self.assertLess(
            ghq_ai.infantry_isolation_penalty(staggered, engine.RED),
            ghq_ai.infantry_isolation_penalty(vertical, engine.RED),
        )
        self.assertGreater(
            ghq_ai.evaluation_breakdown(staggered)["total_red"],
            ghq_ai.evaluation_breakdown(vertical)["total_red"],
        )

    def test_dispersion_metric_penalizes_disconnected_material(self):
        connected = engine.BaseBoard("7q/8/8/8/8/8/8/III4Q - - r")
        dispersed = engine.BaseBoard("7q/8/8/8/8/8/8/I3I2Q - - r")
        self.assertGreater(
            ghq_ai.dispersion_penalty(dispersed, engine.RED),
            ghq_ai.dispersion_penalty(connected, engine.RED),
        )
        metrics = ghq_ai.structure_metrics(dispersed, engine.RED)
        self.assertGreater(metrics["components"], 1.0)
        self.assertGreater(metrics["isolated_units"], 0.0)

    def test_optionality_penalizes_a_boxed_home_rank(self):
        boxed = engine.BaseBoard(
            "7q/8/8/8/8/8/2IIII2/2IIII1Q - - r"
        )
        staggered = engine.BaseBoard(
            "7q/8/8/8/8/8/1I1I1I2/I1I1I2Q - - r"
        )
        boxed_metrics = ghq_ai.optionality_metrics(boxed, engine.RED)
        staggered_metrics = ghq_ai.optionality_metrics(staggered, engine.RED)
        self.assertGreater(boxed_metrics["immobile_units"], 0.0)
        self.assertEqual(staggered_metrics["immobile_units"], 0.0)
        self.assertGreater(
            ghq_ai.congestion_penalty(boxed, engine.RED),
            ghq_ai.congestion_penalty(staggered, engine.RED),
        )
        self.assertGreater(
            ghq_ai.optionality_score(staggered, engine.RED),
            ghq_ai.optionality_score(boxed, engine.RED),
        )

    def test_frontier_schedule_and_extension_penalty_follow_opening_phase(self):
        self.assertEqual(
            [ghq_ai.early_frontier_rank(turn) for turn in (1, 2, 3, 4, 5, 6, 7)],
            [2, 2, 3, 3, 4, 4, 5],
        )
        rank_four = engine.BaseBoard("7q/8/8/8/I7/8/8/7Q - - r")
        self.assertGreater(
            ghq_ai.phase_extension_penalty(rank_four, engine.RED, 3), 0.0
        )
        self.assertEqual(
            ghq_ai.phase_extension_penalty(rank_four, engine.RED, 5), 0.0
        )


class SearchTests(unittest.TestCase):
    def test_data_backed_opening_book_plays_both_sides_first_two_turns(self):
        board = engine.BaseBoard()
        for turn_number in range(1, 5):
            result = ghq_ai.search(
                board,
                "balanced",
                time_ms=500,
                max_depth=1,
                beam_width=4,
                turn_number=turn_number,
                opening_seed=17,
            )
            self.assertEqual(result["recommendation_label"], "opening book")
            self.assertTrue(result["search"]["opening_book_used"])
            self.assertEqual(len(result["best_turn"]["actions"]), 3)
            for uci in result["best_turn"]["all_moves"]:
                move = next(
                    candidate
                    for candidate in board.generate_legal_moves()
                    if candidate.uci() == uci
                )
                board.push(move)

    def test_recent_opening_book_produces_seeded_variety(self):
        openings = {
            tuple(
                ghq_ai.search(
                    engine.BaseBoard(),
                    "balanced",
                    time_ms=500,
                    max_depth=1,
                    beam_width=4,
                    turn_number=1,
                    opening_seed=seed,
                )["best_turn"]["actions"]
            )
            for seed in range(30)
        }
        self.assertGreaterEqual(len(openings), 4)
        self.assertTrue(openings.issubset({moves for moves, _ in ghq_ai.OPENING_FIRST_TURNS}))

    def test_infantry_screen_keeps_armored_infantry_in_reserve(self):
        board = engine.BaseBoard()
        for uci in ("ric1", "rid1", "rie1"):
            board.push(next(move for move in board.generate_legal_moves() if move.uci() == uci))
        self.assertEqual(
            ghq_ai.normalized_opening_signature(board, engine.RED),
            ghq_ai.OPENING_SIGNATURE_KEYS["D"],
        )
        self.assertEqual(board.get_reserve_count(engine.ARMORED_INFANTRY, engine.RED), 3)
        for uci in ("rhe8", "rtd8", "rpg8"):
            board.push(next(move for move in board.generate_legal_moves() if move.uci() == uci))
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=2_000, beam_width=4, turn_number=3
        )
        continuation = ghq_ai.opening_book_turn(
            board, 3, searcher, opening_seed=1
        )
        self.assertIsNotNone(continuation)
        self.assertEqual(
            [move.uci() for move in continuation.moves],
            ["d1d2", "e1e2", "rre1"],
        )

    def test_opening_book_is_bypassed_when_position_does_not_match(self):
        board = engine.BaseBoard()
        searcher = ghq_ai.Searcher("balanced", time_ms=1000, beam_width=4)
        self.assertIsNone(ghq_ai.opening_book_turn(board, 3, searcher))

    def test_missionless_paratrooper_sortie_gets_large_penalty(self):
        before = engine.BaseBoard("7q/8/8/8/8/8/8/1P5Q - - r")
        after = before.copy()
        move = next(
            candidate
            for candidate in after.generate_legal_moves()
            if candidate.uci() == "b1b5"
        )
        after.push(move)
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=6)
        purpose = searcher.turn_purpose_breakdown(
            before, after, [move], engine.RED
        )
        self.assertGreaterEqual(
            purpose["paratrooper_mission_penalty"],
            ghq_ai.MISSIONLESS_PARATROOPER_PENALTY,
        )
        self.assertNotIn(
            "b1b5",
            [
                candidate.uci()
                for _, candidate in searcher.diverse_moves(before)
            ],
        )

    def test_single_capture_target_is_enough_to_allow_a_paradrop(self):
        board = engine.BaseBoard("7q/8/8/2h↓5/8/8/8/1P5Q - - r")
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=6)
        moves = [candidate.uci() for _, candidate in searcher.diverse_moves(board)]
        self.assertIn("b1b4", moves)

    def test_opponent_home_rank_para_is_trapped_when_infantry_can_deploy(self):
        no_reserve = engine.BaseBoard("q6P/8/8/8/8/8/8/7Q - - r")
        infantry_ready = engine.BaseBoard("q6P/8/8/8/8/8/8/7Q - i r")
        self.assertGreater(
            ghq_ai.airborne_survival_penalty(infantry_ready, engine.RED),
            ghq_ai.airborne_survival_penalty(no_reserve, engine.RED) + 8.0,
        )

    def test_multiple_valuable_para_threats_count_as_a_mission(self):
        before = engine.BaseBoard("7q/8/8/r↓1h↓5/8/8/8/1P5Q - - r")
        after = before.copy()
        move = next(
            candidate
            for candidate in after.generate_legal_moves()
            if candidate.uci() == "b1b4"
        )
        after.push(move)
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=6)
        self.assertEqual(
            searcher.paratrooper_mission_penalty(
                before, after, [move], engine.RED
            ),
            0.0,
        )

    def test_square_swapping_turn_has_net_purpose_penalty(self):
        before = engine.BaseBoard("7q/8/8/8/8/8/1P6/2T↑H↑3Q - - r")
        after = before.copy()
        moves = []
        for uci in ("d1d2↑", "c1b1→", "b2c1"):
            move = next(
                candidate
                for candidate in after.generate_legal_moves()
                if candidate.uci() == uci
            )
            moves.append(move)
            after.push(move)
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=6, turn_number=10
        )
        purpose = searcher.turn_purpose_breakdown(
            before, after, moves, engine.RED
        )
        self.assertEqual(purpose["backfills"], 1.0)
        self.assertGreater(purpose["net_purpose_penalty"], 1.5)

    def test_turn_five_filters_idle_paradrop_and_focuses_development(self):
        board = engine.BaseBoard()
        for uci in (
            "rhd1", "rte1", "rpb1",
            "rhe8", "rtd8", "rpg8",
            "e1e3↑", "d1d2↑", "rfc1",
            "d8d6↓", "e8e7↓", "rff8",
        ):
            move = next(
                candidate
                for candidate in board.generate_legal_moves()
                if candidate.uci() == uci
            )
            board.push(move)

        searcher = ghq_ai.Searcher(
            "battery_commander", time_ms=3000, beam_width=8, turn_number=5
        )
        self.assertNotIn(
            "b1h8",
            [candidate.uci() for _, candidate in searcher.diverse_moves(board)],
        )
        result = ghq_ai.search(
            board,
            "battery_commander",
            time_ms=3000,
            max_depth=1,
            beam_width=8,
            turn_number=5,
        )
        self.assertGreaterEqual(
            result["best_turn"]["purpose"]["development_actions"], 2.0
        )
        self.assertTrue(
            all(
                "no_new_effect" not in item["roles"]
                for item in result["best_turn"]["action_purposes"]
            )
        )
        self.assertNotIn("b1h8", result["best_turn"]["actions"])

    def test_quiet_move_cannot_cross_the_phase_frontier(self):
        board = engine.BaseBoard("7q/8/8/8/8/I7/8/7Q - - r")
        early = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=6, turn_number=3
        )
        later = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=6, turn_number=5
        )
        self.assertNotIn(
            "a3a4", [move.uci() for _, move in early.diverse_moves(board)]
        )
        self.assertIn(
            "a3a4", [move.uci() for _, move in later.diverse_moves(board)]
        )

    def test_three_purposeless_infantry_pushes_are_not_a_valid_early_plan(self):
        before = engine.BaseBoard("7q/8/8/8/8/8/I1I1I3/7Q - - r")
        after = before.copy()
        moves = []
        for uci in ("a2a3", "c2c3", "e2e3"):
            move = next(
                candidate
                for candidate in after.generate_legal_moves()
                if candidate.uci() == uci
            )
            moves.append(move)
            after.push(move)
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=6, turn_number=5
        )
        purposes = searcher.action_purpose_labels(before, moves, engine.RED)
        self.assertEqual(
            searcher.forward_infantry_actions(before, moves, engine.RED), 3
        )
        self.assertFalse(
            searcher.early_structure_allowed(
                before, after, moves, engine.RED, purposes
            )
        )

    def test_blocked_non_threatening_artillery_rotation_is_discarded(self):
        board = engine.BaseBoard(VERTICAL_INFANTRY_FEN[:-1] + "b")
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=8)
        self.assertFalse(
            searcher.artillery_move_allowed(
                board, engine.Move.from_uci("g7g7←")
            )
        )

    def test_greedy_fallback_preserves_idle_paratrooper_and_breaks_column(self):
        board = engine.BaseBoard(VERTICAL_INFANTRY_FEN)
        before_shape = ghq_ai.infantry_shape_score(board, engine.RED)
        result = ghq_ai.greedy_complete_turn(board, "balanced", turn_number=12)
        ucis = [move.uci() for move in result.pv]
        self.assertNotEqual(ucis, ["h5h6", "h4h5", "h3h4"])
        self.assertFalse(any(uci.startswith("d1") for uci in ucis))
        after = board.copy()
        for move in result.pv:
            after.push(move)
        self.assertGreater(
            ghq_ai.infantry_shape_score(after, engine.RED), before_shape
        )

    def test_greedy_fallback_does_not_stage_para_just_to_spend_an_action(self):
        result = ghq_ai.greedy_complete_turn(
            engine.BaseBoard(), "balanced", turn_number=2
        )
        self.assertFalse(
            any(
                move.name == "Reinforce"
                and move.unit_type == engine.AIRBORNE_INFANTRY
                for move in result.pv
            )
        )

    def test_greedy_fallback_does_not_double_skip_when_quiet_moves_exist(self):
        # Batch vercel-6101331c-mrkqp2q4 reached this position with 28 legal
        # non-skip actions. The old per-action purpose veto rejected all 28,
        # selected skip, and the opponent repeated it for a false draw.
        board = engine.BaseBoard(
            "1i6/2i5/1i6/q4i2/5Ii1/4I1I1/7I/5Q2 - - b"
        )
        self.assertGreater(
            sum(move.name != "Skip" for move in board.generate_legal_moves()),
            0,
        )
        result = ghq_ai.greedy_complete_turn(
            board, "fortress", turn_number=70
        )
        self.assertGreaterEqual(len(result.pv), 2)
        self.assertNotEqual(result.pv[0].name, "Skip")
        after = board.copy()
        for move in result.pv:
            after.push(move)
        self.assertNotEqual(after.turn, board.turn)

    def test_diagonal_infantry_cover_removes_clean_para_capture(self):
        exposed = engine.BaseBoard(
            "2p4q/8/8/8/8/2R↑5/8/7Q - - b"
        )
        covered = engine.BaseBoard(
            "2p4q/8/8/8/8/2R↑5/1I1I4/7Q - - b"
        )
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=4)
        self.assertEqual(searcher.tactical_risk(exposed, engine.RED)[0], 3.0)
        self.assertEqual(searcher.tactical_risk(covered, engine.RED)[0], 0.0)

    def test_leaving_bombarded_heavy_artillery_is_tactically_unsafe(self):
        before = engine.BaseBoard(PARATROOPER_EXTRACTION_FEN)
        after = before.copy()
        for uci in ("g3f4", "h2g3", "skip"):
            after.push(engine.Move.from_uci(uci))
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=6)
        safety = searcher.assess_turn_safety(before, after, engine.BLUE)
        self.assertFalse(safety.tactically_safe)
        self.assertGreaterEqual(safety.forced_loss_value, 6.0)

    def test_value_model_is_used_in_leaf_score(self):
        searcher = ghq_ai.Searcher(
            "balanced",
            time_ms=1000,
            beam_width=4,
            value_function=lambda _fen, _turn: 0.8,
        )
        board = engine.BaseBoard()
        self.assertGreater(searcher.static_score(board), searcher.heuristic_score(board))
        self.assertEqual(searcher.value_model_evaluations, 1)

    def test_non_forcing_rotation_turns_have_a_quota(self):
        board = engine.BaseBoard("7q/8/8/8/i2R→4/8/8/7Q - - r")
        searcher = ghq_ai.Searcher("balanced", time_ms=3000, beam_width=4)
        turns = searcher.generate_turn_candidates(board)
        wasteful = sum(
            searcher.turn_action_classes(board, turn.moves)[1] for turn in turns
        )
        self.assertLessEqual(wasteful, 1)

    def test_regression_trapped_paratrooper_capture_stays_inside_beam(self):
        """The c1 paratrooper cannot be valued as a clean armored trade.

        Red can deploy on d1 to engage it, vacate b1 with the artillery,
        then move the a1 armored infantry to b1 and capture c1.
        """
        board = engine.BaseBoard(PARATROOPER_TRAP_FEN)
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=6)

        self.assertIn("rid1", [move.uci() for move in searcher.ordered_moves(board)])
        for uci in ("rid1", "b1c2↑", "a1b1xc1"):
            move = engine.Move.from_uci(uci)
            self.assertIn(move, list(board.generate_legal_moves()))
            if uci == "b1c2↑":
                candidates = searcher.ordered_moves(board)
                move = next(
                    candidate
                    for candidate in candidates
                    if candidate.from_square == engine.parse_square("b1")
                    and candidate.to_square == engine.parse_square("c2")
                )
            elif uci != "rid1":
                self.assertIn(uci, [candidate.uci() for candidate in searcher.ordered_moves(board)])
            board.push(move)

        self.assertFalse(board.airborne_infantry & board.occupied_co[engine.BLUE])

    def test_skip_is_not_preserved_before_final_optional_action(self):
        searcher = ghq_ai.Searcher("balanced", time_ms=1000, beam_width=1)
        moves = searcher.ordered_moves(engine.BaseBoard())
        self.assertNotIn("skip", [move.uci() for move in moves])

    def test_artillery_orientations_do_not_crowd_out_destinations(self):
        board = engine.BaseBoard(PARATROOPER_EXTRACTION_FEN)
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=20)
        moves = searcher.ordered_moves(board)
        artillery_moves = [move for move in moves if move.from_square == engine.parse_square("d5")]
        destinations = [move.to_square for move in artillery_moves]
        self.assertEqual(len(destinations), len(set(destinations)))
        self.assertTrue(all(not searcher.points_toward_home(engine.BLUE, move.orientation) for move in artillery_moves))

    def test_distant_artillery_cannot_spend_action_only_rotating(self):
        board = engine.BaseBoard()
        searcher = ghq_ai.Searcher("balanced", time_ms=1000, beam_width=8)
        self.assertFalse(searcher.artillery_move_allowed(board, engine.Move.from_uci("g1g1→")))
        self.assertFalse(searcher.artillery_move_allowed(board, engine.Move.from_uci("g1g1↓")))
        self.assertFalse(searcher.artillery_move_allowed(board, engine.Move.from_uci("g1f1↑")))
        self.assertTrue(searcher.artillery_move_allowed(board, engine.Move.from_uci("g1f1↖")))

    def test_quiet_unblock_and_paratrooper_extraction_stay_in_beam(self):
        board = engine.BaseBoard(PARATROOPER_EXTRACTION_FEN)
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=3000, beam_width=6, turn_number=27
        )

        # Save the bombarded heavy artillery first.
        first = engine.Move.from_uci("d5e6↘")
        self.assertIn(first, list(board.generate_legal_moves()))
        board.push(first)

        # Moving the armored infantry quietly unlocks h2-g3 and must therefore
        # survive a narrow beam.
        self.assertIn("g3f4", [move.uci() for move in searcher.ordered_moves(board)])
        board.push(engine.Move.from_uci("g3f4"))
        self.assertIn("h2g3", [move.uci() for move in searcher.ordered_moves(board)])

    def test_complete_turn_generation_preserves_paratrooper_extraction(self):
        board = engine.BaseBoard(PARATROOPER_EXTRACTION_FEN)
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=5000, beam_width=6, turn_number=27
        )
        turns = searcher.generate_turn_candidates(board)
        sequences = [[move.uci() for move in turn.moves] for turn in turns]
        self.assertTrue(
            any(
                "g3f4" in sequence
                and "h2g3" in sequence
                and sequence.index("g3f4") < sequence.index("h2g3")
                for sequence in sequences
            )
        )
        self.assertGreater(searcher.complete_turns_deduplicated, 0)

    def test_search_reports_best_found_when_it_prunes(self):
        result = ghq_ai.search(
            engine.BaseBoard(PARATROOPER_EXTRACTION_FEN),
            "balanced",
            time_ms=2500,
            max_depth=1,
            beam_width=6,
            turn_number=27,
        )
        self.assertEqual(result["recommendation_label"], "best found")
        self.assertFalse(result["search"]["exhaustive_within_requested_horizon"])

    def test_forcing_artillery_pressure_displaces_empty_infantry_move(self):
        board = engine.BaseBoard(POST_EXTRACTION_PRESSURE_FEN)
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=6, turn_number=29
        )
        moves = [move.uci() for move in searcher.ordered_moves(board)]
        self.assertIn("d2e1↗", moves)
        self.assertNotIn("b1b2", moves)

    def test_search_returns_a_complete_three_action_turn(self):
        board = engine.BaseBoard()
        result = ghq_ai.search(board, "balanced", time_ms=1200, max_depth=1, beam_width=8)
        moves = result["best_turn"]["all_moves"]
        actions = result["best_turn"]["actions"]
        self.assertLessEqual(len(actions), 3)
        self.assertGreater(len(actions), 0)

        replay = board.copy()
        for uci in moves:
            move = engine.Move.from_uci(uci)
            self.assertIn(move, list(replay.generate_legal_moves()))
            replay.push(move)
        self.assertNotEqual(replay.turn, board.turn)
        self.assertEqual(replay.board_fen(), result["best_turn"]["resulting_fen"])

    def test_search_takes_an_immediate_hq_capture(self):
        board = engine.BaseBoard("qI6/8/1I6/8/8/8/8/7Q - - r")
        result = ghq_ai.search(board, "balanced", time_ms=1000, max_depth=1, beam_width=8)
        self.assertEqual(result["best_turn"]["actions"], ["b6a7xa8"])
        self.assertGreaterEqual(result["score"]["red"], ghq_ai.MATE_SCORE)

    def test_search_finds_two_action_hq_combination(self):
        board = engine.BaseBoard(TURN_28_FORCED_MATE_FEN)
        result = ghq_ai.search(
            board,
            "balanced",
            time_ms=4000,
            max_depth=1,
            beam_width=6,
            turn_number=28,
        )
        self.assertEqual(result["best_turn"]["actions"], ["h2g1xf1", "g3h2xh1"])
        self.assertLessEqual(result["score"]["red"], -ghq_ai.MATE_SCORE)


if __name__ == "__main__":
    unittest.main()
