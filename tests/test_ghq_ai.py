import importlib.util
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch


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
THREE_ACTION_PARATROOPER_MATE_FEN = (
    "5i1i/6qi/6f1/5I2/3H↗2I1/1F6/F7/1P3I1Q III - b"
)
VERTICAL_INFANTRY_FEN = (
    "qr↓h↓r↓iiii/iiiffr↓t←p/5f2/7I/7I/7I/1R↑1R↑4/"
    "FIFPT↑H↑R↑Q - - r"
)
STAGGERED_INFANTRY_FEN = (
    "qr↓h↓r↓iiii/iiiffr↓t←p/5f2/8/6I1/5I1I/1R↑1R↑4/"
    "FIFPT↑H↑R↑Q - - r"
)
STALL_CONVEYOR_FEN = "q2i4/6f1/i3ii1i/1i6/8/2I5/1F3I2/I5IQ I i b"
TURN_FIVE_DEVELOPMENT_FEN = (
    "qr↓1r↓1i2/iiiii3/8/8/8/8/3IIIII/2I1R↑1R↑Q "
    "IIFFFPRTH iifffprth r"
)
TURN_23_COMPLEX_ROOT_FEN = (
    "qr↓4ii/ii4i1/3f1f1r↓/8/6fh↓/2F5/2R←2I1I/"
    "IIF1FR↖R↑Q III ii r"
)
TURN_12_SLOW_REPLY_FEN = (
    "q1ir↓fffp/iir↙h↓r→3/2it→4/8/5R↖2/4R←I2/3II1II/"
    "FR↑IH↑PIT↑Q IFF iiii b"
)
TURN_6_FEN = (
    "qr↓1r↓1i2/iiiii3/8/8/8/5T↑2/4H↑III/"
    "PFR↑FF1R↑Q IIIIIR iifffprth b"
)
SELF_PLAY_HQ_UNLOCK_FEN = "8/2q3i1/2I4i/6f1/2F5/8/1F4I1/I1I2I1Q I - b"
SELF_PLAY_HQ_ENGAGEMENT_FEN = "q7/i5i1/3I1i2/2I5/8/1I5I/I1I5/3I3Q - - b"
SELF_PLAY_FORCED_SKIP_MATE_FEN = "8/8/5q2/4F3/5I2/8/1I6/I2Q4 - - b"
SELF_PLAY_THREE_ACTION_HQ_FEN = "8/1i3q2/8/8/3ii3/4i1I1/2I4I/5Q2 - - r"
USER_REPORTED_PARA_AND_ARTILLERY_GAME = (
    ("rhd1", "rte1", "rfc1"),
    ("rhe8", "rtd8", "rpg8"),
    ("rib1", "ria1", "e1e3↑"),
    ("d8d6↓", "e8e7↓", "rff8"),
    ("d1d2↑", "b1b2", "a1a2"),
    ("rrc8", "rrd8", "rfe8"),
    ("b2b3", "a2a3", "rrb1"),
    ("c8d7↘", "g8a1xb1", "rfg8"),
    ("a3a2", "rpb1xa1", "rra1"),
    ("g8g7", "rih8", "b7b6"),
    ("a2a3", "a1a2↑", "d2d3↑"),
    ("d6f6↙", "b8b7↙", "rib8"),
    ("b3b4", "d3c3↑", "a3a4"),
    ("e7e7↙", "b7a6↓", "skip"),
    ("a4a3", "b4b3", "rid1"),
    ("f6f7↓", "e7f6↙", "a6b5↓"),
    ("a3a4", "b3c4", "c3d3↑"),
    ("a7a6", "b8a7", "b5b5↘"),
    ("c4b4xb5", "e3f4↑", "d3e4↑"),
    ("ric8", "rib8", "f6g6↙"),
    ("a2a3↑", "f4g5↑", "e4f4↑"),
    ("g7h7", "g6h6↙", "f7g8↓"),
    ("g5h5↑", "f4g4↑", "skip"),
    ("f8d6", "d7e6↘", "h6h6↓"),
    ("sbh7", "sbh6", "g4f4↖", "a4a5", "a3a4↑"),
    ("h8g7", "e6f6↓", "g8f7↘"),
    ("sbc7", "sbd6", "sba6", "f4g5↑", "h5h6↑", "b4b5"),
    ("f6e7↘", "f7f6→", "g7f7"),
    ("g5f5↑", "h6h4↑", "a5a6"),
    ("f7g6", "f6g5←", "d8d7↙"),
    ("b1d6xd7", "f5g4↑", "f2f3"),
    ("e8c6", "c8d7xd6", "g5f6↘"),
    ("sbg6", "a4b4↑", "g4f5←", "h4f4↑"),
    ("b6b7", "rih8", "f6g6↙"),
    ("c1c3", "f5e6←", "f4e5←"),
    ("e7f6←", "d7e7xe6", "c6e6xe5"),
    ("b5b6", "b4b5↑", "c3c5"),
    ("h8g7", "b7c7", "e6d6"),
    ("d1d2", "rfc1", "rfb1"),
    ("f6e5←", "g6h5↙", "d6d4"),
    ("d2d3", "f3e4xe5", "c5d5xd4"),
)
SELF_PLAY_PURPOSELESS_FILLER_FEN = (
    "1q2i1i1/3i1i1i/1i4fr↓/8/F7/1H↑F2I2/4I1II/1IR↖1R↖I1Q IF - b"
)
SELF_PLAY_LATE_HQ_REPLY_FEN = (
    "q6i/6f1/i7/8/i2i4/1ir↙1f3/I7/Q7 - i r"
)
SLOW_VERCEL_REPLY_FEN = (
    "i1if2i1/qi2fi2/i7/2h↓5/5r↙2/1I1I1r↓2/3R↑2Q1/8 - i b"
)
SMOKE_IMMEDIATE_HQ_LOSS_CASES = (
    (81, SELF_PLAY_LATE_HQ_REPLY_FEN),
    (69, "q3i1i1/2ii4/1i6/8/1I6/F5i1/1I5f/2II3Q - - r"),
    (76, "1qF4i/1F2i2r↓/6ff/I7/8/4I3/5If1/6Q1 - - b"),
    (115, "q4if1/7f/4i3/8/1i6/6f1/6i1/6Q1 - ii r"),
    (136, "8/8/8/8/3I1I1q/2I1I1I1/1I6/F6Q - - b"),
    (123, "q7/8/5i2/2I3i1/1I6/2I5/1I4if/6Q1 - - r"),
)
SMOKE_HQ_ESCAPE_CASES = (
    (
        92,
        "q7/5i2/2F3i1/3F4/8/5I2/2I3I1/4IIQ1 - - b",
        ["g6h7", "f7g8", "a8b8"],
    ),
    (
        50,
        "q3i1ii/1F2r←1f1/F7/4i1r↓1/8/3f2I1/5I2/I6Q III ii b",
        ["sbg3", "g5f4↓", "e7f8↓", "skip"],
    ),
    (
        41,
        "q2i1i2/3ii3/4f3/8/8/1F6/IF1I1f2/1P2f1Q1 II i r",
        ["b2c2", "b3a4", "skip"],
    ),
    (
        109,
        "2q1i2f/5ii1/8/1i6/I1i5/1Q6/5I2/4I3 I - r",
        ["e1d2", "f2e1", "a4a3"],
    ),
    (
        84,
        "1q6/6i1/F1i4r←/1F5f/I7/8/7i/6IQ - i b",
        ["h5g4", "h6h7↓", "h2h3"],
    ),
    (
        75,
        "q2i3f/4i1i1/1i1i4/8/1F3f2/2I1I1i1/I2I2Q1/1I2I3 - - r",
        ["b4b3", "a2a1", "b1a2"],
    ),
    (
        102,
        "5q2/8/3I1I2/3F1R←2/7I/1I6/2F5/3I2Q1 I - b",
        ["f8e7", "skip"],
    ),
    (
        77,
        "q7/i5ii/2i3if/1i6/7f/I6f/F1I4Q/I2I4 - ii r",
        ["a2b3", "a1a2", "a3a4"],
    ),
    (
        44,
        "1q2i1i1/7i/2F5/3F4/8/5f2/4I1I1/R↑I1F1IQ1 I - b",
        ["f3f5", "h7h8", "g8h7"],
    ),
)
SELF_PLAY_AVOIDABLE_IMMEDIATE_HQ_LOSSES = (
    (59, "qi1i4/i1i3i1/1i3i2/8/F5f1/1FF1f1r↓1/I2I1f2/1I4Q1 - - r"),
    (76, "3q4/4i1i1/3I4/4F2i/3R↗2II/2I5/1I1I4/I1I1P2Q - - b"),
    (80, "5q2/8/3FI1I1/5I2/2II4/1I6/8/7Q - - b"),
    (90, "7q/8/7F/6F1/4II1I/7I/1I6/I6Q - - b"),
    (85, "1q6/2i5/Fi1i4/2i1ii2/8/6f1/3I2fi/4I1Q1 - - r"),
    (117, "q7/1I6/8/8/2if1i2/4f3/8/4Q3 - - r"),
    (77, "8/q7/1i6/I1i5/6i1/5i2/5f2/4i2Q - - r"),
    (57, "8/q1i5/1i6/2i5/2F3i1/1F1I3f/I1I5/1I1I1fQ1 - - r"),
    (79, "qi3i2/2i1i3/3I4/4I3/1i6/2if4/8/3Q4 - - r"),
    (76, "q1i3r→1/F2iiii1/1I6/2F5/8/8/3I1I2/3P2IQ - - b"),
)
SELF_PLAY_FORCED_IMMEDIATE_HQ_LOSS = (
    76,
    "qI2f3/2F2i2/2I5/8/6I1/5I2/6I1/7Q - - b",
)
MISSIONLESS_PARA_SURVIVAL_OVERRIDE_FEN = (
    "p5i1/r↓q3i1f/1i1if3/2it↓r↓fi1/1Q3r↓2/IFR↑5/"
    "1H↑1R↑1I2/2P1I3 - - r"
)
AVOIDABLE_HQ_LOSS_REGRESSIONS = (
    (
        142,
        "balanced",
        "4qp2/8/4I1i1/2I2I2/1T↑1IH↑1R↑1/2F4Q/1I6/3P4 - - b",
    ),
    (
        95,
        "fortress",
        "4p3/qi6/i1i5/8/I7/6i1/1H↑I4i/5P1Q - - r",
    ),
    (
        71,
        "para_specialist",
        "p5i1/r↓q1i1i2/4f1i1/5f1f/1QF2r↓2/1H↑1R↑4/3R↑I3/8 - - r",
    ),
    (
        149,
        "balanced",
        "qp6/h↓i3r↓2/8/4i3/4r↓3/8/8/5PfQ - - r",
    ),
)
LIVE_MATE_DELAY_REPLY_CASES = (
    (
        99,
        "balanced",
        "2q5/8/8/8/i5i1/1i3iI1/2i1i3/7Q - - r",
        ("skip",),
    ),
    (
        87,
        "tactical_gambler",
        "q1i5/3i4/1I6/8/8/3iii1Q/8/8 - - r",
        ("h3h2", "skip"),
    ),
    (
        93,
        "tactical_gambler",
        "2i5/q5i1/2i2i2/Q2i4/3i4/2f5/2r↙5/2f5 - - r",
        ("skip",),
    ),
)


class EvaluationTests(unittest.TestCase):
    def test_chebyshev_lookup_matches_production_engine(self):
        for left in range(64):
            for right in range(64):
                self.assertEqual(
                    ghq_ai.chebyshev(left, right),
                    engine.square_distance(left, right),
                )

    def test_mirror_clears_empty_orientation_bits_and_reinforces_forward(self):
        board = engine.BaseBoard(TURN_6_FEN)
        mirrored = board.mirror()
        reinforce = next(
            move
            for move in mirrored.generate_legal_moves()
            if move.uci() == "rrb1"
        )
        mirrored.push(reinforce)
        self.assertEqual(mirrored.get_orientation(engine.B1), engine.ORIENT_N)

        # Replaying a normalized color-swapped turn must produce the exact
        # mirror of the original resulting state, including orientations.
        original = board.copy()
        original_moves = []
        for uci in ("rrg8", "rhe8", "rfh8"):
            move = next(
                candidate
                for candidate in original.generate_legal_moves()
                if candidate.uci() == uci
            )
            original_moves.append(move)
            original.push(move)

        replay = board.mirror()
        for move in original_moves:
            normalized = ghq_ai.normalized_move_uci(move, engine.BLUE)
            replay.push(
                next(
                    candidate
                    for candidate in replay.generate_legal_moves()
                    if candidate.uci() == normalized
                )
            )
        self.assertEqual(original.mirror().serialize(), replay.serialize())

    def test_set_orientation_replaces_all_old_direction_bits(self):
        board = engine.BaseBoard()
        board.set_orientation(engine.G1, engine.ORIENT_NW)
        board.set_orientation(engine.G1, engine.ORIENT_N)
        self.assertEqual(board.get_orientation(engine.G1), engine.ORIENT_N)

    def test_early_turn_candidate_beam_is_color_symmetric(self):
        original = engine.BaseBoard(TURN_6_FEN)
        mirrored = original.mirror()
        beams = []
        for board in (original, mirrored):
            searcher = ghq_ai.Searcher(
                "fortress", time_ms=60_000, beam_width=6, turn_number=6
            )
            searcher.root_key = board.serialize()
            candidates = searcher.generate_turn_candidates(board)
            beams.append(
                [
                    (
                        ghq_ai.normalized_turn_key(candidate.moves, board.turn),
                        round(
                            candidate.static_score
                            if board.turn == engine.RED
                            else -candidate.static_score,
                            6,
                        ),
                        candidate.tactically_safe,
                        round(candidate.safety_penalty, 6),
                    )
                    for candidate in candidates
                ]
            )
        self.assertEqual(beams[0], beams[1])

    def test_starting_position_is_symmetric(self):
        breakdown = ghq_ai.evaluation_breakdown(engine.BaseBoard())
        self.assertAlmostEqual(breakdown["total_red"], 0.0)
        self.assertAlmostEqual(
            sum(breakdown["weighted_components"].values()),
            breakdown["total_red"],
            places=3,
        )

    def test_every_evaluation_component_is_color_antisymmetric(self):
        positions = (
            engine.BaseBoard(PARATROOPER_EXTRACTION_FEN),
            engine.BaseBoard(
                "q2pi3/1ii1f1f1/ir↓3r↓2/2f5/t↓r↓6/5R↑1I/"
                "FIR↑T↑H↑1I1/2F1P1Q1 IIIII iiii r"
            ),
            engine.BaseBoard(
                "1i6/2i5/1i6/q4i2/5Ii1/4I1I1/7I/5Q2 - - b"
            ),
        )
        for board in positions:
            mirrored = board.mirror()
            for personality in ghq_ai.PERSONALITIES:
                original = ghq_ai.evaluation_breakdown(
                    board, personality, turn_number=27
                )
                inverse = ghq_ai.evaluation_breakdown(
                    mirrored, personality, turn_number=27
                )
                for component, value in original["components"].items():
                    self.assertAlmostEqual(
                        value,
                        -inverse["components"][component],
                        places=4,
                        msg=f"{personality} {component}",
                    )
                self.assertAlmostEqual(
                    original["total_red"], -inverse["total_red"], places=4
                )
            self.assertAlmostEqual(
                ghq_ai.quick_evaluation(board, 27),
                -ghq_ai.quick_evaluation(mirrored, 27),
                places=8,
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
    def test_deadline_seed_prioritizes_proven_hq_defenses(self):
        positions = (
            (
                "8/3T→q3/8/8/8/R↑F2FIi1/H↖R↑1I2Q1/1P2F3 II - b",
                72,
            ),
            (
                "5f2/1q1i2ir↓/8/1R↑6/3F1R↗2/4I3/3IFR↖II/2IP2T↑Q II ii b",
                36,
            ),
            (
                "q1F4i/1F4i1/3ifi2/6i1/2I5/I2I4/5I1Q/8 I i b",
                86,
            ),
        )
        for fen, turn_number in positions:
            with self.subTest(turn_number=turn_number):
                board = engine.BaseBoard(fen)
                seed = ghq_ai.purposeful_complete_turn_seed(
                    board,
                    "balanced",
                    turn_number=turn_number,
                )
                moves, resulting = ghq_ai.first_turn_from_pv(board, seed.pv)
                detector = ghq_ai.Searcher(
                    "balanced",
                    time_ms=5000,
                    beam_width=6,
                    turn_number=turn_number + 1,
                )

                self.assertTrue(moves)
                self.assertFalse(detector.has_same_turn_hq_capture(resulting))

    def test_quiet_setup_is_labeled_when_it_unlocks_an_hq_defense(self):
        board = engine.BaseBoard(
            "q1F4i/1F4i1/3ifi2/6i1/2I5/I2I4/5I1Q/8 I i b"
        )
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=5000, beam_width=6, turn_number=86
        )
        setup = next(
            move for move in board.generate_legal_moves() if move.uci() == "rib8"
        )

        self.assertTrue(searcher.unlocks_hq_defense(board, setup))
        self.assertIn(
            "hq_defense_unlock",
            searcher.action_purpose_labels(
                board, [setup], board.turn, retrospective=False
            )[0]["roles"],
        )

    def test_short_search_keeps_safe_seed_over_an_immediate_hq_loss(self):
        board = engine.BaseBoard(
            "q1F4i/1F4i1/3ifi2/6i1/2I5/I2I4/5I1Q/8 I i b"
        )
        result = ghq_ai.search(
            board,
            "balanced",
            time_ms=4000,
            max_depth=2,
            beam_width=6,
            turn_number=86,
            max_actions=3,
        )
        resulting = engine.BaseBoard(result["best_turn"]["resulting_fen"])
        detector = ghq_ai.Searcher(
            "balanced", time_ms=5000, beam_width=6, turn_number=87
        )

        self.assertFalse(detector.has_same_turn_hq_capture(resulting))
        self.assertGreater(result["score"]["current_player"], -ghq_ai.MATE_SCORE)

    def test_seed_telemetry_cannot_leak_a_search_timeout(self):
        original = ghq_ai.Searcher.turn_purpose_breakdown

        def timeout_tactical_telemetry(
            searcher,
            before,
            after,
            moves,
            mover,
            retrospective=True,
            include_tactical_roles=True,
        ):
            if include_tactical_roles:
                raise ghq_ai.SearchTimeout
            return original(
                searcher,
                before,
                after,
                moves,
                mover,
                retrospective=retrospective,
                include_tactical_roles=False,
            )

        with patch.object(
            ghq_ai.Searcher,
            "turn_purpose_breakdown",
            timeout_tactical_telemetry,
        ):
            result = ghq_ai.search(
                engine.BaseBoard(),
                "balanced",
                time_ms=1000,
                max_depth=2,
                beam_width=4,
                turn_number=8,
            )

        self.assertTrue(result["best_turn"]["actions"])
        self.assertTrue(result["search"]["timed_out"])

    def test_emergency_seed_tactical_timeout_still_finishes_a_legal_turn(self):
        board = engine.BaseBoard()
        with patch.object(
            ghq_ai.Searcher,
            "tactical_risk",
            side_effect=ghq_ai.SearchTimeout,
        ):
            seed = ghq_ai.purposeful_complete_turn_seed(
                board, "balanced", turn_number=8, max_actions=3
            )
        moves, resulting = ghq_ai.first_turn_from_pv(board, seed.pv)

        self.assertEqual(
            sum(move.name not in ("AutoCapture", "Skip") for move in moves),
            3,
        )
        self.assertNotEqual(resulting.turn, board.turn)

    def test_emergency_seed_safety_uses_an_isolated_bounded_searcher(self):
        board = engine.BaseBoard()
        seed = ghq_ai.purposeful_complete_turn_seed(
            board,
            "balanced",
            turn_number=8,
            max_actions=3,
            time_ms=100,
        )
        _, seed_board = ghq_ai.first_turn_from_pv(board, seed.pv)
        observed: list[tuple[int, object]] = []

        def record_probe(searcher, before, after, mover):
            observed.append((searcher.time_ms, searcher.root_key))
            return ghq_ai.TacticalSafety(0, 0, 0, 0, 0, True)

        with patch.object(
            ghq_ai.Searcher,
            "assess_turn_safety",
            autospec=True,
            side_effect=record_probe,
        ):
            safety = ghq_ai.bounded_seed_safety(
                board,
                seed_board,
                "balanced",
                turn_number=8,
                beam_width=6,
                max_actions=3,
                time_ms=80,
            )

        self.assertTrue(safety and safety.tactically_safe)
        self.assertEqual(observed, [(80, None)])

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

    def test_single_target_does_not_justify_a_paradrop(self):
        board = engine.BaseBoard("7q/8/8/2h↓5/8/8/8/1P5Q - - r")
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=6)
        moves = [candidate.uci() for _, candidate in searcher.diverse_moves(board)]
        self.assertNotIn("b1b4", moves)

    def test_even_para_trade_for_armored_artillery_is_rejected(self):
        board = engine.BaseBoard("7q/8/8/2t↓5/8/8/8/1P5Q - - r")
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=12)
        moves = [candidate.uci() for _, candidate in searcher.diverse_moves(board)]
        self.assertNotIn("b1b5xc5", moves)

    def test_single_heavy_artillery_capture_does_not_justify_a_paradrop(self):
        board = engine.BaseBoard("7q/8/8/2h↓5/8/8/8/1P5Q - - r")
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=12)
        moves = [candidate.uci() for _, candidate in searcher.diverse_moves(board)]
        self.assertNotIn("b1b5xc5", moves)

    def test_reported_para_trade_and_artillery_exposure_are_rejected(self):
        board = engine.BaseBoard()
        positions = {}
        for turn_number, turn in enumerate(
            USER_REPORTED_PARA_AND_ARTILLERY_GAME, start=1
        ):
            positions[turn_number] = board.copy()
            for uci in turn:
                board.push(engine.Move.from_uci(uci))

        turn_eight = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=12, turn_number=4
        )
        paradrop = engine.Move.from_uci("g8a1xb1")
        self.assertFalse(turn_eight.paradrop_allowed(positions[8], paradrop))

        turn_sixteen = ghq_ai.Searcher(
            "balanced", time_ms=10000, beam_width=8, turn_number=8
        )
        exposed = positions[16].copy()
        for uci in USER_REPORTED_PARA_AND_ARTILLERY_GAME[15]:
            exposed.push(engine.Move.from_uci(uci))
        safety = turn_sixteen.assess_turn_safety(
            positions[16], exposed, positions[16].turn
        )
        self.assertFalse(safety.tactically_safe)
        self.assertEqual(safety.new_risk_value, 3.0)
        candidates = turn_sixteen.generate_turn_candidates(positions[16])
        candidate_lines = [[move.uci() for move in item.moves] for item in candidates]
        self.assertNotIn(
            list(USER_REPORTED_PARA_AND_ARTILLERY_GAME[15]), candidate_lines
        )
        self.assertFalse(any("a6b5↓" in line for line in candidate_lines))

        turn_eighteen = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=12, turn_number=9
        )
        self.assertFalse(
            turn_eighteen.artillery_move_allowed(
                positions[18], engine.Move.from_uci("b5b5↘")
            )
        )

        turn_forty = ghq_ai.Searcher(
            "balanced", time_ms=20000, beam_width=8, turn_number=20
        )
        exposed_pair = positions[40].copy()
        for uci in USER_REPORTED_PARA_AND_ARTILLERY_GAME[39]:
            exposed_pair.push(engine.Move.from_uci(uci))
        setup = next(
            move
            for move in exposed_pair.generate_legal_moves()
            if move.uci() == "d2d3"
        )
        self.assertTrue(
            turn_forty.unlocks_capture_this_turn(exposed_pair, setup)
        )
        after_setup = exposed_pair.copy()
        after_setup.push(setup)
        chain_capture = next(
            move
            for move in after_setup.generate_legal_moves()
            if move.uci() == "f3e4xe5"
        )
        self.assertEqual(
            turn_forty.followup_capture_value(after_setup, chain_capture), 3.0
        )
        replies = turn_forty.generate_turn_candidates(exposed_pair)
        self.assertIn(
            ["d2d3", "f3e4xe5", "c5d5xd4"],
            [[move.uci() for move in item.moves] for item in replies],
        )

        blue_candidates = turn_forty.generate_turn_candidates(positions[40])
        blue_lines = [[move.uci() for move in item.moves] for item in blue_candidates]
        self.assertNotIn(
            list(USER_REPORTED_PARA_AND_ARTILLERY_GAME[39]), blue_lines
        )
        safe_artillery_only = positions[40].copy()
        for uci in ("f6e5←", "skip"):
            safe_artillery_only.push(engine.Move.from_uci(uci))
        self.assertFalse(
            turn_forty.unlocks_capture_this_turn(
                safe_artillery_only, engine.Move.from_uci("d2d3")
            )
        )

    def test_cycle_seed_cannot_prune_progressive_verification_roots(self):
        board = engine.BaseBoard(
            "2i1t↓p2/qih↓ir↓3/2i5/8/1I5Q/I5I1/"
            "2IR↑H↑T↑R↑R↑/4IPI1 - - r"
        )
        searcher = ghq_ai.Searcher(
            "tactical_gambler",
            time_ms=20_000,
            beam_width=6,
            turn_number=81,
            stagnation_turns=18,
        )
        searcher.root_key = ghq_ai.board_key(board)
        cycle_moves = [
            engine.Move.from_uci(uci)
            for uci in ("c2c3", "b4b3", "a3a2")
        ]
        cycled = board.copy()
        for move in cycle_moves:
            cycled.push(move)
        purpose = searcher.turn_purpose_breakdown(
            board, cycled, cycle_moves, board.turn
        )
        action_purposes = searcher.action_purpose_labels(
            board, cycle_moves, board.turn
        )
        searcher.root_fallback = ghq_ai.TurnCandidate(
            cycle_moves,
            cycled,
            0.0,
            searcher.heuristic_score(cycled),
            purpose_penalty=purpose["net_purpose_penalty"],
            paratrooper_mission_penalty=purpose[
                "paratrooper_mission_penalty"
            ],
            action_purposes=action_purposes,
            progress_score=purpose["stagnation_progress"],
            conveyor_actions=purpose["backfills"] + purpose["reversals"],
        )
        searcher.verification_mode = True

        candidates = searcher.generate_turn_candidates(board)
        lines = [[move.uci() for move in item.moves] for item in candidates]

        self.assertNotIn(["c2c3", "b4b3", "a3a2"], lines)
        self.assertTrue(
            any(
                item.progress_score >= 1.9
                and item.moves[0].uci() in ("b4a5", "b4b5", "a3a4")
                for item in candidates
            )
        )

    def test_stagnation_deadline_seed_does_not_repeat_the_infantry_cycle(self):
        board = engine.BaseBoard(
            "2i1t↓p2/qih↓ir↓3/2i5/8/1I5Q/I5I1/"
            "2IR↑H↑T↑R↑R↑/4IPI1 - - r"
        )
        seed = ghq_ai.purposeful_complete_turn_seed(
            board,
            "tactical_gambler",
            turn_number=81,
            max_actions=3,
            time_ms=2_000,
            stagnation_turns=18,
        )
        moves = [move.uci() for move in seed.pv]
        self.assertNotEqual(moves, ["c2c3", "b4b3", "a3a2"])
        self.assertEqual(moves, ["c2c3", "a3a4", "b4b5"])

    def test_safety_prefers_reducing_an_unavoidable_baseline_loss(self):
        board = engine.BaseBoard(
            "qi2iii1/ir↓if3f/1i2r↓3/2If4/2R↑H↑4/"
            "I3R↑1I1/R↑1PI3I/4I2Q FFF i b"
        )
        searcher = ghq_ai.Searcher(
            "para_specialist",
            time_ms=20_000,
            beam_width=16,
            turn_number=26,
        )

        historical = board.copy()
        for uci in ("h7f5", "b6b5xc5", "skip"):
            historical.push(engine.Move.from_uci(uci))
        reduced = board.copy()
        for uci in ("d7d8", "b6b5xc5", "h7f5"):
            reduced.push(engine.Move.from_uci(uci))

        historical_safety = searcher.assess_turn_safety(
            board, historical, board.turn
        )
        reduced_safety = searcher.assess_turn_safety(
            board, reduced, board.turn
        )
        self.assertEqual(historical_safety.forced_loss_value, 6.0)
        self.assertFalse(historical_safety.tactically_safe)
        self.assertEqual(reduced_safety.forced_loss_value, 3.0)
        self.assertTrue(reduced_safety.tactically_safe)

    def test_armored_infantry_may_not_walk_into_immediate_forced_capture(self):
        board = engine.BaseBoard(
            "qi2iii1/ir↓i5/4r↓3/1i3f2/1IR↑H↑4/"
            "1R↑2R↑1I1/3I3I/2P1I2Q FFF i b"
        )
        searcher = ghq_ai.Searcher(
            "para_specialist",
            time_ms=20_000,
            beam_width=16,
            turn_number=28,
        )

        suicide = board.copy()
        for uci in ("f5h3", "b7b6↓", "e6f7↓"):
            suicide.push(engine.Move.from_uci(uci))
        escape = board.copy()
        for uci in ("b5a6", "b7b6↓", "skip"):
            escape.push(engine.Move.from_uci(uci))

        suicide_safety = searcher.assess_turn_safety(
            board, suicide, board.turn
        )
        escape_safety = searcher.assess_turn_safety(
            board, escape, board.turn
        )
        self.assertEqual(suicide_safety.forced_loss_value, 4.0)
        self.assertFalse(suicide_safety.tactically_safe)
        self.assertEqual(escape_safety.forced_loss_value, 0.0)
        self.assertTrue(escape_safety.tactically_safe)

    def test_exact_hq_probe_retains_sparse_artillery_orientation_mate(self):
        root = engine.BaseBoard(
            "4qT←2/5H↑2/4r→3/1I3R↑2/8/2R→5/1R↑2Q3/8 - - b"
        )
        exposed = root.copy()
        for uci in ("e6d6↓", "e8e7", "skip"):
            exposed.push(engine.Move.from_uci(uci))
        searcher = ghq_ai.Searcher(
            "mobile_raider",
            time_ms=60_000,
            beam_width=6,
            turn_number=158,
        )
        remaining_nodes = [100_000]

        self.assertTrue(
            searcher.exact_same_turn_hq_capture(
                exposed, exposed.turn, remaining_nodes
            )
        )
        self.assertGreater(remaining_nodes[0], 0)
        survival = searcher.find_hq_survival_turn(
            root, max_reply_nodes=1_000_000
        )
        self.assertIsNotNone(survival)
        assert survival is not None
        _, survived = survival
        verification_nodes = [100_000]
        self.assertFalse(
            searcher.exact_same_turn_hq_capture(
                survived, survived.turn, verification_nodes
            )
        )

    def test_sparse_orientation_proof_reclassifies_old_turn_107_escape(self):
        root = engine.BaseBoard(
            "q5pr↓/i1i5/1i6/8/3f4/t↓1r↓1i3/"
            "QT↑2h←3/8 - - r"
        )
        alleged_escape = root.copy()
        for uci in ("a2b1", "b2b4↘", "skip"):
            alleged_escape.push(engine.Move.from_uci(uci))
        searcher = ghq_ai.Searcher(
            "tactical_gambler",
            time_ms=60_000,
            beam_width=6,
            turn_number=107,
        )
        remaining_nodes = [100_000]

        self.assertTrue(
            searcher.exact_same_turn_hq_capture(
                alleged_escape,
                alleged_escape.turn,
                remaining_nodes,
            )
        )
        winning = alleged_escape.copy()
        for uci in ("g8b3xb4", "a3a1→", "c3b2←"):
            legal = {move.uci(): move for move in winning.generate_legal_moves()}
            self.assertIn(uci, legal)
            winning.push(legal[uci])
        self.assertEqual(winning.outcome().winner, engine.BLUE)

    def test_opponent_home_rank_para_is_trapped_when_infantry_can_deploy(self):
        no_reserve = engine.BaseBoard("q6P/8/8/8/8/8/8/7Q - - r")
        infantry_ready = engine.BaseBoard("q6P/8/8/8/8/8/8/7Q - i r")
        self.assertGreater(
            ghq_ai.airborne_survival_penalty(infantry_ready, engine.RED),
            ghq_ai.airborne_survival_penalty(no_reserve, engine.RED) + 8.0,
        )

    def test_future_para_threats_do_not_count_as_a_same_turn_mission(self):
        before = engine.BaseBoard("7q/8/8/r↓1h↓5/8/8/8/1P5Q - - r")
        after = before.copy()
        move = next(
            candidate
            for candidate in after.generate_legal_moves()
            if candidate.uci() == "b1b4"
        )
        after.push(move)
        searcher = ghq_ai.Searcher("balanced", time_ms=2000, beam_width=6)
        self.assertGreaterEqual(
            searcher.paratrooper_mission_penalty(
                before, after, [move], engine.RED
            ),
            ghq_ai.MISSIONLESS_PARATROOPER_PENALTY,
        )

    def test_reported_para_specialist_opening_trade_is_rejected(self):
        board = engine.BaseBoard()
        for turn in (
            ("rib1", "ric1", "ria1"),
            ("rhe8", "rtd8", "rpg8"),
            ("c1c2", "b1b2", "a1a2"),
            ("d8d6↓", "e8e7↓", "rff8"),
            ("rra1", "rrb1", "rtc1"),
        ):
            for uci in turn:
                board.push(engine.Move.from_uci(uci))

        searcher = ghq_ai.Searcher(
            "para_specialist", time_ms=5000, beam_width=12, turn_number=3
        )
        move = engine.Move.from_uci("g8d1xc1")
        self.assertFalse(searcher.paradrop_allowed(board, move))
        candidates = searcher.generate_turn_candidates(board)
        self.assertFalse(
            any(
                "g8d1xc1" in [candidate_move.uci() for candidate_move in item.moves]
                for item in candidates
            )
        )

    def test_deadline_seed_cannot_bypass_single_capture_para_rule(self):
        cases = (
            (
                "qr↓f1ffpi/1ir←2r↓2/i1i3h↓1/5t↓2/8/3FT↑1I1/"
                "FR↑IH↑IIR→I/1P1I1R↑FQ II iiii r",
                "tactical_gambler",
                15,
                "b1h6xg6",
            ),
            (
                "q1f1f1p1/1ir↓r↓3i/i1i4f/8/4IP2/1FH↑FT↑R↑2/"
                "1R↑I1II1I/1I1I1R↑FQ I iiii b",
                "battery_commander",
                20,
                "g8g3xf3",
            ),
        )
        for fen, personality, turn_number, rejected_move in cases:
            with self.subTest(turn_number=turn_number):
                board = engine.BaseBoard(fen)
                seed = ghq_ai.purposeful_complete_turn_seed(
                    board,
                    personality,
                    turn_number=turn_number,
                    max_actions=3,
                    time_ms=500,
                )
                moves, _ = ghq_ai.first_turn_from_pv(board, seed.pv)
                self.assertNotIn(rejected_move, [move.uci() for move in moves])

    def test_para_capture_setup_must_be_converted_in_the_same_turn(self):
        before = engine.BaseBoard(
            "q2fi1i1/1ir↓2i1i/i1F3f1/4F3/3IT↑2I/R→FH↑2R→2/"
            "2I2I1Q/1I1I1PI1 - - r"
        )
        searcher = ghq_ai.Searcher(
            "tactical_gambler", time_ms=2000, beam_width=6, turn_number=35
        )

        def replay(ucis):
            board = before.copy()
            moves = []
            for uci in ucis:
                move = next(
                    move
                    for move in board.generate_legal_moves()
                    if move.uci() == uci
                )
                moves.append(move)
                board.push(move)
            return board, moves

        converted, converted_moves = replay(
            ("e4g4↑", "f1c8", "c6d7xc7")
        )
        abandoned, abandoned_moves = replay(("e4g4↑", "f1c8", "skip"))

        self.assertEqual(
            searcher.paratrooper_mission_penalty(
                before, converted, converted_moves, engine.RED
            ),
            0.0,
        )
        self.assertGreaterEqual(
            searcher.paratrooper_mission_penalty(
                before, abandoned, abandoned_moves, engine.RED
            ),
            ghq_ai.MISSIONLESS_PARATROOPER_PENALTY,
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

    def test_quiet_action_gets_credit_for_setting_up_later_protection(self):
        before = engine.BaseBoard(
            "6q1/8/8/8/I6i/1F4i1/I1I3i1/6Q1 - - r"
        )
        after = before.copy()
        moves = []
        for uci in ("c2c1", "a2b2", "g1f1"):
            move = next(
                candidate
                for candidate in after.generate_legal_moves()
                if candidate.uci() == uci
            )
            moves.append(move)
            after.push(move)
        searcher = ghq_ai.Searcher(
            "fortress", time_ms=2000, beam_width=6, turn_number=115
        )
        purposes = searcher.action_purpose_labels(before, moves, engine.RED)
        self.assertEqual(purposes[0]["roles"], ["setup"])
        self.assertIn("protect", purposes[1]["roles"])
        self.assertIn("protect", purposes[2]["roles"])
        self.assertEqual(
            searcher.turn_purpose_breakdown(
                before, after, moves, engine.RED
            )["unpurposed_actions"],
            0.0,
        )

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
        seed = ghq_ai.purposeful_complete_turn_seed(
            board, "battery_commander", turn_number=5
        )
        seed_moves, seed_board = ghq_ai.first_turn_from_pv(board, seed.pv)
        seed_purpose = searcher.turn_purpose_breakdown(
            board, seed_board, seed_moves, board.turn
        )
        self.assertGreaterEqual(seed_purpose["development_actions"], 2.0)
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

    def test_complete_turn_seed_preserves_idle_paratrooper_and_breaks_column(self):
        board = engine.BaseBoard(VERTICAL_INFANTRY_FEN)
        before_shape = ghq_ai.infantry_shape_score(board, engine.RED)
        result = ghq_ai.purposeful_complete_turn_seed(
            board, "balanced", turn_number=12
        )
        ucis = [move.uci() for move in result.pv]
        self.assertNotEqual(ucis, ["h5h6", "h4h5", "h3h4"])
        self.assertFalse(any(uci.startswith("d1") for uci in ucis))
        after = board.copy()
        for move in result.pv:
            after.push(move)
        self.assertGreater(
            ghq_ai.infantry_shape_score(after, engine.RED), before_shape
        )

    def test_complete_turn_seed_does_not_stage_para_just_to_spend_an_action(self):
        result = ghq_ai.purposeful_complete_turn_seed(
            engine.BaseBoard(), "balanced", turn_number=2
        )
        self.assertFalse(
            any(
                move.name == "Reinforce"
                and move.unit_type == engine.AIRBORNE_INFANTRY
                for move in result.pv
            )
        )

    def test_complete_turn_seed_does_not_double_skip_when_quiet_moves_exist(self):
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
        result = ghq_ai.purposeful_complete_turn_seed(
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

    def test_policy_model_is_separate_bounded_and_cached(self):
        calls = []

        def policy(fen, turn_number, mover):
            calls.append((fen, turn_number, mover))
            return 4.0

        searcher = ghq_ai.Searcher(
            "balanced",
            time_ms=1000,
            beam_width=4,
            policy_function=policy,
        )
        board = engine.BaseBoard()
        value_score = searcher.static_score(board)
        first = searcher.transition_policy_score(board, engine.RED)
        second = searcher.transition_policy_score(board, engine.RED)

        self.assertEqual(value_score, searcher.heuristic_score(board))
        self.assertEqual(first, 3.0 * ghq_ai.POLICY_SCORE_WEIGHT)
        self.assertEqual(second, first)
        self.assertEqual(len(calls), 1)
        self.assertEqual(searcher.policy_model_evaluations, 1)

    def test_deadline_safe_policy_score_does_not_break_fallback(self):
        calls = []

        def policy(fen, turn_number, mover):
            calls.append((fen, turn_number, mover))
            return 1.0

        searcher = ghq_ai.Searcher(
            "balanced",
            time_ms=100,
            beam_width=4,
            policy_function=policy,
        )
        board = engine.BaseBoard()
        searcher.deadline = 0.0

        self.assertEqual(
            searcher.deadline_safe_transition_policy_score(board, engine.RED),
            0.0,
        )
        self.assertEqual(calls, [])

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
        self.assertTrue(searcher.artillery_move_allowed(board, engine.Move.from_uci("g1f1↑")))
        self.assertFalse(searcher.artillery_move_allowed(board, engine.Move.from_uci("g1f1↖")))

    def test_edge_facing_artillery_relocation_without_a_target_is_rejected(self):
        board = engine.BaseBoard(
            "qr↓1fi1if/i1ir↙h↓f2/1i1t↓4/8/8/III5/"
            "R↑R↑1H↑1III/2I1I1R↑Q FFFP iiir b"
        )
        board.push(engine.Move.from_uci("h8h7"))
        board.push(engine.Move.from_uci("rrc8"))
        move = engine.Move.from_uci("b8b7←")
        searcher = ghq_ai.Searcher(
            "para_specialist", time_ms=2000, beam_width=16, turn_number=6
        )

        self.assertIn(move, list(board.generate_legal_moves()))
        self.assertFalse(searcher.artillery_move_allowed(board, move))

    def test_unprotected_high_value_guns_cannot_advance_into_para_range(self):
        board = engine.BaseBoard(
            "q1r↓fi1i1/ir←ir↙h↓f1f/1i1t↓4/8/8/III2II1/"
            "R↑R↑1H↑2R↑I/2I1I2Q FFFP iii b"
        )
        searcher = ghq_ai.Searcher(
            "para_specialist", time_ms=2000, beam_width=16, turn_number=7
        )

        self.assertFalse(
            searcher.artillery_move_allowed(
                board, engine.Move.from_uci("e7f6↓")
            )
        )
        self.assertFalse(
            searcher.artillery_move_allowed(
                board, engine.Move.from_uci("d6c5↓")
            )
        )

    def test_threatened_high_value_guns_can_retreat_from_para_range(self):
        board = engine.BaseBoard(
            "q1r↓fi1i1/ir←ir↙1f1f/1i3h↓2/2t↓5/3I4/II3II1/"
            "R↑R↑H↑3R↑I/2I1IP1Q FFF iii b"
        )
        board.push(engine.Move.from_uci("sbf3"))
        searcher = ghq_ai.Searcher(
            "para_specialist", time_ms=2000, beam_width=16, turn_number=8
        )

        self.assertFalse(
            searcher.artillery_move_allowed(
                board, engine.Move.from_uci("c5b5↓")
            )
        )
        self.assertFalse(
            searcher.artillery_move_allowed(
                board, engine.Move.from_uci("f6g5↓")
            )
        )
        self.assertTrue(
            searcher.artillery_move_allowed(
                board, engine.Move.from_uci("c5d6↓")
            )
        )
        self.assertTrue(
            searcher.artillery_move_allowed(
                board, engine.Move.from_uci("f6e7↓")
            )
        )

    def test_reported_para_trap_moves_cannot_immediately_follow_bombardment(self):
        """Do not let partial-turn expansion bypass the valuable-gun floor."""
        board = engine.BaseBoard(
            "q1r↓fi1i1/ir←ir↙1f1f/1i3h↓2/2t↓5/3I4/II3II1/"
            "R↑R↑H↑3R↑I/2I1IP1Q FFF iii b"
        )
        board.push(engine.Move.from_uci("sbf3"))
        searcher = ghq_ai.Searcher(
            "para_specialist", time_ms=5000, beam_width=16, turn_number=8
        )

        candidates = searcher.generate_turn_candidates(board)
        forbidden = {"c5b5↓", "f6g5↓"}

        self.assertTrue(candidates)
        self.assertTrue(
            all(
                candidate.moves
                and candidate.moves[0].uci() not in forbidden
                for candidate in candidates
            )
        )

    def test_protected_multi_target_diagonal_artillery_is_allowed(self):
        board = engine.BaseBoard(
            "p6q/8/5r↓2/4r↓3/4I3/2IR↑4/8/7Q - - r"
        )
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=12, turn_number=8
        )
        forward = engine.Move.from_uci("d3d4↑")
        diagonal = engine.Move.from_uci("d3d4↗")
        self.assertTrue(searcher.artillery_move_allowed(board, forward))
        self.assertTrue(searcher.artillery_move_allowed(board, diagonal))
        self.assertGreater(
            searcher.move_priority(board, diagonal),
            searcher.move_priority(board, forward),
        )

    def test_reported_two_target_artillery_lane_retains_forcing_value(self):
        board = engine.BaseBoard(
            "qiir↓fft↓i/i1ir↘3f/ii5h↙/6T↑1/II3H↑2/"
            "R↑7/5III/1PFI2R↑Q IIFF i r"
        )
        move = engine.Move.from_uci("g5h5↑")
        after = board.copy()
        after.push(move)
        self.assertEqual(
            ghq_ai.artillery_forced_response_burden(after, engine.RED),
            2.75,
        )

        result = ghq_ai.search(
            board,
            "balanced",
            time_ms=10000,
            max_depth=2,
            beam_width=8,
            turn_number=12,
            opening_seed=0,
        )
        self.assertIn("g5h5↑", result["best_turn"]["all_moves"])

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
            time_ms=6000,
            max_depth=1,
            beam_width=6,
            turn_number=27,
        )
        self.assertEqual(result["recommendation_label"], "best found")
        self.assertFalse(result["search"]["exhaustive_within_requested_horizon"])

    def test_search_choice_is_equivalent_after_color_mirror(self):
        board = engine.BaseBoard(PARATROOPER_EXTRACTION_FEN)
        mirrored = board.mirror()
        original = ghq_ai.search(
            board,
            "balanced",
            time_ms=3000,
            max_depth=1,
            beam_width=6,
            turn_number=27,
        )
        inverse = ghq_ai.search(
            mirrored,
            "balanced",
            time_ms=3000,
            max_depth=1,
            beam_width=6,
            turn_number=27,
        )
        original_key = tuple(
            ghq_ai.normalized_move_uci(engine.Move.from_uci(uci), board.turn)
            for uci in original["best_turn"]["all_moves"]
        )
        inverse_key = tuple(
            ghq_ai.normalized_move_uci(engine.Move.from_uci(uci), mirrored.turn)
            for uci in inverse["best_turn"]["all_moves"]
        )
        self.assertEqual(original_key, inverse_key)
        self.assertAlmostEqual(
            original["score"]["red"], -inverse["score"]["red"], places=4
        )

    def test_reply_first_search_completes_depth_two_before_widening(self):
        result = ghq_ai.search(
            engine.BaseBoard(PARATROOPER_EXTRACTION_FEN),
            "balanced",
            time_ms=4000,
            max_depth=2,
            beam_width=6,
            turn_number=27,
        )
        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertNotEqual(result["search"]["fallback_used"], "seeded")

    def test_reserved_final_slice_retries_the_emergency_seed_reply(self):
        calls = []

        def staged_alphabeta(searcher, board, depth, alpha, beta):
            calls.append((board.turn, depth, searcher.hq_leaf_extension_enabled))
            if len(calls) <= 2:
                raise ghq_ai.SearchTimeout
            return ghq_ai.SearchResult(ghq_ai.MATE_SCORE + 100.0, [])

        with patch.object(
            ghq_ai.Searcher,
            "alphabeta",
            staged_alphabeta,
        ):
            result = ghq_ai.search(
                engine.BaseBoard(),
                "balanced",
                time_ms=1000,
                max_depth=2,
                beam_width=6,
                turn_number=8,
            )

        self.assertEqual(len(calls), 3)
        self.assertEqual(
            [leaf_extension for _, _, leaf_extension in calls],
            [False, True, False],
        )
        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertEqual(result["search"]["fallback_used"], "safe")
        self.assertTrue(result["search"]["seed_reply_verified"])
        self.assertTrue(result["search"]["seed_reply_retry_used"])
        self.assertTrue(result["best_turn"]["actions"])

    def test_seed_floor_gets_contiguous_budget_then_root_gets_remainder(self):
        remaining_deadlines = []

        def staged_alphabeta(searcher, board, depth, alpha, beta):
            remaining_deadlines.append(searcher.deadline - time.monotonic())
            if len(remaining_deadlines) == 1:
                return ghq_ai.SearchResult(0.0, [])
            raise ghq_ai.SearchTimeout

        with patch.object(
            ghq_ai.Searcher,
            "alphabeta",
            staged_alphabeta,
        ):
            result = ghq_ai.search(
                engine.BaseBoard(),
                "balanced",
                time_ms=1_000,
                max_depth=2,
                beam_width=6,
                turn_number=8,
            )

        self.assertEqual(len(remaining_deadlines), 2)
        self.assertGreater(remaining_deadlines[0], 0.30)
        self.assertLessEqual(remaining_deadlines[0], 0.41)
        self.assertGreater(remaining_deadlines[1], 0.80)
        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertEqual(result["search"]["fallback_used"], "safe")
        self.assertTrue(result["search"]["seed_reply_verified"])

    def test_timeout_keeps_verified_root_development_instead_of_seed_backfill(self):
        result = ghq_ai.search(
            engine.BaseBoard(TURN_FIVE_DEVELOPMENT_FEN),
            "fortress",
            time_ms=2000,
            max_depth=2,
            beam_width=6,
            turn_number=5,
        )
        moves = result["best_turn"]["all_moves"]
        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertNotIn("f2f1", moves)
        self.assertEqual(result["best_turn"]["purpose"]["development_actions"], 3)
        self.assertIn(moves, [turn["all_moves"] for turn in result["candidate_turns"]])

    def test_verification_bounds_atomic_breadth_before_comparing_complex_root(self):
        result = ghq_ai.search(
            engine.BaseBoard(TURN_23_COMPLEX_ROOT_FEN),
            "balanced",
            time_ms=4000,
            max_depth=2,
            beam_width=6,
            turn_number=23,
        )
        candidates = result["candidate_turns"]
        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertTrue(
            any(
                "hq_escape_unlock" in purpose["roles"]
                for purpose in result["best_turn"]["action_purposes"]
            )
        )
        self.assertTrue(
            any(
                "hq_defense" in purpose["roles"]
                for purpose in result["best_turn"]["action_purposes"]
            )
        )
        escaped = engine.BaseBoard(result["best_turn"]["resulting_fen"])
        verifier = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=24
        )
        self.assertFalse(
            verifier.exact_same_turn_hq_capture(
                escaped, escaped.turn, [100_000]
            )
        )
        self.assertGreater(result["score"]["current_player"], -1000000.0)
        self.assertIn(
            result["best_turn"]["all_moves"],
            [turn["all_moves"] for turn in candidates],
        )

    def test_verification_finishes_reply_before_spending_budget_on_breadth(self):
        result = ghq_ai.search(
            engine.BaseBoard(TURN_12_SLOW_REPLY_FEN),
            "fortress",
            time_ms=6000,
            max_depth=2,
            beam_width=6,
            turn_number=12,
            value_function=lambda _fen, _turn: 0.5,
        )
        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertNotEqual(result["search"]["fallback_used"], "seeded")

    def test_stagnation_keeps_a_noncycling_root_alternative(self):
        board = engine.BaseBoard(STALL_CONVEYOR_FEN)
        searcher = ghq_ai.Searcher(
            "balanced",
            time_ms=1000,
            beam_width=6,
            turn_number=82,
            stagnation_turns=17,
        )
        searcher.root_key = board.serialize()
        cycle = ghq_ai.TurnCandidate(
            [], board, 0.0, progress_score=0.12, conveyor_actions=1.0
        )
        central_break = ghq_ai.TurnCandidate(
            [], board, 0.0, progress_score=0.90, conveyor_actions=1.0
        )

        self.assertLess(
            searcher.candidate_sort_key(central_break, engine.BLUE, True),
            searcher.candidate_sort_key(cycle, engine.BLUE, True),
        )

    def test_late_stagnation_beam_reserves_a_capture_line(self):
        board = engine.BaseBoard(
            "q7/iii4r←/3i1ii1/2F1i3/1I6/F5f1/1II3Q1/3I3I - i b"
        )
        searcher = ghq_ai.Searcher(
            "fortress",
            time_ms=60_000,
            beam_width=6,
            turn_number=88,
            stagnation_turns=22,
        )

        def candidate(ucis):
            working = board.copy()
            moves = []
            for uci in ucis:
                move = next(
                    move
                    for move in working.generate_legal_moves()
                    if move.uci() == uci
                )
                moves.append(move)
                working.push(move)
            return ghq_ai.TurnCandidate(moves, working, 0.0)

        retreat = candidate(("g3g5", "g6h6", "skip"))
        capture = candidate(("b7c6", "c7b7", "d6d5xc5"))
        setup = next(
            move for move in board.generate_legal_moves() if move.uci() == "b7c6"
        )
        self.assertTrue(searcher.unlocks_capture_this_turn(board, setup))
        self.assertGreater(searcher.move_priority(board, setup), 3000.0)

        ready = board.copy()
        for uci in ("b7c6", "c7b7"):
            ready.push(
                next(
                    move
                    for move in ready.generate_legal_moves()
                    if move.uci() == uci
                )
            )
        ready = searcher.board_as_turn(ready, engine.BLUE)
        self.assertTrue(
            any(
                move.capture_preference is not None
                for move in ready.generate_legal_moves()
            )
        )
        unnecessary = next(
            move for move in ready.generate_legal_moves() if move.uci() == "g3g5"
        )
        self.assertFalse(
            searcher.unlocks_capture_this_turn(ready, unnecessary)
        )
        selected = searcher.select_diverse_turns(
            board, [retreat, capture], turn_width=1
        )

        self.assertIn(
            "capture", searcher.turn_action_classes(board, capture.moves)[0]
        )
        self.assertIn(capture, selected)

    def test_quiet_objective_approach_has_durable_stagnation_purpose(self):
        board = engine.BaseBoard(
            "q1i3i1/3i3i/2i3i1/8/8/8/1IIII1I1/1F3I1Q - - r"
        )
        searcher = ghq_ai.Searcher(
            "balanced",
            time_ms=1000,
            beam_width=6,
            turn_number=95,
            stagnation_turns=17,
        )
        moves = [engine.Move.from_uci(uci) for uci in ("d2c3", "c2d2", "b1a1")]
        after = board.copy()
        for move in moves:
            after.push(move)
        purpose = searcher.turn_purpose_breakdown(board, after, moves, engine.RED)
        roles = {
            role
            for action in searcher.action_purpose_labels(board, moves, engine.RED)
            for role in action["roles"]
        }

        self.assertIn("advance", roles)
        self.assertGreater(purpose["stagnation_progress"], 0.0)

    def test_forcing_artillery_pressure_displaces_empty_infantry_move(self):
        board = engine.BaseBoard(POST_EXTRACTION_PRESSURE_FEN)
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=2000, beam_width=6, turn_number=29
        )
        moves = [move.uci() for move in searcher.ordered_moves(board)]
        self.assertIn("d2e3↑", moves)
        self.assertNotIn("b1b2", moves)

    def test_search_returns_a_complete_three_action_turn(self):
        board = engine.BaseBoard()
        result = ghq_ai.search(board, "balanced", time_ms=1200, max_depth=1, beam_width=8)
        moves = result["best_turn"]["all_moves"]
        actions = result["best_turn"]["actions"]
        self.assertEqual(len([uci for uci in actions if uci != "skip"]), 3)

        replay = board.copy()
        for uci in moves:
            move = engine.Move.from_uci(uci)
            self.assertIn(move, list(replay.generate_legal_moves()))
            replay.push(move)
        self.assertNotEqual(replay.turn, board.turn)
        self.assertEqual(replay.board_fen(), result["best_turn"]["resulting_fen"])

    def test_search_can_plan_and_complete_a_two_action_turn(self):
        board = engine.BaseBoard()
        result = ghq_ai.search(
            board,
            "balanced",
            time_ms=1200,
            max_depth=1,
            beam_width=8,
            max_actions=2,
        )
        actions = result["best_turn"]["actions"]
        self.assertEqual(len([uci for uci in actions if uci != "skip"]), 2)
        self.assertEqual(actions[-1], "skip")
        self.assertEqual(result["search"]["max_actions"], 2)

        replay = board.copy()
        for uci in result["best_turn"]["all_moves"]:
            move = engine.Move.from_uci(uci)
            self.assertIn(move, list(replay.generate_legal_moves()))
            replay.push(move)
        self.assertNotEqual(replay.turn, board.turn)

    def test_search_does_not_return_a_replayable_no_effect_action(self):
        board = engine.BaseBoard(SELF_PLAY_PURPOSELESS_FILLER_FEN)
        result = ghq_ai.search(
            board,
            "battery_commander",
            time_ms=5000,
            max_depth=2,
            beam_width=6,
            turn_number=36,
        )

        voluntary = [
            purpose
            for move, purpose in zip(
                result["best_turn"]["all_moves"],
                result["best_turn"]["action_purposes"],
            )
            if move != "skip" and not move.startswith("sb")
        ]
        self.assertTrue(voluntary)
        self.assertTrue(
            all("no_new_effect" not in purpose["roles"] for purpose in voluntary)
        )
        self.assertEqual(
            result["best_turn"]["purpose"]["unpurposed_actions"], 0.0
        )
        self.assertGreater(
            result["search"]["purposeful_early_stops_generated"], 0
        )

    def test_late_three_action_hq_mate_survives_the_narrow_reply_frontier(self):
        board = engine.BaseBoard(SELF_PLAY_LATE_HQ_REPLY_FEN)
        result = ghq_ai.search(
            board,
            "mobile_raider",
            time_ms=5000,
            max_depth=2,
            beam_width=6,
            turn_number=81,
        )

        self.assertLessEqual(result["score"]["red"], -ghq_ai.MATE_SCORE)
        replay = board.copy()
        for uci in result["principal_variation"]:
            move = next(
                move for move in replay.generate_legal_moves() if move.uci() == uci
            )
            replay.push(move)
        self.assertTrue(replay.is_game_over())
        self.assertEqual(replay.outcome().winner, engine.BLUE)

    def test_two_quiet_setups_preserve_a_paratrooper_hq_mate(self):
        root = engine.BaseBoard(THREE_ACTION_PARATROOPER_MATE_FEN)
        reply = root.copy()
        for uci in ["g7h6", "g6h5", "h8g8"]:
            move = next(
                move for move in reply.generate_legal_moves() if move.uci() == uci
            )
            reply.push(move)

        searcher = ghq_ai.Searcher(
            "para_specialist", time_ms=2000, beam_width=6, turn_number=56
        )
        first_setup = next(
            move for move in reply.generate_legal_moves() if move.uci() == "g4h4"
        )
        self.assertTrue(
            searcher.unlocks_immediate_hq_capture(reply, first_setup)
        )
        self.assertEqual(len(searcher.hq_capture_unlock_move_cache), 1)
        self.assertTrue(
            searcher.unlocks_immediate_hq_capture(reply, first_setup)
        )
        self.assertEqual(len(searcher.hq_capture_unlock_move_cache), 1)

        searcher.root_key = root.serialize()
        searcher.verification_mode = True
        replies = searcher.generate_turn_candidates(reply)
        self.assertTrue(
            any(
                len(candidate.moves) == 3
                and any(move.from_square == engine.B1 for move in candidate.moves)
                and candidate.board.is_game_over()
                and candidate.board.outcome().winner == reply.turn
                for candidate in replies
            )
        )

    def test_slow_vercel_reply_position_completes_tactical_floor(self):
        result = ghq_ai.search(
            engine.BaseBoard(SLOW_VERCEL_REPLY_FEN),
            "fortress",
            time_ms=5_000,
            max_depth=2,
            beam_width=6,
            turn_number=44,
            stagnation_turns=4,
        )

        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertEqual(result["search"]["fallback_used"], "none")
        self.assertNotEqual(result["recommendation_label"], "complete-turn seed")

    def test_search_escapes_two_quiet_setup_paratrooper_mate(self):
        board = engine.BaseBoard(THREE_ACTION_PARATROOPER_MATE_FEN)
        result = ghq_ai.search(
            board,
            "para_specialist",
            time_ms=4000,
            max_depth=2,
            beam_width=6,
            turn_number=56,
        )

        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertGreater(result["score"]["current_player"], -ghq_ai.MATE_SCORE)
        self.assertNotEqual(
            result["best_turn"]["all_moves"],
            ["g7h6", "g6h5", "h8g8"],
        )
        escaped = engine.BaseBoard(result["best_turn"]["resulting_fen"])
        searcher = ghq_ai.Searcher(
            "para_specialist", time_ms=1000, beam_width=6, turn_number=57
        )
        self.assertFalse(
            searcher.has_same_turn_hq_capture(
                searcher.board_as_turn(escaped, engine.RED)
            )
        )

    def test_completed_smoke_hq_losses_are_recognized_before_the_move(self):
        for turn_number, fen in SMOKE_IMMEDIATE_HQ_LOSS_CASES:
            with self.subTest(turn_number=turn_number, fen=fen):
                result = ghq_ai.search(
                    engine.BaseBoard(fen),
                    "balanced",
                    time_ms=10_000,
                    max_depth=2,
                    beam_width=6,
                    turn_number=turn_number,
                )
                self.assertLessEqual(
                    result["score"]["current_player"], -ghq_ai.MATE_SCORE
                )

    def test_completed_smoke_hq_losses_with_escapes_are_reclassified(self):
        for turn_number, fen, deployed_losing_turn in SMOKE_HQ_ESCAPE_CASES:
            with self.subTest(turn_number=turn_number, fen=fen):
                board = engine.BaseBoard(fen)
                result = ghq_ai.search(
                    board,
                    "balanced",
                    time_ms=3000,
                    max_depth=2,
                    beam_width=6,
                    turn_number=turn_number,
                )
                self.assertGreater(
                    result["score"]["current_player"], -ghq_ai.MATE_SCORE
                )
                self.assertNotEqual(
                    result["best_turn"]["all_moves"], deployed_losing_turn
                )
                escaped = engine.BaseBoard(result["best_turn"]["resulting_fen"])
                searcher = ghq_ai.Searcher(
                    "balanced",
                    time_ms=1000,
                    beam_width=6,
                    turn_number=turn_number + 1,
                )
                self.assertFalse(
                    searcher.has_same_turn_hq_capture(
                        searcher.board_as_turn(escaped, escaped.turn)
                    )
                )

    def test_exact_survival_floor_recovers_all_avoidable_batch_hq_losses(self):
        for turn_number, fen in SELF_PLAY_AVOIDABLE_IMMEDIATE_HQ_LOSSES:
            with self.subTest(turn_number=turn_number):
                board = engine.BaseBoard(fen)
                mover = board.turn
                searcher = ghq_ai.Searcher(
                    "balanced", time_ms=60_000, beam_width=6, turn_number=turn_number
                )
                survival = searcher.find_hq_survival_turn(board)
                self.assertIsNotNone(survival)
                moves, escaped = survival
                self.assertTrue(moves)
                reply_budget = [100_000]
                self.assertFalse(
                    searcher.exact_same_turn_hq_capture(
                        escaped, not mover, reply_budget
                    )
                )

    def test_hq_survival_floor_cannot_cash_para_for_one_piece(self):
        board = engine.BaseBoard(MISSIONLESS_PARA_SURVIVAL_OVERRIDE_FEN)
        searcher = ghq_ai.Searcher(
            "para_specialist", time_ms=60_000, beam_width=6, turn_number=63
        )

        survival = searcher.find_hq_survival_turn(board)

        self.assertIsNotNone(survival)
        moves, escaped = survival
        self.assertNotIn("c1c5xd5", [move.uci() for move in moves])
        self.assertEqual(
            searcher.paratrooper_mission_penalty(
                board, escaped, moves, engine.RED
            ),
            0.0,
        )

    def test_hq_survival_floor_recovers_newly_audited_avoidable_losses(self):
        for turn_number, personality, fen in AVOIDABLE_HQ_LOSS_REGRESSIONS:
            with self.subTest(turn_number=turn_number):
                board = engine.BaseBoard(fen)
                searcher = ghq_ai.Searcher(
                    personality,
                    time_ms=60_000,
                    beam_width=6,
                    turn_number=turn_number,
                )

                survival = searcher.find_hq_survival_turn(board)

                self.assertIsNotNone(survival)
                moves, escaped = survival
                self.assertTrue(moves)
                self.assertFalse(searcher.has_same_turn_hq_capture(escaped))

    def test_remote_artillery_lane_can_interdict_a_para_hq_combination(self):
        board = engine.BaseBoard(
            "4p3/qi6/i1i5/8/I7/6i1/1H↑I4i/5P1Q - - r"
        )
        searcher = ghq_ai.Searcher(
            "fortress", time_ms=60_000, beam_width=6, turn_number=95
        )
        east = engine.Move.from_uci("b2c1→")
        north = engine.Move.from_uci("b2c1↑")
        hq_square = engine.parse_square("h1")

        self.assertTrue(
            searcher.artillery_hq_interdiction_squares(
                board, east, hq_square, engine.RED
            )
            & engine.BB_SQUARES[engine.parse_square("f1")]
        )
        self.assertEqual(
            searcher.artillery_hq_interdiction_squares(
                board, north, hq_square, engine.RED
            ),
            engine.BB_EMPTY,
        )
        survival = searcher.find_hq_survival_turn(board)
        self.assertIsNotNone(survival)
        self.assertIn("b2c1→", [move.uci() for move in survival[0]])

    def test_non_capturing_para_can_complete_a_proven_hq_evasion(self):
        root = engine.BaseBoard(
            "4qp2/8/4I1i1/2I2I2/1T↑1IH↑1R↑1/2F4Q/1I6/3P4 - - b"
        )
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=142
        )
        safe = root.copy()
        safe_moves = []
        for uci in ("e8d8", "f8d7", "g6f7"):
            move = next(move for move in safe.generate_legal_moves() if move.uci() == uci)
            safe_moves.append(move)
            safe.push(move)

        self.assertTrue(
            searcher.paratrooper_hq_defense_mission(
                root, safe, safe_moves, engine.BLUE
            )
        )
        self.assertEqual(
            searcher.paratrooper_mission_penalty(
                root, safe, safe_moves, engine.BLUE
            ),
            0.0,
        )
        labels = searcher.action_purpose_labels(
            root, safe_moves, engine.BLUE, retrospective=False
        )
        self.assertIn("hq_defense", labels[1]["roles"])

    def test_deterministic_policy_floor_never_moves_a_paratrooper(self):
        board = engine.BaseBoard(MISSIONLESS_PARA_SURVIVAL_OVERRIDE_FEN)

        moves, after = ghq_ai.deterministic_skip_turn(board)

        self.assertNotEqual(after.turn, board.turn)
        self.assertTrue(moves)
        self.assertTrue(
            all(move.name in ("AutoCapture", "Skip") for move in moves)
        )

    def test_exact_hq_capture_moves_retain_sparse_artillery_and_prune_remote_actions(self):
        board = engine.BaseBoard("q7/8/2R5/8/8/8/8/7Q - - r")
        moves = list(ghq_ai.Searcher.exact_hq_capture_moves(board))

        self.assertNotIn("Skip", {move.name for move in moves})
        stationary = [
            move
            for move in moves
            if move.name == "MoveAndOrient"
            and move.from_square == move.to_square
        ]
        self.assertTrue(stationary)
        enemy_hq = board.hq & board.occupied_co[engine.BLUE]
        for move in stationary:
            piece_type = board.piece_type_at(move.from_square)
            distance = 3 if piece_type == engine.HEAVY_ARTILLERY else 2
            target = board.get_bombardment_target(
                move.to_square, move.orientation, distance
            )
            self.assertIsNotNone(target)
            assert target is not None
            self.assertTrue(
                engine.between_inclusive_end(move.to_square, target)
                & enemy_hq
            )
        self.assertFalse(
            any(
                move.name == "Reinforce"
                and move.unit_type is not None
                and engine.is_artillery(move.unit_type)
                for move in moves
            )
        )
        relocations = [
            (move.from_square, move.to_square)
            for move in moves
            if move.name == "MoveAndOrient"
        ]
        self.assertGreater(len(relocations), len(set(relocations)))
        move_ucis = {move.uci() for move in moves}
        self.assertIn("c6b5↑", move_ucis)
        self.assertIn("c6b5↗", move_ucis)
        self.assertIn("c6c6↖", move_ucis)
        remote = engine.BaseBoard("q7/8/8/8/8/8/8/R6Q R - r")
        self.assertEqual(
            list(ghq_ai.Searcher.exact_hq_capture_moves(remote)), []
        )

    def test_exact_hq_capture_order_checks_the_hq_capture_first(self):
        board = engine.BaseBoard("q7/FF6/8/8/8/8/8/7Q - - r")
        moves = list(ghq_ai.Searcher.exact_hq_capture_moves(board))
        moves.sort(
            key=lambda move: ghq_ai.Searcher.exact_hq_capture_move_priority(
                board, move
            )
        )
        self.assertEqual(moves[-1].uci(), "b7b8xa8")
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=1000, beam_width=6, turn_number=1
        )
        self.assertTrue(
            searcher.exact_same_turn_hq_capture(board, engine.RED, [2])
        )

    def test_masked_immediate_hq_detector_matches_full_legal_generation(self):
        fens = (
            SELF_PLAY_HQ_UNLOCK_FEN,
            SELF_PLAY_HQ_ENGAGEMENT_FEN,
            SELF_PLAY_THREE_ACTION_HQ_FEN,
            *(fen for _, fen in SMOKE_IMMEDIATE_HQ_LOSS_CASES),
            *(fen for _, fen in SELF_PLAY_AVOIDABLE_IMMEDIATE_HQ_LOSSES),
        )
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=80
        )
        for fen in fens:
            board = engine.BaseBoard(fen)
            probes = [board]
            for move in list(board.generate_legal_moves())[:12]:
                child = board.copy()
                child.push(move)
                probes.append(child)
            for probe in probes:
                with self.subTest(fen=fen, position=probe.board_fen()):
                    exhaustive = any(
                        move.capture_preference is not None
                        and probe.piece_type_at(move.capture_preference)
                        == engine.HQ
                        for move in probe.generate_legal_moves()
                    )
                    self.assertEqual(
                        searcher.has_immediate_hq_capture(probe), exhaustive
                    )

    def test_exact_survival_floor_does_not_invent_an_escape_from_forced_mate(self):
        turn_number, fen = SELF_PLAY_FORCED_IMMEDIATE_HQ_LOSS
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=turn_number
        )
        self.assertIsNone(
            searcher.find_hq_survival_turn(engine.BaseBoard(fen))
        )

    def test_search_delays_mate_instead_of_walking_hq_onto_capture(self):
        turn_number, fen = SELF_PLAY_AVOIDABLE_IMMEDIATE_HQ_LOSSES[0]
        board = engine.BaseBoard(fen)
        mover = board.turn
        result = ghq_ai.search(
            board,
            "balanced",
            time_ms=1_000,
            max_depth=2,
            beam_width=6,
            turn_number=turn_number,
        )
        escaped = engine.BaseBoard(result["best_turn"]["resulting_fen"])
        verifier = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=turn_number
        )
        self.assertFalse(
            verifier.exact_same_turn_hq_capture(
                escaped, not mover, [100_000]
            )
        )
        self.assertEqual(result["search"]["fallback_used"], "safe")
        self.assertGreater(result["search"]["hq_survival_reply_nodes"], 0)

    def test_live_mate_delay_fallbacks_complete_an_opponent_reply(self):
        for turn_number, personality, fen, expected_moves in LIVE_MATE_DELAY_REPLY_CASES:
            with self.subTest(turn_number=turn_number):
                result = ghq_ai.search(
                    engine.BaseBoard(fen),
                    personality,
                    time_ms=2_000,
                    max_depth=2,
                    beam_width=6,
                    turn_number=turn_number,
                )
                self.assertEqual(
                    tuple(result["best_turn"]["all_moves"]), expected_moves
                )
                self.assertEqual(
                    result["search"]["completed_depth_in_turns"], 2
                )
                self.assertEqual(result["search"]["fallback_used"], "safe")
                self.assertTrue(
                    result["search"]["hq_survival_override_used"]
                )
                self.assertTrue(
                    result["search"]["hq_survival_reply_verified"]
                )

    def test_newly_reclassified_smoke_escapes_survive_every_immediate_reply(self):
        cases = (
            (
                "1q6/6i1/F1i4r←/1F5f/I7/8/7i/6IQ - i b",
                ("c6c7", "rif8", "h5f7"),
                1593,
            ),
            (
                "2q1i2f/5ii1/8/1i6/I1i5/1Q6/5I2/4I3 I - r",
                ("e1e2", "rid1", "skip"),
                5933,
            ),
            (
                "q2i1i2/3ii3/4f3/8/8/1F6/IF1I1f2/1P2f1Q1 II i r",
                ("rif1", "d2e2", "skip"),
                10525,
            ),
            (
                "q3i1ii/1F2r←1f1/F7/4i1r↓1/8/3f2I1/5I2/I6Q III ii b",
                ("sbg3", "rid8", "ric8", "g7f7"),
                3521,
            ),
            (
                "q7/5i2/2F3i1/3F4/8/5I2/2I3I1/4IIQ1 - - b",
                ("g6f5", "f7g8", "skip"),
                16698,
            ),
        )
        for fen, ucis, expected_replies in cases:
            with self.subTest(fen=fen):
                escaped = engine.BaseBoard(fen)
                for uci in ucis:
                    escaped.push(
                        next(
                            move
                            for move in escaped.generate_legal_moves()
                            if move.uci() == uci
                        )
                    )

                opponent = escaped.turn
                frontier = [escaped]
                completed = set()
                while frontier:
                    partial = frontier.pop()
                    if partial.is_game_over() or partial.turn != opponent:
                        key = partial.serialize()
                        if key in completed:
                            continue
                        completed.add(key)
                        outcome = partial.outcome()
                        self.assertFalse(
                            outcome is not None and outcome.winner == opponent,
                            "the reclassified line still permits an immediate HQ win",
                        )
                        continue
                    for move in partial.generate_legal_moves():
                        child = partial.copy()
                        child.push(move)
                        frontier.append(child)
                self.assertEqual(len(completed), expected_replies)

    def test_horizon_extension_preserves_the_hq_evasion(self):
        # Red has completed the three-action battery setup from production
        # smoke game 0008. Blue can clear b8 and move its HQ there, but the old
        # safety gate discarded escape sequences when minor material remained
        # hanging after the 100-point HQ threat was resolved.
        board = engine.BaseBoard(
            "qr↓1f2ii/iii1r→3/6f1/H↑6f/8/2FF2I1/"
            "R↑3FI2/IR↑3IR↑Q III iii b"
        )
        searcher = ghq_ai.Searcher(
            "battery_commander",
            time_ms=60_000,
            beam_width=6,
            turn_number=26,
        )
        searcher.hq_leaf_extension_enabled = True
        result = searcher.alphabeta(
            board, 0, -float("inf"), float("inf")
        )
        self.assertGreater(result.score, -ghq_ai.MATE_SCORE)
        self.assertLess(result.score, ghq_ai.MATE_SCORE)
        self.assertIn("a8b8", [move.uci() for move in result.pv])

        working = board.copy()
        moves = []
        for uci in ("h5h7", "b8c8↘", "a8b8"):
            move = next(
                candidate
                for candidate in working.generate_legal_moves()
                if candidate.uci() == uci
            )
            moves.append(move)
            working.push(move)
        safety = searcher.assess_turn_safety(board, working, engine.BLUE)
        self.assertTrue(safety.tactically_safe)
        self.assertLess(safety.forced_loss_value, ghq_ai.PIECE_VALUES[engine.HQ])

    def test_reinforcement_interposition_survives_the_narrow_beam(self):
        board = engine.BaseBoard(
            "q3i1ii/1F2r←1f1/F7/4i1r↓1/8/3f2I1/5I2/I6Q III ii b"
        )
        automatic = next(
            move for move in board.generate_legal_moves() if move.uci() == "sbg3"
        )
        board.push(automatic)
        searcher = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=50
        )
        interpose = next(
            move for move in board.generate_legal_moves() if move.uci() == "ric8"
        )
        self.assertTrue(searcher.resolves_hq_threat(board, interpose))
        self.assertGreater(searcher.move_priority(board, interpose), 8000.0)

        candidates = searcher.generate_turn_candidates(board)
        self.assertTrue(
            any(
                "ric8" in [move.uci() for move in candidate.moves]
                and candidate.tactically_safe
                for candidate in candidates
            )
        )

    def test_blocker_vacate_and_hq_escape_survive_the_narrow_beam(self):
        board = engine.BaseBoard(
            "qr↓1f2ii/iii1r→3/6f1/H↑6f/8/2FF2I1/"
            "R↑3FI2/IR↑3IR↑Q III iii b"
        )
        searcher = ghq_ai.Searcher(
            "battery_commander",
            time_ms=60_000,
            beam_width=6,
            turn_number=26,
        )
        searcher.root_key = board.serialize()
        searcher.verification_mode = True
        bounded = [
            move.uci() for _, move in searcher.bounded_diverse_moves(board)
        ]
        self.assertIn("b8c8↓", bounded)

        candidates = searcher.generate_turn_candidates(board)
        escapes = [
            candidate
            for candidate in candidates
            if "a8b8" in [move.uci() for move in candidate.moves]
        ]
        self.assertTrue(escapes)
        self.assertTrue(any(candidate.tactically_safe for candidate in escapes))

    def test_nonadjacent_interposition_unlocks_a_safe_hq_evacuation(self):
        board = engine.BaseBoard(
            "1q6/FF2i1ii/5i1i/8/8/8/1I5Q/I1I1II2 - - b"
        )
        searcher = ghq_ai.Searcher(
            "battery_commander",
            time_ms=60_000,
            beam_width=6,
            turn_number=104,
        )
        interpose = next(
            move for move in board.generate_legal_moves() if move.uci() == "e7d7"
        )
        self.assertTrue(searcher.unlocks_hq_escape(board, interpose))
        self.assertGreater(searcher.move_priority(board, interpose), 8000.0)

        searcher.root_key = board.serialize()
        searcher.verification_mode = True
        candidates = searcher.generate_turn_candidates(board)
        escapes = [
            candidate
            for candidate in candidates
            if "e7d7" in [move.uci() for move in candidate.moves]
            and "b8c8" in [move.uci() for move in candidate.moves]
        ]
        self.assertTrue(escapes)
        self.assertTrue(all(candidate.tactically_safe for candidate in escapes))

    def test_nonadjacent_infantry_move_unlocks_a_second_hq_evacuation(self):
        board = engine.BaseBoard(
            "2Fq3i/2F4i/6i1/8/6i1/1I6/2I4f/2I4Q I - b"
        )
        searcher = ghq_ai.Searcher(
            "battery_commander",
            time_ms=60_000,
            beam_width=6,
            turn_number=98,
        )
        setup = next(
            move for move in board.generate_legal_moves() if move.uci() == "g6f7"
        )
        self.assertTrue(searcher.unlocks_hq_escape(board, setup))
        self.assertGreater(searcher.move_priority(board, setup), 8000.0)

        searcher.root_key = board.serialize()
        searcher.verification_mode = True
        candidates = searcher.generate_turn_candidates(board)
        escapes = [
            candidate
            for candidate in candidates
            if "g6f7" in [move.uci() for move in candidate.moves]
            and "d8e8" in [move.uci() for move in candidate.moves]
        ]
        self.assertTrue(escapes)
        self.assertTrue(all(candidate.tactically_safe for candidate in escapes))

    def test_stagnation_beam_preserves_multi_piece_hq_encirclement(self):
        board = engine.BaseBoard(
            "8/8/5Iq1/8/2I5/1F2I2I/F4I2/7Q - - r"
        )
        searcher = ghq_ai.Searcher(
            "para_specialist",
            time_ms=60_000,
            beam_width=6,
            turn_number=131,
            stagnation_turns=23,
        )
        searcher.root_key = board.serialize()
        searcher.verification_mode = True

        candidates = searcher.generate_turn_candidates(board)

        self.assertTrue(candidates)
        self.assertGreaterEqual(
            max(candidate.progress_score for candidate in candidates), 3.0
        )
        self.assertTrue(
            any(
                candidate.conveyor_actions == 0
                and candidate.skip_actions == 0
                and any(
                    "advance" in action["roles"]
                    for action in candidate.action_purposes
                )
                for candidate in candidates
            )
        )

    def test_mild_stagnation_keeps_reply_verification_root_narrow(self):
        board = engine.BaseBoard(
            "6i1/3q3i/1i6/F1r↓1i3/8/8/1FI1IF1I/2PI2IQ II ii b"
        )
        searcher = ghq_ai.Searcher(
            "mobile_raider",
            time_ms=60_000,
            beam_width=6,
            turn_number=38,
            stagnation_turns=8,
        )
        searcher.root_key = board.serialize()
        searcher.verification_mode = True

        candidates = searcher.generate_turn_candidates(board)

        self.assertTrue(candidates)
        self.assertLessEqual(len(candidates), 2)

    def test_purpose_filter_cannot_delete_a_quiet_hq_escape(self):
        board = engine.BaseBoard(
            "q2i1i2/3ii3/4f3/8/8/1F6/IF1I1f2/1P2f1Q1 II i r"
        )
        searcher = ghq_ai.Searcher(
            "balanced",
            time_ms=60_000,
            beam_width=6,
            turn_number=41,
        )
        searcher.root_key = board.serialize()
        searcher.verification_mode = True
        candidates = searcher.generate_turn_candidates(board)
        escapes = [
            candidate
            for candidate in candidates
            if "g1h2" in [move.uci() for move in candidate.moves]
        ]
        self.assertTrue(escapes)
        self.assertTrue(all(candidate.tactically_safe for candidate in escapes))

    def test_purposeful_early_stop_finds_escape_from_smoke_hq_mate(self):
        board = engine.BaseBoard("8/2q5/8/1F6/I7/2i5/1i6/Q7 - - r")
        result = ghq_ai.search(
            board,
            "fortress",
            time_ms=3000,
            max_depth=2,
            beam_width=6,
            turn_number=141,
        )

        self.assertEqual(result["best_turn"]["actions"][-1], "skip")
        self.assertEqual(len(result["best_turn"]["actions"]), 3)
        self.assertGreater(result["score"]["current_player"], -ghq_ai.MATE_SCORE)
        self.assertEqual(result["best_turn"]["purpose"]["unpurposed_actions"], 0.0)

        escaped = board.copy()
        for uci in result["best_turn"]["all_moves"]:
            move = next(
                move for move in escaped.generate_legal_moves() if move.uci() == uci
            )
            escaped.push(move)
        opponent = escaped.turn
        frontier = [escaped]
        completed = set()
        while frontier:
            partial = frontier.pop()
            if partial.is_game_over() or partial.turn != opponent:
                key = partial.serialize()
                if key in completed:
                    continue
                completed.add(key)
                outcome = partial.outcome()
                self.assertFalse(
                    outcome is not None and outcome.winner == opponent,
                    "the supposedly safe turn still permits an immediate HQ capture",
                )
                continue
            for move in partial.generate_legal_moves():
                child = partial.copy()
                child.push(move)
                frontier.append(child)
        self.assertGreater(len(completed), 400)

    def test_complete_turn_seed_completes_a_one_action_hq_only_turn(self):
        board = engine.BaseBoard(
            "q6i/8/8/2ii1i2/1i4i1/r→7/8/5Q2 - - r"
        )
        board.push(
            next(
                move
                for move in board.generate_legal_moves()
                if move.uci() == "f1g2"
            )
        )
        self.assertEqual(
            [move.uci() for move in board.generate_legal_moves()], ["skip"]
        )

        result = ghq_ai.purposeful_complete_turn_seed(
            board, "balanced", turn_number=61
        )
        self.assertEqual([move.uci() for move in result.pv], ["skip"])
        for move in result.pv:
            board.push(move)
        self.assertEqual(board.turn, engine.BLUE)

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
            time_ms=8000,
            max_depth=1,
            beam_width=6,
            turn_number=28,
        )
        self.assertEqual(result["best_turn"]["actions"], ["h2g1xf1", "g3h2xh1"])
        self.assertLessEqual(result["score"]["red"], -ghq_ai.MATE_SCORE)

    def test_quiet_hq_capture_unlock_survives_action_and_turn_beams(self):
        board = engine.BaseBoard(SELF_PLAY_HQ_UNLOCK_FEN)
        root_key = board.serialize()
        for uci in ("g5h4", "h6g5", "g7h6"):
            board.push(
                next(move for move in board.generate_legal_moves() if move.uci() == uci)
            )

        searcher = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=63
        )
        # Exercise the deliberately narrow opponent-reply generator used by
        # the production depth-two verification pass, not only the wide root.
        searcher.root_key = root_key
        searcher.verification_mode = True
        bounded = [move.uci() for _, move in searcher.bounded_diverse_moves(board)]
        self.assertIn("c6b7", bounded)

        candidates = searcher.generate_turn_candidates(board)
        mating_turns = [
            candidate
            for candidate in candidates
            if candidate.board.outcome() is not None
            and candidate.board.outcome().winner == engine.RED
        ]
        self.assertTrue(mating_turns)
        self.assertTrue(
            any(
                [move.uci() for move in candidate.moves][:2]
                == ["c6b7", "c4c6xc7"]
                for candidate in mating_turns
            )
        )

    def test_hq_engagement_setup_survives_narrow_reply_beam(self):
        board = engine.BaseBoard(SELF_PLAY_HQ_ENGAGEMENT_FEN)
        root_key = board.serialize()
        for uci in ("f6e6", "g7f6", "a8b7"):
            board.push(
                next(move for move in board.generate_legal_moves() if move.uci() == uci)
            )

        searcher = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=83
        )
        searcher.root_key = root_key
        searcher.verification_mode = True
        bounded = [move.uci() for _, move in searcher.bounded_diverse_moves(board)]
        self.assertIn("c5b6", bounded)

        candidates = searcher.generate_turn_candidates(board)
        self.assertTrue(
            any(
                [move.uci() for move in candidate.moves][:2]
                == ["c5b6", "d6c7xb7"]
                and candidate.board.outcome() is not None
                and candidate.board.outcome().winner == engine.RED
                for candidate in candidates
            )
        )

    def test_one_action_turn_finds_a_safe_hq_escape(self):
        board = engine.BaseBoard(SELF_PLAY_FORCED_SKIP_MATE_FEN)
        result = ghq_ai.search(
            board,
            "tactical_gambler",
            time_ms=4_000,
            max_depth=2,
            beam_width=6,
            turn_number=86,
        )
        self.assertEqual(result["best_turn"]["actions"][-1], "skip")
        self.assertEqual(len(result["best_turn"]["actions"]), 2)
        self.assertTrue(result["best_turn"]["actions"][0].startswith("f6"))
        self.assertGreater(result["score"]["current_player"], -ghq_ai.MATE_SCORE)
        self.assertEqual(result["search"]["completed_depth_in_turns"], 2)
        self.assertEqual(result["search"]["fallback_used"], "none")

        escaped = engine.BaseBoard(result["best_turn"]["resulting_fen"])
        opponent = escaped.turn
        frontier = [escaped]
        completed = set()
        while frontier:
            partial = frontier.pop()
            if partial.is_game_over() or partial.turn != opponent:
                key = partial.serialize()
                if key in completed:
                    continue
                completed.add(key)
                outcome = partial.outcome()
                self.assertFalse(
                    outcome is not None and outcome.winner == opponent,
                    "the selected HQ escape still permits an immediate win",
                )
                continue
            for move in partial.generate_legal_moves():
                child = partial.copy()
                child.push(move)
                frontier.append(child)
        self.assertGreater(len(completed), 3000)

    def test_three_action_hq_setup_survives_narrow_reply_beam(self):
        board = engine.BaseBoard(SELF_PLAY_THREE_ACTION_HQ_FEN)
        root_key = board.serialize()
        for uci in ("f1e2", "g3f2", "h2g2"):
            board.push(
                next(move for move in board.generate_legal_moves() if move.uci() == uci)
            )

        searcher = ghq_ai.Searcher(
            "balanced", time_ms=60_000, beam_width=6, turn_number=106
        )
        searcher.root_key = root_key
        searcher.verification_mode = True
        bounded = [move.uci() for _, move in searcher.bounded_diverse_moves(board)]
        self.assertIn("e3d2", bounded)

        candidates = searcher.generate_turn_candidates(board)
        self.assertTrue(
            any(
                len(candidate.moves) <= 3
                and candidate.board.outcome() is not None
                and candidate.board.outcome().winner == engine.BLUE
                for candidate in candidates
            )
        )


if __name__ == "__main__":
    unittest.main()
