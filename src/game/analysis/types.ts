import type { Player } from "@/game/engine";
import type {
  PersonalityEvaluation,
  StyleContribution,
} from "@/game/value-model/styled-evaluation";
import type { PersonalityId } from "@/game/value-model/personalities";
import type { ValueModelVersion } from "@/game/value-model/inference";

export const GHQ_STARTING_FEN =
  "qr↓6/iii5/8/8/8/8/5III/6R↑Q IIIIIFFFPRRTH iiiiifffprrth r";

export interface FenAnalysisRequest {
  fen?: string;
  serializedState?: string;
  personality?: PersonalityId;
  turnNumber?: number;
  timeMs?: number;
  maxDepth?: number;
  beamWidth?: number;
  /** Experimental policy cap; production GHQ permits three voluntary actions. */
  maxActions?: 2 | 3;
  /** Selects a staged value checkpoint; production defaults to incumbent. */
  valueModel?: ValueModelVersion;
  /** Zero is deterministic; larger values sample more broadly among safe near-best turns. */
  explorationTemperature?: number;
  explorationSeed?: number;
  /** Recent full-turn positions used to avoid self-play repetition loops. */
  recentFens?: string[];
  /** The same player's previous turn, used to detect immediate undo cycles. */
  previousOwnTurnMoves?: string[];
  /** Several same-player turns, ordered oldest to newest, for longer cycles. */
  previousOwnTurns?: string[][];
  /** Current durable-game quiet clock, used to escalate anti-stagnation policy. */
  turnsWithoutProgress?: number;
}

export interface SearchEvaluationBreakdown {
  perspective: "red-positive";
  personality: PersonalityId;
  components: Record<string, number>;
  weights: Record<string, number>;
  weighted_components: Record<string, number>;
  total_red: number;
}

export interface GhqTurnPurpose {
  capture_gain: number;
  deployment_gain: number;
  threat_gain: number;
  protection_gain: number;
  development_gain: number;
  formation_gain: number;
  dispersion_increase: number;
  uncompensated_dispersion: number;
  optionality_gain: number;
  congestion_increase: number;
  immobile_units: number;
  relocation_options: number;
  extension_increase: number;
  frontier_rank: number;
  frontier_limit: number;
  forward_infantry_actions: number;
  coordinated_overpush: number;
  escape_actions: number;
  purposeful_actions: number;
  unpurposed_actions: number;
  setup_actions?: number;
  development_actions: number;
  formation_actions: number;
  quiet_actions: number;
  counted_actions: number;
  unused_actions: number;
  backfills: number;
  reversals: number;
  pure_rotations: number;
  forcing_gain: number;
  net_purpose_penalty: number;
  paratrooper_mission_penalty: number;
  total_penalty: number;
}

export interface GhqCandidateTurn {
  rank: number;
  automatic_captures: string[];
  actions: string[];
  all_moves: string[];
  resulting_fen: string;
  score: number;
  action_purposes: Array<{ move: string; roles: string[] }>;
  purpose: GhqTurnPurpose;
}

export interface GhqSearchResult {
  recommendation_label:
    | "best move"
    | "best found"
    | "safe fallback"
    | "complete-turn seed"
    | "opening book"
    | "exploratory"
    | "history avoidance";
  input_fen: string;
  side_to_move: "red" | "blue";
  best_turn: {
    automatic_captures: string[];
    actions: string[];
    all_moves: string[];
    resulting_fen: string;
    action_purposes: Array<{
      move: string;
      roles: string[];
    }>;
    purpose: GhqTurnPurpose;
  };
  principal_variation: string[];
  candidate_turns: GhqCandidateTurn[];
  score: {
    current_player: number;
    red: number;
  };
  search: {
    completed_depth_in_turns: number;
    requested_depth_in_turns: number;
    base_complete_turn_width: number;
    max_actions: number;
    nodes: number;
    elapsed_ms: number;
    timed_out: boolean;
    fallback_used: "none" | "safe" | "seeded";
    opening_book_used: boolean;
    early_game_focus: boolean;
    approximate: boolean;
    exhaustive_within_requested_horizon: boolean;
    rule_filtered_actions: number;
    beam_pruned_actions: number;
    partial_turns_pruned: number;
    complete_turns_generated: number;
    complete_turns_deduplicated: number;
    complete_turns_pruned: number;
    /** Clean two-action candidates derived by removing a replayable filler. */
    purposeful_early_stops_generated?: number;
    tactically_unsafe_turns: number;
    rotation_quota_pruned: number;
    purpose_filtered_turns: number;
    value_model_evaluations: number;
    turn_cache_hits: number;
    transposition_hits: number;
    /** True when a shared reply-verified early search was reused. */
    persistent_cache_hit?: boolean;
  };
  evaluation: {
    before: SearchEvaluationBreakdown;
    after_best_turn: SearchEvaluationBreakdown;
  };
  exploration?: {
    temperature: number;
    seed: number;
    selectedRank: number;
    candidateCount: number;
  };
}

export interface ModelPositionOutput {
  redWinProbability: number;
  blueWinProbability: number;
  personality: PersonalityEvaluation;
}

export interface FenAnalysisResponse {
  fen: string;
  resultingFen: string;
  serializedState: string;
  sideToMove: Player;
  turnNumber: number;
  personality: PersonalityId;
  effectiveConfig: {
    timeMs: number;
    maxDepth: number;
    beamWidth: number;
    explorationTemperature: number;
    explorationSeed: number;
    maxActions: 2 | 3;
    valueModel: ValueModelVersion;
  };
  outcome?: {
    winner?: Player;
    termination: string;
  };
  model: {
    before: ModelPositionOutput;
    after: ModelPositionOutput;
  };
  search: GhqSearchResult;
}

export type { PersonalityId, StyleContribution };
