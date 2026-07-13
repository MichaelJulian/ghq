import type { Player } from "@/game/engine";
import type {
  PersonalityEvaluation,
  StyleContribution,
} from "@/game/value-model/styled-evaluation";
import type { PersonalityId } from "@/game/value-model/personalities";

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
}

export interface SearchEvaluationBreakdown {
  perspective: "red-positive";
  personality: PersonalityId;
  components: Record<string, number>;
  weights: Record<string, number>;
  weighted_components: Record<string, number>;
  total_red: number;
}

export interface GhqSearchResult {
  recommendation_label: "best move" | "best found";
  input_fen: string;
  side_to_move: "red" | "blue";
  best_turn: {
    automatic_captures: string[];
    actions: string[];
    all_moves: string[];
    resulting_fen: string;
  };
  principal_variation: string[];
  score: {
    current_player: number;
    red: number;
  };
  search: {
    completed_depth_in_turns: number;
    requested_depth_in_turns: number;
    base_complete_turn_width: number;
    nodes: number;
    elapsed_ms: number;
    timed_out: boolean;
    approximate: boolean;
    exhaustive_within_requested_horizon: boolean;
    rule_filtered_actions: number;
    beam_pruned_actions: number;
    partial_turns_pruned: number;
    complete_turns_generated: number;
    complete_turns_deduplicated: number;
    complete_turns_pruned: number;
    turn_cache_hits: number;
    transposition_hits: number;
  };
  evaluation: {
    before: SearchEvaluationBreakdown;
    after_best_turn: SearchEvaluationBreakdown;
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
