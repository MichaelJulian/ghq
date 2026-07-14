import type {
  FenAnalysisRequest,
  FenAnalysisResponse,
  GhqCandidateTurn,
  PersonalityId,
} from "@/game/analysis/types";
import type { Player } from "@/game/engine";
import type { SelfPlayAgent } from "@/game/self-play/play-one-game";

export type AnalyzePosition = (
  request: FenAnalysisRequest
) => Promise<FenAnalysisResponse>;

export interface SearchDecisionRecord {
  turnNumber: number;
  player: Player;
  fen: string;
  personality: PersonalityId;
  selectedRank: number;
  selectedMoves: string[];
  candidateTurns: GhqCandidateTurn[];
  currentPlayerScore: number;
  winProbability: number;
  completedDepth: number;
  timedOut: boolean;
  fallback: "none" | "safe" | "greedy";
  explorationSeed: number;
  explorationTemperature: number;
}

export interface SearchSelfPlayAgentOptions {
  id: string;
  personality: PersonalityId;
  analyze: AnalyzePosition;
  timeMs: number;
  maxDepth: number;
  beamWidth: number;
  explorationTemperature: number;
  onDecision?: (record: SearchDecisionRecord) => void | Promise<void>;
}

/**
 * Adapt the complete-turn production search to playOneGame's atomic-action
 * interface. Search is invoked once per turn and the already-validated action
 * sequence is consumed as the production engine advances.
 */
export function createSearchSelfPlayAgent(
  options: SearchSelfPlayAgentOptions
): SelfPlayAgent {
  let queuedMoves: string[] = [];
  let queuedTurn = -1;

  return {
    id: options.id,
    async selectMove(context) {
      if (queuedTurn !== context.turnNumber) {
        queuedMoves = [];
        queuedTurn = context.turnNumber;
      }

      if (queuedMoves.length === 0) {
        const explorationSeed = Math.floor(
          context.random() * 0x1_0000_0000
        );
        const analysis = await options.analyze({
          serializedState: context.board.serialize(),
          turnNumber: context.turnNumber,
          personality: options.personality,
          timeMs: options.timeMs,
          maxDepth: options.maxDepth,
          beamWidth: options.beamWidth,
          explorationTemperature: options.explorationTemperature,
          explorationSeed,
        });
        queuedMoves = [...analysis.search.best_turn.all_moves];
        if (queuedMoves.length === 0) {
          throw new Error(`${options.id} search returned an empty turn`);
        }
        await options.onDecision?.({
          turnNumber: context.turnNumber,
          player: context.player,
          fen: analysis.fen,
          personality: options.personality,
          selectedRank: analysis.search.exploration?.selectedRank ?? 1,
          selectedMoves: [...queuedMoves],
          candidateTurns: analysis.search.candidate_turns ?? [],
          currentPlayerScore: analysis.search.score.current_player,
          winProbability:
            context.player === "RED"
              ? analysis.model.before.redWinProbability
              : analysis.model.before.blueWinProbability,
          completedDepth: analysis.search.search.completed_depth_in_turns,
          timedOut: analysis.search.search.timed_out,
          fallback: analysis.search.search.fallback_used,
          explorationSeed,
          explorationTemperature: options.explorationTemperature,
        });
      }

      const selected = queuedMoves.shift();
      if (!selected || !context.legalMoves.some((move) => move.uci() === selected)) {
        throw new Error(
          `${options.id} queued stale search move ${JSON.stringify(selected)}`
        );
      }
      return selected;
    },
  };
}
