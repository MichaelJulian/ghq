"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { AllowedMove, GHQState, Player } from "@/game/engine";
import { BoardProps } from "boardgame.io/react";
import { UserActionState } from "./state";
import { playCaptureSound, playMoveSound } from "@/game/audio";
import { Ctx } from "boardgame.io";

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

export default function useBoard({
  ctx,
  G,
  moves,
  userActionState,
  currentPlayer,
  currentPlayerTurn,
}: {
  ctx: Ctx;
  G: GHQState;
  moves: BoardProps<GHQState>["moves"];
  plugins: BoardProps<GHQState>["plugins"];
  currentPlayer: Player;
  currentPlayerTurn: Player;
  userActionState: UserActionState;
}) {
  const [board, setBoard] = useState<GHQState["board"]>(G.board);
  const [mostRecentMove, setMostRecentMove] = useState<
    AllowedMove | undefined
  >();

  const animateOpponentsTurnToLatestBoardState = useCallback(() => {
    // Only animate when it's our turn (opponent's move has ended)
    if (currentPlayerTurn === currentPlayer) {
      // Slowly re-apply the state to allow for animations.
      for (const [i, board] of G.lastTurnBoards.entries()) {
        sleep(i * 250).then(() => {
          setBoard(board);

          const lastMove = G.lastPlayerMoves[i];
          setMostRecentMove(lastMove);

          if (lastMove.name === "Move" && lastMove.args[2]) {
            playCaptureSound();
          } else {
            playMoveSound();
          }
        });
      }

      // Wait for all the animations to finish before setting the final board state.
      sleep(G.lastTurnBoards.length * 250).then(() => {
        setMostRecentMove(undefined);
        setBoard(G.board);
      });
    }
  }, [currentPlayerTurn, currentPlayer, G.lastTurnBoards, G.board]);

  // Change the board state when the current turn changes or it's game over.
  useEffect(() => {
    if (G.isReplayMode) {
      return;
    }

    animateOpponentsTurnToLatestBoardState();
  }, [currentPlayerTurn, ctx.gameover]);

  useEffect(() => {
    // Animate when it's our turn (i.e. we just made move 1 or 2, or hit undo to go to move 0)
    if (currentPlayerTurn === currentPlayer && G.thisTurnMoves.length >= 0) {
      setBoard(G.board);
    }

    // Also if it's our opponents turn and they have made 0 moves (i.e. we just made our move)
    if (currentPlayerTurn !== currentPlayer && G.thisTurnMoves.length === 0) {
      setBoard(G.board);
    }
  }, [currentPlayerTurn, G.thisTurnMoves]);

  // In replay mode, don't animate the board state when the game state changes, just set it immediately.
  useEffect(() => {
    if (G.isReplayMode) {
      setBoard(G.board);
    }
  }, [G.board]);

  // Actually make the move that's been chosen by the user.
  useEffect(() => {
    if (userActionState.chosenMove) {
      const { name, args } = userActionState.chosenMove;
      moves[name](...args);

      if (name === "Move" && args[2]) {
        playCaptureSound();
      } else {
        playMoveSound();
      }
    }
  }, [userActionState.chosenMove]);

  // Play capture sounds when a start-of-turn capture has occurred.
  useEffect(() => {
    const startOfTurnCaptures = G.historyLog?.find(
      ({ turn, isCapture }) => turn === ctx.turn && isCapture
    );
    if (startOfTurnCaptures) {
      playCaptureSound();
    }
  }, [ctx.turn, G.historyLog]);

  return {
    board,
    mostRecentMove,
    replay: () => animateOpponentsTurnToLatestBoardState(),
  };
}