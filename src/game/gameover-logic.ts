import { GameoverState, GHQState, Player } from "./engine";

export function getGameoverState(G: GHQState): GameoverState | undefined {
  // No gameover during replays
  if (G.isReplayMode) {
    return;
  }

  if (G.timeControl > 0 && G.redElapsed > G.timeControl) {
    return newWinner("BLUE", "on time");
  }
  if (G.timeControl > 0 && G.blueElapsed > G.timeControl) {
    return newWinner("RED", "on time");
  }

  if (!isHqOnBoard(G.board, "RED")) {
    return newWinner("BLUE", "by HQ capture");
  }
  if (!isHqOnBoard(G.board, "BLUE")) {
    return newWinner("RED", "by HQ capture");
  }

  return undefined;
}

function newWinner(player: Player, reason: string): GameoverState {
  return {
    status: "WIN",
    winner: player,
    reason,
  };
}

export function isHqOnBoard(board: GHQState["board"], player: Player): boolean {
  return board.some((rows) =>
    rows.some((square) => square?.type === "HQ" && square.player === player)
  );
}
