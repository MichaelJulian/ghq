import { Coordinate, GHQState, Player, Units } from "@/game/engine";
import type { Ctx } from "boardgame.io";
import { bombardedSquares } from "@/game/move-logic";

export function captureCandidates(
  lastMovedInfantry: Coordinate,
  board: GHQState["board"]
): Coordinate[] {
  const engagedPairs = maximizeEngagement(lastMovedInfantry, board);
  const attacker = board[lastMovedInfantry[0]][lastMovedInfantry[1]];

  if (!attacker) {
    throw new Error("No piece at the last moved infantry position");
  }

  const engagedInfantry: Record<string, Player> = {};
  for (const pairs of engagedPairs) {
    engagedInfantry[`${pairs.RED[0]},${pairs.RED[1]}`] = "RED";
    engagedInfantry[`${pairs.BLUE[0]},${pairs.BLUE[1]}`] = "BLUE";
  }

  // Find the adjacent pieces to the last moved infantry
  const attackerAdjacentPieces: Coordinate[] = [];
  const directions = [
    [-1, 0], // Up
    [1, 0], // Down
    [0, -1], // Left
    [0, 1], // Right
  ];

  for (const [dx, dy] of directions) {
    const x = lastMovedInfantry[0] + dx;
    const y = lastMovedInfantry[1] + dy;

    if (x >= 0 && x < board.length && y >= 0 && y < board[0].length) {
      attackerAdjacentPieces.push([x, y]);
    }
  }

  // Find out if the opponent has any unengaged infantry near us, if so we need to engage with it
  for (const coord of attackerAdjacentPieces) {
    const piece = board[coord[0]][coord[1]];
    if (
      piece &&
      piece.player !== attacker.player &&
      piece.type === "INFANTRY" &&
      !engagedInfantry[`${coord[0]},${coord[1]}`]
    ) {
      return [];
    }
  }

  // Filter for only already-engaged infantry of the opponent or non-infantry pieces
  const attackablePieces = attackerAdjacentPieces.filter((coord) => {
    const piece = board[coord[0]][coord[1]];
    return (
      piece &&
      (engagedInfantry[`${coord[0]},${coord[1]}`] ||
        piece.type !== "INFANTRY") &&
      piece.player !== attacker.player
    );
  });

  return attackablePieces;
}

function maximizeEngagement(
  excludeCoordinate: Coordinate,
  board: GHQState["board"]
): Record<Player, Coordinate>[] {
  const N = 8; // Size of the board

  // Arrays to hold the positions of '0' and '1' pieces
  const pieces0: { x: number; y: number }[] = [];
  const pieces1: { x: number; y: number }[] = [];

  // Maps to assign an index to each piece for easy reference
  const piece0Index: Record<string, number> = {};
  const piece1Index: Record<string, number> = {};

  let index0 = 0;
  let index1 = 0;

  // Populate pieces0 and pieces1 arrays and their indices
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      // Skip the last moved piece, we want to calculate engagement as if they weren't there
      if (excludeCoordinate[0] === i && excludeCoordinate[1] === j) {
        continue;
      }

      const unit = board[i]?.[j];
      if (!unit) {
        continue;
      }

      // Only infantry can be engaged
      if (unit.type !== "INFANTRY") {
        continue;
      }

      if (unit.player === "RED") {
        pieces0.push({ x: i, y: j });
        piece0Index[`${i},${j}`] = index0++;
      } else if (unit.player === "BLUE") {
        pieces1.push({ x: i, y: j });
        piece1Index[`${i},${j}`] = index1++;
      }
    }
  }

  // Build the adjacency list for pieces of '0' to their adjacent '1's
  const adjList: number[][] = Array.from({ length: pieces0.length }, () => []);
  const directions = [
    [-1, 0], // Up
    [1, 0], // Down
    [0, -1], // Left
    [0, 1], // Right
  ];

  for (let i = 0; i < pieces0.length; i++) {
    const { x: x0, y: y0 } = pieces0[i];

    for (const [dx, dy] of directions) {
      const x1 = x0 + dx;
      const y1 = y0 + dy;

      // Check if the adjacent position is within bounds and has a '1'
      if (
        x1 >= 0 &&
        x1 < N &&
        y1 >= 0 &&
        y1 < N &&
        board[x1]?.[y1]?.player === "BLUE"
      ) {
        if (excludeCoordinate[0] === x1 && excludeCoordinate[1] === y1) {
          continue;
        }
        const idx1 = piece1Index[`${x1},${y1}`];
        adjList[i].push(idx1);
      }
    }
  }

  // Initialize matching array for pieces of '1'
  const matchTo1 = Array(pieces1.length).fill(-1);

  // Helper function to perform DFS and find augmenting paths
  function bpm(u: number, seen: boolean[]) {
    for (const v of adjList[u]) {
      if (!seen[v]) {
        seen[v] = true;

        // If the '1' piece is not matched or can be rematched
        if (matchTo1[v] === -1 || bpm(matchTo1[v], seen)) {
          matchTo1[v] = u;
          return true;
        }
      }
    }
    return false;
  }

  // Attempt to find a matching for each '0' piece
  for (let u = 0; u < pieces0.length; u++) {
    const seen = Array(pieces1.length).fill(false);
    bpm(u, seen);
  }

  // Build the list of engaged pairs
  const engagedPairs: Record<Player, Coordinate>[] = [];
  for (let v = 0; v < matchTo1.length; v++) {
    if (matchTo1[v] !== -1) {
      const u = matchTo1[v];
      const piece0 = pieces0[u];
      const piece1 = pieces1[v];
      engagedPairs.push({
        RED: [piece0.x, piece0.y],
        BLUE: [piece1.x, piece1.y],
      });
    }
  }

  // Return the list of engaged pairs
  return engagedPairs;
}

export function clearBombardedSquares(G: GHQState, ctx: Ctx) {
  const bombarded = bombardedSquares(G.board);

  G.board.forEach((rows, x) => {
    rows.forEach((square, y) => {
      const bombardedSquare = bombarded[`${x},${y}`];

      // If there is nothing here or it's not bombarded, do nothing.
      if (!square || !bombardedSquare) {
        return;
      }

      // If it's our turn, and we bombard the square, and the square is occupied by an enemy piece, remove it.
      const currentPlayerColor = ctx.currentPlayer === "0" ? "RED" : "BLUE";
      if (
        bombardedSquare[currentPlayerColor] &&
        square.player !== currentPlayerColor
      ) {
        G.board[x][y] = null;
      }
    });
  });

  // TODO: add log message for user indicating that a bombarded piece was destroyed
}