"use client";

import { useMemo } from "react";
import type { Square as GhqSquare } from "@/game/engine";
import { FENtoBoardState } from "@/game/notation";
import { bombardedSquares } from "@/game/move-logic";
import { pieceSizes, squareSizes } from "@/game/constants";
import Square, { type SquareState } from "@/components/board-v3/Square";
import { ReserveBankV2 } from "@/components/board-v3/ReserveBankV2";

export function PositionBoard({ fen }: { fen: string }) {
  const parsed = useMemo(() => {
    try {
      return { state: FENtoBoardState(fen) };
    } catch (error) {
      return {
        error: error instanceof Error ? error.message : "Invalid GHQ FEN",
      };
    }
  }, [fen]);

  if (!parsed.state) {
    return (
      <div className="flex h-[360px] w-[360px] items-center justify-center rounded border border-red-300 bg-red-50 p-6 text-center text-sm text-red-700">
        {parsed.error}
      </div>
    );
  }

  const { board, redReserve, blueReserve, currentPlayerTurn } = parsed.state;
  const bombarded = bombardedSquares(board);
  return (
    <div className="flex w-[360px] flex-col gap-2" aria-label="GHQ position">
      <StaticReserve
        label="Blue reserves"
        player="BLUE"
        reserve={blueReserve}
      />
      <div className="overflow-hidden rounded border border-slate-600 shadow-lg">
        {board.map((row, rowIndex) => (
          <div key={rowIndex} className="flex">
            {row.map((square, colIndex) => (
              <Square
                key={colIndex}
                squareSize={squareSizes.small}
                pieceSize={pieceSizes.small}
                squareState={staticSquareState(
                  rowIndex,
                  colIndex,
                  square,
                  bombarded[`${rowIndex},${colIndex}`]
                )}
                isFlipped={false}
              />
            ))}
          </div>
        ))}
      </div>
      <StaticReserve label="Red reserves" player="RED" reserve={redReserve} />
      <div className="text-center text-xs font-medium uppercase tracking-[0.2em] text-slate-500">
        {currentPlayerTurn} to move
      </div>
    </div>
  );
}

function StaticReserve({
  label,
  player,
  reserve,
}: {
  label: string;
  player: "RED" | "BLUE";
  reserve: ReturnType<typeof FENtoBoardState>["redReserve"];
}) {
  return (
    <div>
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div className="flex min-h-10 items-center rounded border bg-white/80 px-2 py-1">
        <ReserveBankV2
          player={player}
          reserve={reserve}
          selectable={false}
          selectReserve={() => {}}
          squareSize={34}
          hideHQ
        />
      </div>
    </div>
  );
}

function staticSquareState(
  rowIndex: number,
  colIndex: number,
  square: GhqSquare,
  bombardment?: { RED?: true; BLUE?: true }
): SquareState {
  return {
    rowIndex,
    colIndex,
    square,
    stagedSquare: null,
    isRedBombarded: bombardment?.RED === true,
    isBlueBombarded: bombardment?.BLUE === true,
    isSelected: false,
    isCaptureCandidate: false,
    isBombardCandidate: false,
    isHighlightedBombardCandidate: false,
    showTarget: false,
    wasRecentlyCapturedPiece: undefined,
    wasRecentlyMovedTo: false,
    isMovable: false,
    isRightClicked: false,
    isHovered: false,
    isMidMove: false,
    shouldAnimateTo: undefined,
    engagedOrientation: undefined,
  };
}
