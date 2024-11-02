import React, { useCallback, useEffect, useMemo, useRef } from "react";
import {
  GHQState,
  Orientation,
  Player,
  ReserveFleet,
  Units,
} from "@/game/engine";
import { BoardProps } from "boardgame.io/react";
import { useMachine } from "@xstate/react";
import { turnStateMachine } from "@/game/board-state";
import classNames from "classnames";
import { useHotkeys } from "react-hotkeys-hook";
import { bombardedSquares } from "@/game/move-logic";
import Image from "next/image";
import { SelectOrientation } from "@/game/select-orientation";

const rows = 8;
const columns = 8;
//coordinate string x,y
type Annotations = {
  [key: string]: {
    moveTo?: true;
    bombardedBy?: { RED?: true; BLUE?: true };
    selectedPiece?: true;
  };
};

export function GHQBoard({ ctx, G, moves }: BoardProps<GHQState>) {
  const divRef = useRef(null); // Create a ref

  const [state, send] = useMachine(
    turnStateMachine.provide({
      actions: {
        movePiece: ({ context, event }) => {
          if ("at" in event) moves.Move(context.selectedPiece!.at, event.at);
        },
        changeOrientation: ({ context, event }) => {
          if ("orientation" in event)
            moves.ChangeOrientation(
              context.selectedPiece!.at,
              event.orientation
            );
        },
        spawnPiece: ({ context, event }) => {
          if (context.unitKind && "at" in event)
            moves.Reinforce(context.unitKind, event.at);
        },
      },
    })
  );

  useHotkeys("escape", () => send({ type: "DESELECT" }), [send]);

  useEffect(() => {
    send({
      type: "START_TURN",
      player: ctx.currentPlayer === "0" ? "RED" : "BLUE",
    });
  }, [ctx.currentPlayer]);

  const selectReserve = useCallback(
    (kind: keyof ReserveFleet) => {
      send({
        type: "SELECT_RESERVE_PIECE",
        currentBoard: G.board,
        reserve: ctx.currentPlayer === "0" ? G.redReserve : G.blueReserve,
        kind,
      });
    },
    [G.board, G.redReserve, G.blueReserve]
  );

  const annotations = useMemo(() => {
    const annotate: Annotations = {};

    (state.context.allowedMoves || []).forEach((i) => {
      annotate[`${i[0]},${i[1]}`] = annotate[`${i[0]},${i[1]}`]
        ? { ...annotate[`${i[0]},${i[1]}`], moveTo: true }
        : { moveTo: true };
    });

    const bombarded = bombardedSquares(G.board);
    Object.entries(bombarded).forEach(([key, value]) => {
      if (annotate[key]) {
        annotate[key].bombardedBy = value;
      } else {
        annotate[key] = { bombardedBy: value };
      }
    });

    const { selectedPiece } = state.context;
    if (selectedPiece) {
      annotate[`${selectedPiece.at[0]},${selectedPiece.at[1]}`] = {
        ...annotate[`${selectedPiece.at[0]},${selectedPiece.at[1]}`],
        selectedPiece: true,
      };
    }

    return annotate;
  }, [state.context]);

  const cells = Array.from({ length: rows }).map((_, rowIndex) => (
    <tr key={rowIndex}>
      {Array.from({ length: columns }).map((_, colIndex) => {
        const square = G.board[rowIndex][colIndex];

        const annotationsForSquare = annotations[`${rowIndex},${colIndex}`];

        const rotation =
          square && "orientation" in square
            ? square && square.orientation
            : undefined;

        const add180 =
          square &&
          ((ctx.currentPlayer === "0" && square.player === "BLUE") ||
            (ctx.currentPlayer === "1" && square.player === "RED"));

        const bombardmentClass =
          annotationsForSquare && annotationsForSquare.bombardedBy
            ? annotationsForSquare.bombardedBy
              ? annotationsForSquare.bombardedBy.BLUE &&
                annotationsForSquare.bombardedBy.RED
                ? "stripe-red-blue"
                : annotationsForSquare.bombardedBy.BLUE
                ? "stripe-blue-transparent"
                : annotationsForSquare.bombardedBy.RED
                ? "stripe-red-transparent"
                : ""
              : ""
            : "";

        return (
          <td
            onClick={() => {
              if (square) {
                send({
                  type: "SELECT_ACTIVE_PIECE",
                  at: [rowIndex, colIndex],
                  piece: square,
                  currentBoard: G.board,
                });
              } else {
                send({
                  type: "SELECT_SQUARE",
                  at: [rowIndex, colIndex],
                  currentBoard: G.board,
                });
              }
            }}
            key={colIndex}
            className={classNames("relative", bombardmentClass, {
              ["cursor-pointer"]:
                annotationsForSquare?.moveTo ||
                square?.player === (ctx.currentPlayer === "0" ? "RED" : "BLUE"),
            })}
            style={{
              border: "1px solid black",
              boxShadow:
                annotationsForSquare?.selectedPiece && "inset 0 0 8px darkgray",
              textAlign: "center",
              width: "90px",
              height: "90px",
            }}
          >
            {square ? (
              <div
                className={classNames(
                  "flex items-center justify-center select-none font-bold text-3xl",
                  square.player === "RED" ? "text-red-600" : "text-blue-600",
                  {
                    // @todo this is really only for infantry. Adjust when we do orientation
                    // ["rotate-180"]:
                    //   (ctx.currentPlayer === "0" && square.player === "BLUE") ||
                    //   (ctx.currentPlayer === "1" && square.player === "RED"),
                  }
                )}
              >
                <Image
                  src={`/${Units[square.type].imagePathPrefix}-${
                    square.player
                  }.png`}
                  width="64"
                  height="64"
                  className="select-none"
                  draggable="false"
                  style={{
                    transform: square.orientation
                      ? ctx.currentPlayer === "1"
                        ? `rotate(${180 - square.orientation}deg)`
                        : `rotate(${square.orientation}deg)`
                      : `rotate(${add180 ? 180 : 0}deg)`,
                  }}
                  alt={Units[square.type].imagePathPrefix}
                />
              </div>
            ) : null}
            {square &&
            Units[square.type].artilleryRange &&
            annotationsForSquare?.selectedPiece ? (
              <SelectOrientation
                player={square.player}
                onChange={(orientation: Orientation) => {
                  send({
                    type: "CHANGE_ORIENTATION",
                    orientation: orientation,
                  });
                }}
              />
            ) : null}
            {annotationsForSquare?.moveTo ? (
              <div className="rounded-full w-8 h-8 m-auto bg-gray-300" />
            ) : null}
          </td>
        );
      })}
    </tr>
  ));

  return (
    <div className="grid bg-gray-200 absolute w-full h-full grid-cols-7">
      <div className="col-span-5 border-r-2 border-gray-100 flex items-center justify-center">
        <table ref={divRef} style={{ borderCollapse: "collapse" }}>
          {/*flip board*/}
          <tbody>{ctx.currentPlayer === "0" ? cells : cells.reverse()}</tbody>
        </table>
      </div>
      <div
        className={classNames(
          "col-span-2 bg-white pt-10 flex flex-col ",
          ctx.currentPlayer === "0" ? "bg-red-50" : "bg-blue-50",
          { ["flex-col-reverse"]: ctx.currentPlayer === "1" }
        )}
      >
        <div className="flex-1 items-center justify-center flex">
          <ReserveBank
            player="BLUE"
            reserve={G.blueReserve}
            selectable={
              ctx.currentPlayer === "1" && !state.matches("activePieceSelected")
            }
            selectReserve={selectReserve}
          />
        </div>
        <h2
          className={classNames(
            "text-center font-semibold text-2xl",
            ctx.currentPlayer === "0" ? "text-red-500" : "text-blue-500"
          )}
        >
          {ctx.currentPlayer === "0" ? "Red's " : "Blue's"} Turn
          <div className="text-lg text-gray-600 font-mono">
            {3 - ctx.numMoves!} remaining move{ctx.numMoves !== 2 ? "s" : ""}{" "}
            <button
              onClick={() => moves.Skip()}
              className="bg-black text-white p-0.5 text-sm px-2"
            >
              Skip
            </button>
          </div>
        </h2>
        <div className="flex-1 items-center justify-center flex">
          <ReserveBank
            player="RED"
            reserve={G.redReserve}
            selectable={
              ctx.currentPlayer === "0" && !state.matches("activePieceSelected")
            }
            selectReserve={selectReserve}
          />
        </div>
      </div>
    </div>
  );
}

function ReserveBank(props: {
  player: Player;
  reserve: ReserveFleet;
  selectable: boolean;
  selectReserve: (kind: keyof ReserveFleet) => void;
}) {
  const kinds = [
    "INFANTRY",
    "ARMORED_INFANTRY",
    "AIRBORNE_INFANTRY",
    "ARTILLERY",
    "ARMORED_ARTILLERY",
    "HEAVY_ARTILLERY",
  ] as (keyof ReserveFleet)[];

  const reserves = kinds.flatMap((kind) => {
    const count = props.reserve[kind as keyof ReserveFleet];

    return Array.from({ length: count }, (_, index) => (
      <div
        onClick={() => {
          props.selectReserve(kind);
        }}
        key={`${kind}-${index++}`}
        className={[
          "flex items-end justify-end col-span-1 select-none font-bold text-3xl p-1",
          props.player === "RED" ? "text-red-600" : "text-blue-600",
          { ["cursor-pointer"]: props.selectable },
        ].join(" ")}
      >
        <Image
          src={`/${Units[kind].imagePathPrefix}-${props.player}.png`}
          width="64"
          height="64"
          alt={Units[kind].imagePathPrefix}
        />
      </div>
    ));
  });

  return <div className="grid grid-cols-8 text-center">{reserves}</div>;
}
