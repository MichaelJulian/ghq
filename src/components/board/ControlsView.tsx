import { Redo, Repeat, SkipForward, Undo } from "lucide-react";
import { Button } from "../ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import useControls from "./Controls";
import { BoardProps } from "boardgame.io/react";
import { GHQState } from "@/game/engine";
import { useCallback, useMemo } from "react";
import { Ctx, LogEntry } from "boardgame.io";

export default function ControlsView({
  ctx,
  undo,
  redo,
  replay,
  moves,
  isMyTurn,
  log,
  cancel,
}: {
  ctx: Ctx;
  undo: () => void;
  redo: () => void;
  replay: () => void;
  cancel: () => void;
  log: LogEntry[];
  moves: BoardProps<GHQState>["moves"];
  isMyTurn: boolean;
}) {
  const lastLog = useMemo(() => log.slice(-1)[0], [log]);
  const canUndo = useMemo(
    () => isMyTurn && ctx.numMoves && ctx.numMoves >= 0,
    [ctx.numMoves, isMyTurn]
  );
  const canRedo = useMemo(
    () => isMyTurn && lastLog && lastLog.action.type === "UNDO",
    [lastLog, isMyTurn]
  );
  const canSkip = useMemo(
    () => isMyTurn && ctx.numMoves && ctx.numMoves > 0,
    [ctx.numMoves, isMyTurn]
  );
  const canReplay = useMemo(
    () => isMyTurn && ctx.turn > 1,
    [ctx.turn, isMyTurn]
  );

  const doUndo = useCallback(() => {
    if (canUndo) {
      undo();
    }
    cancel();
  }, [canUndo, undo, cancel]);

  const doRedo = useCallback(() => {
    if (canRedo) {
      redo();
    }
  }, [canRedo, redo]);

  const doSkip = useCallback(() => {
    if (canSkip) {
      moves.Skip();
    }
  }, [canSkip, moves]);

  const doReplay = useCallback(() => {
    if (canReplay) {
      replay();
    }
    cancel();
  }, [replay, cancel]);

  useControls({
    undo: doUndo,
    redo: doRedo,
    cancel,
    skip: doSkip,
    replay: doReplay,
  });

  return (
    <div className="flex gap-1 m-1">
      <ActionButton
        Icon={Undo}
        tooltip="Undo the most recent move you made this turn (shortcut: left-arrow)"
        onClick={doUndo}
        disabled={!canUndo}
      />
      <ActionButton
        Icon={Redo}
        tooltip="Redo the most recent move you undid this turn (shortcut: right-arrow)"
        onClick={doRedo}
        disabled={!canRedo}
      />
      <ActionButton
        Icon={SkipForward}
        tooltip='Skip the remainder of your turn after making at least one move (shortcut: ".")'
        onClick={doSkip}
        disabled={!canSkip}
      />
      <ActionButton
        Icon={Repeat}
        tooltip="Replay the animation of your opponent's most recent turn (shortcut: spacebar)"
        onClick={doReplay}
        disabled={!canReplay}
      />
    </div>
  );
}

function ActionButton({
  Icon,
  tooltip,
  onClick,
  disabled,
}: {
  Icon: React.FC;
  tooltip: string;
  onClick: () => void;
  disabled: boolean;
}) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            className="w-full"
            onClick={onClick}
            disabled={disabled}
          >
            <Icon />
          </Button>
        </TooltipTrigger>
        <TooltipContent>
          <p>{tooltip}</p>
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}
