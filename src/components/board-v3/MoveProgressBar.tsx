"use client";

import React from "react";

interface MoveProgressBarProps {
  numMoves: number;
  maxMoves?: number;
}

export default function MoveProgressBar({
  numMoves,
  maxMoves = 3,
}: MoveProgressBarProps) {
  const progress = Math.min((numMoves / maxMoves) * 100, 100);

  return (
    <div className="w-full h-1 bg-white overflow-hidden">
      <div
        className="h-full bg-blue-500 transition-all duration-300 ease-in-out"
        style={{ width: `${progress}%` }}
      />
    </div>
  );
}
