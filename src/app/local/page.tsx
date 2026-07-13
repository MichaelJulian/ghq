"use client";

import { GHQBoardV3 } from "@/components/board-v3/boardv3";
import { useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";

function LocalPage() {
  const searchParams = useSearchParams();
  const fen = useMemo(
    () => searchParams.get("fen") ?? undefined,
    [searchParams]
  );
  return (
    <div>
      <GHQBoardV3 isPassAndPlayMode={true} fen={fen} />
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <LocalPage />
    </Suspense>
  );
}
