"use client";

import { GHQBoardV3 } from "@/components/board-v3/boardv3";
import { Suspense, useMemo, useEffect } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { nanoid } from "nanoid";

function BotPage() {
  const searchParams = useSearchParams();
  const router = useRouter();
  const fen = useMemo(
    () => searchParams.get("fen") ?? undefined,
    [searchParams]
  );
  const id = useMemo(() => searchParams.get("id") ?? nanoid(), [searchParams]);

  useEffect(() => {
    if (!searchParams.get("id")) {
      const params = new URLSearchParams(searchParams.toString());
      params.set("id", id);
      router.replace(`?${params.toString()}`);
    }
  }, [id, searchParams, router]);

  return (
    <div>
      <GHQBoardV3 fen={fen} bot={true} playerId="0" id={id} />
    </div>
  );
}

export default function Page() {
  return (
    <Suspense fallback={null}>
      <BotPage />
    </Suspense>
  );
}
