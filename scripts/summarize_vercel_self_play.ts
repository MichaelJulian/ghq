import "dotenv/config";
import { get, list, type ListBlobResultBlob } from "@vercel/blob";

interface StoredGame {
  gameId?: string;
  generationId: string;
  decisions: Array<{
    completedDepth: number;
    fallback: "none" | "safe" | "greedy" | "seeded";
    timedOut: boolean;
    completedTurn?: boolean;
    player?: "RED" | "BLUE";
    selectedMoves?: string[];
    turnNumber?: number;
    fen?: string;
  }>;
  outcome: { winner?: "RED" | "BLUE"; termination: string };
  trainingPositions: number;
}

async function listGameBlobs(): Promise<ListBlobResultBlob[]> {
  const blobs: ListBlobResultBlob[] = [];
  let cursor: string | undefined;
  do {
    const page = await list({
      prefix: "self-play/generations/",
      cursor,
      limit: 1000,
    });
    blobs.push(
      ...page.blobs.filter(
        (blob) =>
          blob.pathname.includes("/games/") && blob.pathname.endsWith(".json")
      )
    );
    cursor = page.hasMore ? page.cursor : undefined;
  } while (cursor);
  return blobs;
}

async function readGame(blob: ListBlobResultBlob): Promise<StoredGame> {
  const result = await get(blob.pathname, {
    access: "private",
    useCache: false,
  });
  if (!result || result.statusCode !== 200 || !result.stream) {
    throw new Error(`Unable to read ${blob.pathname}`);
  }
  return JSON.parse(await new Response(result.stream).text()) as StoredGame;
}

async function main() {
  const blobs = await listGameBlobs();
  const games: StoredGame[] = [];
  for (let index = 0; index < blobs.length; index += 8) {
    games.push(
      ...(await Promise.all(blobs.slice(index, index + 8).map(readGame)))
    );
  }
  const decisions = games.flatMap((game) => game.decisions);
  const countBy = (values: string[]) =>
    Object.fromEntries(
      [...new Set(values)]
        .sort()
        .map((value) => [
          value,
          values.filter((candidate) => candidate === value).length,
        ])
    );
  const byGeneration = Object.fromEntries(
    [...new Set(games.map((game) => game.generationId))]
      .sort()
      .map((generationId) => {
        const generationGames = games.filter(
          (game) => game.generationId === generationId
        );
        return [
          generationId,
          {
            games: generationGames.length,
            decisive: generationGames.filter((game) => game.outcome.winner)
              .length,
            trainingPositions: generationGames.reduce(
              (sum, game) => sum + game.trainingPositions,
              0
            ),
          },
        ];
      })
  );

  console.log(
    JSON.stringify(
      {
        games: games.length,
        outcomes: countBy(games.map((game) => game.outcome.winner ?? "DRAW")),
        terminations: countBy(games.map((game) => game.outcome.termination)),
        decisions: decisions.length,
        trainingPositions: games.reduce(
          (sum, game) => sum + game.trainingPositions,
          0
        ),
        fallbackDecisions: decisions.filter(
          (decision) => decision.fallback !== "none"
        ).length,
        timedOutDecisions: decisions.filter((decision) => decision.timedOut)
          .length,
        depth: countBy(
          decisions.map((decision) => String(decision.completedDepth))
        ),
        incompleteTurnFragments: decisions.filter(
          (decision) => decision.completedTurn === false
        ).length,
        incompleteExamples: games
          .filter((game) => game.outcome.termination === "incomplete-turn")
          .slice(0, 5)
          .map((game) => ({
            gameId: game.gameId,
            finalDecisions: game.decisions.slice(-5).map((decision) => ({
              turnNumber: decision.turnNumber,
              player: decision.player,
              fen: decision.fen,
              moves: decision.selectedMoves,
              fallback: decision.fallback,
              completedTurn: decision.completedTurn,
            })),
          })),
        byGeneration,
      },
      null,
      2
    )
  );
}

void main();
