import path from "path";
import { readFile } from "fs/promises";
import type { GameEngine } from "@/game/engine-v2";
import { loadServerPyodide } from "@/server/pyodide";

export async function loadV2Engine(): Promise<GameEngine> {
  const pyodide = await loadServerPyodide();

  const engineCode = await readFile(
    path.join(__dirname, "../../public/engine.py"),
    "utf8"
  );
  pyodide.FS.writeFile("engine.py", new TextEncoder().encode(engineCode));
  return pyodide.pyimport("engine");
}
