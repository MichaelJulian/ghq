import { realpath } from "fs/promises";
import path from "path";
import { loadPyodide } from "pyodide";

/**
 * Next rewrites module locations while bundling server components. Letting
 * Pyodide infer its data directory (or using require.resolve here) can
 * therefore point at Next's virtual `(rsc)` directory instead of the package
 * on disk. Resolve from the deployed application root so its JS, WASM, lock
 * file, and standard library all come from the traced pyodide package.
 */
export async function loadServerPyodide() {
  const packageDirectory = await realpath(
    path.join(process.cwd(), "node_modules", "pyodide")
  );
  return loadPyodide({ indexURL: `${packageDirectory}${path.sep}` });
}
