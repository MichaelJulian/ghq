import { createHash } from "node:crypto";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

export const NATIVE_BUNDLE_PAIRS = [
  { canonical: "public/engine.py", production: "api/_engine.py" },
  { canonical: "scripts/ghq_ai.py", production: "api/_ghq_ai.py" },
];

function sha256(contents) {
  return createHash("sha256").update(contents).digest("hex");
}

export async function nativeBundleMismatches(
  rootDir = process.cwd(),
  pairs = NATIVE_BUNDLE_PAIRS
) {
  const mismatches = [];
  for (const pair of pairs) {
    const canonicalPath = path.join(rootDir, pair.canonical);
    const productionPath = path.join(rootDir, pair.production);
    const [canonical, production] = await Promise.all([
      readFile(canonicalPath),
      readFile(productionPath),
    ]);
    if (!canonical.equals(production)) {
      mismatches.push(
        `${pair.production} does not match ${pair.canonical} ` +
          `(${sha256(production)} != ${sha256(canonical)})`
      );
    }
  }
  return mismatches;
}

export async function verifyNativeBundle(
  rootDir = process.cwd(),
  pairs = NATIVE_BUNDLE_PAIRS
) {
  const mismatches = await nativeBundleMismatches(rootDir, pairs);
  if (mismatches.length) {
    throw new Error(
      `Native production bundle drift detected:\n${mismatches
        .map((message) => `- ${message}`)
        .join("\n")}`
    );
  }
}

const invokedPath = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (invokedPath === fileURLToPath(import.meta.url)) {
  verifyNativeBundle()
    .then(() => {
      console.log("Native production bundle matches canonical Python sources.");
    })
    .catch((error) => {
      console.error(error instanceof Error ? error.message : error);
      process.exitCode = 1;
    });
}
