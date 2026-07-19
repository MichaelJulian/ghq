import assert from "node:assert/strict";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  nativeBundleMismatches,
  verifyNativeBundle,
} from "./verify_native_bundle.mjs";

const pairs = [{ canonical: "canonical.py", production: "api/bundle.py" }];

async function fixture(context, canonical, production) {
  const root = await mkdtemp(path.join(os.tmpdir(), "ghq-native-bundle-"));
  context.after(() => rm(root, { recursive: true }));
  await mkdir(path.join(root, "api"));
  await Promise.all([
    writeFile(path.join(root, "canonical.py"), canonical),
    writeFile(path.join(root, "api/bundle.py"), production),
  ]);
  return root;
}

test("accepts byte-identical canonical and production files", async (context) => {
  const root = await fixture(context, "same\n", "same\n");

  assert.deepEqual(await nativeBundleMismatches(root, pairs), []);
});

test("reports paths and hashes when production drifts", async (context) => {
  const root = await fixture(context, "canonical\n", "stale\n");

  const mismatches = await nativeBundleMismatches(root, pairs);
  assert.equal(mismatches.length, 1);
  assert.match(mismatches[0], /api\/bundle\.py does not match canonical\.py/);
  await assert.rejects(
    verifyNativeBundle(root, pairs),
    /Native production bundle drift detected/
  );
});
