/** @jest-environment node */

import { afterEach, describe, expect, it, jest } from "@jest/globals";
import { nativeSearchUrl, searchNatively } from "./native-search";

const ORIGINAL = {
  configured: process.env.GHQ_NATIVE_SEARCH_URL,
  environment: process.env.VERCEL_ENV,
  deployment: process.env.VERCEL_URL,
  production: process.env.VERCEL_PROJECT_PRODUCTION_URL,
  commit: process.env.VERCEL_GIT_COMMIT_SHA,
  oidc: process.env.VERCEL_OIDC_TOKEN,
};

afterEach(() => {
  if (ORIGINAL.configured === undefined) delete process.env.GHQ_NATIVE_SEARCH_URL;
  else process.env.GHQ_NATIVE_SEARCH_URL = ORIGINAL.configured;
  if (ORIGINAL.environment === undefined) delete process.env.VERCEL_ENV;
  else process.env.VERCEL_ENV = ORIGINAL.environment;
  if (ORIGINAL.deployment === undefined) delete process.env.VERCEL_URL;
  else process.env.VERCEL_URL = ORIGINAL.deployment;
  if (ORIGINAL.production === undefined)
    delete process.env.VERCEL_PROJECT_PRODUCTION_URL;
  else process.env.VERCEL_PROJECT_PRODUCTION_URL = ORIGINAL.production;
  if (ORIGINAL.commit === undefined) delete process.env.VERCEL_GIT_COMMIT_SHA;
  else process.env.VERCEL_GIT_COMMIT_SHA = ORIGINAL.commit;
  if (ORIGINAL.oidc === undefined) delete process.env.VERCEL_OIDC_TOKEN;
  else process.env.VERCEL_OIDC_TOKEN = ORIGINAL.oidc;
  jest.restoreAllMocks();
});

describe("native search source attestation", () => {
  it("rejects a native function from a stale deployment", async () => {
    process.env.VERCEL_GIT_COMMIT_SHA = "current-revision";
    jest.spyOn(global, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          codeVersion: "stale-revision",
          search: {
            search: {
              backend: "native-python",
              value_model_backend: "native-gbdt",
              value_model_version: "incumbent",
              code_version: "stale-revision",
            },
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );

    await expect(
      searchNatively(
        "https://example.test/api/native_search",
        { fen: "position" },
        {
          personality: "balanced",
          turnNumber: 1,
          timeMs: 1_000,
          maxDepth: 2,
          beamWidth: 6,
          openingSeed: 1,
          maxActions: 3,
          stagnationTurns: 0,
          valueModel: "incumbent",
        }
      )
    ).rejects.toThrow(
      "expected current-revision, received stale-revision"
    );
  });
});

describe("native search endpoint resolution", () => {
  it("uses explicit local or preview configuration", () => {
    process.env.GHQ_NATIVE_SEARCH_URL = "http://127.0.0.1:9000/api/";
    expect(nativeSearchUrl()).toBe("http://127.0.0.1:9000/api");
  });

  it("does not self-fetch through protected preview SSO", () => {
    delete process.env.GHQ_NATIVE_SEARCH_URL;
    process.env.VERCEL_ENV = "preview";
    process.env.VERCEL_URL = "preview.example.test";
    expect(nativeSearchUrl()).toBeUndefined();
  });

  it("targets the immutable production deployment", () => {
    delete process.env.GHQ_NATIVE_SEARCH_URL;
    process.env.VERCEL_ENV = "production";
    process.env.VERCEL_URL = "deployment.example.test";
    process.env.VERCEL_PROJECT_PRODUCTION_URL = "public.example.test";
    expect(nativeSearchUrl()).toBe(
      "https://deployment.example.test/api/native_search"
    );
  });

  it("falls back to the public project domain without deployment metadata", () => {
    delete process.env.GHQ_NATIVE_SEARCH_URL;
    delete process.env.VERCEL_URL;
    process.env.VERCEL_ENV = "production";
    process.env.VERCEL_PROJECT_PRODUCTION_URL = "public.example.test";
    expect(nativeSearchUrl()).toBe(
      "https://public.example.test/api/native_search"
    );
  });

  it("authenticates immutable deployment requests with the Vercel OIDC token", async () => {
    process.env.VERCEL_ENV = "production";
    process.env.VERCEL_URL = "deployment.example.test";
    process.env.VERCEL_GIT_COMMIT_SHA = "current-revision";
    const payload = Buffer.from(
      JSON.stringify({ exp: Math.floor(Date.now() / 1_000) + 3_600 })
    ).toString("base64url");
    process.env.VERCEL_OIDC_TOKEN = `eyJhbGciOiJub25lIn0.${payload}.signature`;
    const fetchMock = jest.spyOn(global, "fetch").mockResolvedValue(
      new Response(
        JSON.stringify({
          codeVersion: "current-revision",
          search: {
            search: {
              backend: "native-python",
              value_model_backend: "native-gbdt",
              value_model_version: "incumbent",
              code_version: "current-revision",
            },
          },
        }),
        { status: 200, headers: { "content-type": "application/json" } }
      )
    );

    await searchNatively(
      nativeSearchUrl()!,
      { fen: "position" },
      {
        personality: "balanced",
        turnNumber: 1,
        timeMs: 1_000,
        maxDepth: 2,
        beamWidth: 6,
        openingSeed: 1,
        maxActions: 3,
        stagnationTurns: 0,
        valueModel: "incumbent",
      }
    );

    expect(fetchMock).toHaveBeenCalledWith(
      "https://deployment.example.test/api/native_search",
      expect.objectContaining({
        headers: expect.objectContaining({
          "x-vercel-trusted-oidc-idp-token":
            process.env.VERCEL_OIDC_TOKEN,
        }),
      })
    );
  });
});
