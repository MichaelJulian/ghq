/** @jest-environment node */

import { afterEach, describe, expect, it } from "@jest/globals";
import { nativeSearchUrl } from "./native-search";

const ORIGINAL = {
  configured: process.env.GHQ_NATIVE_SEARCH_URL,
  environment: process.env.VERCEL_ENV,
  deployment: process.env.VERCEL_URL,
  production: process.env.VERCEL_PROJECT_PRODUCTION_URL,
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

  it("targets the public production project domain", () => {
    delete process.env.GHQ_NATIVE_SEARCH_URL;
    process.env.VERCEL_ENV = "production";
    process.env.VERCEL_URL = "deployment.example.test";
    process.env.VERCEL_PROJECT_PRODUCTION_URL = "public.example.test";
    expect(nativeSearchUrl()).toBe(
      "https://public.example.test/api/native_search"
    );
  });

  it("does not fall back to a protected immutable production domain", () => {
    delete process.env.GHQ_NATIVE_SEARCH_URL;
    delete process.env.VERCEL_PROJECT_PRODUCTION_URL;
    process.env.VERCEL_ENV = "production";
    process.env.VERCEL_URL = "protected.example.test";
    expect(nativeSearchUrl()).toBeUndefined();
  });
});
