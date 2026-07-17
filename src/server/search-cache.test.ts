/** @jest-environment node */

import { describe, expect, it } from "@jest/globals";
import type { GhqSearchResult } from "@/game/analysis/types";
import {
  searchCachePathname,
  shouldPersistSearch,
  type SearchCacheKey,
} from "./search-cache";

const key: SearchCacheKey = {
  serializedPosition: "position",
  searchCodeVersion: "commit-a",
  personality: "balanced",
  turnNumber: 12,
  timeMs: 20_000,
  maxDepth: 2,
  beamWidth: 6,
  maxActions: 3,
  stagnationTurns: 0,
  valueModel: "incumbent",
  valueModelCheckpoint: "three-actions:incumbent:test",
};

function result(overrides: Partial<GhqSearchResult["search"]> = {}) {
  return {
    candidate_turns: [{}],
    search: {
      completed_depth_in_turns: 2,
      fallback_used: "none",
      ...overrides,
    },
  } as GhqSearchResult;
}

describe("persistent early search cache", () => {
  it("isolates different artifacts that occupy the challenger slot", () => {
    expect(searchCachePathname(key)).not.toBe(
      searchCachePathname({
        ...key,
        valueModelCheckpoint: "three-actions:incumbent:replacement",
      })
    );
  });

  it("isolates cached searches produced by different code revisions", () => {
    expect(searchCachePathname(key)).not.toBe(
      searchCachePathname({ ...key, searchCodeVersion: "commit-b" })
    );
  });

  it("admits only reply-complete non-fallback opening searches", () => {
    expect(shouldPersistSearch(key, result())).toBe(true);
    expect(
      shouldPersistSearch(key, result({ completed_depth_in_turns: 1 }))
    ).toBe(false);
    expect(shouldPersistSearch(key, result({ fallback_used: "safe" }))).toBe(
      false
    );
    expect(shouldPersistSearch({ ...key, turnNumber: 17 }, result())).toBe(
      false
    );
    expect(shouldPersistSearch({ ...key, turnNumber: 4 }, result())).toBe(
      false
    );
  });

  it("keeps checkpoints separated in the cache key", () => {
    expect({ ...key, valueModel: "challenger" }).not.toEqual(key);
  });
});
