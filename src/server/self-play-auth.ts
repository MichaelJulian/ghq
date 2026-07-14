import { timingSafeEqual } from "node:crypto";

export function selfPlayAuthorized(request: Request): boolean {
  const expected = process.env.SELF_PLAY_SECRET;
  if (!expected) return process.env.NODE_ENV !== "production";
  const authorization = request.headers.get("authorization") ?? "";
  const provided = authorization.startsWith("Bearer ")
    ? authorization.slice("Bearer ".length)
    : "";
  const expectedBytes = Buffer.from(expected);
  const providedBytes = Buffer.from(provided);
  return (
    expectedBytes.length === providedBytes.length &&
    timingSafeEqual(expectedBytes, providedBytes)
  );
}
