import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  outputFileTracingIncludes: {
    "/api/ai/analyze": [
      "./public/engine.py",
      "./scripts/ghq_ai.py",
      "./node_modules/pyodide/**/*",
    ],
  },
  typescript: {
    tsconfigPath: "./tsconfig.json",
    // @todo figure out why tsconfig doesn't exclude node modules
    ignoreBuildErrors: true,
  },
};

export default nextConfig;
