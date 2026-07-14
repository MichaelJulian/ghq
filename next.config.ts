import type { NextConfig } from "next";
import { withWorkflow } from "workflow/next";

const aiRuntimeFiles = [
  "./public/engine.py",
  "./scripts/ghq_ai.py",
  "./node_modules/pyodide/**/*",
];

const nextConfig: NextConfig = {
  /* config options here */
  serverExternalPackages: ["pyodide"],
  outputFileTracingIncludes: {
    "/api/ai/analyze": aiRuntimeFiles,
    "/.well-known/workflow/v1/step": aiRuntimeFiles,
  },
  typescript: {
    tsconfigPath: "./tsconfig.json",
    // @todo figure out why tsconfig doesn't exclude node modules
    ignoreBuildErrors: true,
  },
};

export default withWorkflow(nextConfig);
