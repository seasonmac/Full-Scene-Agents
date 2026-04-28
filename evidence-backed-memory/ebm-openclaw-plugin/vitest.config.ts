import { defineConfig } from "vitest/config";

export default defineConfig({
  test: {
    globals: true,
    environment: "node",
    pool: "forks",
    include: ["tests/**/*.test.ts"],
    coverage: {
      provider: "v8",
      include: ["src/**/*.ts"],
      exclude: ["src/**/*.d.ts"],
      thresholds: {
        // index.ts and process-manager.ts require live sidecar / OS integration
        // — covered by e2e tests, not unit tests. Per-file thresholds for
        // pure-logic modules that *can* be fully unit-tested:
        "src/config.ts": { lines: 90, branches: 90, functions: 90, statements: 90 },
        "src/client.ts": { lines: 85, branches: 70, functions: 85, statements: 85 },
        "src/context-engine.ts": { lines: 85, branches: 70, functions: 90, statements: 85 },
      },
    },
  },
});
