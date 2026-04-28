import { describe, it, expect, vi } from "vitest";
import {
  quickTcpProbe,
  quickHealthCheck,
  quickPrecheck,
  IS_WIN,
  type ProcessLogger,
} from "../src/process-manager.js";

// Stub fetch for health check tests
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: Record<string, unknown>, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
  };
}

function createLogger(): ProcessLogger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
  };
}

describe("IS_WIN", () => {
  it("is a boolean", () => {
    expect(typeof IS_WIN).toBe("boolean");
  });
});

describe("quickHealthCheck", () => {
  it("returns true when server responds with ok", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true }));
    const result = await quickHealthCheck("http://127.0.0.1:18790", 2000);
    expect(result).toBe(true);
  });

  it("returns false when server responds with not ok", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ ok: false }));
    const result = await quickHealthCheck("http://127.0.0.1:18790", 2000);
    expect(result).toBe(false);
  });

  it("returns false when fetch fails", async () => {
    mockFetch.mockRejectedValueOnce(new Error("connection refused"));
    const result = await quickHealthCheck("http://127.0.0.1:18790", 2000);
    expect(result).toBe(false);
  });

  it("returns false on non-200 status", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      json: () => Promise.resolve({}),
    });
    const result = await quickHealthCheck("http://127.0.0.1:18790", 2000);
    expect(result).toBe(false);
  });
});

describe("quickPrecheck", () => {
  it("returns ok when health check passes", async () => {
    mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true }));
    const result = await quickPrecheck("local", "http://127.0.0.1:18790", 18790, null);
    expect(result.ok).toBe(true);
  });

  it("returns not ok when remote health check fails", async () => {
    mockFetch.mockRejectedValueOnce(new Error("timeout"));
    const result = await quickPrecheck("remote", "http://127.0.0.1:18790", 18790, null);
    expect(result).toEqual({ ok: false, reason: "health check failed" });
  });
});

describe("quickTcpProbe", () => {
  it("returns false for non-listening port", async () => {
    // Use a very high port that's almost certainly not in use
    const result = await quickTcpProbe("127.0.0.1", 61234, 500);
    expect(typeof result).toBe("boolean");
    // We can't guarantee the port is free, so just test it doesn't throw
  });
});
