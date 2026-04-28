import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { EbmPyClient, getCachedLocalClient, setCachedLocalClient, clearCachedLocalClient } from "../src/client.js";

// Mock fetch for all client tests
const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

function jsonResponse(body: Record<string, unknown>, status = 200) {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: () => Promise.resolve(body),
    text: () => Promise.resolve(JSON.stringify(body)),
  };
}

describe("EbmPyClient", () => {
  let client: EbmPyClient;

  beforeEach(() => {
    client = new EbmPyClient("http://127.0.0.1:18790", 5000);
    mockFetch.mockReset();
  });

  describe("constructor", () => {
    it("strips trailing slashes from baseUrl", () => {
      const c = new EbmPyClient("http://host:9000///");
      expect(c.baseUrl).toBe("http://host:9000");
    });
  });

  describe("healthCheck", () => {
    it("returns health result on success", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true, engine: "ebm_context_engine" }));
      const result = await client.healthCheck();
      expect(result.ok).toBe(true);
      expect(result.engine).toBe("ebm_context_engine");
      expect(mockFetch).toHaveBeenCalledWith(
        "http://127.0.0.1:18790/health",
        expect.objectContaining({ method: "GET" }),
      );
    });
  });

  describe("bootstrap", () => {
    it("returns bootstrap result", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ ok: true, bootstrapped: true, importedMessages: 42 }),
      );
      const result = await client.bootstrap({
        sessionId: "test-session",
        sessionKey: "agent:main:test",
        sessionFile: "/tmp/session.jsonl",
      });
      expect(result.bootstrapped).toBe(true);
      expect(result.importedMessages).toBe(42);
    });

    it("defaults importedMessages to 0", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true, bootstrapped: true }));
      const result = await client.bootstrap({ sessionId: "s1" });
      expect(result.importedMessages).toBe(0);
    });
  });

  describe("ingest", () => {
    it("posts message to /ingest", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true, ingested: true }));
      const result = await client.ingest({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/tmp/s1.jsonl",
        message: { role: "user", content: "hello" },
      });
      expect(result.ingested).toBe(true);
    });
  });

  describe("ingestBatch", () => {
    it("posts messages to /ingest-batch", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true, ingestedCount: 3 }));
      const result = await client.ingestBatch({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/tmp/s1.jsonl",
        messages: [
          { role: "user", content: "one" },
          { role: "assistant", content: "two" },
          { role: "user", content: "three" },
        ],
      });
      expect(result.ingestedCount).toBe(3);
    });
  });

  describe("assemble", () => {
    it("returns assembled messages and token count", async () => {
      const assembled = [{ role: "user", content: "hello" }];
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          ok: true,
          messages: assembled,
          estimatedTokens: 500,
          systemPromptAddition: "Memory context here",
        }),
      );
      const result = await client.assemble({
        sessionId: "s1",
        sessionKey: "sk1",
        messages: [{ role: "user", content: "hello" }],
        tokenBudget: 16000,
        prompt: "hello",
        runtimeContext: { source: "test" },
      });
      expect(result.messages).toEqual(assembled);
      expect(result.estimatedTokens).toBe(500);
      expect(result.systemPromptAddition).toBe("Memory context here");
      const [, options] = mockFetch.mock.calls.at(-1) as [string, RequestInit];
      expect(JSON.parse(String(options.body))).toMatchObject({
        sessionKey: "sk1",
        prompt: "hello",
        runtimeContext: { source: "test" },
      });
    });

    it("falls back to original messages when response has no messages", async () => {
      const original = [{ role: "user", content: "test" }];
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ ok: true, estimatedTokens: 0 }),
      );
      const result = await client.assemble({
        sessionId: "s1",
        messages: original,
      });
      expect(result.messages).toEqual(original);
    });

    it("omits systemPromptAddition when not present", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ ok: true, messages: [], estimatedTokens: 0 }),
      );
      const result = await client.assemble({ sessionId: "s1", messages: [] });
      expect(result.systemPromptAddition).toBeUndefined();
    });
  });

  describe("afterTurn", () => {
    it("posts to /after-turn", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true }));
      await client.afterTurn({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/path/to/session.jsonl",
        messages: [],
        prePromptMessageCount: 0,
        tokenBudget: 1000,
        runtimeContext: { source: "test" },
      });
      expect(mockFetch).toHaveBeenCalledWith(
        "http://127.0.0.1:18790/after-turn",
        expect.objectContaining({ method: "POST" }),
      );
    });
  });

  describe("compact", () => {
    it("returns compact result", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          ok: true,
          compacted: false,
          reason: "not implemented",
          result: { tokensBefore: 128, tokensAfter: 64 },
        }),
      );
      const result = await client.compact({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/tmp/s1.jsonl",
        currentTokenCount: 128,
        runtimeContext: { source: "test" },
      });
      expect(result.ok).toBe(true);
      expect(result.compacted).toBe(false);
      expect(result.result?.tokensBefore).toBe(128);
      expect(result.result?.tokensAfter).toBe(64);
    });
  });

  describe("status", () => {
    it("returns slow path status", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ ok: true, status: { pending: 1, running: 0, done: 5, failed: 0 } }),
      );
      const result = await client.status();
      expect(result.pending).toBe(1);
      expect(result.done).toBe(5);
    });
  });

  describe("flush", () => {
    it("returns flush result with status", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ ok: true, status: { pending: 0, running: 0, done: 10, failed: 0 } }),
      );
      const result = await client.flush();
      expect(result.status.done).toBe(10);
    });
  });

  describe("retryFailed", () => {
    it("returns retried count", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true, retried: 3 }));
      const result = await client.retryFailed();
      expect(result.retried).toBe(3);
    });
  });

  describe("memory tools endpoints", () => {
    it("gets memory item", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true, item: { id: "fact:1" } }));
      const result = await client.memoryGet("fact:1");
      expect(result.item?.id).toBe("fact:1");
      expect(mockFetch).toHaveBeenCalledWith(
        "http://127.0.0.1:18790/memory-get",
        expect.objectContaining({ method: "POST" }),
      );
    });

    it("forgets memory item", async () => {
      mockFetch.mockResolvedValueOnce(jsonResponse({ ok: true, forgotten: true, id: "fact:1", type: "FACT" }));
      const result = await client.memoryForget("fact:1");
      expect(result.forgotten).toBe(true);
      expect(result.type).toBe("FACT");
    });

    it("expands archive", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({
          ok: true,
          archiveId: "s1",
          messages: [{ role: "user", text: "hello", messageIndex: 0 }],
          source: "transcript",
        }),
      );
      const result = await client.archiveExpand({ archiveId: "s1", sessionId: "s1" });
      expect(result.messages[0]?.text).toBe("hello");
      expect(result.source).toBe("transcript");
    });
  });

  describe("dispose", () => {
    it("does not throw when server is gone", async () => {
      mockFetch.mockRejectedValueOnce(new Error("connection refused"));
      await expect(client.dispose()).resolves.not.toThrow();
    });
  });

  describe("error handling", () => {
    it("throws on HTTP error status", async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        text: () => Promise.resolve("Internal Server Error"),
      });
      await expect(client.healthCheck()).rejects.toThrow("EBM-PY GET /health failed: 500");
    });

    it("throws on payload ok=false", async () => {
      mockFetch.mockResolvedValueOnce(
        jsonResponse({ ok: false, error: "bad request" }),
      );
      await expect(client.status()).rejects.toThrow("EBM-PY /status error: bad request");
    });
  });
});

describe("client cache", () => {
  afterEach(() => {
    clearCachedLocalClient("default");
  });

  it("starts with null cache", () => {
    expect(getCachedLocalClient()).toBeNull();
  });

  it("set and get cached client", () => {
    const client = new EbmPyClient("http://localhost:18790");
    setCachedLocalClient(client, "default");
    expect(getCachedLocalClient()).toBe(client);
  });

  it("clear resets cache to null", () => {
    setCachedLocalClient(new EbmPyClient("http://localhost:18790"), "default");
    clearCachedLocalClient("default");
    expect(getCachedLocalClient()).toBeNull();
  });

  it("isolates runtime state by key", () => {
    const one = new EbmPyClient("http://localhost:18790");
    const two = new EbmPyClient("http://localhost:18791");
    setCachedLocalClient(one, "local:a");
    setCachedLocalClient(two, "local:b");
    expect(getCachedLocalClient("local:a")).toBe(one);
    expect(getCachedLocalClient("local:b")).toBe(two);
    clearCachedLocalClient("local:a");
    clearCachedLocalClient("local:b");
  });
});
