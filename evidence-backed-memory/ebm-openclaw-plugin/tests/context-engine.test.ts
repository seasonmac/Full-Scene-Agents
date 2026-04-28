import { describe, it, expect, vi } from "vitest";
import { createEbmContextEngine, type EngineLogger, type AgentMessage } from "../src/context-engine.js";
import type { EbmPyClient } from "../src/client.js";

function createMockClient(overrides: Partial<EbmPyClient> = {}): EbmPyClient {
  return {
    baseUrl: "http://127.0.0.1:18790",
    healthCheck: vi.fn().mockResolvedValue({ ok: true, engine: "ebm_context_engine" }),
    bootstrap: vi.fn().mockResolvedValue({ bootstrapped: true, importedMessages: 0 }),
    ingest: vi.fn().mockResolvedValue({ ingested: true }),
    ingestBatch: vi.fn().mockResolvedValue({ ingestedCount: 0 }),
    assemble: vi.fn().mockResolvedValue({ messages: [], estimatedTokens: 0 }),
    afterTurn: vi.fn().mockResolvedValue(undefined),
    compact: vi.fn().mockResolvedValue({ ok: true, compacted: false }),
    status: vi.fn().mockResolvedValue({ pending: 0, running: 0, done: 0, failed: 0 }),
    flush: vi.fn().mockResolvedValue({ status: { pending: 0, running: 0, done: 0, failed: 0 } }),
    retryFailed: vi.fn().mockResolvedValue({ retried: 0 }),
    dispose: vi.fn().mockResolvedValue(undefined),
    query: vi.fn().mockResolvedValue({ answer: "", context: "", debug: {} }),
    memorySearch: vi.fn().mockResolvedValue([]),
    memoryGet: vi.fn().mockResolvedValue({ item: null }),
    memoryForget: vi.fn().mockResolvedValue({ forgotten: false, id: "", reason: "not_found" }),
    archiveExpand: vi.fn().mockResolvedValue({ archiveId: "", messages: [], source: "not_found" }),
    ...overrides,
  } as unknown as EbmPyClient;
}

function createMockLogger(): EngineLogger {
  return {
    info: vi.fn(),
    warn: vi.fn(),
    error: vi.fn(),
  };
}

describe("createEbmContextEngine", () => {
  const logger = createMockLogger();

  describe("info", () => {
    it("has correct id and name", () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      expect(engine.info.id).toBe("ebm-context-engine");
      expect(engine.info.name).toBe("EBM Python Memory Engine");
      expect(engine.info.ownsCompaction).toBe(false);
    });
  });

  describe("bootstrap", () => {
    it("calls client bootstrap and returns result", async () => {
      const client = createMockClient({
        bootstrap: vi.fn().mockResolvedValue({ bootstrapped: true, importedMessages: 5 }),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.bootstrap({ sessionId: "s1" });
      expect(result.bootstrapped).toBe(true);
      expect(client.bootstrap).toHaveBeenCalledWith({
        sessionId: "s1",
        sessionKey: undefined,
        sessionFile: undefined,
      });
    });

    it("returns false on error", async () => {
      const client = createMockClient({
        bootstrap: vi.fn().mockRejectedValue(new Error("fail")),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.bootstrap({ sessionId: "s1" });
      expect(result.bootstrapped).toBe(false);
    });
  });

  describe("ingest", () => {
    it("skips heartbeat messages", async () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.ingest({
        sessionId: "s1",
        message: { role: "user", content: "hello" },
        isHeartbeat: true,
      });
      expect(result.ingested).toBe(false);
      expect(client.ingest).not.toHaveBeenCalled();
    });

    it("extracts text content from message", async () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      await engine.ingest({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/tmp/s1.jsonl",
        message: { role: "user", content: "hello world" },
      });
      expect(client.ingest).toHaveBeenCalledWith({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/tmp/s1.jsonl",
        message: { role: "user", content: "hello world" },
      });
    });

    it("extracts text from content array", async () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      await engine.ingest({
        sessionId: "s1",
        message: {
          role: "assistant",
          content: [
            { type: "text", text: "part one" },
            { type: "toolUse", name: "search" },
            { type: "text", text: "part two" },
          ],
        },
      });
      expect(client.ingest).toHaveBeenCalledWith({
        sessionId: "s1",
        message: { role: "assistant", content: "part one\npart two" },
      });
    });

    it("skips messages with empty content", async () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.ingest({
        sessionId: "s1",
        message: { role: "user", content: "   " },
      });
      expect(result.ingested).toBe(false);
      expect(client.ingest).not.toHaveBeenCalled();
    });

    it("returns false on error", async () => {
      const client = createMockClient({
        ingest: vi.fn().mockRejectedValue(new Error("fail")),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.ingest({
        sessionId: "s1",
        message: { role: "user", content: "hello" },
      });
      expect(result.ingested).toBe(false);
    });
  });

  describe("ingestBatch", () => {
    it("skips heartbeat", async () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.ingestBatch({
        sessionId: "s1",
        messages: [{ role: "user", content: "hello" }],
        isHeartbeat: true,
      });
      expect(result.ingestedCount).toBe(0);
    });

    it("filters out empty messages", async () => {
      const client = createMockClient({
        ingestBatch: vi.fn().mockResolvedValue({ ingestedCount: 1 }),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      await engine.ingestBatch({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/tmp/s1.jsonl",
        messages: [
          { role: "user", content: "hello" },
          { role: "user", content: "" },
          { role: "user", content: "   " },
        ],
      });
      expect(client.ingestBatch).toHaveBeenCalledWith({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/tmp/s1.jsonl",
        messages: [{ role: "user", content: "hello" }],
      });
    });
  });

  describe("assemble", () => {
    it("returns assembled messages from client", async () => {
      const assembled = [{ role: "user", content: "hello" }];
      const client = createMockClient({
        assemble: vi.fn().mockResolvedValue({
          messages: assembled,
          estimatedTokens: 100,
          systemPromptAddition: "memory",
        }),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.assemble({
        sessionId: "s1",
        sessionKey: "sk1",
        messages: [{ role: "user", content: "hello" }],
        tokenBudget: 16000,
        prompt: "hello",
        runtimeContext: { source: "test" },
      });
      expect(result.messages).toEqual(assembled);
      expect(result.estimatedTokens).toBe(100);
      expect(result.systemPromptAddition).toBe("memory");
      expect(client.assemble).toHaveBeenCalledWith({
        sessionId: "s1",
        sessionKey: "sk1",
        messages: [{ role: "user", content: "hello" }],
        tokenBudget: 16000,
        prompt: "hello",
        runtimeContext: { source: "test" },
      });
    });

    it("passes through original messages on error", async () => {
      const original: AgentMessage[] = [{ role: "user", content: "test" }];
      const client = createMockClient({
        assemble: vi.fn().mockRejectedValue(new Error("timeout")),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.assemble({
        sessionId: "s1",
        messages: original,
      });
      expect(result.messages).toBe(original);
      expect(result.estimatedTokens).toBeGreaterThan(0);
    });

    it("appends ebm_archive_expand guidance when systemPromptAddition has Session History", async () => {
      const addition = "## Memory context\nSome facts\n\n## Session History\nSession s1: discussed tea";
      const client = createMockClient({
        assemble: vi.fn().mockResolvedValue({
          messages: [],
          estimatedTokens: 50,
          systemPromptAddition: addition,
        }),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.assemble({
        sessionId: "s1",
        messages: [{ role: "user", content: "hello" }],
      });
      expect(result.systemPromptAddition).toContain("ebm_archive_expand");
      expect(result.systemPromptAddition).toContain("[session:<id>]");
      expect(result.systemPromptAddition).toContain(addition);
    });

    it("appends ebm_archive_expand guidance when systemPromptAddition has Evidence Traces", async () => {
      const addition = "## Evidence Traces\nTrace 1: some evidence";
      const client = createMockClient({
        assemble: vi.fn().mockResolvedValue({
          messages: [],
          estimatedTokens: 50,
          systemPromptAddition: addition,
        }),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.assemble({
        sessionId: "s1",
        messages: [{ role: "user", content: "hello" }],
      });
      expect(result.systemPromptAddition).toContain("ebm_archive_expand");
    });

    it("does not append ebm_archive_expand guidance when systemPromptAddition has no session context", async () => {
      const client = createMockClient({
        assemble: vi.fn().mockResolvedValue({
          messages: [],
          estimatedTokens: 50,
          systemPromptAddition: "just some plain memory context",
        }),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.assemble({
        sessionId: "s1",
        messages: [{ role: "user", content: "hello" }],
      });
      expect(result.systemPromptAddition).not.toContain("ebm_archive_expand");
    });

    it("does not duplicate ebm_archive_expand guidance if already present", async () => {
      const addition = "## Session History\nSession s1\n\nuse ebm_archive_expand to expand";
      const client = createMockClient({
        assemble: vi.fn().mockResolvedValue({
          messages: [],
          estimatedTokens: 50,
          systemPromptAddition: addition,
        }),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.assemble({
        sessionId: "s1",
        messages: [{ role: "user", content: "hello" }],
      });
      const count = (result.systemPromptAddition?.match(/ebm_archive_expand/g) ?? []).length;
      expect(count).toBe(1);
    });
  });

  describe("afterTurn", () => {
    it("skips heartbeat", async () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      await engine.afterTurn({
        sessionId: "s1",
        sessionFile: "",
        messages: [],
        prePromptMessageCount: 0,
        isHeartbeat: true,
      });
      expect(client.afterTurn).not.toHaveBeenCalled();
    });

    it("calls client afterTurn", async () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      await engine.afterTurn({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/session.jsonl",
        messages: [{ role: "user", content: "hi" }],
        prePromptMessageCount: 0,
        tokenBudget: 1000,
        runtimeContext: { source: "test" },
      });
      expect(client.afterTurn).toHaveBeenCalledWith({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "/session.jsonl",
        messages: [{ role: "user", content: "hi" }],
        prePromptMessageCount: 0,
        tokenBudget: 1000,
        runtimeContext: { source: "test" },
      });
    });

    it("does not throw on error", async () => {
      const client = createMockClient({
        afterTurn: vi.fn().mockRejectedValue(new Error("fail")),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      await expect(
        engine.afterTurn({
          sessionId: "s1",
          sessionFile: "",
          messages: [],
          prePromptMessageCount: 0,
        }),
      ).resolves.not.toThrow();
    });
  });

  describe("compact", () => {
    it("returns compact result from client", async () => {
      const client = createMockClient({
        compact: vi.fn().mockResolvedValue({ ok: true, compacted: false, reason: "stub" }),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.compact({
        sessionId: "s1",
        sessionKey: "sk1",
        sessionFile: "",
        currentTokenCount: 123,
        runtimeContext: { source: "test" },
      });
      expect(result.ok).toBe(true);
      expect(result.compacted).toBe(false);
    });

    it("returns error result on failure", async () => {
      const client = createMockClient({
        compact: vi.fn().mockRejectedValue(new Error("boom")),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      const result = await engine.compact({ sessionId: "s1", sessionFile: "" });
      expect(result.ok).toBe(false);
      expect(result.reason).toContain("compact_error");
      expect(result.result?.tokensBefore).toBe(0);
    });
  });

  describe("dispose", () => {
    it("calls client dispose", async () => {
      const client = createMockClient();
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      await engine.dispose();
      expect(client.dispose).toHaveBeenCalled();
    });

    it("does not throw on error", async () => {
      const client = createMockClient({
        dispose: vi.fn().mockRejectedValue(new Error("fail")),
      });
      const engine = createEbmContextEngine(() => Promise.resolve(client), logger);
      await expect(engine.dispose()).resolves.not.toThrow();
    });
  });
});
