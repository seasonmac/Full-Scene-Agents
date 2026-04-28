import { describe, expect, it, vi, beforeEach } from "vitest";
import plugin from "../src/index.js";

const mockFetch = vi.fn();
vi.stubGlobal("fetch", mockFetch);

const {
  mockQuickHealthCheck,
  mockPrepareLocalPort,
  mockResolvePythonCommand,
  mockWaitForHealthOrExit,
  mockSpawn,
} = vi.hoisted(() => ({
  mockQuickHealthCheck: vi.fn(),
  mockPrepareLocalPort: vi.fn(),
  mockResolvePythonCommand: vi.fn(),
  mockWaitForHealthOrExit: vi.fn(),
  mockSpawn: vi.fn(),
}));

vi.mock("../src/process-manager.js", async () => {
  const actual = await vi.importActual<typeof import("../src/process-manager.js")>("../src/process-manager.js");
  return {
    ...actual,
    quickHealthCheck: mockQuickHealthCheck,
    prepareLocalPort: mockPrepareLocalPort,
    resolvePythonCommand: mockResolvePythonCommand,
    waitForHealthOrExit: mockWaitForHealthOrExit,
  };
});

vi.mock("node:child_process", async () => {
  const actual = await vi.importActual<typeof import("node:child_process")>("node:child_process");
  return {
    ...actual,
    spawn: mockSpawn,
  };
});

function createApi(pluginConfig: Record<string, unknown>) {
  const service: { start?: () => Promise<void>; stop?: () => Promise<void> } = {};
  return {
    api: {
      pluginConfig,
      logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn(), debug: vi.fn() },
      resolvePath: (input: string) => input,
      registerTool: vi.fn(),
      on: vi.fn(),
      registerContextEngine: vi.fn(),
      registerHttpRoute: vi.fn(),
      registerService: vi.fn((s) => {
        service.start = s.start;
        service.stop = s.stop;
      }),
    },
    service,
  };
}

function createChildProcessStub() {
  const handlers = new Map<string, Function[]>();
  return {
    stdout: { on: vi.fn() },
    stderr: { on: vi.fn() },
    on: vi.fn((event: string, handler: Function) => {
      handlers.set(event, [...(handlers.get(event) || []), handler]);
    }),
    kill: vi.fn(),
    killed: false,
    exitCode: null,
    signalCode: null,
  };
}

describe("plugin index", () => {
  beforeEach(() => {
    mockFetch.mockReset();
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ ok: true }),
      text: () => Promise.resolve(JSON.stringify({ ok: true })),
    });
    mockQuickHealthCheck.mockReset();
    mockPrepareLocalPort.mockReset();
    mockResolvePythonCommand.mockReset();
    mockWaitForHealthOrExit.mockReset();
    mockSpawn.mockReset();
  });

  it("registers EBM memory tools and hooks", () => {
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);

    const toolNames = api.registerTool.mock.calls.flatMap((call) => {
      const opts = call[1] as { name?: string; names?: string[] } | undefined;
      return opts?.names ?? (opts?.name ? [opts.name] : []);
    });
    expect(toolNames).toEqual([
      "memory_recall",
      "memory_store",
      "memory_forget",
      "ebm_archive_expand",
    ]);

    const hookNames = api.on.mock.calls.map((call) => call[0]);
    expect(hookNames).toEqual([
      "session_start",
      "session_end",
      "agent_end",
      "after_compaction",
      "before_prompt_build",
      "before_reset",
    ]);
  });

  it("memory_recall tool calls memory-search and formats hits", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({
        ok: true,
        results: [{
          id: "fact:1",
          title: "Preference",
          content: "prefers jasmine tea",
          source: "ledger",
          score: 0.91,
        }],
      }),
      text: () => Promise.resolve(""),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const recallFactory = api.registerTool.mock.calls[0]?.[0] as (ctx: unknown) => {
      execute: (toolCallId: string, params: Record<string, unknown>) => Promise<{ content: Array<{ text: string }> }>;
    };
    const result = await recallFactory({}).execute("tc1", { query: "tea" });
    expect(result.content[0]?.text).toContain("prefers jasmine tea");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://remote:18790/memory-search",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("before_prompt_build injects recalled memory context", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({
        ok: true,
        results: [{
          id: "fact:1",
          title: "Preference",
          content: "prefers jasmine tea",
          source: "ledger",
          score: 0.91,
        }],
      }),
      text: () => Promise.resolve(""),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: "What tea do I prefer?" }],
      prompt: "What tea do I prefer?",
    });
    expect(result?.prependContext).toContain("<relevant-memories>");
    expect(result?.prependContext).toContain("prefers jasmine tea");
  });

  it("before_prompt_build skips auto-recall when assemble already produced memory context", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({
        ok: true,
        messages: [{ role: "user", content: "What tea do I prefer?" }],
        estimatedTokens: 128,
        systemPromptAddition:
          "Memory context for this query. Use these facts to answer directly and confidently.\n\n" +
          "If the memory contains relevant information, use it to answer even if not perfectly verified.\n\n" +
          "Session History:\n- [session:s1] User prefers jasmine tea.",
      }),
      text: () => Promise.resolve(""),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const engineFactory = api.registerContextEngine.mock.calls[0]?.[1] as () => {
      assemble: (params: {
        sessionId: string;
        messages: Array<{ role: string; content: string }>;
      }) => Promise<unknown>;
    };
    await engineFactory().assemble({
      sessionId: "s1",
      messages: [{ role: "user", content: "What tea do I prefer?" }],
    });
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt(
      { messages: [{ role: "user", content: "What tea do I prefer?" }] },
      { sessionId: "s1" },
    );
    expect(result).toBeUndefined();
    expect(mockFetch).toHaveBeenCalledTimes(1);
    expect(api.logger.debug).toHaveBeenCalledWith(
      expect.stringContaining("assemble already provided memory context"),
    );
  });

  it("before_prompt_build falls back to auto-recall when assemble produced no meaningful memory context", async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          ok: true,
          messages: [{ role: "user", content: "What tea do I prefer?" }],
          estimatedTokens: 32,
          systemPromptAddition:
            "Memory context for this query. Use these facts to answer directly and confidently.\n\n" +
            "If the memory contains relevant information, use it to answer even if not perfectly verified.",
        }),
        text: () => Promise.resolve(""),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          ok: true,
          results: [{
            id: "fact:1",
            title: "Preference",
            content: "prefers jasmine tea",
            source: "ledger",
            score: 0.91,
          }],
        }),
        text: () => Promise.resolve(""),
      });
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const engineFactory = api.registerContextEngine.mock.calls[0]?.[1] as () => {
      assemble: (params: {
        sessionId: string;
        messages: Array<{ role: string; content: string }>;
      }) => Promise<unknown>;
    };
    await engineFactory().assemble({
      sessionId: "s1",
      messages: [{ role: "user", content: "What tea do I prefer?" }],
    });
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt(
      { messages: [{ role: "user", content: "What tea do I prefer?" }] },
      { sessionId: "s1" },
    );
    expect(result?.prependContext).toContain("<relevant-memories>");
    expect(result?.prependContext).toContain("prefers jasmine tea");
    expect(mockFetch).toHaveBeenCalledTimes(2);
  });

  it("does not spawn in remote mode when backend is unhealthy", async () => {
    mockQuickHealthCheck.mockResolvedValue(false);
    const { api, service } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    await service.start?.();
    expect(mockSpawn).not.toHaveBeenCalled();
  });

  it("reuses shared local startup and only kills on final stop", async () => {
    mockPrepareLocalPort.mockResolvedValue(18790);
    mockResolvePythonCommand.mockReturnValue("python3");
    mockWaitForHealthOrExit.mockResolvedValue(undefined);
    const child = createChildProcessStub();
    mockSpawn.mockReturnValue(child);

    const first = createApi({ mode: "local", ebmPyPath: "/tmp/project" });
    const second = createApi({ mode: "local", ebmPyPath: "/tmp/project" });
    (plugin as { register(api: unknown): void }).register(first.api);
    (plugin as { register(api: unknown): void }).register(second.api);

    await first.service.start?.();
    await second.service.start?.();
    expect(mockSpawn).toHaveBeenCalledTimes(1);

    await first.service.stop?.();
    expect(child.kill).not.toHaveBeenCalled();
    await second.service.stop?.();
    expect(child.kill).toHaveBeenCalledWith("SIGTERM");
  });

  it("before_prompt_build skips auto-recall when autoRecall is false", async () => {
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790", autoRecall: false });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: "What tea do I prefer?" }],
    });
    expect(result).toBeUndefined();
    expect(mockFetch).not.toHaveBeenCalledWith(
      expect.stringContaining("/memory-search"),
      expect.anything(),
    );
  });

  it("before_prompt_build skips when query is too short", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: "hi" }],
    });
    expect(result).toBeUndefined();
  });

  it("before_prompt_build skips recall when health precheck fails", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(false);
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: "What tea do I prefer?" }],
    });
    expect(result).toBeUndefined();
    expect(api.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("health precheck failed"),
    );
  });

  it("before_prompt_build returns undefined when no memories match", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ ok: true, results: [] }),
      text: () => Promise.resolve(""),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: "What tea do I prefer?" }],
    });
    expect(result).toBeUndefined();
  });

  it("before_prompt_build injects ingest-reply-assist for transcript input", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ ok: true, results: [] }),
      text: () => Promise.resolve(""),
    });
    const transcript =
      "Alice: I think we should use React for the frontend.\n" +
      "Bob: I agree, but we need to consider performance.\n" +
      "Alice: Let's benchmark both options before deciding.";
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: transcript }],
    });
    expect(result?.prependContext).toContain("<ingest-reply-assist>");
    expect(result?.prependContext).toContain("multi-speaker transcript");
  });

  it("before_prompt_build skips ingest-reply-assist when disabled", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ ok: true, results: [] }),
      text: () => Promise.resolve(""),
    });
    const transcript =
      "Alice: I think we should use React.\n" +
      "Bob: I agree, performance matters.\n" +
      "Alice: Let's benchmark both options before deciding.";
    const { api } = createApi({
      mode: "remote",
      baseUrl: "http://remote:18790",
      ingestReplyAssist: false,
    });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: transcript }],
    });
    expect(result?.prependContext ?? "").not.toContain("<ingest-reply-assist>");
  });

  it("before_prompt_build skips ingest-reply-assist for matching session pattern", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ ok: true, results: [] }),
      text: () => Promise.resolve(""),
    });
    const transcript =
      "Alice: I think we should use React.\n" +
      "Bob: I agree, performance matters.\n" +
      "Alice: Let's benchmark both options before deciding.";
    const { api } = createApi({
      mode: "remote",
      baseUrl: "http://remote:18790",
      ingestReplyAssistIgnoreSessionPatterns: ["ci:**"],
    });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt(
      { messages: [{ role: "user", content: transcript }] },
      { sessionKey: "ci:build:123" },
    );
    expect(result?.prependContext ?? "").not.toContain("<ingest-reply-assist>");
  });

  it("before_prompt_build combines recall and ingest-reply-assist", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({
        ok: true,
        results: [{
          id: "fact:1",
          title: "Framework",
          content: "team prefers React",
          source: "ledger",
          score: 0.88,
        }],
      }),
      text: () => Promise.resolve(""),
    });
    const transcript =
      "Alice: I think we should use React for the frontend.\n" +
      "Bob: I agree, but we need to consider performance.\n" +
      "Alice: Let's benchmark both options before deciding.";
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: transcript }],
    });
    expect(result?.prependContext).toContain("<relevant-memories>");
    expect(result?.prependContext).toContain("<ingest-reply-assist>");
  });

  it("before_prompt_build extracts text from prompt fallback", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({
        ok: true,
        results: [{
          id: "fact:1",
          title: "Pref",
          content: "likes green tea",
          source: "ledger",
          score: 0.9,
        }],
      }),
      text: () => Promise.resolve(""),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      prompt: "What tea do I like?",
    });
    expect(result?.prependContext).toContain("<relevant-memories>");
  });

  it("before_prompt_build gracefully handles recall timeout", async () => {
    mockQuickHealthCheck.mockResolvedValueOnce(true);
    mockFetch.mockImplementationOnce(
      () => new Promise((resolve) => setTimeout(resolve, 60_000)),
    );
    const { api } = createApi({
      mode: "remote",
      baseUrl: "http://remote:18790",
      recallTimeoutMs: 500,
    });
    (plugin as { register(api: unknown): void }).register(api);
    const beforePrompt = api.on.mock.calls.find((call) => call[0] === "before_prompt_build")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<{ prependContext?: string } | undefined>;
    const result = await beforePrompt({
      messages: [{ role: "user", content: "What tea do I prefer?" }],
    });
    expect(result).toBeUndefined();
    expect(api.logger.warn).toHaveBeenCalledWith(
      expect.stringContaining("auto-recall failed"),
    );
  });

  it("releases startup owners when shared startup fails", async () => {
    mockPrepareLocalPort.mockResolvedValue(18790);
    mockResolvePythonCommand.mockReturnValue("python3");
    mockWaitForHealthOrExit.mockRejectedValue(new Error("boom"));
    const child = createChildProcessStub();
    mockSpawn.mockReturnValue(child);

    const first = createApi({ mode: "local", ebmPyPath: "/tmp/project" });
    const second = createApi({ mode: "local", ebmPyPath: "/tmp/project" });
    (plugin as { register(api: unknown): void }).register(first.api);
    (plugin as { register(api: unknown): void }).register(second.api);

    await expect(first.service.start?.()).rejects.toThrow("boom");
    await expect(second.service.start?.()).rejects.toThrow("boom");
  });

  it("keeps local runtimes isolated by config key", async () => {
    mockPrepareLocalPort
      .mockResolvedValueOnce(18790)
      .mockResolvedValueOnce(18791);
    mockResolvePythonCommand.mockReturnValue("python3");
    mockWaitForHealthOrExit.mockResolvedValue(undefined);
    const childA = createChildProcessStub();
    const childB = createChildProcessStub();
    mockSpawn
      .mockReturnValueOnce(childA)
      .mockReturnValueOnce(childB);

    const first = createApi({ mode: "local", ebmPyPath: "/tmp/project-a", port: 18790 });
    const second = createApi({ mode: "local", ebmPyPath: "/tmp/project-b", port: 18791 });
    (plugin as { register(api: unknown): void }).register(first.api);
    (plugin as { register(api: unknown): void }).register(second.api);

    await first.service.start?.();
    await second.service.start?.();
    expect(mockSpawn).toHaveBeenCalledTimes(2);
  });

  // ── Fix High 1: memory_store uses only afterTurn (no duplicate ingest) ──

  it("memory_store calls afterTurn then flush without separate ingest", async () => {
    const callOrder: string[] = [];
    mockFetch.mockImplementation(async (url: string) => {
      const path = new URL(url).pathname;
      callOrder.push(path);
      return {
        ok: true,
        status: 200,
        json: () => Promise.resolve({ ok: true, ingested: 1, status: { pending: 0, running: 1, done: 1, failed: 0 } }),
        text: () => Promise.resolve(JSON.stringify({ ok: true })),
      };
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const storeFactory = api.registerTool.mock.calls[1]?.[0] as (ctx: unknown) => {
      execute: (toolCallId: string, params: Record<string, unknown>) => Promise<{ content: Array<{ text: string }> }>;
    };
    const result = await storeFactory({ sessionId: "s1" }).execute("tc1", { text: "remember this" });
    expect(result.content[0]?.text).toContain("Stored EBM memory");
    expect(callOrder).not.toContain("/ingest");
    const afterTurnIdx = callOrder.indexOf("/after-turn");
    const flushIdx = callOrder.indexOf("/flush");
    expect(afterTurnIdx).toBeGreaterThanOrEqual(0);
    expect(flushIdx).toBeGreaterThan(afterTurnIdx);
  });

  // ── Fix Medium 4: memory_forget source type check ──

  it("memory_forget skips auto-forget for unsupported source types including graph", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: () => Promise.resolve({
        ok: true,
        results: [
          { id: "community:42", title: "Community Topic", content: "some community content", source: "community", score: 0.95 },
          { id: "graph:entity:7", title: "Entity", content: "entity content", source: "graph", score: 0.92 },
        ],
      }),
      text: () => Promise.resolve(""),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const forgetFactory = api.registerTool.mock.calls[2]?.[0] as (ctx: unknown) => {
      execute: (toolCallId: string, params: Record<string, unknown>) => Promise<{ content: Array<{ text: string }> }>;
    };
    const result = await forgetFactory({}).execute("tc1", { query: "community topic" });
    expect(result.content[0]?.text).toContain("candidates");
    expect(mockFetch).toHaveBeenCalledTimes(1);
  });

  it("memory_forget auto-forgets when source is a forgettable type", async () => {
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          ok: true,
          results: [{
            id: "fact:99",
            title: "Forgettable",
            content: "some fact",
            source: "ledger",
            score: 0.90,
          }],
        }),
        text: () => Promise.resolve(""),
      })
      .mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ ok: true, forgotten: true, id: "fact:99" }),
        text: () => Promise.resolve(""),
      });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const forgetFactory = api.registerTool.mock.calls[2]?.[0] as (ctx: unknown) => {
      execute: (toolCallId: string, params: Record<string, unknown>) => Promise<{ content: Array<{ text: string }> }>;
    };
    const result = await forgetFactory({}).execute("tc1", { query: "some fact" });
    expect(result.content[0]?.text).toContain("Forgotten");
    expect(mockFetch).toHaveBeenCalledWith(
      "http://remote:18790/memory-forget",
      expect.objectContaining({ method: "POST" }),
    );
  });

  // ── Fix Medium 5: before_reset filters empty messages ──

  it("before_reset always flushes even when all messages are empty", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ ok: true, status: { pending: 0, running: 1, done: 0, failed: 0 } }),
      text: () => Promise.resolve(JSON.stringify({ ok: true })),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforeReset = api.on.mock.calls.find((call) => call[0] === "before_reset")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<void>;
    await beforeReset(
      { messages: [{ role: "user", content: "" }, { role: "assistant", content: "  " }] },
      { sessionId: "s1" },
    );
    expect(mockFetch).not.toHaveBeenCalledWith(
      expect.stringContaining("/ingest-batch"),
      expect.anything(),
    );
    expect(mockFetch).toHaveBeenCalledWith(
      "http://remote:18790/flush",
      expect.objectContaining({ method: "POST" }),
    );
  });

  it("before_reset ingests non-empty messages and flushes", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ ok: true, ingested: 1, status: { pending: 0, running: 1, done: 0, failed: 0 } }),
      text: () => Promise.resolve(JSON.stringify({ ok: true })),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforeReset = api.on.mock.calls.find((call) => call[0] === "before_reset")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<void>;
    await beforeReset(
      { messages: [{ role: "user", content: "hello" }, { role: "assistant", content: "" }] },
      { sessionId: "s1" },
    );
    expect(mockFetch).toHaveBeenCalledWith(
      "http://remote:18790/ingest-batch",
      expect.objectContaining({ method: "POST" }),
    );
    expect(mockFetch).toHaveBeenCalledWith(
      "http://remote:18790/flush",
      expect.objectContaining({ method: "POST" }),
    );
    expect(api.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("ingested 1 messages"),
    );
    expect(api.logger.info).toHaveBeenCalledWith(
      expect.stringContaining("flush triggered"),
    );
  });

  it("before_reset flushes even when ctx has no sessionId", async () => {
    mockFetch.mockResolvedValue({
      ok: true,
      status: 200,
      json: () => Promise.resolve({ ok: true, status: { pending: 0, running: 1, done: 0, failed: 0 } }),
      text: () => Promise.resolve(JSON.stringify({ ok: true })),
    });
    const { api } = createApi({ mode: "remote", baseUrl: "http://remote:18790" });
    (plugin as { register(api: unknown): void }).register(api);
    const beforeReset = api.on.mock.calls.find((call) => call[0] === "before_reset")?.[1] as (
      event: unknown,
      ctx?: unknown,
    ) => Promise<void>;
    await beforeReset(
      { messages: [{ role: "user", content: "hello" }] },
      {},
    );
    expect(mockFetch).not.toHaveBeenCalledWith(
      expect.stringContaining("/ingest-batch"),
      expect.anything(),
    );
    expect(mockFetch).toHaveBeenCalledWith(
      "http://remote:18790/flush",
      expect.objectContaining({ method: "POST" }),
    );
  });
});
