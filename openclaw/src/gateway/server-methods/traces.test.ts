import { afterEach, describe, expect, it, vi } from "vitest";
import { createLlmTraceRecorder, resetLlmTracesForTest } from "../../infra/llm-traces.js";
import { tracesHandlers } from "./traces.js";

describe("tracesHandlers", () => {
  afterEach(() => {
    resetLlmTracesForTest();
  });

  it("tails recorded traces", async () => {
    const recorder = createLlmTraceRecorder({
      runId: "run-1",
      sessionId: "session-1",
      sessionKey: "agent:session-1",
      provider: "openai",
      modelId: "gpt-5.4",
      modelApi: "openai-responses",
    });

    const streamFn = recorder.wrapStreamFn((_model, _context, options) => {
      options?.onPayload?.({ input: [{ role: "user", content: "hello" }] }, {} as never);
      return {
        async result() {
          return {
            role: "assistant",
            content: "world",
            usage: { input: 10, output: 3, total: 13 },
            cost: { total: 0.0024 },
          };
        },
      } as never;
    });

    void streamFn(
      { id: "gpt-5.4", api: "openai-responses", provider: "openai" } as never,
      {} as never,
      {} as never,
    );
    await new Promise((resolve) => setTimeout(resolve, 10));

    const respond = vi.fn();
    await tracesHandlers["traces.tail"]({
      params: {},
      respond,
      context: {} as never,
      client: null,
      req: { id: "req-1", type: "req", method: "traces.tail" },
      isWebchatConnect: () => false,
    });

    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        records: [
          expect.objectContaining({
            runId: "run-1",
            provider: "openai",
            modelId: "gpt-5.4",
            status: "ok",
            requestCount: 1,
            requestPreview: expect.stringContaining("hello"),
            costTotal: 0.0024,
          }),
        ],
      }),
      undefined,
    );
  });

  it("loads a single trace by id", async () => {
    const recorder = createLlmTraceRecorder({
      runId: "run-2",
      sessionId: "session-2",
      provider: "anthropic",
      modelId: "sonnet-4.6",
      modelApi: "anthropic-messages",
    });

    const streamFn = recorder.wrapStreamFn((_model, _context, options) => {
      options?.onPayload?.({ messages: [{ role: "user", content: "ping" }] }, {} as never);
      return {
        async result() {
          return {
            role: "assistant",
            content: [{ text: "pong" }],
            usage: { input: 8, output: 4, total: 12 },
          };
        },
      } as never;
    });

    void streamFn(
      { id: "sonnet-4.6", api: "anthropic-messages", provider: "anthropic" } as never,
      {} as never,
      {} as never,
    );
    await new Promise((resolve) => setTimeout(resolve, 10));

    const listRespond = vi.fn();
    await tracesHandlers["traces.tail"]({
      params: {},
      respond: listRespond,
      context: {} as never,
      client: null,
      req: { id: "req-2", type: "req", method: "traces.tail" },
      isWebchatConnect: () => false,
    });

    const payload = listRespond.mock.calls[0]?.[1] as { records: Array<{ traceId: string }> };
    const traceId = payload.records[0]?.traceId;

    const getRespond = vi.fn();
    await tracesHandlers["traces.get"]({
      params: { traceId },
      respond: getRespond,
      context: {} as never,
      client: null,
      req: { id: "req-3", type: "req", method: "traces.get" },
      isWebchatConnect: () => false,
    });

    expect(getRespond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        traceId,
        requests: [expect.anything()],
        response: expect.anything(),
      }),
      undefined,
    );
  });

  it("filters traces by provider, status, and query", async () => {
    const okRecorder = createLlmTraceRecorder({
      runId: "run-ok",
      sessionId: "session-ok",
      sessionKey: "agent:session-ok",
      provider: "openai",
      modelId: "gpt-5.4",
      modelApi: "openai-responses",
    });
    const errorRecorder = createLlmTraceRecorder({
      runId: "run-error",
      sessionId: "session-error",
      sessionKey: "agent:session-error",
      provider: "anthropic",
      modelId: "sonnet-4.6",
      modelApi: "anthropic-messages",
    });

    const okStreamFn = okRecorder.wrapStreamFn((_model, _context, options) => {
      options?.onPayload?.({ input: [{ role: "user", content: "hello openai" }] }, {} as never);
      return {
        async result() {
          return {
            role: "assistant",
            content: "world",
            usage: { input: 10, output: 3, total: 13 },
          };
        },
      } as never;
    });
    const errorStreamFn = errorRecorder.wrapStreamFn((_model, _context, options) => {
      options?.onPayload?.(
        { messages: [{ role: "user", content: "hello anthropic" }] },
        {} as never,
      );
      return {
        async result() {
          throw new Error("upstream failed");
        },
      } as never;
    });

    void okStreamFn(
      { id: "gpt-5.4", api: "openai-responses", provider: "openai" } as never,
      {} as never,
      {} as never,
    );
    const errorStream = await Promise.resolve(
      errorStreamFn(
        { id: "sonnet-4.6", api: "anthropic-messages", provider: "anthropic" } as never,
        {} as never,
        {} as never,
      ),
    );
    void errorStream.result().catch(() => undefined);
    await new Promise((resolve) => setTimeout(resolve, 10));

    const providerRespond = vi.fn();
    await tracesHandlers["traces.tail"]({
      params: { provider: "openai" },
      respond: providerRespond,
      context: {} as never,
      client: null,
      req: { id: "req-4", type: "req", method: "traces.tail" },
      isWebchatConnect: () => false,
    });
    const providerPayload = providerRespond.mock.calls[0]?.[1] as {
      records: Array<{ provider?: string; runId?: string; status?: string }>;
    };
    expect(providerPayload.records.length).toBeGreaterThan(0);
    expect(providerPayload.records.every((record) => record.provider === "openai")).toBe(true);
    expect(providerPayload.records.some((record) => record.runId === "run-ok")).toBe(true);

    const statusRespond = vi.fn();
    await tracesHandlers["traces.tail"]({
      params: { status: "error" },
      respond: statusRespond,
      context: {} as never,
      client: null,
      req: { id: "req-5", type: "req", method: "traces.tail" },
      isWebchatConnect: () => false,
    });
    const statusPayload = statusRespond.mock.calls[0]?.[1] as {
      records: Array<{ provider?: string; status?: string; runId?: string }>;
    };
    expect(statusPayload.records.length).toBeGreaterThan(0);
    expect(statusPayload.records.every((record) => record.status === "error")).toBe(true);
    expect(statusPayload.records.some((record) => record.runId === "run-error")).toBe(true);

    const queryRespond = vi.fn();
    await tracesHandlers["traces.tail"]({
      params: { query: "session-ok" },
      respond: queryRespond,
      context: {} as never,
      client: null,
      req: { id: "req-6", type: "req", method: "traces.tail" },
      isWebchatConnect: () => false,
    });
    const queryPayload = queryRespond.mock.calls[0]?.[1] as {
      records: Array<{ sessionKey?: string; runId?: string }>;
    };
    expect(queryPayload.records.length).toBeGreaterThan(0);
    expect(queryPayload.records.every((record) => record.sessionKey === "agent:session-ok")).toBe(
      true,
    );
    expect(queryPayload.records.some((record) => record.runId === "run-ok")).toBe(true);
  });

  it("derives request preview and token totals from stream context and alias usage fields", async () => {
    const recorder = createLlmTraceRecorder({
      runId: "run-context",
      sessionId: "session-context",
      provider: "openai",
      modelId: "gpt-5.4",
      modelApi: "openai-responses",
    });

    const streamFn = recorder.wrapStreamFn(() => {
      return {
        async result() {
          return {
            role: "assistant",
            content: "done",
            usage: { inputTokens: 120, outputTokens: 30, totalTokens: 150 },
          };
        },
      } as never;
    });

    void streamFn(
      { id: "gpt-5.4", api: "openai-responses", provider: "openai" } as never,
      {
        messages: [{ role: "user", content: "explain why traces disappear" }],
        system: "You are helpful.",
      } as never,
      {} as never,
    );
    await new Promise((resolve) => setTimeout(resolve, 10));

    const respond = vi.fn();
    await tracesHandlers["traces.tail"]({
      params: { query: "disappear" },
      respond,
      context: {} as never,
      client: null,
      req: { id: "req-7", type: "req", method: "traces.tail" },
      isWebchatConnect: () => false,
    });

    expect(respond).toHaveBeenCalledWith(
      true,
      expect.objectContaining({
        records: [
          expect.objectContaining({
            runId: "run-context",
            requestPreview: expect.stringContaining("explain why traces disappear"),
            usage: expect.objectContaining({
              input: 120,
              output: 30,
              total: 150,
            }),
          }),
        ],
      }),
      undefined,
    );
  });
});
