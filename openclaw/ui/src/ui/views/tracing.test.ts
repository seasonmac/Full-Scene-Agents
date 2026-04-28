/* @vitest-environment jsdom */

import { render } from "lit";
import { describe, expect, it, vi } from "vitest";
import type { LlmTraceRecord, LlmTraceSummary } from "../types.ts";
import { renderTracing, type TracingProps } from "./tracing.ts";

function buildSummary(): LlmTraceSummary {
  return {
    traceId: "trace-1",
    provider: "openai",
    modelId: "gpt-5.4",
    startedAt: Date.now(),
    updatedAt: Date.now(),
    status: "ok",
    durationMs: 240,
    requestCount: 1,
    requestPreview: "Write a haiku about traces",
    responsePreview: "Silent spans arrive",
    usage: { total: 42 },
    costTotal: 0.0024,
  };
}

function buildRecord(): LlmTraceRecord {
  return {
    ...buildSummary(),
    requests: [
      {
        input: [{ role: "user", content: "Write a haiku about traces with AGENTS.md context" }],
        messages: [{ role: "user", content: "Write a haiku about traces" }],
        tools: [{ name: "search", input_schema: { type: "object" } }],
        system:
          "Load AGENTS.md and skills/openclaw-tracing/SKILL.md before answering with the available tool schemas.",
      },
    ],
    response: { role: "assistant", content: "Silent spans arrive" },
    timeline: [
      { ts: Date.now() - 20, kind: "request", label: "Stream started" },
      { ts: Date.now() - 10, kind: "request", label: "Request payload captured" },
      { ts: Date.now(), kind: "response", label: "Response completed" },
    ],
  };
}

function buildProps(overrides: Partial<TracingProps> = {}): TracingProps {
  return {
    loading: false,
    error: null,
    file: null,
    entries: [buildSummary()],
    selected: null,
    selectedId: null,
    filterText: "",
    providerFilter: "",
    statusFilter: "all",
    autoFollow: true,
    sortKey: "time",
    sortDir: "desc",
    truncated: false,
    onFilterTextChange: () => undefined,
    onProviderFilterChange: () => undefined,
    onStatusFilterChange: () => undefined,
    onAutoFollowChange: () => undefined,
    onSortChange: () => undefined,
    onRefresh: () => undefined,
    onSelect: () => undefined,
    onCloseDetail: () => undefined,
    onExport: () => undefined,
    ...overrides,
  };
}

describe("tracing view", () => {
  it("renders Langfuse-style list columns", async () => {
    const container = document.createElement("div");
    render(renderTracing(buildProps()), container);
    await Promise.resolve();

    const headerText = container.querySelector(".tracing-table__header")?.textContent ?? "";
    expect(headerText).toContain("Time");
    expect(headerText).toContain("Name");
    expect(headerText).toContain("Status");
    expect(headerText).toContain("Tokens");
    expect(headerText).toContain("Cost");
    expect(headerText).toContain("Latency");
    expect(headerText).toContain("LLM Input");
    expect(headerText).toContain("LLM Output");
    expect(container.querySelector(".tracing-table__row")?.textContent).toContain(
      "Write a haiku about traces",
    );
    expect(container.querySelector(".tracing-table__row")?.textContent).toContain("$0.0024");
  });

  it("sorts rows by tokens when requested", async () => {
    const container = document.createElement("div");
    render(
      renderTracing(
        buildProps({
          entries: [
            buildSummary(),
            {
              ...buildSummary(),
              traceId: "trace-2",
              requestPreview: "A smaller request",
              responsePreview: "Tiny output",
              usage: { total: 5 },
            },
          ],
          sortKey: "tokens",
          sortDir: "asc",
        }),
      ),
      container,
    );
    await Promise.resolve();

    const rows = Array.from(container.querySelectorAll(".tracing-table__row"));
    expect(rows[0]?.textContent).toContain("A smaller request");
    expect(rows[1]?.textContent).toContain("Write a haiku about traces");
  });

  it("renders the floating detail drawer with copyable JSON inspectors", async () => {
    const onCloseDetail = vi.fn();
    const container = document.createElement("div");
    render(
      renderTracing(
        buildProps({
          selectedId: "trace-1",
          selected: buildRecord(),
          onCloseDetail,
        }),
      ),
      container,
    );
    await Promise.resolve();

    expect(container.querySelector(".tracing-detail-overlay")).not.toBeNull();
    expect(container.querySelectorAll(".tracing-json-card").length).toBe(2);
    expect(container.textContent).toContain("Request Payloads");
    expect(container.textContent).toContain("Response Snapshot");
    expect(container.textContent).toContain("Summary");
    expect(container.textContent).toContain("Trace Summary JSON");
    expect(container.textContent).toContain("Input Token Composition");
    expect(container.textContent).toContain("System prompts");
    expect(container.textContent).toContain("Conversation and state history");
    expect(container.textContent).toContain("Stream started");
    expect(container.textContent).toContain("Request payload captured");
    expect(container.textContent).toContain("Response completed");
    expect(container.querySelector(".tracing-summary-json")?.textContent).toContain(
      '"provider": "openai"',
    );
    const metaText = container.querySelector(".tracing-detail-meta")?.textContent ?? "";
    expect(metaText).not.toContain("Trace ID");
    expect(metaText).not.toContain("Run");

    (container.querySelector(".tracing-detail-overlay") as HTMLElement).click();
    expect(onCloseDetail).toHaveBeenCalledTimes(1);
  });
});
