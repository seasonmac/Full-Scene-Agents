import { describe, expect, it, vi } from "vitest";
import { loadTraces, type TracingState } from "./tracing.ts";

function createHarness(): { state: TracingState; request: ReturnType<typeof vi.fn> } {
  const request = vi
    .fn()
    .mockResolvedValueOnce({
      cursor: 1,
      records: [
        {
          traceId: "trace-1",
          startedAt: 100,
          updatedAt: 100,
          status: "ok",
          requestCount: 1,
          requestPreview: "first input",
        },
      ],
    })
    .mockResolvedValueOnce({
      cursor: 2,
      records: [],
    });

  return {
    state: {
      client: {
        request,
      } as never,
      connected: true,
      tracingLoading: false,
      tracingError: null,
      tracingCursor: null,
      tracingFile: null,
      tracingEntries: [],
      tracingTruncated: false,
      tracingLimit: 100,
      tracingMaxBytes: 500000,
      tracingSelectedId: null,
      tracingSelected: null,
      tracingFilterText: "",
      tracingProviderFilter: "",
      tracingStatusFilter: "all",
      tracingAutoFollow: true,
    },
    request,
  };
}

describe("loadTraces", () => {
  it("keeps current entries when quiet polling returns no new trace records", async () => {
    const { state, request } = createHarness();

    await loadTraces(state, { reset: true });
    expect(state.tracingEntries).toHaveLength(1);
    expect(state.tracingEntries[0]?.traceId).toBe("trace-1");
    expect(state.tracingSelectedId).toBeNull();
    expect(request.mock.calls).toHaveLength(1);

    await loadTraces(state, { quiet: true });
    expect(state.tracingEntries).toHaveLength(1);
    expect(state.tracingEntries[0]?.requestPreview).toBe("first input");
    expect(state.tracingSelectedId).toBeNull();
    expect(request.mock.calls).toHaveLength(2);
  });
});
