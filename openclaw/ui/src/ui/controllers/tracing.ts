import type { GatewayBrowserClient } from "../gateway.ts";
import type { LlmTraceRecord, LlmTraceSummary } from "../types.ts";
import {
  formatMissingOperatorReadScopeMessage,
  isMissingOperatorReadScopeError,
} from "./scope-errors.ts";

export type TracingState = {
  client: GatewayBrowserClient | null;
  connected: boolean;
  tracingLoading: boolean;
  tracingError: string | null;
  tracingCursor: number | null;
  tracingFile: string | null;
  tracingEntries: LlmTraceSummary[];
  tracingTruncated: boolean;
  tracingLimit: number;
  tracingMaxBytes: number;
  tracingSelectedId: string | null;
  tracingSelected: LlmTraceRecord | null;
  tracingFilterText: string;
  tracingProviderFilter: string;
  tracingStatusFilter: "all" | "ok" | "error" | "in_progress";
  tracingAutoFollow: boolean;
};

const TRACE_BUFFER_LIMIT = 300;

function mergeTraceEntries(
  current: LlmTraceSummary[],
  incoming: LlmTraceSummary[],
): LlmTraceSummary[] {
  const merged = new Map<string, LlmTraceSummary>();
  for (const entry of current) {
    merged.set(entry.traceId, entry);
  }
  for (const entry of incoming) {
    merged.set(entry.traceId, entry);
  }
  return Array.from(merged.values())
    .toSorted(
      (left, right) => (right.updatedAt ?? right.startedAt) - (left.updatedAt ?? left.startedAt),
    )
    .slice(0, TRACE_BUFFER_LIMIT);
}

export async function loadTraces(state: TracingState, opts?: { reset?: boolean; quiet?: boolean }) {
  if (!state.client || !state.connected) {
    return;
  }
  if (state.tracingLoading && !opts?.quiet) {
    return;
  }
  if (!opts?.quiet) {
    state.tracingLoading = true;
  }
  state.tracingError = null;
  try {
    const res = await state.client.request("traces.tail", {
      cursor: opts?.reset ? undefined : (state.tracingCursor ?? undefined),
      limit: state.tracingLimit,
      maxBytes: state.tracingMaxBytes,
      query: state.tracingFilterText || undefined,
      provider: state.tracingProviderFilter || undefined,
      status: state.tracingStatusFilter === "all" ? undefined : state.tracingStatusFilter,
    });
    const payload = res as {
      file?: string;
      cursor?: number;
      records?: unknown[];
      truncated?: boolean;
      reset?: boolean;
    };
    const entries = Array.isArray(payload.records) ? (payload.records as LlmTraceSummary[]) : [];
    const shouldReset = Boolean(opts?.reset || payload.reset || state.tracingCursor == null);
    state.tracingEntries = shouldReset ? entries : mergeTraceEntries(state.tracingEntries, entries);
    state.tracingCursor = typeof payload.cursor === "number" ? payload.cursor : state.tracingCursor;
    state.tracingFile = typeof payload.file === "string" ? payload.file : state.tracingFile;
    state.tracingTruncated = Boolean(payload.truncated);
    if (
      state.tracingSelectedId &&
      !state.tracingEntries.some((entry) => entry.traceId === state.tracingSelectedId)
    ) {
      state.tracingSelectedId = null;
      state.tracingSelected = null;
    }
    if (state.tracingSelectedId) {
      await loadTraceDetail(state, state.tracingSelectedId);
    }
  } catch (error) {
    if (isMissingOperatorReadScopeError(error)) {
      state.tracingEntries = [];
      state.tracingSelected = null;
      state.tracingError = formatMissingOperatorReadScopeMessage("traces");
    } else {
      state.tracingError = String(error);
    }
  } finally {
    if (!opts?.quiet) {
      state.tracingLoading = false;
    }
  }
}

export async function loadTraceDetail(state: TracingState, traceId: string) {
  if (!state.client || !state.connected || !traceId) {
    return;
  }
  state.tracingSelectedId = traceId;
  try {
    const res = await state.client.request("traces.get", { traceId });
    state.tracingSelected = res as LlmTraceRecord;
  } catch (error) {
    if (isMissingOperatorReadScopeError(error)) {
      state.tracingSelected = null;
      state.tracingError = formatMissingOperatorReadScopeMessage("traces");
    } else {
      state.tracingError = String(error);
    }
  }
}
