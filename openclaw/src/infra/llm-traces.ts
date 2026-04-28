import crypto from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type { StreamFn } from "@mariozechner/pi-agent-core";
import type { Api, AssistantMessageEventStream, Model } from "@mariozechner/pi-ai";
import { sanitizeDiagnosticPayload } from "../agents/payload-redaction.js";
import { getQueuedFileWriter, type QueuedFileWriter } from "../agents/queued-file-writer.js";
import { resolveStateDir } from "../config/paths.js";
import { clamp } from "../utils.js";
import { safeJsonStringify } from "../utils/safe-json.js";

export type LlmTraceStatus = "ok" | "error" | "in_progress";

export type LlmTraceTimelineEntry = {
  ts: number;
  kind: "request" | "response" | "error";
  label: string;
  detail?: string;
};

export type LlmTraceUsage = {
  input?: number;
  output?: number;
  cacheRead?: number;
  cacheWrite?: number;
  promptTokens?: number;
  total?: number;
};

export type LlmTraceSummary = {
  traceId: string;
  runId?: string;
  sessionId?: string;
  sessionKey?: string;
  provider?: string;
  modelId?: string;
  modelApi?: string | null;
  workspaceDir?: string;
  startedAt: number;
  endedAt?: number;
  updatedAt: number;
  status: LlmTraceStatus;
  durationMs?: number;
  usage?: LlmTraceUsage;
  costTotal?: number;
  error?: string;
  requestCount: number;
  requestPreview?: string;
  responsePreview?: string;
};

export type LlmTraceRecord = LlmTraceSummary & {
  requests: unknown[];
  response?: unknown;
  timeline: LlmTraceTimelineEntry[];
};

type TraceState = {
  byId: Map<string, LlmTraceRecord>;
  order: string[];
  maxEntries: number;
};

type TraceFileSlice = {
  file: string;
  cursor: number;
  size: number;
  records: LlmTraceSummary[];
  truncated: boolean;
  reset: boolean;
};

type TraceTailFilters = {
  query?: string;
  provider?: string;
  status?: LlmTraceStatus;
};

type LlmTraceRecorderParams = {
  env?: NodeJS.ProcessEnv;
  runId?: string;
  sessionId?: string;
  sessionKey?: string;
  provider?: string;
  modelId?: string;
  modelApi?: string | null;
  workspaceDir?: string;
  writer?: QueuedFileWriter;
};

type PersistedTraceRecord = LlmTraceRecord & { persistedAt: string };

const TRACE_BUFFER_LIMIT = 300;
const DEFAULT_LIMIT = 100;
const DEFAULT_MAX_BYTES = 500_000;
const MAX_LIMIT = 1000;
const MAX_BYTES = 2_000_000;
const TRACE_FILE_NAME = "llm-trace.jsonl";
const writers = new Map<string, QueuedFileWriter>();

function getTraceState(): TraceState {
  const globalStore = globalThis as typeof globalThis & {
    __openclawLlmTraceState?: TraceState;
  };
  if (!globalStore.__openclawLlmTraceState) {
    globalStore.__openclawLlmTraceState = {
      byId: new Map<string, LlmTraceRecord>(),
      order: [],
      maxEntries: TRACE_BUFFER_LIMIT,
    };
  }
  return globalStore.__openclawLlmTraceState;
}

function resolveTraceFilePath(env: NodeJS.ProcessEnv = process.env): string {
  return path.join(resolveStateDir(env), "logs", TRACE_FILE_NAME);
}

function getTraceWriter(filePath: string): QueuedFileWriter {
  return getQueuedFileWriter(writers, filePath);
}

function truncateString(value: string, maxLength = 16_000): string {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength)}...[truncated ${value.length - maxLength} chars]`;
}

function truncateTraceValue(value: unknown, depth = 0): unknown {
  if (depth > 6) {
    return "[truncated:depth]";
  }
  if (typeof value === "string") {
    return truncateString(value);
  }
  if (Array.isArray(value)) {
    const maxItems = 100;
    const items = value.slice(0, maxItems).map((entry) => truncateTraceValue(entry, depth + 1));
    if (value.length > maxItems) {
      items.push(`[truncated ${value.length - maxItems} items]`);
    }
    return items;
  }
  if (!value || typeof value !== "object") {
    return value;
  }
  const entries = Object.entries(value as Record<string, unknown>);
  const maxEntries = 100;
  const output: Record<string, unknown> = {};
  for (const [index, [key, entryValue]] of entries.entries()) {
    if (index >= maxEntries) {
      output.__truncated__ = `${entries.length - maxEntries} entries omitted`;
      break;
    }
    output[key] = truncateTraceValue(entryValue, depth + 1);
  }
  return output;
}

function sanitizeTraceValue(value: unknown): unknown {
  return truncateTraceValue(sanitizeDiagnosticPayload(value));
}

function toTraceUsage(value: unknown): LlmTraceUsage | undefined {
  if (!value || typeof value !== "object") {
    return undefined;
  }
  const usage = value as Record<string, unknown>;
  const out: LlmTraceUsage = {};
  if (typeof usage.input === "number") {
    out.input = usage.input;
  } else if (typeof usage.inputTokens === "number") {
    out.input = usage.inputTokens;
  } else if (typeof usage.input_tokens === "number") {
    out.input = usage.input_tokens;
  }
  if (typeof usage.output === "number") {
    out.output = usage.output;
  } else if (typeof usage.outputTokens === "number") {
    out.output = usage.outputTokens;
  } else if (typeof usage.output_tokens === "number") {
    out.output = usage.output_tokens;
  }
  if (typeof usage.cacheRead === "number") {
    out.cacheRead = usage.cacheRead;
  } else if (typeof usage.cache_read_input_tokens === "number") {
    out.cacheRead = usage.cache_read_input_tokens;
  }
  if (typeof usage.cacheWrite === "number") {
    out.cacheWrite = usage.cacheWrite;
  } else if (typeof usage.cache_creation_input_tokens === "number") {
    out.cacheWrite = usage.cache_creation_input_tokens;
  }
  if (typeof usage.promptTokens === "number") {
    out.promptTokens = usage.promptTokens;
  } else if (typeof usage.prompt_tokens === "number") {
    out.promptTokens = usage.prompt_tokens;
  }
  if (typeof usage.total === "number") {
    out.total = usage.total;
  } else if (typeof usage.totalTokens === "number") {
    out.total = usage.totalTokens;
  } else if (typeof usage.total_tokens === "number") {
    out.total = usage.total_tokens;
  } else if (typeof out.input === "number" || typeof out.output === "number") {
    out.total = (out.input ?? 0) + (out.output ?? 0) + (out.cacheRead ?? 0) + (out.cacheWrite ?? 0);
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

function extractContextRequestPreview(context: unknown): string | undefined {
  if (!context || typeof context !== "object") {
    return undefined;
  }
  const record = context as Record<string, unknown>;
  const candidateEntries = Object.entries({
    messages: record.messages,
    system: record.system,
    prompt: record.prompt,
    instructions: record.instructions,
    input: record.input,
  }).filter(([, value]) => value != null);
  if (candidateEntries.length === 0) {
    return undefined;
  }
  const candidate = Object.fromEntries(candidateEntries);
  return extractRequestPreview(candidate);
}

function extractResponsePreview(response: unknown): string | undefined {
  if (!response || typeof response !== "object") {
    return undefined;
  }
  const record = response as Record<string, unknown>;
  const content = record.content;
  if (typeof content === "string") {
    return truncateString(content, 240);
  }
  if (Array.isArray(content)) {
    const joined = content
      .map((entry) => {
        if (typeof entry === "string") {
          return entry;
        }
        if (entry && typeof entry === "object") {
          const text = (entry as Record<string, unknown>).text;
          return typeof text === "string" ? text : (safeJsonStringify(entry) ?? "");
        }
        return "";
      })
      .filter(Boolean)
      .join(" ");
    return joined ? truncateString(joined, 240) : undefined;
  }
  return undefined;
}

function collectPreviewSegments(value: unknown, depth = 0): string[] {
  if (depth > 5 || value == null) {
    return [];
  }
  if (typeof value === "string") {
    const trimmed = value.trim();
    return trimmed ? [trimmed] : [];
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return [String(value)];
  }
  if (Array.isArray(value)) {
    return value.slice(0, 6).flatMap((entry) => collectPreviewSegments(entry, depth + 1));
  }
  if (typeof value !== "object") {
    return [];
  }

  const record = value as Record<string, unknown>;
  const prioritizedKeys = [
    "input",
    "messages",
    "message",
    "prompt",
    "content",
    "text",
    "instructions",
    "system",
  ];
  const prioritized = prioritizedKeys.flatMap((key) =>
    collectPreviewSegments(record[key], depth + 1),
  );
  if (prioritized.length > 0) {
    return prioritized;
  }
  return Object.values(record)
    .slice(0, 6)
    .flatMap((entry) => collectPreviewSegments(entry, depth + 1));
}

function extractRequestPreview(payload: unknown): string | undefined {
  const segments = collectPreviewSegments(payload)
    .map((segment) => segment.replace(/\s+/g, " ").trim())
    .filter(Boolean);
  if (segments.length === 0) {
    const fallback = safeJsonStringify(payload);
    return fallback ? truncateString(fallback, 240) : undefined;
  }
  return truncateString(segments.join(" "), 240);
}

function formatError(error: unknown): string | undefined {
  if (error instanceof Error) {
    return error.message;
  }
  if (typeof error === "string") {
    return error;
  }
  if (error == null) {
    return undefined;
  }
  return safeJsonStringify(error) ?? Object.prototype.toString.call(error);
}

function extractCostTotal(message: unknown): number | undefined {
  if (!message || typeof message !== "object") {
    return undefined;
  }
  const cost = (message as Record<string, unknown>).cost;
  if (!cost || typeof cost !== "object") {
    return undefined;
  }
  const total = (cost as Record<string, unknown>).total;
  return typeof total === "number" ? total : undefined;
}

function extractUsageFromAssistantMessage(message: unknown): LlmTraceUsage | undefined {
  if (!message || typeof message !== "object") {
    return undefined;
  }
  return toTraceUsage((message as Record<string, unknown>).usage);
}

function toSummary(record: LlmTraceRecord): LlmTraceSummary {
  return {
    traceId: record.traceId,
    runId: record.runId,
    sessionId: record.sessionId,
    sessionKey: record.sessionKey,
    provider: record.provider,
    modelId: record.modelId,
    modelApi: record.modelApi,
    workspaceDir: record.workspaceDir,
    startedAt: record.startedAt,
    endedAt: record.endedAt,
    updatedAt: record.updatedAt,
    status: record.status,
    durationMs: record.durationMs,
    usage: record.usage,
    costTotal: record.costTotal,
    error: record.error,
    requestCount: record.requestCount,
    requestPreview: record.requestPreview,
    responsePreview: record.responsePreview,
  };
}

function upsertTraceRecord(record: LlmTraceRecord) {
  const state = getTraceState();
  state.byId.set(record.traceId, record);
  const existingIndex = state.order.indexOf(record.traceId);
  if (existingIndex >= 0) {
    state.order.splice(existingIndex, 1);
  }
  state.order.unshift(record.traceId);
  while (state.order.length > state.maxEntries) {
    const dropped = state.order.pop();
    if (dropped) {
      state.byId.delete(dropped);
    }
  }
}

function persistTraceRecord(writer: QueuedFileWriter, record: LlmTraceRecord) {
  const persisted: PersistedTraceRecord = {
    ...record,
    persistedAt: new Date().toISOString(),
  };
  const line = safeJsonStringify(persisted);
  if (!line) {
    return;
  }
  writer.write(`${line}\n`);
}

function recordFromJsonLine(line: string): LlmTraceRecord | null {
  if (!line.trim()) {
    return null;
  }
  try {
    const parsed = JSON.parse(line) as Partial<LlmTraceRecord>;
    if (!parsed || typeof parsed.traceId !== "string" || typeof parsed.startedAt !== "number") {
      return null;
    }
    return {
      traceId: parsed.traceId,
      runId: typeof parsed.runId === "string" ? parsed.runId : undefined,
      sessionId: typeof parsed.sessionId === "string" ? parsed.sessionId : undefined,
      sessionKey: typeof parsed.sessionKey === "string" ? parsed.sessionKey : undefined,
      provider: typeof parsed.provider === "string" ? parsed.provider : undefined,
      modelId: typeof parsed.modelId === "string" ? parsed.modelId : undefined,
      modelApi: typeof parsed.modelApi === "string" ? parsed.modelApi : null,
      workspaceDir: typeof parsed.workspaceDir === "string" ? parsed.workspaceDir : undefined,
      startedAt: parsed.startedAt,
      endedAt: typeof parsed.endedAt === "number" ? parsed.endedAt : undefined,
      updatedAt: typeof parsed.updatedAt === "number" ? parsed.updatedAt : parsed.startedAt,
      status:
        parsed.status === "ok" || parsed.status === "error" || parsed.status === "in_progress"
          ? parsed.status
          : "ok",
      durationMs: typeof parsed.durationMs === "number" ? parsed.durationMs : undefined,
      usage: toTraceUsage(parsed.usage),
      costTotal: typeof parsed.costTotal === "number" ? parsed.costTotal : undefined,
      error: typeof parsed.error === "string" ? parsed.error : undefined,
      requestCount: typeof parsed.requestCount === "number" ? parsed.requestCount : 0,
      requestPreview: typeof parsed.requestPreview === "string" ? parsed.requestPreview : undefined,
      responsePreview:
        typeof parsed.responsePreview === "string" ? parsed.responsePreview : undefined,
      requests: Array.isArray(parsed.requests) ? parsed.requests : [],
      response: parsed.response,
      timeline: Array.isArray(parsed.timeline)
        ? parsed.timeline.filter((entry): entry is LlmTraceTimelineEntry => {
            if (!entry || typeof entry !== "object") {
              return false;
            }
            const timelineEntry = entry as Record<string, unknown>;
            return (
              typeof timelineEntry.ts === "number" &&
              typeof timelineEntry.label === "string" &&
              typeof timelineEntry.kind === "string"
            );
          })
        : [],
    };
  } catch {
    return null;
  }
}

async function resolveTraceRecordFromFile(
  traceId: string,
  filePath: string,
): Promise<LlmTraceRecord | null> {
  const content = await fs.readFile(filePath, "utf8").catch(() => null);
  if (!content) {
    return null;
  }
  const lines = content.split("\n");
  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const record = recordFromJsonLine(lines[index] ?? "");
    if (record?.traceId === traceId) {
      return record;
    }
  }
  return null;
}

async function readTraceSlice(params: {
  file: string;
  cursor?: number;
  limit: number;
  maxBytes: number;
}): Promise<TraceFileSlice> {
  const stat = await fs.stat(params.file).catch(() => null);
  if (!stat) {
    return {
      file: params.file,
      cursor: 0,
      size: 0,
      records: [],
      truncated: false,
      reset: false,
    };
  }

  const size = stat.size;
  const maxBytes = clamp(params.maxBytes, 1, MAX_BYTES);
  const limit = clamp(params.limit, 1, MAX_LIMIT);
  let cursor =
    typeof params.cursor === "number" && Number.isFinite(params.cursor)
      ? Math.max(0, Math.floor(params.cursor))
      : undefined;
  let reset = false;
  let truncated = false;
  let start = 0;

  if (cursor != null) {
    if (cursor > size) {
      reset = true;
      start = Math.max(0, size - maxBytes);
      truncated = start > 0;
    } else {
      start = cursor;
      if (size - start > maxBytes) {
        reset = true;
        truncated = true;
        start = Math.max(0, size - maxBytes);
      }
    }
  } else {
    start = Math.max(0, size - maxBytes);
    truncated = start > 0;
  }

  if (size === 0 || size <= start) {
    return {
      file: params.file,
      cursor: size,
      size,
      records: [],
      truncated,
      reset,
    };
  }

  const handle = await fs.open(params.file, "r");
  try {
    let prefix = "";
    if (start > 0) {
      const prefixBuffer = Buffer.alloc(1);
      const prefixRead = await handle.read(prefixBuffer, 0, 1, start - 1);
      prefix = prefixBuffer.toString("utf8", 0, prefixRead.bytesRead);
    }

    const length = Math.max(0, size - start);
    const buffer = Buffer.alloc(length);
    const readResult = await handle.read(buffer, 0, length, start);
    let lines = buffer.toString("utf8", 0, readResult.bytesRead).split("\n");
    if (start > 0 && prefix !== "\n") {
      lines = lines.slice(1);
    }
    if (lines.length > 0 && lines[lines.length - 1] === "") {
      lines = lines.slice(0, -1);
    }

    const records = lines
      .map((line) => recordFromJsonLine(line))
      .filter((record): record is LlmTraceRecord => Boolean(record))
      .map((record) => toSummary(record));

    return {
      file: params.file,
      cursor: size,
      size,
      records: records.slice(-limit),
      truncated,
      reset,
    };
  } finally {
    await handle.close();
  }
}

function matchesTraceFilters(record: LlmTraceSummary, filters: TraceTailFilters): boolean {
  const provider = filters.provider?.trim().toLowerCase();
  if (provider && (record.provider ?? "").toLowerCase() !== provider) {
    return false;
  }
  if (filters.status && record.status !== filters.status) {
    return false;
  }
  const query = filters.query?.trim().toLowerCase();
  if (!query) {
    return true;
  }
  const haystack = [
    record.traceId,
    record.runId,
    record.sessionId,
    record.sessionKey,
    record.provider,
    record.modelId,
    record.modelApi,
    record.workspaceDir,
    record.error,
    record.requestPreview,
    record.responsePreview,
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

export async function tailLlmTraces(params: {
  cursor?: number;
  limit?: number;
  maxBytes?: number;
  query?: string;
  provider?: string;
  status?: LlmTraceStatus;
  env?: NodeJS.ProcessEnv;
}): Promise<TraceFileSlice> {
  const file = resolveTraceFilePath(params.env);
  const fromFile = await readTraceSlice({
    file,
    cursor: params.cursor,
    limit: params.limit ?? DEFAULT_LIMIT,
    maxBytes: params.maxBytes ?? DEFAULT_MAX_BYTES,
  });
  const state = getTraceState();
  const merged = new Map<string, LlmTraceSummary>();
  for (const summary of fromFile.records) {
    merged.set(summary.traceId, summary);
  }
  for (const traceId of state.order) {
    const record = state.byId.get(traceId);
    if (!record) {
      continue;
    }
    merged.set(traceId, toSummary(record));
  }
  return {
    ...fromFile,
    records: Array.from(merged.values())
      .filter((record) =>
        matchesTraceFilters(record, {
          query: params.query,
          provider: params.provider,
          status: params.status,
        }),
      )
      .toSorted((left, right) => right.updatedAt - left.updatedAt)
      .slice(0, clamp(params.limit ?? DEFAULT_LIMIT, 1, MAX_LIMIT)),
  };
}

export async function getLlmTrace(params: {
  traceId: string;
  env?: NodeJS.ProcessEnv;
}): Promise<LlmTraceRecord | null> {
  const state = getTraceState();
  const inMemory = state.byId.get(params.traceId);
  if (inMemory) {
    return inMemory;
  }
  return await resolveTraceRecordFromFile(params.traceId, resolveTraceFilePath(params.env));
}

export type LlmTraceRecorder = {
  wrapStreamFn: (streamFn: StreamFn) => StreamFn;
};

export function createLlmTraceRecorder(params: LlmTraceRecorderParams): LlmTraceRecorder {
  const writer = params.writer ?? getTraceWriter(resolveTraceFilePath(params.env ?? process.env));

  const wrapStreamFn: LlmTraceRecorder["wrapStreamFn"] = (streamFn) => {
    const wrapped: StreamFn = (model, context, options) => {
      const traceId = crypto.randomUUID();
      const startedAt = Date.now();
      const requestPreview = extractContextRequestPreview(context);
      const record: LlmTraceRecord = {
        traceId,
        runId: params.runId,
        sessionId: params.sessionId,
        sessionKey: params.sessionKey,
        provider: params.provider,
        modelId: params.modelId ?? model.id,
        modelApi: params.modelApi ?? model.api,
        workspaceDir: params.workspaceDir,
        startedAt,
        updatedAt: startedAt,
        status: "in_progress",
        requestCount: 0,
        requestPreview,
        requests: [],
        timeline: [{ ts: startedAt, kind: "request", label: "Stream started" }],
      };
      upsertTraceRecord(record);

      const originalOnPayload = options?.onPayload;
      const nextOptions = {
        ...options,
        onPayload: (payload: unknown, payloadModel: Model<Api>) => {
          record.requests.push(sanitizeTraceValue(payload));
          record.requestCount = record.requests.length;
          record.requestPreview ??= extractRequestPreview(payload);
          record.updatedAt = Date.now();
          record.timeline.push({
            ts: record.updatedAt,
            kind: "request",
            label: record.requestCount === 1 ? "Request payload captured" : "Additional request",
            detail: `${payloadModel.provider ?? params.provider ?? "unknown"}/${payloadModel.id ?? params.modelId ?? "unknown"}`,
          });
          upsertTraceRecord(record);
          return originalOnPayload?.(payload, payloadModel);
        },
      } as Parameters<StreamFn>[2];

      const finalize = (result: unknown, error?: unknown) => {
        const finishedAt = Date.now();
        record.updatedAt = finishedAt;
        record.endedAt = finishedAt;
        record.durationMs = Math.max(0, finishedAt - startedAt);
        if (error) {
          record.status = "error";
          record.error = formatError(error);
          record.timeline.push({
            ts: finishedAt,
            kind: "error",
            label: "Stream failed",
            detail: record.error,
          });
        } else {
          record.status = "ok";
          record.response = sanitizeTraceValue(result);
          record.usage = extractUsageFromAssistantMessage(result);
          record.costTotal = extractCostTotal(result);
          record.responsePreview = extractResponsePreview(result);
          record.timeline.push({ ts: finishedAt, kind: "response", label: "Response completed" });
        }
        upsertTraceRecord(record);
        persistTraceRecord(writer, record);
      };

      const attachResultFinalizer = (
        value: AssistantMessageEventStream,
      ): AssistantMessageEventStream => {
        const maybeStream = value as { result?: () => Promise<unknown> } | null;
        if (maybeStream && typeof maybeStream.result === "function") {
          void maybeStream
            .result()
            .then((result) => finalize(result))
            .catch((error) => finalize(undefined, error));
          return value;
        }
        finalize(undefined);
        return value;
      };

      try {
        const stream = streamFn(model, context, nextOptions);
        if (stream && typeof (stream as PromiseLike<unknown>).then === "function") {
          return (stream as Promise<AssistantMessageEventStream>).then((resolved) =>
            attachResultFinalizer(resolved),
          );
        }
        return attachResultFinalizer(stream as AssistantMessageEventStream);
      } catch (error) {
        finalize(undefined, error);
        throw error;
      }
    };
    return wrapped;
  };

  return { wrapStreamFn };
}

export function resetLlmTracesForTest() {
  const state = getTraceState();
  state.byId.clear();
  state.order = [];
}
