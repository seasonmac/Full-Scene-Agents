/**
 * EBM Python sidecar HTTP client.
 *
 * Wraps all ebm_context_engine.server endpoints with typed request/response shapes,
 * timeout via AbortController, and unified error handling.
 */

// ── Response types ──────────────────────────────────────────

export interface HealthResult {
  ok: boolean;
  engine: string;
}

export interface StatusResult {
  pending: number;
  running: number;
  done: number;
  failed: number;
}

export interface BootstrapResult {
  bootstrapped: boolean;
  importedMessages: number;
}

export interface IngestResult {
  ingested: boolean;
}

export interface IngestBatchResult {
  ingestedCount: number;
}

export interface AssembleResult {
  messages: unknown[];
  estimatedTokens: number;
  systemPromptAddition?: string;
}

export interface CompactResult {
  ok: boolean;
  compacted: boolean;
  reason?: string;
  result?: {
    summary?: string;
    firstKeptEntryId?: string;
    tokensBefore: number;
    tokensAfter?: number;
    details?: unknown;
  };
}

export interface QueryResult {
  answer: string;
  context: string;
  debug: Record<string, unknown>;
}

export interface MemorySearchHit {
  id: string;
  title: string;
  content: string;
  source: string;
  score: number;
  reason?: string;
  evidence?: unknown;
  session_key?: string;
  turn_index?: number;
  verified?: boolean;
  verificationNote?: string;
}

export interface MemoryGetResult {
  item: Record<string, unknown> | null;
}

export interface MemoryForgetResult {
  forgotten: boolean;
  id: string;
  type?: string;
  reason?: string;
}

export interface ArchiveExpandMessage {
  role: string;
  text: string;
  messageIndex?: number;
  sessionId?: string;
  sessionKey?: string;
  sessionFile?: string;
  evidence?: unknown;
}

export interface ArchiveExpandResult {
  archiveId: string;
  sessionId?: string;
  sessionKey?: string;
  summary?: Record<string, unknown> | null;
  messages: ArchiveExpandMessage[];
  source: string;
}

export interface FlushResult {
  status: StatusResult;
}

export interface RetryFailedResult {
  retried: number;
}

// ── Client ──────────────────────────────────────────────────

export class EbmPyClient {
  readonly baseUrl: string;
  private readonly timeoutMs: number;

  constructor(baseUrl: string, timeoutMs = 120_000) {
    this.baseUrl = baseUrl.replace(/\/+$/, "");
    this.timeoutMs = timeoutMs;
  }

  // ── Core context engine endpoints ───────────────────────

  async healthCheck(timeoutMs?: number): Promise<HealthResult> {
    return this.get<HealthResult>("/health", timeoutMs ?? 5_000);
  }

  async bootstrap(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile?: string;
  }): Promise<BootstrapResult> {
    const res = await this.post<{ ok: boolean } & BootstrapResult>("/bootstrap", params);
    return { bootstrapped: res.bootstrapped ?? true, importedMessages: res.importedMessages ?? 0 };
  }

  async ingest(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile?: string;
    message: { role: string; content: string; timestamp?: number };
  }): Promise<IngestResult> {
    const res = await this.post<{ ok: boolean; ingested?: boolean }>("/ingest", params);
    return { ingested: res.ingested ?? true };
  }

  async ingestBatch(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile?: string;
    messages: Array<{ role: string; content: string; timestamp?: number }>;
  }): Promise<IngestBatchResult> {
    const res = await this.post<{ ok: boolean; ingestedCount?: number }>("/ingest-batch", params);
    return { ingestedCount: res.ingestedCount ?? 0 };
  }

  async assemble(params: {
    sessionId: string;
    sessionKey?: string;
    messages: unknown[];
    tokenBudget?: number;
    prompt?: string;
    runtimeContext?: Record<string, unknown>;
  }): Promise<AssembleResult> {
    const res = await this.post<{ ok: boolean } & AssembleResult>("/assemble", {
      ...params,
      tokenBudget: params.tokenBudget ?? 16_000,
      prompt: params.prompt ?? "",
    });
    return {
      messages: res.messages ?? params.messages,
      estimatedTokens: res.estimatedTokens ?? 0,
      ...(res.systemPromptAddition ? { systemPromptAddition: res.systemPromptAddition } : {}),
    };
  }

  async afterTurn(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    messages: unknown[];
    prePromptMessageCount: number;
    tokenBudget?: number;
    runtimeContext?: Record<string, unknown>;
  }): Promise<void> {
    await this.post("/after-turn", params);
  }

  async compact(params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile?: string;
    tokenBudget?: number;
    force?: boolean;
    currentTokenCount?: number;
    compactionTarget?: "budget" | "threshold";
    customInstructions?: string;
    runtimeContext?: Record<string, unknown>;
  }): Promise<CompactResult> {
    const res = await this.post<{
      ok: boolean;
      compacted?: boolean;
      reason?: string;
      result?: CompactResult["result"];
    }>(
      "/compact",
      params,
    );
    return {
      ok: res.ok ?? true,
      compacted: res.compacted ?? false,
      reason: res.reason,
      result: res.result,
    };
  }

  // ── Operational endpoints ─────────────────────────────

  async status(): Promise<StatusResult> {
    const res = await this.get<{ ok: boolean; status: StatusResult }>("/status");
    return res.status;
  }

  async flush(): Promise<FlushResult> {
    const res = await this.post<{ ok: boolean; status: StatusResult }>("/flush");
    return { status: res.status };
  }

  async retryFailed(): Promise<RetryFailedResult> {
    const res = await this.post<{ ok: boolean; retried: number }>("/retry-failed");
    return { retried: res.retried ?? 0 };
  }

  async dispose(): Promise<void> {
    try {
      await this.post("/dispose", {}, 5_000);
    } catch {
      // best-effort — sidecar may already be gone
    }
  }

  // ── Diagnostic endpoints ──────────────────────────────

  async query(question: string, useAaak = false): Promise<QueryResult> {
    return this.post<{ ok: boolean } & QueryResult>("/query", { question, useAaak });
  }

  async memorySearch(query: string, limit = 10): Promise<MemorySearchHit[]> {
    const res = await this.post<{ ok: boolean; results: MemorySearchHit[] }>("/memory-search", {
      query,
      limit,
    });
    return res.results ?? [];
  }

  async memoryGet(id: string): Promise<MemoryGetResult> {
    const res = await this.post<{ ok: boolean; item?: Record<string, unknown> | null }>(
      "/memory-get",
      { id },
    );
    return { item: res.item ?? null };
  }

  async memoryForget(id: string): Promise<MemoryForgetResult> {
    const res = await this.post<{ ok: boolean } & MemoryForgetResult>("/memory-forget", { id });
    return {
      forgotten: res.forgotten ?? false,
      id: res.id ?? id,
      type: res.type,
      reason: res.reason,
    };
  }

  async archiveExpand(params: {
    archiveId: string;
    sessionId?: string;
    sessionKey?: string;
    limit?: number;
  }): Promise<ArchiveExpandResult> {
    const res = await this.post<{ ok: boolean } & ArchiveExpandResult>("/archive-expand", params);
    return {
      archiveId: res.archiveId ?? params.archiveId,
      sessionId: res.sessionId,
      sessionKey: res.sessionKey,
      summary: res.summary ?? null,
      messages: res.messages ?? [],
      source: res.source ?? "unknown",
    };
  }

  // ── Internal HTTP primitives ──────────────────────────

  private async get<T>(path: string, timeoutMs?: number): Promise<T> {
    return this.request<T>("GET", path, undefined, timeoutMs);
  }

  private async post<T>(
    path: string,
    body?: Record<string, unknown>,
    timeoutMs?: number,
  ): Promise<T> {
    return this.request<T>("POST", path, body, timeoutMs);
  }

  private async request<T>(
    method: string,
    path: string,
    body?: Record<string, unknown>,
    timeoutMs?: number,
  ): Promise<T> {
    const url = `${this.baseUrl}${path}`;
    const timeout = timeoutMs ?? this.timeoutMs;

    const response = await fetch(url, {
      method,
      headers: body != null ? { "Content-Type": "application/json" } : undefined,
      body: body != null ? JSON.stringify(body) : undefined,
      signal: AbortSignal.timeout(timeout),
    });

    if (!response.ok) {
      const text = await response.text().catch(() => "");
      throw new Error(`EBM-PY ${method} ${path} failed: ${response.status} ${text.slice(0, 500)}`);
    }

    const payload = (await response.json()) as Record<string, unknown>;
    if (payload.ok === false) {
      const msg =
        typeof payload.error === "string" ? payload.error : JSON.stringify(payload.error);
      throw new Error(`EBM-PY ${path} error: ${msg}`);
    }

    return payload as T;
  }
}

// ── Module-level local client cache ─────────────────────────

export type LocalRuntimeState = {
  client: EbmPyClient | null;
  process: import("node:child_process").ChildProcess | null;
  startupPromise: Promise<EbmPyClient> | null;
  refCount: number;
  startupOwners: number;
};

const EMPTY_LOCAL_RUNTIME_STATE: LocalRuntimeState = {
  client: null,
  process: null,
  startupPromise: null,
  refCount: 0,
  startupOwners: 0,
};

const localRuntimeStates = new Map<string, LocalRuntimeState>();

export function getCachedLocalClient(runtimeKey = "default"): EbmPyClient | null {
  return getLocalRuntimeState(runtimeKey).client;
}

export function setCachedLocalClient(client: EbmPyClient, runtimeKey = "default"): void {
  setLocalRuntimeState(runtimeKey, { client });
}

export function clearCachedLocalClient(runtimeKey = "default"): void {
  localRuntimeStates.delete(runtimeKey);
}

export function getLocalRuntimeState(runtimeKey = "default"): LocalRuntimeState {
  return localRuntimeStates.get(runtimeKey) ?? { ...EMPTY_LOCAL_RUNTIME_STATE };
}

export function setLocalRuntimeState(
  runtimeKey: string,
  patch: Partial<LocalRuntimeState>,
): LocalRuntimeState {
  const current = getLocalRuntimeState(runtimeKey);
  const next = {
    ...current,
    ...patch,
  };
  localRuntimeStates.set(runtimeKey, next);
  return next;
}
