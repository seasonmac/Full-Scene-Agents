/**
 * ebm-context-engine — OpenClaw context engine plugin (thin TS proxy → Python sidecar)
 *
 * Architecture:
 *   OpenClaw gateway ←→ this TS plugin ←→ HTTP ←→ ebm_context_engine.server (Python)
 *
 * Supports two modes:
 *   - local: spawns a Python sidecar as a child process (default)
 *   - remote: connects to an already-running ebm_context_engine server
 *
 * Follows the same plugin structure as EBM:
 *   config.ts → client.ts → process-manager.ts → context-engine.ts → index.ts
 */
import { definePluginEntry } from "openclaw/plugin-sdk/core";
import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { spawn, type ChildProcess } from "node:child_process";
import { resolve as resolvePath } from "node:path";

import { ebmPyConfigSchema } from "./config.js";
import {
  EbmPyClient,
  clearCachedLocalClient,
  getLocalRuntimeState,
  setLocalRuntimeState,
} from "./client.js";
import type { MemorySearchHit } from "./client.js";
import {
  prepareLocalPort,
  waitForHealthOrExit,
  quickHealthCheck,
  resolvePythonCommand,
} from "./process-manager.js";
import { createEbmContextEngine } from "./context-engine.js";
import { buildMemoryLinesWithBudget } from "./memory-ranking.js";

// ── Constants ───────────────────────────────────────────────

const PLUGIN_ID = "ebm-context-engine";
const HEALTH_POLL_INTERVAL_MS = 500;
const MAX_STDERR_LINES = 200;
const MAX_STDERR_CHARS = 256_000;
const IMPLICIT_RUNTIME_OWNER = Symbol("ebm-py-implicit-runtime-owner");

type ToolContext = {
  sessionId?: string;
  sessionKey?: string;
  agentId?: string;
};

type HookAgentContext = ToolContext;

type AssembleFallbackState = {
  hasMeaningfulMemoryContext: boolean;
};

type ToolDefinition = {
  name: string;
  label: string;
  description: string;
  parameters: unknown;
  execute: (_toolCallId: string, params: Record<string, unknown>) => Promise<unknown>;
};

function textParam(params: Record<string, unknown>, key: string): string {
  const value = params[key];
  return typeof value === "string" ? value.trim() : "";
}

function numberParam(params: Record<string, unknown>, key: string, fallback: number): number {
  const value = params[key];
  const parsed = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
  return Number.isFinite(parsed) ? parsed : fallback;
}

function clampLimit(raw: number, fallback = 6, max = 50): number {
  return Math.max(1, Math.min(max, Math.floor(Number.isFinite(raw) ? raw : fallback)));
}

function extractLatestUserText(messages: unknown[] | undefined): string {
  if (!Array.isArray(messages)) {
    return "";
  }
  for (let i = messages.length - 1; i >= 0; i -= 1) {
    const msg = messages[i] as Record<string, unknown> | undefined;
    if (!msg || msg.role !== "user") {
      continue;
    }
    return messageText(msg).trim();
  }
  return "";
}

function messageText(message: Record<string, unknown>): string {
  const content = message.content;
  if (typeof content === "string") {
    return content;
  }
  if (Array.isArray(content)) {
    return content
      .map((block) => {
        if (!block || typeof block !== "object") {
          return "";
        }
        const b = block as Record<string, unknown>;
        return b.type === "text" ? String(b.text ?? "") : "";
      })
      .filter(Boolean)
      .join("\n");
  }
  return content == null ? "" : String(content);
}

function formatHit(hit: MemorySearchHit, index: number): string {
  const score = Number.isFinite(hit.score) ? ` score=${hit.score.toFixed(3)}` : "";
  const source = hit.source ? ` source=${hit.source}` : "";
  const title = hit.title || hit.id;
  const content = (hit.content || "").replace(/\s+/g, " ").trim();
  return `${index + 1}. ${title} (${hit.id}${source}${score})\n${content}`;
}

function formatMemoryHits(hits: MemorySearchHit[]): string {
  return hits.map(formatHit).join("\n\n");
}

function isTranscriptLike(text: string, minTurns = 2, minChars = 120): boolean {
  if (text.trim().length < minChars) {
    return false;
  }
  const speakerTurns = text
    .split(/\n+/)
    .filter((line) => /^[\p{L}\p{N}_ .-]{1,40}[:：]/u.test(line.trim())).length;
  return speakerTurns >= minTurns;
}

function shouldSkipSession(
  ctx: HookAgentContext | undefined,
  patterns: string[],
): boolean {
  if (!patterns.length) return false;
  const key = ctx?.sessionKey || ctx?.sessionId || "";
  if (!key) return false;
  return patterns.some((p) => matchGlob(p, key));
}

function matchGlob(pattern: string, value: string): boolean {
  const regex = pattern
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*/g, "\0")
    .replace(/\*/g, "[^:]*")
    .replace(/\0/g, ".*");
  return new RegExp(`^${regex}$`).test(value);
}

function resolveSessionStateKey(ctx: HookAgentContext | undefined): string {
  return (ctx?.sessionKey || ctx?.sessionId || "").trim();
}

function hasMeaningfulAssembleResult(systemPromptAddition: string | undefined): boolean {
  const addition = systemPromptAddition?.trim();
  if (!addition) {
    return false;
  }
  const normalized = addition.replace(/\s+/g, " ").trim();
  const boilerplate = (
    "Memory context for this query. Use these facts to answer directly and confidently. " +
    "If the memory contains relevant information, use it to answer even if not perfectly verified."
  ).replace(/\s+/g, " ").trim();
  return normalized !== boilerplate;
}

function withTimeout<T>(promise: Promise<T>, ms: number, label: string): Promise<T> {
  return Promise.race([
    promise,
    new Promise<never>((_, reject) =>
      setTimeout(() => reject(new Error(`${label} (${ms}ms)`)), ms),
    ),
  ]);
}

// ── Plugin registration ─────────────────────────────────────

function register(api: OpenClawPluginApi): void {
  const cfg = ebmPyConfigSchema.parse(api.pluginConfig);
  const assembleFallbackState = new Map<string, AssembleFallbackState>();

  api.logger.info(
    `[EBM-PY] config: mode=${cfg.mode} port=${cfg.port} timeout=${cfg.timeoutMs}ms` +
      (cfg.mode === "local" ? ` python=${cfg.pythonCommand} ebmPyPath=${cfg.ebmPyPath || "(auto)"}` : ` baseUrl=${cfg.baseUrl}`),
  );

  // ── Client promise — local mode defers until service start ─

  let localProcess: ChildProcess | null = null;
  let clientPromise: Promise<EbmPyClient> | null = null;
  let implicitRuntimeAcquired = false;

  const runtimeKey = cfg.mode === "remote"
    ? `remote:${cfg.baseUrl}`
    : `local:${cfg.port}:${cfg.ebmPyPath || ""}:${cfg.configJsonPath || ""}:${cfg.dbPath}`;

  if (cfg.mode === "remote") {
    // Remote mode: client is immediately available
    const client = new EbmPyClient(cfg.baseUrl, cfg.timeoutMs);
    clientPromise = Promise.resolve(client);
  } else {
    const runtime = getLocalRuntimeState(runtimeKey);
    if (runtime.client) {
      clientPromise = Promise.resolve(runtime.client);
    } else if (runtime.startupPromise) {
      clientPromise = runtime.startupPromise;
    }
  }

  const createStartupPromise = async (): Promise<EbmPyClient> => {
    const healthBaseUrl = `http://127.0.0.1:${cfg.port}`;
    const healthyExisting = await quickHealthCheck(healthBaseUrl, 2_000);
    if (healthyExisting) {
      const client = new EbmPyClient(healthBaseUrl, cfg.timeoutMs);
      setLocalRuntimeState(runtimeKey, {
        client,
        process: null,
        startupPromise: null,
      });
      api.logger.info(`[EBM-PY] reusing healthy sidecar at ${healthBaseUrl}`);
      return client;
    }

    const actualPort = await prepareLocalPort(cfg.port, api.logger, cfg.portScanRange);
    const baseUrl = `http://127.0.0.1:${actualPort}`;
    const pythonCmd = resolvePythonCommand(
      cfg.pythonCommand || undefined,
      api.logger,
    );

    const ebmPyPath = cfg.ebmPyPath || resolveEbmPyPath(api) || process.cwd();
    const configJsonPath = cfg.configJsonPath || "";
    const dbPath = cfg.dbPath;

    const args = [
      "-m", "ebm_context_engine.server",
      "--port", String(actualPort),
      ...(configJsonPath ? ["--config", configJsonPath] : []),
      "--db", dbPath,
      "--log-level", "INFO",
    ];

    api.logger.info(`[EBM-PY] spawning: ${pythonCmd} ${args.join(" ")} (cwd=${ebmPyPath})`);

    const {
      ALL_PROXY,
      all_proxy,
      HTTP_PROXY,
      http_proxy,
      HTTPS_PROXY,
      https_proxy,
      ...filteredEnv
    } = process.env;
    const child = spawn(pythonCmd, args, {
      cwd: ebmPyPath,
      env: {
        ...filteredEnv,
        PYTHONUNBUFFERED: "1",
        PYTHONPATH: ebmPyPath,
        EBM_PY_SLOWPATH_ENABLED: cfg.slowPathEnabled ? "1" : "0",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });
    localProcess = child;
    setLocalRuntimeState(runtimeKey, { process: child });

    const stderrChunks: string[] = [];
    let stderrCharCount = 0;
    let stderrDropped = 0;
    const pushStderr = (chunk: string) => {
      if (!chunk) return;
      stderrChunks.push(chunk);
      stderrCharCount += chunk.length;
      while (stderrChunks.length > MAX_STDERR_LINES || stderrCharCount > MAX_STDERR_CHARS) {
        const dropped = stderrChunks.shift();
        if (!dropped) break;
        stderrCharCount -= dropped.length;
        stderrDropped += 1;
      }
    };
    const formatStderr = () => {
      if (!stderrChunks.length && !stderrDropped) return "";
      const prefix = stderrDropped > 0 ? `[truncated ${stderrDropped} earlier chunks]\n` : "";
      return `\n[EBM-PY stderr]\n${prefix}${stderrChunks.join("\n")}`;
    };

    child.stdout?.on("data", (data: Buffer) => {
      const line = data.toString().trim();
      if (line) api.logger.info(`[EBM-PY:stdout] ${line}`);
    });
    child.stderr?.on("data", (data: Buffer) => {
      const line = data.toString().trim();
      pushStderr(line);
      api.logger.debug?.(`[EBM-PY:stderr] ${line}`);
    });
    child.on("error", (err: Error) => {
      api.logger.warn(`[EBM-PY] sidecar error: ${err}`);
    });
    child.on("exit", (code, signal) => {
      const runtime = getLocalRuntimeState(runtimeKey);
      if (runtime.process === child) {
        setLocalRuntimeState(runtimeKey, {
          client: null,
          process: null,
          startupPromise: null,
          startupOwners: 0,
          refCount: 0,
        });
      }
      if (localProcess === child) {
        localProcess = null;
      }
      api.logger.warn(`[EBM-PY] sidecar exited (code=${code}, signal=${signal})${formatStderr()}`);
    });

    try {
      await waitForHealthOrExit(baseUrl, cfg.healthTimeoutMs, HEALTH_POLL_INTERVAL_MS, child);
      const client = new EbmPyClient(baseUrl, cfg.timeoutMs);
      setLocalRuntimeState(runtimeKey, {
        client,
        process: child,
        startupPromise: null,
        startupOwners: 0,
      });
      api.logger.info(`[EBM-PY] local sidecar started at ${baseUrl} (port=${actualPort})`);
      return client;
    } catch (err) {
      localProcess = null;
      child.kill("SIGTERM");
      setLocalRuntimeState(runtimeKey, {
        client: null,
        process: null,
        startupPromise: null,
        startupOwners: 0,
      });
      const errObj = err instanceof Error ? err : new Error(String(err));
      if (stderrChunks.length) {
        api.logger.warn(`[EBM-PY] startup failed.${formatStderr()}`);
      }
      throw errObj;
    }
  };

  const acquireLocalRuntime = async (implicit = false): Promise<EbmPyClient> => {
    const runtimeBefore = getLocalRuntimeState(runtimeKey);
    if (runtimeBefore.client) {
      setLocalRuntimeState(runtimeKey, { refCount: runtimeBefore.refCount + 1 });
      localProcess = runtimeBefore.process;
      if (implicit) {
        implicitRuntimeAcquired = true;
      }
      return runtimeBefore.client;
    }
    if (runtimeBefore.startupPromise) {
      setLocalRuntimeState(runtimeKey, { startupOwners: runtimeBefore.startupOwners + 1 });
      try {
        const client = await runtimeBefore.startupPromise;
        const runtimeAfter = getLocalRuntimeState(runtimeKey);
        setLocalRuntimeState(runtimeKey, {
          startupOwners: Math.max(0, runtimeAfter.startupOwners - 1),
          refCount: runtimeAfter.refCount + 1,
        });
        localProcess = getLocalRuntimeState(runtimeKey).process;
        if (implicit) {
          implicitRuntimeAcquired = true;
        }
        return client;
      } catch (err) {
        const runtimeAfter = getLocalRuntimeState(runtimeKey);
        setLocalRuntimeState(runtimeKey, {
          startupOwners: Math.max(0, runtimeAfter.startupOwners - 1),
        });
        throw err;
      }
    }

    const startupPromise = createStartupPromise();
    setLocalRuntimeState(runtimeKey, {
      startupPromise,
      startupOwners: 1,
    });
    try {
      const client = await startupPromise;
      const runtimeAfter = getLocalRuntimeState(runtimeKey);
      setLocalRuntimeState(runtimeKey, {
        startupOwners: Math.max(0, runtimeAfter.startupOwners - 1),
        refCount: runtimeAfter.refCount + 1,
      });
      localProcess = getLocalRuntimeState(runtimeKey).process;
      if (implicit) {
        implicitRuntimeAcquired = true;
      }
      return client;
    } catch (err) {
      const runtimeAfter = getLocalRuntimeState(runtimeKey);
      setLocalRuntimeState(runtimeKey, {
        startupOwners: Math.max(0, runtimeAfter.startupOwners - 1),
      });
      throw err;
    }
  };

  const releaseLocalRuntime = async (): Promise<void> => {
    const runtime = getLocalRuntimeState(runtimeKey);
    if (runtime.client === null && runtime.startupPromise) {
      const nextOwners = Math.max(0, runtime.startupOwners - 1);
      setLocalRuntimeState(runtimeKey, { startupOwners: nextOwners });
      api.logger.info(`[EBM-PY] local startup owner released (startupOwners=${nextOwners})`);
      return;
    }
    const nextRefCount = Math.max(0, runtime.refCount - 1);
    if (nextRefCount > 0) {
      setLocalRuntimeState(runtimeKey, { refCount: nextRefCount });
      api.logger.info(`[EBM-PY] shared local sidecar retained (refCount=${nextRefCount})`);
      return;
    }

    if (runtime.process) {
      try {
        if (runtime.client) await runtime.client.dispose();
      } catch {
        // best-effort
      }
      runtime.process.kill("SIGTERM");
    }
    localProcess = null;
    clearCachedLocalClient(runtimeKey);
    api.logger.info(`[EBM-PY] service stopped`);
  };

  const getClient = (): Promise<EbmPyClient> => {
    if (clientPromise) {
      return clientPromise;
    }
    const runtime = getLocalRuntimeState(runtimeKey);
    if (runtime.client) {
      clientPromise = Promise.resolve(runtime.client);
      return clientPromise;
    }
    if (runtime.startupPromise) {
      clientPromise = runtime.startupPromise;
      return clientPromise;
    }
    if (cfg.mode === "local") {
      clientPromise = acquireLocalRuntime(true);
      return clientPromise;
    }
    return Promise.reject(new Error("EBM-PY remote runtime not started"));
  };

  const registerTool = (
    toolOrFactory: ToolDefinition | ((ctx: ToolContext) => ToolDefinition),
    opts?: { name?: string; names?: string[] },
  ) => {
    const apiWithTools = api as OpenClawPluginApi & {
      registerTool?: (
        tool: ToolDefinition | ((ctx: ToolContext) => ToolDefinition),
        opts?: { name?: string; names?: string[] },
      ) => void;
    };
    if (typeof apiWithTools.registerTool === "function") {
      apiWithTools.registerTool(toolOrFactory, opts);
    } else {
      api.logger.warn(`[EBM-PY] registerTool unavailable; skipping ${opts?.name ?? opts?.names?.join(",") ?? "tool"}`);
    }
  };

  registerTool(
    (_ctx: ToolContext) => ({
      name: "memory_recall",
      label: "Memory Recall (EBM-PY)",
      description: "Search EBM long-term memory for relevant user preferences, facts, decisions, and transcript evidence.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string", description: "Search query" },
          limit: { type: "number", description: "Maximum results, default 6" },
        },
        required: ["query"],
        additionalProperties: false,
      },
      async execute(_toolCallId, params) {
        const query = textParam(params, "query");
        if (!query) {
          return { content: [{ type: "text", text: "Provide query." }], details: { error: "missing_query" } };
        }
        const limit = clampLimit(numberParam(params, "limit", 6));
        const hits = await (await getClient()).memorySearch(query, limit);
        if (hits.length === 0) {
          return {
            content: [{ type: "text", text: "No relevant EBM memories found." }],
            details: { count: 0, query, limit },
          };
        }
        return {
          content: [{ type: "text", text: `Found ${hits.length} EBM memories:\n\n${formatMemoryHits(hits)}` }],
          details: { count: hits.length, memories: hits, query, limit },
        };
      },
    }),
    { name: "memory_recall" },
  );

  registerTool(
    (ctx: ToolContext) => ({
      name: "memory_store",
      label: "Memory Store (EBM-PY)",
      description: "Store text in EBM memory and trigger background slow-path extraction.",
      parameters: {
        type: "object",
        properties: {
          text: { type: "string", description: "Information to store" },
          role: { type: "string", description: "Message role, default user" },
          sessionId: { type: "string", description: "Optional session id override" },
          sessionKey: { type: "string", description: "Optional session key override" },
        },
        required: ["text"],
        additionalProperties: false,
      },
      async execute(_toolCallId, params) {
        const text = textParam(params, "text");
        if (!text) {
          return { content: [{ type: "text", text: "Provide text." }], details: { error: "missing_text" } };
        }
        const sessionId = textParam(params, "sessionId") || ctx.sessionId || `memory-store-${Date.now()}`;
        const sessionKey = textParam(params, "sessionKey") || ctx.sessionKey;
        const role = textParam(params, "role") || "user";
        const client = await getClient();
        // afterTurn stores transcript rows AND creates the slow-path
        // extraction job in a single call — no separate ingest() needed.
        await client.afterTurn({
          sessionId,
          sessionKey,
          sessionFile: "",
          messages: [{ role, content: text }],
          prePromptMessageCount: 0,
        });
        const flush = await client.flush();
        return {
          content: [{ type: "text", text: `Stored EBM memory in session ${sessionId}; extraction queued (pending=${flush.status.pending}, running=${flush.status.running}, done=${flush.status.done}, failed=${flush.status.failed}).` }],
          details: { action: "stored", sessionId, sessionKey, flushStatus: flush.status },
        };
      },
    }),
    { name: "memory_store" },
  );

  registerTool(
    (_ctx: ToolContext) => ({
      name: "memory_forget",
      label: "Memory Forget (EBM-PY)",
      description: "Forget an EBM memory by id, or search first and forget one strong match.",
      parameters: {
        type: "object",
        properties: {
          id: { type: "string", description: "Exact memory id to forget" },
          query: { type: "string", description: "Search query when id is unknown" },
          limit: { type: "number", description: "Search limit, default 5" },
        },
        additionalProperties: false,
      },
      async execute(_toolCallId, params) {
        const client = await getClient();
        const id = textParam(params, "id");
        if (id) {
          const result = await client.memoryForget(id);
          const message = result.forgotten
            ? `Forgotten EBM memory: ${result.id}`
            : `Could not forget ${result.id}: ${result.reason ?? "unknown"}`;
          return { content: [{ type: "text", text: message }], details: result };
        }
        const query = textParam(params, "query");
        if (!query) {
          return { content: [{ type: "text", text: "Provide id or query." }], details: { error: "missing_param" } };
        }
        const hits = await client.memorySearch(query, clampLimit(numberParam(params, "limit", 5), 5));
        const FORGETTABLE_SOURCES = new Set(["fact", "unified_fact", "hm_fact", "ledger"]);
        const candidates = hits.filter(
          (hit) => hit.id && hit.score >= 0.85 && FORGETTABLE_SOURCES.has(hit.source ?? ""),
        );
        if (candidates.length === 1) {
          const result = await client.memoryForget(candidates[0]!.id);
          return { content: [{ type: "text", text: result.forgotten ? `Forgotten EBM memory: ${result.id}` : `Could not forget ${result.id}: ${result.reason ?? "unknown"}` }], details: result };
        }
        if (hits.length === 0) {
          return { content: [{ type: "text", text: "No matching EBM memory candidates found." }], details: { action: "none" } };
        }
        return {
          content: [{ type: "text", text: `Found ${hits.length} candidates. Specify id:\n${hits.map((hit) => `- ${hit.id} (${hit.score.toFixed(3)}) ${hit.title}`).join("\n")}` }],
          details: { action: "candidates", candidates: hits },
        };
      },
    }),
    { name: "memory_forget" },
  );

  registerTool(
    (ctx: ToolContext) => ({
      name: "ebm_archive_expand",
      label: "Archive Expand (EBM-PY)",
      description: "Expand an EBM session summary or transcript reference into original messages. Check the Memory context / Evidence Traces in the system prompt for session IDs to pass as archiveId.",
      parameters: {
        type: "object",
        properties: {
          archiveId: { type: "string", description: "EBM summary id, session id, session key, or transcript reference" },
          limit: { type: "number", description: "Maximum messages, default 200" },
        },
        required: ["archiveId"],
        additionalProperties: false,
      },
      async execute(_toolCallId, params) {
        const archiveId = textParam(params, "archiveId");
        if (!archiveId) {
          return { content: [{ type: "text", text: "Error: archiveId is required." }], details: { error: "missing_param" } };
        }
        const result = await (await getClient()).archiveExpand({
          archiveId,
          sessionId: ctx.sessionId,
          sessionKey: ctx.sessionKey,
          limit: clampLimit(numberParam(params, "limit", 200), 200, 1000),
        });
        const summary = result.summary
          ? `## Summary\n${String(result.summary.overview ?? result.summary.abstract ?? "").trim()}\n\n`
          : "";
        const body = result.messages
          .map((message) => `[${message.role} #${message.messageIndex ?? "?"}]\n${message.text}`)
          .join("\n\n");
        if (!summary && !body) {
          return { content: [{ type: "text", text: `No EBM archive or transcript found for ${archiveId}.` }], details: result };
        }
        return {
          content: [{ type: "text", text: `# EBM Archive ${archiveId}\n\n${summary}## Messages (${result.messages.length})\n${body}` }],
          details: result,
        };
      },
    }),
    { name: "ebm_archive_expand" },
  );

  const registerHook = <TEvent, TContext>(
    name: string,
    handler: (event: TEvent, ctx?: TContext) => unknown,
    opts?: { priority?: number },
  ) => {
    const apiWithHooks = api as OpenClawPluginApi & {
      on?: (name: string, handler: (event: TEvent, ctx?: TContext) => unknown, opts?: { priority?: number }) => void;
    };
    if (typeof apiWithHooks.on === "function") {
      apiWithHooks.on(name, handler, opts);
    } else {
      api.logger.warn(`[EBM-PY] hook API unavailable; skipping ${name}`);
    }
  };

  registerHook("session_start", async (_event: unknown, ctx?: HookAgentContext) => {
    api.logger.debug?.(`[EBM-PY] session_start: session=${ctx?.sessionId ?? "unknown"}`);
  });
  registerHook("session_end", async (_event: unknown, ctx?: HookAgentContext) => {
    const key = resolveSessionStateKey(ctx);
    if (key) {
      assembleFallbackState.delete(key);
    }
    api.logger.debug?.(`[EBM-PY] session_end: session=${ctx?.sessionId ?? "unknown"}`);
  });
  registerHook("agent_end", async (_event: unknown, ctx?: HookAgentContext) => {
    const key = resolveSessionStateKey(ctx);
    if (key) {
      assembleFallbackState.delete(key);
    }
    api.logger.debug?.(`[EBM-PY] agent_end: session=${ctx?.sessionId ?? "unknown"}`);
  });
  registerHook("after_compaction", async (_event: unknown, ctx?: HookAgentContext) => {
    api.logger.debug?.(`[EBM-PY] after_compaction: session=${ctx?.sessionId ?? "unknown"}`);
  });

  registerHook(
    "before_prompt_build",
    async (event: unknown, ctx?: HookAgentContext) => {
      const eventObj = (event ?? {}) as { messages?: unknown[]; prompt?: string };
      const query = extractLatestUserText(eventObj.messages) || (typeof eventObj.prompt === "string" ? eventObj.prompt.trim() : "");
      if (!query || query.length < 5) {
        return undefined;
      }
      const parts: string[] = [];
      const sessionStateKey = resolveSessionStateKey(ctx);
      const assembleHadUsefulContext = sessionStateKey
        ? (assembleFallbackState.get(sessionStateKey)?.hasMeaningfulMemoryContext ?? false)
        : false;

      // ── Auto-recall with token budget ──
      if (cfg.autoRecall && !assembleHadUsefulContext) {
        try {
          const client = await withTimeout(getClient(), 5000, "[EBM-PY] client init timeout");
          const healthy = await quickHealthCheck(client.baseUrl, 2000);
          if (healthy) {
            await withTimeout(
              (async () => {
                const candidateLimit = Math.max(cfg.recallLimit * 4, 20);
                const hits = await client.memorySearch(query, candidateLimit);
                const { lines, estimatedTokens } = buildMemoryLinesWithBudget(hits, {
                  recallTokenBudget: cfg.recallTokenBudget,
                  recallMaxContentChars: cfg.recallMaxContentChars,
                  recallScoreThreshold: cfg.recallScoreThreshold,
                });
                if (lines.length > 0) {
                  api.logger.info(
                    `[EBM-PY] auto-recall: injecting ${lines.length} memories (~${estimatedTokens} tokens, budget=${cfg.recallTokenBudget})`,
                  );
                  parts.push(
                    "<relevant-memories>\nThe following EBM memories may be relevant:\n" +
                      lines.join("\n") +
                      "\n</relevant-memories>",
                  );
                }
              })(),
              cfg.recallTimeoutMs,
              "[EBM-PY] auto-recall timeout",
            );
          } else {
            api.logger.warn(`[EBM-PY] auto-recall skipped: health precheck failed`);
          }
        } catch (err) {
          api.logger.warn(`[EBM-PY] auto-recall failed: ${err}`);
        }
      } else if (cfg.autoRecall && assembleHadUsefulContext) {
        api.logger.debug?.("[EBM-PY] auto-recall skipped: assemble already provided memory context");
      }

      // ── Ingest-reply-assist ──
      if (cfg.ingestReplyAssist) {
        if (!shouldSkipSession(ctx, cfg.ingestReplyAssistIgnoreSessionPatterns)) {
          if (isTranscriptLike(query, cfg.ingestReplyAssistMinSpeakerTurns, cfg.ingestReplyAssistMinChars)) {
            parts.push(
              "<ingest-reply-assist>\n" +
                "The latest user input looks like a multi-speaker transcript used for memory ingestion.\n" +
                "Reply with 1-2 concise sentences to acknowledge or summarize key points.\n" +
                "Do not output NO_REPLY or an empty reply.\n" +
                "Do not fabricate facts beyond the provided transcript and recalled memories.\n" +
                "</ingest-reply-assist>",
            );
          }
        }
      }

      return parts.length > 0 ? { prependContext: parts.join("\n\n") } : undefined;
    },
    { priority: 0 },
  );

  registerHook(
    "before_reset",
    async (event: unknown, ctx?: HookAgentContext) => {
      const eventObj = (event ?? {}) as { messages?: unknown[]; sessionFile?: string };
      try {
        const client = await getClient();
        if (ctx?.sessionId && Array.isArray(eventObj.messages) && eventObj.messages.length > 0) {
          const mapped = eventObj.messages
            .map((message) => {
              const msg = (message ?? {}) as Record<string, unknown>;
              const content = messageText(msg);
              if (!content.trim()) return null;
              return {
                role: typeof msg.role === "string" ? String(msg.role) : "user",
                content,
              };
            })
            .filter((m): m is { role: string; content: string } => m !== null);
          if (mapped.length > 0) {
            await client.ingestBatch({
              sessionId: ctx.sessionId,
              sessionKey: ctx.sessionKey,
              sessionFile: eventObj.sessionFile,
              messages: mapped,
            });
            api.logger.info(`[EBM-PY] before_reset: ingested ${mapped.length} messages`);
          }
        }
        await client.flush();
        api.logger.info(`[EBM-PY] before_reset: flush triggered`);
      } catch (err) {
        api.logger.warn(`[EBM-PY] before_reset flush failed: ${err}`);
      }
    },
  );

  // ── Context engine registration ───────────────────────────

  const engine = createEbmContextEngine(
    getClient,
    api.logger,
    async () => {
      if (!implicitRuntimeAcquired || cfg.mode !== "local") {
        return;
      }
      implicitRuntimeAcquired = false;
      await releaseLocalRuntime();
    },
  );
  const baseAssemble = engine.assemble.bind(engine);
  engine.assemble = async (params) => {
    const result = await baseAssemble(params);
    const sessionStateKey = (params.sessionKey || params.sessionId || "").trim();
    if (sessionStateKey) {
      assembleFallbackState.set(sessionStateKey, {
        hasMeaningfulMemoryContext: hasMeaningfulAssembleResult(result.systemPromptAddition),
      });
    }
    return result;
  };

  api.registerContextEngine(PLUGIN_ID, () => engine);
  api.logger.info(
    `[EBM-PY] registered context-engine (assemble=memory-recall, afterTurn=slow-path, ingest/ingestBatch=transcript)`,
  );

  // ── Service lifecycle ─────────────────────────────────────

  api.registerService({
    id: PLUGIN_ID,
    async start() {
      if (cfg.mode === "remote") {
        // Remote mode: health-check the existing server
        const ok = await quickHealthCheck(cfg.baseUrl, 5_000);
        if (ok) {
          api.logger.info(`[EBM-PY] remote server healthy at ${cfg.baseUrl}`);
        } else {
          api.logger.warn(`[EBM-PY] remote server not responding at ${cfg.baseUrl}`);
        }
        return;
      }

      clientPromise = acquireLocalRuntime(false);
      await clientPromise;
      localProcess = getLocalRuntimeState(runtimeKey).process;
    },

    async stop() {
      if (cfg.mode === "remote") {
        api.logger.info(`[EBM-PY] service stopped`);
        return;
      }
      await releaseLocalRuntime();
    },
  });

  // ── Gateway HTTP debug routes ─────────────────────────────

  api.registerHttpRoute({
    path: "/v1/extensions/ebm-py/status",
    auth: "gateway",
    handler: async (_req, res) => {
      try {
        const client = await getClient();
        const data = await client.status();
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.statusCode = 200;
        res.end(JSON.stringify({ ok: true, status: data }));
      } catch (err) {
        res.statusCode = 502;
        res.end(JSON.stringify({ ok: false, error: String(err) }));
      }
      return true;
    },
  });

  api.registerHttpRoute({
    path: "/v1/extensions/ebm-py/flush",
    auth: "gateway",
    handler: async (req, res) => {
      if (req.method !== "POST") {
        res.statusCode = 405;
        res.end(JSON.stringify({ ok: false, error: "Method Not Allowed" }));
        return true;
      }
      try {
        const client = await getClient();
        const data = await client.flush();
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.statusCode = 200;
        res.end(JSON.stringify({ ok: true, ...data }));
      } catch (err) {
        res.statusCode = 502;
        res.end(JSON.stringify({ ok: false, error: String(err) }));
      }
      return true;
    },
  });

  api.registerHttpRoute({
    path: "/v1/extensions/ebm-py/retry-failed",
    auth: "gateway",
    handler: async (req, res) => {
      if (req.method !== "POST") {
        res.statusCode = 405;
        res.end(JSON.stringify({ ok: false, error: "Method Not Allowed" }));
        return true;
      }
      try {
        const client = await getClient();
        const data = await client.retryFailed();
        res.setHeader("Content-Type", "application/json; charset=utf-8");
        res.statusCode = 200;
        res.end(JSON.stringify({ ok: true, ...data }));
      } catch (err) {
        res.statusCode = 502;
        res.end(JSON.stringify({ ok: false, error: String(err) }));
      }
      return true;
    },
  });

  api.logger.info(`[EBM-PY] plugin registered: context-engine + service + 3 HTTP routes`);
}

// ── Helpers ─────────────────────────────────────────────────

function resolveEbmPyPath(api: OpenClawPluginApi): string {
  try {
    // Plugin installed at ~/.openclaw/extensions/ebm-context-engine/src/index.ts
    // ebm_context_engine lives at the workspace root — go up from plugin install path
    return resolvePath(api.resolvePath(".."), "..");
  } catch {
    return "";
  }
}

// ── Export ───────────────────────────────────────────────────

export default definePluginEntry({
  id: PLUGIN_ID,
  name: "EBM Python Memory Engine",
  description:
    "Evidence-Backed Memory (Python) — three-plane memory with HTTP sidecar proxy to ebm_context_engine engine.",
  kind: "context-engine",
  register,
});
