/**
 * EBM Python context engine — implements the OpenClaw ContextEngine contract
 * by proxying all calls to the EBM Python sidecar via EbmPyClient.
 *
 * Follows the same API shape as EBM's context-engine.ts:
 * info, ingest, ingestBatch, assemble, afterTurn, compact.
 */
import type { EbmPyClient, AssembleResult as ClientAssembleResult } from "./client.js";

// ── Types aligned with OpenClaw ContextEngine contract ──────

export type AgentMessage = {
  role?: string;
  content?: unknown;
};

export type ContextEngineInfo = {
  id: string;
  name: string;
  version?: string;
  ownsCompaction: boolean;
};

export type IngestResult = {
  ingested: boolean;
};

export type IngestBatchResult = {
  ingestedCount: number;
};

export type AssembleResult = {
  messages: AgentMessage[];
  estimatedTokens: number;
  systemPromptAddition?: string;
};

export type CompactResult = {
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
};

export type EbmContextEngine = {
  info: ContextEngineInfo;
  ingest: (params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile?: string;
    message: AgentMessage;
    isHeartbeat?: boolean;
  }) => Promise<IngestResult>;
  ingestBatch: (params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile?: string;
    messages: AgentMessage[];
    isHeartbeat?: boolean;
  }) => Promise<IngestBatchResult>;
  assemble: (params: {
    sessionId: string;
    sessionKey?: string;
    messages: AgentMessage[];
    tokenBudget?: number;
    prompt?: string;
    runtimeContext?: Record<string, unknown>;
  }) => Promise<AssembleResult>;
  afterTurn: (params: {
    sessionId: string;
    sessionFile: string;
    messages: AgentMessage[];
    prePromptMessageCount: number;
    isHeartbeat?: boolean;
    tokenBudget?: number;
    runtimeContext?: Record<string, unknown>;
    sessionKey?: string;
  }) => Promise<void>;
  compact: (params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile: string;
    tokenBudget?: number;
    force?: boolean;
    currentTokenCount?: number;
    compactionTarget?: "budget" | "threshold";
    customInstructions?: string;
    runtimeContext?: Record<string, unknown>;
  }) => Promise<CompactResult>;
  bootstrap: (params: {
    sessionId: string;
    sessionKey?: string;
    sessionFile?: string;
  }) => Promise<{ bootstrapped: boolean }>;
  dispose: () => Promise<void>;
};

export interface EngineLogger {
  info: (msg: string) => void;
  warn: (msg: string) => void;
  error: (msg: string) => void;
}

// ── Token estimation (chars/4 heuristic) ────────────────────

function roughEstimate(messages: AgentMessage[]): number {
  return Math.ceil(JSON.stringify(messages).length / 4);
}

// ── Factory ─────────────────────────────────────────────────

export function createEbmContextEngine(
  getClient: () => Promise<EbmPyClient>,
  logger: EngineLogger,
  releaseImplicitLocalRuntime?: () => Promise<void>,
): EbmContextEngine {
  return {
    info: {
      id: "ebm-context-engine",
      name: "EBM Python Memory Engine",
      version: "0.3.0",
      ownsCompaction: false,
    },

    async bootstrap({ sessionId, sessionKey, sessionFile }) {
      try {
        const client = await getClient();
        const res = await client.bootstrap({ sessionId, sessionKey, sessionFile });
        logger.info(`[EBM-PY] bootstrap: session=${sessionId} imported=${res.importedMessages}`);
        return { bootstrapped: res.bootstrapped };
      } catch (err) {
        logger.error(`[EBM-PY] bootstrap failed: ${err}`);
        return { bootstrapped: false };
      }
    },

    async ingest({ sessionId, sessionKey, sessionFile, message, isHeartbeat }) {
      if (isHeartbeat) return { ingested: false };
      try {
        const client = await getClient();
        const msg = extractMessageContent(message);
        if (!msg) return { ingested: false };
        return await client.ingest({ sessionId, sessionKey, sessionFile, message: msg });
      } catch (err) {
        logger.error(`[EBM-PY] ingest failed: ${err}`);
        return { ingested: false };
      }
    },

    async ingestBatch({ sessionId, sessionKey, sessionFile, messages, isHeartbeat }) {
      if (isHeartbeat) return { ingestedCount: 0 };
      try {
        const client = await getClient();
        const converted = messages
          .map(extractMessageContent)
          .filter((m): m is NonNullable<typeof m> => m != null);
        if (converted.length === 0) return { ingestedCount: 0 };
        return await client.ingestBatch({
          sessionId,
          sessionKey,
          sessionFile,
          messages: converted,
        });
      } catch (err) {
        logger.error(`[EBM-PY] ingestBatch failed: ${err}`);
        return { ingestedCount: 0 };
      }
    },

    async assemble({ sessionId, sessionKey, messages, tokenBudget, prompt, runtimeContext }) {
      try {
        const client = await getClient();
        const res = await client.assemble({
          sessionId,
          sessionKey,
          messages,
          tokenBudget: tokenBudget ?? 16_000,
          prompt,
          runtimeContext,
        });
        logger.info(
          `[EBM-PY] assemble: session=${sessionId} tokens=${res.estimatedTokens} ` +
            `hasSystemAddition=${!!res.systemPromptAddition}`,
        );
        const addition = appendArchiveExpandGuidance(res.systemPromptAddition, sessionId);
        return {
          messages: (res.messages ?? messages) as AgentMessage[],
          estimatedTokens: res.estimatedTokens ?? roughEstimate(messages),
          ...(addition ? { systemPromptAddition: addition } : {}),
        };
      } catch (err) {
        logger.error(`[EBM-PY] assemble failed, passthrough: ${err}`);
        return { messages, estimatedTokens: roughEstimate(messages) };
      }
    },

    async afterTurn({
      sessionId,
      sessionKey,
      sessionFile,
      messages,
      prePromptMessageCount,
      isHeartbeat,
      tokenBudget,
      runtimeContext,
    }) {
      if (isHeartbeat) return;
      try {
        const client = await getClient();
        await client.afterTurn({
          sessionId,
          sessionKey,
          sessionFile: sessionFile ?? "",
          messages,
          prePromptMessageCount: prePromptMessageCount ?? 0,
          tokenBudget,
          runtimeContext,
        });
        logger.info(`[EBM-PY] afterTurn: session=${sessionId} messages=${messages.length}`);
      } catch (err) {
        logger.error(`[EBM-PY] afterTurn failed: ${err}`);
      }
    },

    async compact({
      sessionId,
      sessionKey,
      sessionFile,
      tokenBudget,
      force,
      currentTokenCount,
      compactionTarget,
      customInstructions,
      runtimeContext,
    }) {
      try {
        const client = await getClient();
        const res = await client.compact({
          sessionId,
          sessionKey,
          sessionFile,
          tokenBudget,
          force,
          currentTokenCount,
          compactionTarget,
          customInstructions,
          runtimeContext,
        });
        logger.info(`[EBM-PY] compact: session=${sessionId} compacted=${res.compacted}`);
        return res;
      } catch (err) {
        logger.error(`[EBM-PY] compact failed: ${err}`);
        return {
          ok: false,
          compacted: false,
          reason: `compact_error: ${err}`,
          result: {
            tokensBefore: currentTokenCount ?? 0,
            details: { error: String(err) },
          },
        };
      }
    },

    async dispose() {
      try {
        const client = await getClient();
        await client.dispose();
      } catch {
        // best-effort
      }
      if (releaseImplicitLocalRuntime) {
        await releaseImplicitLocalRuntime().catch(() => {
          // best-effort
        });
      }
    },
  };
}

// ── Helpers ─────────────────────────────────────────────────

function extractMessageContent(
  msg: AgentMessage,
): { role: string; content: string; timestamp?: number } | null {
  const role = typeof msg.role === "string" ? msg.role : "user";
  const raw = msg.content;
  let content: string;

  if (typeof raw === "string") {
    content = raw;
  } else if (Array.isArray(raw)) {
    // Extract text blocks from content array
    content = (raw as Array<Record<string, unknown>>)
      .filter((block) => block.type === "text")
      .map((block) => String(block.text ?? ""))
      .join("\n");
  } else if (raw != null) {
    content = String(raw);
  } else {
    return null;
  }

  if (!content.trim()) return null;
  return { role, content };
}

const ARCHIVE_EXPAND_GUIDANCE =
  "\n\nSession IDs appear as [session:<id>] in the history above. " +
  "If the user asks about past conversations or session details, " +
  "use the ebm_archive_expand tool with the session ID to retrieve the original messages.";

function appendArchiveExpandGuidance(
  systemPromptAddition: string | undefined,
  sessionId: string,
): string | undefined {
  if (!systemPromptAddition) return undefined;
  if (systemPromptAddition.includes("ebm_archive_expand")) return systemPromptAddition;
  const hasSessionContext =
    systemPromptAddition.includes("Session History") ||
    systemPromptAddition.includes("Evidence Traces");
  if (!hasSessionContext) return systemPromptAddition;
  return systemPromptAddition + ARCHIVE_EXPAND_GUIDANCE;
}
