/**
 * Type declarations for OpenClaw plugin SDK.
 * These ambient modules let TypeScript compile the plugin without
 * the full openclaw package installed as a real dependency.
 */

declare module "openclaw/plugin-sdk" {
  export type AgentMessage = {
    role?: string;
    content?: unknown;
    timestamp?: number;
    [key: string]: unknown;
  };

  export type ContextEngineInfo = {
    id: string;
    name: string;
    version?: string;
    ownsCompaction?: boolean;
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

  export type IngestResult = {
    ingested: boolean;
  };

  export type IngestBatchResult = {
    ingestedCount: number;
  };

  export type BootstrapResult = {
    bootstrapped: boolean;
    importedMessages?: number;
    reason?: string;
  };

  export type ContextEngineRuntimeContext = Record<string, unknown>;

  export interface ContextEngine {
    readonly info: ContextEngineInfo;
    bootstrap?(params: {
      sessionId: string;
      sessionKey?: string;
      sessionFile: string;
    }): Promise<BootstrapResult>;
    ingest(params: {
      sessionId: string;
      sessionKey?: string;
      message: AgentMessage;
      isHeartbeat?: boolean;
    }): Promise<IngestResult>;
    ingestBatch?(params: {
      sessionId: string;
      sessionKey?: string;
      messages: AgentMessage[];
      isHeartbeat?: boolean;
    }): Promise<IngestBatchResult>;
    afterTurn?(params: {
      sessionId: string;
      sessionKey?: string;
      sessionFile: string;
      messages: AgentMessage[];
      prePromptMessageCount: number;
      isHeartbeat?: boolean;
      tokenBudget?: number;
      runtimeContext?: ContextEngineRuntimeContext;
    }): Promise<void>;
    assemble(params: {
      sessionId: string;
      sessionKey?: string;
      messages: AgentMessage[];
      tokenBudget?: number;
      model?: string;
      prompt?: string;
    }): Promise<AssembleResult>;
    compact(params: {
      sessionId: string;
      sessionKey?: string;
      sessionFile: string;
      tokenBudget?: number;
      force?: boolean;
      currentTokenCount?: number;
      compactionTarget?: "budget" | "threshold";
      customInstructions?: string;
      runtimeContext?: ContextEngineRuntimeContext;
    }): Promise<CompactResult>;
    dispose?(): Promise<void>;
  }

  export type OpenClawPluginConfigSchema = Record<string, unknown>;

  export type PluginLogger = {
    debug?: (message: string) => void;
    info: (message: string) => void;
    warn: (message: string) => void;
    error: (message: string) => void;
  };

  export type OpenClawPluginToolContext = {
    sessionId?: string;
    sessionKey?: string;
    agentId?: string;
    [key: string]: unknown;
  };

  export type OpenClawPluginTool = {
    name: string;
    label?: string;
    description?: string;
    parameters?: unknown;
    execute: (toolCallId: string, params: Record<string, unknown>) => Promise<unknown> | unknown;
  };

  export type OpenClawPluginApi = {
    id: string;
    pluginConfig?: Record<string, unknown>;
    logger: PluginLogger;
    resolvePath: (input: string) => string;
    registerTool?: (
      tool: OpenClawPluginTool | ((ctx: OpenClawPluginToolContext) => OpenClawPluginTool),
      opts?: { name?: string; names?: string[] },
    ) => void;
    registerContextEngine: (
      id: string,
      factory: () => ContextEngine | Promise<ContextEngine>,
    ) => void;
    registerService: (service: {
      id: string;
      start: (ctx?: unknown) => void | Promise<void>;
      stop?: (ctx?: unknown) => void | Promise<void>;
    }) => void;
    registerHttpRoute: (route: {
      path: string;
      auth?: string;
      handler: (
        req: { method?: string; url?: string },
        res: {
          statusCode: number;
          setHeader: (name: string, value: string) => void;
          end: (data?: string) => void;
        },
      ) => Promise<boolean>;
    }) => void;
    on?: (
      hookName: string,
      handler: (event: unknown, ctx?: { agentId?: string; sessionId?: string; sessionKey?: string }) => unknown,
      opts?: { priority?: number },
    ) => void;
  };
}

declare module "openclaw/plugin-sdk/core" {
  import type {
    OpenClawPluginApi,
    OpenClawPluginConfigSchema,
  } from "openclaw/plugin-sdk";

  export function definePluginEntry(options: {
    id: string;
    name: string;
    description: string;
    kind?: "memory" | "context-engine";
    configSchema?: OpenClawPluginConfigSchema | (() => OpenClawPluginConfigSchema);
    register: (api: OpenClawPluginApi) => void;
  }): unknown;
}
