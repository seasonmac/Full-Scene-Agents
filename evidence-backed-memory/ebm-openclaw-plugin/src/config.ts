/**
 * EBM Python plugin — configuration model with local/remote mode support.
 *
 * Follows the same parse-and-validate pattern as EBM's config.ts:
 * all fields optional with sensible defaults, clamped to safe ranges.
 */
import { homedir } from "node:os";
import { resolve as resolvePath, join } from "node:path";

// ── Public types ────────────────────────────────────────────

export type EbmPyPluginConfig = {
  /** "local" = spawn Python sidecar; "remote" = use existing HTTP server */
  mode?: "local" | "remote";
  /** Python executable path. Default "python3" */
  pythonCommand?: string;
  /** Local sidecar HTTP port. Default 18790 */
  port?: number;
  /** API base URL. Local mode auto-derives from port; remote mode reads env or explicit value */
  baseUrl?: string;
  /** ebm_context_engine project root (contains ebm_context_engine/ package). Supports ~ and ${ENV} */
  ebmPyPath?: string;
  /** Path to ebm/config.json. Default: <ebmPyPath>/ebm/config.json */
  configJsonPath?: string;
  /** SQLite database path. Default: ~/.openclaw/memory/ebm_context_engine.sqlite */
  dbPath?: string;
  /** Enable slow path background distillation. Default true */
  slowPathEnabled?: boolean;
  /** Per-request HTTP timeout in ms. Min 1000. Default 120000 */
  timeoutMs?: number;
  /** Health check timeout in ms. Min 1000. Default 30000 */
  healthTimeoutMs?: number;
  /** Max port scan range when preferred port is occupied. Default 10 */
  portScanRange?: number;
  /** Enable auto-recall in before_prompt_build. Default true */
  autoRecall?: boolean;
  /** Max memories to inject after budget filtering. Default 6 */
  recallLimit?: number;
  /** Minimum score (0-1) for auto-recall. Default 0.15 */
  recallScoreThreshold?: number;
  /** Max chars per memory content in injection. Default 500 */
  recallMaxContentChars?: number;
  /** Max estimated tokens for auto-recall injection. Default 2000 */
  recallTokenBudget?: number;
  /** Timeout in ms for auto-recall search. Default 5000 */
  recallTimeoutMs?: number;
  /** Detect transcript-like input and inject reply assist. Default true */
  ingestReplyAssist?: boolean;
  /** Min speaker-tag turns to detect transcript. Default 2 */
  ingestReplyAssistMinSpeakerTurns?: number;
  /** Min text length for ingest-reply-assist. Default 120 */
  ingestReplyAssistMinChars?: number;
  /** Session patterns to skip ingest-reply-assist. Default [] */
  ingestReplyAssistIgnoreSessionPatterns?: string[];
};

// ── Defaults ────────────────────────────────────────────────

const DEFAULT_MODE = "local" as const;
const DEFAULT_PYTHON_COMMAND = "python3";
const DEFAULT_PORT = 18790;
const DEFAULT_TIMEOUT_MS = 120_000;
const DEFAULT_HEALTH_TIMEOUT_MS = 30_000;
const DEFAULT_PORT_SCAN_RANGE = 10;
const DEFAULT_DB_PATH = join(homedir(), ".openclaw", "memory", "ebm_context_engine.sqlite");
const DEFAULT_AUTO_RECALL = true;
const DEFAULT_RECALL_LIMIT = 6;
const DEFAULT_RECALL_SCORE_THRESHOLD = 0.15;
const DEFAULT_RECALL_MAX_CONTENT_CHARS = 500;
const DEFAULT_RECALL_TOKEN_BUDGET = 2000;
const DEFAULT_RECALL_TIMEOUT_MS = 5000;
const DEFAULT_INGEST_REPLY_ASSIST = true;
const DEFAULT_INGEST_REPLY_ASSIST_MIN_SPEAKER_TURNS = 2;
const DEFAULT_INGEST_REPLY_ASSIST_MIN_CHARS = 120;

// ── Helpers ─────────────────────────────────────────────────

function resolveEnvVars(value: string): string {
  return value.replace(/\$\{([^}]+)\}/g, (_, envVar: string) => {
    const envValue = process.env[envVar];
    if (!envValue) {
      throw new Error(`Environment variable ${envVar} is not set`);
    }
    return envValue;
  });
}

function expandHome(value: string): string {
  return value.replace(/^~(?=$|[/\\])/, homedir());
}

function toNumber(value: unknown, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return fallback;
}

function toStringArray(value: unknown, fallback: string[]): string[] {
  if (Array.isArray(value)) {
    return value
      .filter((entry): entry is string => typeof entry === "string")
      .map((entry) => entry.trim())
      .filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split(/[,\n]/)
      .map((entry) => entry.trim())
      .filter(Boolean);
  }
  return fallback;
}

function resolveDefaultBaseUrl(): string {
  return process.env.EBM_PY_BASE_URL ?? `http://127.0.0.1:${DEFAULT_PORT}`;
}

const ALLOWED_KEYS = [
  "mode",
  "pythonCommand",
  "port",
  "baseUrl",
  "ebmPyPath",
  "configJsonPath",
  "dbPath",
  "slowPathEnabled",
  "timeoutMs",
  "healthTimeoutMs",
  "portScanRange",
  "autoRecall",
  "recallLimit",
  "recallScoreThreshold",
  "recallMaxContentChars",
  "recallTokenBudget",
  "recallTimeoutMs",
  "ingestReplyAssist",
  "ingestReplyAssistMinSpeakerTurns",
  "ingestReplyAssistMinChars",
  "ingestReplyAssistIgnoreSessionPatterns",
] as const;

function assertAllowedKeys(value: Record<string, unknown>, label: string): void {
  const allowed = ALLOWED_KEYS as readonly string[];
  const unknown = Object.keys(value).filter((k) => !allowed.includes(k));
  if (unknown.length > 0) {
    throw new Error(`${label} has unknown keys: ${unknown.join(", ")}`);
  }
}

// ── Config schema (parse + validate) ────────────────────────

export const ebmPyConfigSchema = {
  parse(value: unknown): Required<EbmPyPluginConfig> {
    if (!value || typeof value !== "object" || Array.isArray(value)) {
      value = {};
    }
    const cfg = value as Record<string, unknown>;
    assertAllowedKeys(cfg, "ebm-context-engine config");

    const mode = (cfg.mode === "local" || cfg.mode === "remote" ? cfg.mode : DEFAULT_MODE) as
      | "local"
      | "remote";

    const pythonCommand =
      typeof cfg.pythonCommand === "string" && cfg.pythonCommand.trim()
        ? cfg.pythonCommand.trim()
        : DEFAULT_PYTHON_COMMAND;

    const port = Math.max(1, Math.min(65535, Math.floor(toNumber(cfg.port, DEFAULT_PORT))));

    const localBaseUrl = `http://127.0.0.1:${port}`;
    const rawBaseUrl =
      mode === "local"
        ? localBaseUrl
        : typeof cfg.baseUrl === "string" && cfg.baseUrl.trim()
          ? cfg.baseUrl.trim()
          : resolveDefaultBaseUrl();
    const baseUrl = resolveEnvVars(rawBaseUrl).replace(/\/+$/, "");

    const ebmPyPath =
      typeof cfg.ebmPyPath === "string" && cfg.ebmPyPath.trim()
        ? resolvePath(expandHome(resolveEnvVars(cfg.ebmPyPath.trim())))
        : "";

    const configJsonPath =
      typeof cfg.configJsonPath === "string" && cfg.configJsonPath.trim()
        ? resolvePath(expandHome(resolveEnvVars(cfg.configJsonPath.trim())))
        : ebmPyPath
          ? join(ebmPyPath, "ebm", "config.json")
          : "";

    const dbPath =
      typeof cfg.dbPath === "string" && cfg.dbPath.trim()
        ? resolvePath(expandHome(resolveEnvVars(cfg.dbPath.trim())))
        : DEFAULT_DB_PATH;

    const slowPathEnabled = cfg.slowPathEnabled !== false;

    const timeoutMs = Math.max(1000, Math.floor(toNumber(cfg.timeoutMs, DEFAULT_TIMEOUT_MS)));
    const healthTimeoutMs = Math.max(
      1000,
      Math.floor(toNumber(cfg.healthTimeoutMs, DEFAULT_HEALTH_TIMEOUT_MS)),
    );
    const portScanRange = Math.max(
      0,
      Math.min(100, Math.floor(toNumber(cfg.portScanRange, DEFAULT_PORT_SCAN_RANGE))),
    );

    const autoRecall = cfg.autoRecall !== false;
    const recallLimit = Math.max(1, Math.floor(toNumber(cfg.recallLimit, DEFAULT_RECALL_LIMIT)));
    const recallScoreThreshold = Math.min(
      1,
      Math.max(0, toNumber(cfg.recallScoreThreshold, DEFAULT_RECALL_SCORE_THRESHOLD)),
    );
    const recallMaxContentChars = Math.max(
      50,
      Math.min(10000, Math.floor(toNumber(cfg.recallMaxContentChars, DEFAULT_RECALL_MAX_CONTENT_CHARS))),
    );
    const recallTokenBudget = Math.max(
      100,
      Math.min(50000, Math.floor(toNumber(cfg.recallTokenBudget, DEFAULT_RECALL_TOKEN_BUDGET))),
    );
    const recallTimeoutMs = Math.max(
      500,
      Math.min(30000, Math.floor(toNumber(cfg.recallTimeoutMs, DEFAULT_RECALL_TIMEOUT_MS))),
    );
    const ingestReplyAssist = cfg.ingestReplyAssist !== false;
    const ingestReplyAssistMinSpeakerTurns = Math.max(
      1,
      Math.min(12, Math.floor(toNumber(cfg.ingestReplyAssistMinSpeakerTurns, DEFAULT_INGEST_REPLY_ASSIST_MIN_SPEAKER_TURNS))),
    );
    const ingestReplyAssistMinChars = Math.max(
      32,
      Math.min(10000, Math.floor(toNumber(cfg.ingestReplyAssistMinChars, DEFAULT_INGEST_REPLY_ASSIST_MIN_CHARS))),
    );
    const ingestReplyAssistIgnoreSessionPatterns = toStringArray(
      cfg.ingestReplyAssistIgnoreSessionPatterns,
      [],
    );

    return {
      mode,
      pythonCommand,
      port,
      baseUrl,
      ebmPyPath,
      configJsonPath,
      dbPath,
      slowPathEnabled,
      timeoutMs,
      healthTimeoutMs,
      portScanRange,
      autoRecall,
      recallLimit,
      recallScoreThreshold,
      recallMaxContentChars,
      recallTokenBudget,
      recallTimeoutMs,
      ingestReplyAssist,
      ingestReplyAssistMinSpeakerTurns,
      ingestReplyAssistMinChars,
      ingestReplyAssistIgnoreSessionPatterns,
    };
  },
};
