/**
 * EBM Python sidecar process manager.
 *
 * Handles port preparation, health checking, sidecar lifecycle,
 * and defensive re-spawn — following the same patterns as EBM.
 */
import { execSync, execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { Socket } from "node:net";
import { platform, homedir } from "node:os";
import { join } from "node:path";
import type { spawn as spawnFn } from "node:child_process";

export const IS_WIN = platform() === "win32";

// ── Logger interface ────────────────────────────────────────

export interface ProcessLogger {
  info: (msg: string) => void;
  warn: (msg: string) => void;
}

// ── Health checks ───────────────────────────────────────────

export function waitForHealth(
  baseUrl: string,
  timeoutMs: number,
  intervalMs = 500,
): Promise<void> {
  const deadline = Date.now() + timeoutMs;
  return new Promise((resolve, reject) => {
    const tick = () => {
      if (Date.now() > deadline) {
        reject(new Error(`EBM-PY health check timeout at ${baseUrl}`));
        return;
      }
      fetch(`${baseUrl}/health`, { signal: AbortSignal.timeout(2_000) })
        .then((r) => r.json())
        .then((body: Record<string, unknown>) => {
          if (body?.ok === true) {
            resolve();
            return;
          }
          setTimeout(tick, intervalMs);
        })
        .catch(() => setTimeout(tick, intervalMs));
    };
    tick();
  });
}

export function waitForHealthOrExit(
  baseUrl: string,
  timeoutMs: number,
  intervalMs: number,
  child: ReturnType<typeof spawnFn>,
): Promise<void> {
  const exited = child.killed || child.exitCode !== null || child.signalCode !== null;
  if (exited) {
    return Promise.reject(
      new Error(
        `EBM-PY subprocess exited before health check (code=${child.exitCode}, signal=${child.signalCode})`,
      ),
    );
  }

  return new Promise((resolve, reject) => {
    let settled = false;

    const cleanup = () => {
      child.off?.("error", onError);
      child.off?.("exit", onExit);
    };

    const finishResolve = () => {
      if (settled) return;
      settled = true;
      cleanup();
      resolve();
    };

    const finishReject = (err: unknown) => {
      if (settled) return;
      settled = true;
      cleanup();
      reject(err instanceof Error ? err : new Error(String(err)));
    };

    const onError = (err: Error) => finishReject(err);
    const onExit = (code: number | null, signal: string | null) => {
      finishReject(
        new Error(`EBM-PY subprocess exited before health check (code=${code}, signal=${signal})`),
      );
    };

    child.once("error", onError);
    child.once("exit", onExit);
    waitForHealth(baseUrl, timeoutMs, intervalMs).then(finishResolve, finishReject);
  });
}

// ── TCP / quick probes ──────────────────────────────────────

export function quickTcpProbe(host: string, port: number, timeoutMs: number): Promise<boolean> {
  return new Promise((resolve) => {
    const socket = new Socket();
    let done = false;
    const finish = (ok: boolean) => {
      if (done) return;
      done = true;
      socket.destroy();
      resolve(ok);
    };
    socket.setTimeout(timeoutMs);
    socket.once("connect", () => finish(true));
    socket.once("timeout", () => finish(false));
    socket.once("error", () => finish(false));
    try {
      socket.connect(port, host);
    } catch {
      finish(false);
    }
  });
}

export async function quickHealthCheck(baseUrl: string, timeoutMs: number): Promise<boolean> {
  try {
    const response = await fetch(`${baseUrl}/health`, {
      method: "GET",
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!response.ok) return false;
    const body = (await response.json().catch(() => ({}))) as Record<string, unknown>;
    return body.ok === true;
  } catch {
    return false;
  }
}

export async function quickPrecheck(
  mode: "local" | "remote",
  baseUrl: string,
  defaultPort: number,
  localProcess: ReturnType<typeof spawnFn> | null,
): Promise<{ ok: true } | { ok: false; reason: string }> {
  const healthOk = await quickHealthCheck(baseUrl, 500);
  if (healthOk) return { ok: true };

  let host = "127.0.0.1";
  let port = defaultPort;
  try {
    const parsed = new URL(baseUrl);
    if (parsed.hostname) host = parsed.hostname;
    if (parsed.port) {
      const p = Number(parsed.port);
      if (Number.isFinite(p) && p > 0) port = p;
    }
  } catch {
    // malformed URL — use defaults
  }

  if (mode === "local") {
    const portOk = await quickTcpProbe(host, port, 200);
    if (!portOk) return { ok: false, reason: `local port unavailable (${host}:${port})` };
    if (
      localProcess &&
      (localProcess.killed || localProcess.exitCode !== null || localProcess.signalCode !== null)
    ) {
      return { ok: false, reason: "local process is not running" };
    }
    if (localProcess === null) return { ok: true };
  }
  return { ok: false, reason: "health check failed" };
}

// ── Port management ─────────────────────────────────────────

export async function prepareLocalPort(
  port: number,
  logger: ProcessLogger,
  maxRetries = 10,
): Promise<number> {
  // Check if an EBM sidecar is already on this port
  const isEbm = await quickHealthCheck(`http://127.0.0.1:${port}`, 2_000);
  if (isEbm) {
    logger.info(`[EBM-PY] killing stale sidecar on port ${port}`);
    await killProcessOnPort(port, logger);
    return port;
  }

  const occupied = await quickTcpProbe("127.0.0.1", port, 500);
  if (!occupied) return port;

  // Port occupied by something else — scan for free port
  logger.warn(`[EBM-PY] port ${port} occupied, searching for free port...`);
  for (let candidate = port + 1; candidate <= port + maxRetries; candidate++) {
    if (candidate > 65535) break;
    const taken = await quickTcpProbe("127.0.0.1", candidate, 300);
    if (!taken) {
      logger.info(`[EBM-PY] using free port ${candidate} instead of ${port}`);
      return candidate;
    }
  }
  throw new Error(
    `[EBM-PY] port ${port} occupied and no free port in range ${port + 1}–${port + maxRetries}`,
  );
}

// ── Kill helpers ────────────────────────────────────────────

function killProcessOnPort(port: number, logger: ProcessLogger): Promise<void> {
  const safePort = Math.floor(port);
  if (!Number.isFinite(safePort) || safePort < 1 || safePort > 65535) {
    throw new Error(`[EBM-PY] invalid port: ${port}`);
  }
  return IS_WIN ? killProcessOnPortWin(safePort, logger) : killProcessOnPortUnix(safePort, logger);
}

async function killProcessOnPortWin(port: number, logger: ProcessLogger): Promise<void> {
  try {
    const netstatOut = execFileSync("cmd.exe", [
      "/c",
      `netstat -ano | findstr "LISTENING" | findstr ":${String(port)}"`,
    ], { encoding: "utf-8" }).trim();
    if (!netstatOut) return;
    const pids = new Set<number>();
    for (const line of netstatOut.split(/\r?\n/)) {
      const m = line.trim().match(/\s(\d+)\s*$/);
      if (m) pids.add(Number(m[1]));
    }
    for (const pid of pids) {
      if (pid > 0) {
        logger.info(`[EBM-PY] killing pid ${pid} on port ${port}`);
        try {
          execFileSync("taskkill", ["/PID", String(pid), "/F"]);
        } catch {
          /* already gone */
        }
      }
    }
    if (pids.size) await new Promise((r) => setTimeout(r, 500));
  } catch {
    /* netstat not available or no stale process */
  }
}

async function killProcessOnPortUnix(port: number, logger: ProcessLogger): Promise<void> {
  const portStr = String(port); // already validated as safe integer
  try {
    let pids: number[] = [];
    try {
      // lsof supports direct args without shell
      const lsofOut = execFileSync("lsof", ["-ti", `tcp:${portStr}`, "-s", "tcp:listen"], {
        encoding: "utf-8",
        stdio: ["pipe", "pipe", "pipe"],
      }).trim();
      if (lsofOut)
        pids = lsofOut
          .split(/\s+/)
          .map((s) => Number(s))
          .filter((n) => n > 0);
    } catch {
      /* lsof not available */
    }
    if (pids.length === 0) {
      try {
        // ss requires shell pipe — port is pre-validated as safe integer
        const ssOut = execSync(
          `ss -tlnp 2>/dev/null | awk -v p=":${portStr}" '$4 ~ p {gsub(/.*pid=/,""); gsub(/,.*/,""); print; exit}'`,
          { encoding: "utf-8", shell: "/bin/sh" },
        ).trim();
        if (ssOut) {
          const n = Number(ssOut);
          if (n > 0) pids = [n];
        }
      } catch {
        /* ss not available */
      }
    }
    for (const pid of pids) {
      logger.info(`[EBM-PY] killing pid ${pid} on port ${port}`);
      try {
        process.kill(pid, "SIGKILL");
      } catch {
        /* already gone */
      }
    }
    if (pids.length) await new Promise((r) => setTimeout(r, 500));
  } catch {
    /* port check failed */
  }
}

// ── Python resolution ───────────────────────────────────────

export function resolvePythonCommand(configured: string | undefined, logger: ProcessLogger): string {
  // Explicit config takes priority
  if (configured && configured.trim()) return configured.trim();

  // Environment variable
  const fromEnv = process.env.EBM_PY_PYTHON;
  if (fromEnv) return fromEnv;

  // Shell-env file (~/.openclaw/ebm-py.env)
  const envFile = join(homedir(), ".openclaw", "ebm-py.env");
  if (existsSync(envFile)) {
    try {
      const content = readFileSync(envFile, "utf-8");
      const m = IS_WIN
        ? content.match(/set\s+EBM_PY_PYTHON=(.+)/i)
        : content.match(/EBM_PY_PYTHON=['"]([^'"]+)['"]/);
      if (m?.[1]) return m[1].trim();
    } catch {
      /* ignore */
    }
  }

  // which/where fallback
  const defaultPy = IS_WIN ? "python" : "python3";
  try {
    const cmd = IS_WIN
      ? "where python"
      : "command -v python3 || which python3";
    const resolved = execSync(cmd, {
      encoding: "utf-8",
      shell: IS_WIN ? "cmd.exe" : "/bin/sh",
      env: process.env,
    }).split(/\r?\n/)[0].trim();
    if (resolved) return resolved;
  } catch {
    /* not found */
  }

  logger.warn(
    `[EBM-PY] could not resolve python path, falling back to "${defaultPy}". ` +
      `Set EBM_PY_PYTHON or pythonCommand in plugin config.`,
  );
  return defaultPy;
}
