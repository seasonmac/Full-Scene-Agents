import type { DebugState } from "./controllers/debug.ts";
import { loadDebug } from "./controllers/debug.ts";
import type { LogsState } from "./controllers/logs.ts";
import { loadLogs } from "./controllers/logs.ts";
import type { NodesState } from "./controllers/nodes.ts";
import { loadNodes } from "./controllers/nodes.ts";
import { loadTraces } from "./controllers/tracing.ts";

type PollingHost = {
  nodesPollInterval: number | null;
  logsPollInterval: number | null;
  debugPollInterval: number | null;
  tracesPollInterval: number | null;
  tab: string;
  tracingAutoFollow: boolean;
};

export function startNodesPolling(host: PollingHost) {
  if (host.nodesPollInterval != null) {
    return;
  }
  host.nodesPollInterval = window.setInterval(
    () => void loadNodes(host as unknown as NodesState, { quiet: true }),
    5000,
  );
}

export function stopNodesPolling(host: PollingHost) {
  if (host.nodesPollInterval == null) {
    return;
  }
  clearInterval(host.nodesPollInterval);
  host.nodesPollInterval = null;
}

export function startLogsPolling(host: PollingHost) {
  if (host.logsPollInterval != null) {
    return;
  }
  host.logsPollInterval = window.setInterval(() => {
    if (host.tab !== "logs") {
      return;
    }
    void loadLogs(host as unknown as LogsState, { quiet: true });
  }, 2000);
}

export function stopLogsPolling(host: PollingHost) {
  if (host.logsPollInterval == null) {
    return;
  }
  clearInterval(host.logsPollInterval);
  host.logsPollInterval = null;
}

export function startDebugPolling(host: PollingHost) {
  if (host.debugPollInterval != null) {
    return;
  }
  host.debugPollInterval = window.setInterval(() => {
    if (host.tab !== "debug") {
      return;
    }
    void loadDebug(host as unknown as DebugState);
  }, 3000);
}

export function stopDebugPolling(host: PollingHost) {
  if (host.debugPollInterval == null) {
    return;
  }
  clearInterval(host.debugPollInterval);
  host.debugPollInterval = null;
}

export function startTracingPolling(host: PollingHost) {
  if (host.tracesPollInterval != null) {
    return;
  }
  host.tracesPollInterval = window.setInterval(() => {
    if (host.tab !== "tracing" || !host.tracingAutoFollow) {
      return;
    }
    void loadTraces(host as unknown as OpenClawApp, { quiet: true });
  }, 2500);
}

export function stopTracingPolling(host: PollingHost) {
  if (host.tracesPollInterval == null) {
    return;
  }
  clearInterval(host.tracesPollInterval);
  host.tracesPollInterval = null;
}
