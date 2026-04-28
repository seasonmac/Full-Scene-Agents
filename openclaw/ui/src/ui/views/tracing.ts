import { html, nothing, type TemplateResult } from "lit";
import { formatCost, formatTokens } from "../format.ts";
import type { LlmTraceRecord, LlmTraceSummary } from "../types.ts";

type TraceSortKey = "time" | "name" | "status" | "tokens" | "cost" | "latency" | "input" | "output";

export type TracingProps = {
  loading: boolean;
  error: string | null;
  file: string | null;
  entries: LlmTraceSummary[];
  selected: LlmTraceRecord | null;
  selectedId: string | null;
  filterText: string;
  providerFilter: string;
  statusFilter: "all" | "ok" | "error" | "in_progress";
  autoFollow: boolean;
  sortKey: TraceSortKey;
  sortDir: "asc" | "desc";
  truncated: boolean;
  onFilterTextChange: (next: string) => void;
  onProviderFilterChange: (next: string) => void;
  onStatusFilterChange: (next: "all" | "ok" | "error" | "in_progress") => void;
  onAutoFollowChange: (next: boolean) => void;
  onSortChange: (next: TraceSortKey) => void;
  onRefresh: () => void;
  onSelect: (traceId: string) => void;
  onCloseDetail: () => void;
  onExport: (lines: string[], label: string) => void;
};

function formatDateTime(value?: number) {
  if (typeof value !== "number") {
    return "";
  }
  return new Date(value).toLocaleString();
}

function formatDuration(value?: number) {
  if (typeof value !== "number") {
    return "--";
  }
  if (value < 1000) {
    return `${value} ms`;
  }
  return `${(value / 1000).toFixed(2)} s`;
}

function formatCompactTime(value?: number) {
  if (typeof value !== "number") {
    return "--";
  }
  return new Date(value).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function compareText(left?: string, right?: string) {
  return (left ?? "").localeCompare(right ?? "");
}

function sortEntries(entries: LlmTraceSummary[], sortKey: TraceSortKey, sortDir: "asc" | "desc") {
  const direction = sortDir === "asc" ? 1 : -1;
  return entries.toSorted((left, right) => {
    let value = 0;
    switch (sortKey) {
      case "time":
        value = (left.startedAt ?? 0) - (right.startedAt ?? 0);
        break;
      case "name":
        value = compareText(
          `${left.provider ?? ""}/${left.modelId ?? left.modelApi ?? ""}`,
          `${right.provider ?? ""}/${right.modelId ?? right.modelApi ?? ""}`,
        );
        break;
      case "status":
        value = compareText(left.status, right.status);
        break;
      case "tokens":
        value = (left.usage?.total ?? 0) - (right.usage?.total ?? 0);
        break;
      case "cost":
        value = (left.costTotal ?? -1) - (right.costTotal ?? -1);
        break;
      case "latency":
        value = (left.durationMs ?? 0) - (right.durationMs ?? 0);
        break;
      case "input":
        value = compareText(left.requestPreview, right.requestPreview);
        break;
      case "output":
        value = compareText(
          left.responsePreview ?? left.error,
          right.responsePreview ?? right.error,
        );
        break;
    }
    if (value === 0) {
      return ((left.startedAt ?? 0) - (right.startedAt ?? 0)) * direction;
    }
    return value * direction;
  });
}

function renderSortHeader(
  label: string,
  key: TraceSortKey,
  activeKey: TraceSortKey,
  activeDir: "asc" | "desc",
  onSortChange: (next: TraceSortKey) => void,
) {
  const active = activeKey === key;
  const arrow = active ? (activeDir === "asc" ? "↑" : "↓") : "↕";
  return html`
    <button
      class="tracing-table__sort ${active ? "is-active" : ""}"
      @click=${() => onSortChange(key)}
    >
      <span>${label}</span>
      <span class="tracing-table__sort-icon">${arrow}</span>
    </button>
  `;
}

function copyText(text: string) {
  return navigator.clipboard.writeText(text).catch(() => undefined);
}

function jsonString(value: unknown) {
  return JSON.stringify(value ?? null, null, 2);
}

function renderJsonPrimitive(value: unknown): TemplateResult {
  if (typeof value === "string") {
    return html`<span class="tracing-json-string">"${value}"</span>`;
  }
  if (typeof value === "number") {
    return html`<span class="tracing-json-number">${String(value)}</span>`;
  }
  if (typeof value === "boolean") {
    return html`<span class="tracing-json-boolean">${String(value)}</span>`;
  }
  if (value === null || value === undefined) {
    return html` <span class="tracing-json-null">null</span> `;
  }
  const fallback = JSON.stringify(value);
  return html`<span class="tracing-json-string">${fallback ?? '"[unsupported]"'}</span>`;
}

function renderJsonTree(value: unknown, depth = 0): TemplateResult {
  if (value === null || value === undefined || typeof value !== "object") {
    return renderJsonPrimitive(value);
  }

  const isArray = Array.isArray(value);
  const entries = isArray
    ? value.map((entry, index) => [String(index), entry] as const)
    : Object.entries(value as Record<string, unknown>);

  if (entries.length === 0) {
    return html`<span class="tracing-json-punctuation">${isArray ? "[]" : "{}"}</span>`;
  }

  const preview = entries
    .slice(0, 3)
    .map(([key, child]) => {
      if (isArray) {
        return typeof child === "object" && child !== null ? "…" : String(child);
      }
      return key;
    })
    .join(", ");

  return html`
    <details class="tracing-json-node" ?open=${depth < 2}>
      <summary class="tracing-json-node__summary">
        <span class="tracing-json-punctuation">${isArray ? "[" : "{"}</span>
        <span class="tracing-json-node__meta">${entries.length} ${isArray ? "items" : "keys"}</span>
        <span class="tracing-json-node__preview">${preview}</span>
        <span class="tracing-json-punctuation">${isArray ? "]" : "}"}</span>
      </summary>
      <div class="tracing-json-node__children">
        ${entries.map(
          ([key, child], index) => html`
            <div class="tracing-json-entry">
              ${isArray
                ? html`<span class="tracing-json-index">${key}</span>`
                : html`<span class="tracing-json-key">"${key}"</span
                    ><span class="tracing-json-punctuation">:</span>`}
              <span class="tracing-json-value">${renderJsonTree(child, depth + 1)}</span>
              ${index < entries.length - 1
                ? html` <span class="tracing-json-punctuation">,</span> `
                : nothing}
            </div>
          `,
        )}
      </div>
    </details>
  `;
}

function renderJsonInspectorCard(title: string, value: unknown) {
  const raw = jsonString(value);
  return html`
    <details class="tracing-json-card" open>
      <summary class="tracing-json-card__summary">
        <span>${title}</span>
        <span class="tracing-json-card__actions">
          <button
            type="button"
            class="btn btn--sm btn--ghost"
            @click=${(event: Event) => {
              event.preventDefault();
              event.stopPropagation();
              void copyText(raw);
            }}
          >
            Copy
          </button>
        </span>
      </summary>
      <div class="tracing-json-card__body">
        <div class="tracing-json-card__meta">${raw.split("\n").length} lines</div>
        <div class="tracing-json-tree">${renderJsonTree(value)}</div>
      </div>
    </details>
  `;
}

function collectStringMatches(value: unknown, pattern: RegExp, out: Set<string>) {
  if (typeof value === "string") {
    for (const match of value.matchAll(pattern)) {
      if (match[0]) {
        out.add(match[0]);
      }
    }
    return;
  }
  if (Array.isArray(value)) {
    for (const entry of value) {
      collectStringMatches(entry, pattern, out);
    }
    return;
  }
  if (!value || typeof value !== "object") {
    return;
  }
  for (const entry of Object.values(value as Record<string, unknown>)) {
    collectStringMatches(entry, pattern, out);
  }
}

function countToolDefinitions(value: unknown): number {
  if (Array.isArray(value)) {
    let total = 0;
    for (const entry of value) {
      total += countToolDefinitions(entry);
    }
    return total;
  }
  if (!value || typeof value !== "object") {
    return 0;
  }
  const record = value as Record<string, unknown>;
  let total = 0;
  const tools = record.tools;
  if (Array.isArray(tools)) {
    total += tools.filter((entry) => entry && typeof entry === "object").length;
  }
  const functions = record.functions;
  if (Array.isArray(functions)) {
    total += functions.filter((entry) => entry && typeof entry === "object").length;
  }
  for (const entry of Object.values(record)) {
    total += countToolDefinitions(entry);
  }
  return total;
}

function countMessageEntries(value: unknown): number {
  if (Array.isArray(value)) {
    let total = 0;
    for (const entry of value) {
      total += countMessageEntries(entry);
    }
    return total;
  }
  if (!value || typeof value !== "object") {
    return 0;
  }
  const record = value as Record<string, unknown>;
  let total = 0;
  if (Array.isArray(record.messages)) {
    total += record.messages.length;
  }
  if (Array.isArray(record.input)) {
    total += record.input.length;
  }
  for (const entry of Object.values(record)) {
    total += countMessageEntries(entry);
  }
  return total;
}

function buildTokenComposition(record: LlmTraceRecord) {
  const staticFiles = new Set<string>();
  const skills = new Set<string>();
  collectStringMatches(
    record.requests,
    /\b(?:AGENTS|CLAUDE|README|copilot-instructions)\.md\b/g,
    staticFiles,
  );
  collectStringMatches(record.requests, /\b[\w-]+\/SKILL\.md\b/g, skills);
  const toolSchemas = countToolDefinitions(record.requests);
  const dynamicMessages = countMessageEntries(record.requests);

  return {
    staticPrompt: [
      {
        title: "System prompts",
        stat: `${staticFiles.size} core files`,
        detail:
          staticFiles.size > 0
            ? `${Array.from(staticFiles).slice(0, 3).join(", ")} and related runtime instructions`
            : "Base runtime instructions and framework constraints",
      },
      {
        title: "Skills",
        stat: `${skills.size} presets`,
        detail:
          skills.size > 0
            ? `${Array.from(skills).slice(0, 2).join(", ")} text descriptions`
            : "Preset skill descriptions injected for the current run",
      },
      {
        title: "Tool schemas",
        stat: `${toolSchemas} JSON defs`,
        detail: "Structured tool definitions exposed to the model at request time",
      },
      {
        title: "Others",
        stat: "Runtime guardrails",
        detail:
          "System-level safety instructions, provider routing rules, and framework constraints",
      },
    ],
    dynamicPrompt: [
      {
        title: "Conversation and state history",
        stat: `${dynamicMessages} message blocks`,
        detail: "上下关联的往返对话、思考过程与缓冲，以及本轮请求前的状态延续内容",
      },
    ],
  };
}

function renderCompositionCard(record: LlmTraceRecord) {
  const composition = buildTokenComposition(record);
  return html`
    <div class="tracing-summary-card">
      <div class="tracing-summary-card__title">Input Token Composition</div>
      <div class="tracing-composition-section">
        <div class="tracing-composition-section__heading">系统提示词</div>
        ${composition.staticPrompt.map(
          (item) => html`
            <div class="tracing-composition-item">
              <div class="tracing-composition-item__title-row">
                <span>${item.title}</span>
                <span class="tracing-composition-item__stat">${item.stat}</span>
              </div>
              <div class="tracing-composition-item__detail">${item.detail}</div>
            </div>
          `,
        )}
      </div>
      <div class="tracing-composition-section">
        <div class="tracing-composition-section__heading">动态提示词</div>
        ${composition.dynamicPrompt.map(
          (item) => html`
            <div class="tracing-composition-item">
              <div class="tracing-composition-item__title-row">
                <span>${item.title}</span>
                <span class="tracing-composition-item__stat">${item.stat}</span>
              </div>
              <div class="tracing-composition-item__detail">${item.detail}</div>
            </div>
          `,
        )}
      </div>
    </div>
  `;
}

function renderSummaryJsonCard(record: LlmTraceRecord) {
  const summary = {
    provider: record.provider,
    modelId: record.modelId,
    modelApi: record.modelApi,
    usage: record.usage ?? null,
    costTotal: record.costTotal ?? null,
    requestPreview: record.requestPreview ?? null,
    responsePreview: record.responsePreview ?? null,
    error: record.error ?? null,
  };

  return html`
    <div class="tracing-summary-card">
      <div class="tracing-summary-card__title">Trace Summary JSON</div>
      <pre class="tracing-summary-json">${jsonString(summary)}</pre>
    </div>
  `;
}

function renderTimelineFlow(record: LlmTraceRecord) {
  return html`
    <div class="tracing-timeline-flow">
      ${record.timeline.map(
        (entry, index) => html`
          <div class="tracing-timeline-flow__item">
            <span class="tracing-timeline-flow__label">${entry.label}</span>
            <span class="tracing-timeline-flow__time"
              >${new Date(entry.ts).toLocaleTimeString()}</span
            >
          </div>
          ${index < record.timeline.length - 1
            ? html` <span class="tracing-timeline-flow__arrow">→</span> `
            : nothing}
        `,
      )}
    </div>
  `;
}

export function renderTracing(props: TracingProps) {
  const filtered = sortEntries(props.entries, props.sortKey, props.sortDir);
  const exportPayload = props.selected ? JSON.stringify(props.selected, null, 2) : null;
  const providerOptions = Array.from(
    new Set(
      props.entries
        .map((entry) => entry.provider)
        .filter((value): value is string => Boolean(value)),
    ),
  ).toSorted((left, right) => left.localeCompare(right));

  const handleCodeBlockCopy = (event: Event) => {
    const button = (event.target as HTMLElement).closest(".code-block-copy");
    if (!button) {
      return;
    }
    const code = (button as HTMLElement).dataset.code ?? "";
    void navigator.clipboard.writeText(code).then(() => {
      button.classList.add("copied");
      setTimeout(() => button.classList.remove("copied"), 1500);
    });
  };

  return html`
    <section class="tracing-shell" @click=${handleCodeBlockCopy}>
      <div class="card">
        <div class="row" style="justify-content: space-between; align-items: flex-start;">
          <div>
            <div class="card-title">Traces</div>
            <div class="card-sub">Recent sanitized LLM request and response activity.</div>
          </div>
          <button class="btn" ?disabled=${props.loading} @click=${props.onRefresh}>
            ${props.loading ? "Loading…" : "Refresh"}
          </button>
        </div>

        <label class="field" style="margin-top: 14px;">
          <span>Filter</span>
          <input
            .value=${props.filterText}
            @input=${(e: Event) => props.onFilterTextChange((e.target as HTMLInputElement).value)}
            placeholder="Search trace id, provider, model, session"
          />
        </label>

        <div
          class="grid"
          style="grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; margin-top: 12px; align-items: end;"
        >
          <label class="field">
            <span>Provider</span>
            <select
              .value=${props.providerFilter}
              @change=${(e: Event) =>
                props.onProviderFilterChange((e.target as HTMLSelectElement).value)}
            >
              <option value="">All providers</option>
              ${providerOptions.map(
                (provider) => html`<option value=${provider}>${provider}</option>`,
              )}
            </select>
          </label>

          <label class="field">
            <span>Status</span>
            <select
              .value=${props.statusFilter}
              @change=${(e: Event) =>
                props.onStatusFilterChange(
                  (e.target as HTMLSelectElement).value as "all" | "ok" | "error" | "in_progress",
                )}
            >
              <option value="all">All statuses</option>
              <option value="ok">ok</option>
              <option value="error">error</option>
              <option value="in_progress">in_progress</option>
            </select>
          </label>

          <label
            class="field"
            style="display: flex; flex-direction: row; align-items: center; gap: 8px; min-height: 44px;"
          >
            <input
              type="checkbox"
              .checked=${props.autoFollow}
              @change=${(e: Event) =>
                props.onAutoFollowChange((e.target as HTMLInputElement).checked)}
            />
            <span>Auto-follow</span>
          </label>
        </div>

        ${props.file
          ? html`<div class="muted" style="margin-top: 10px;">File: ${props.file}</div>`
          : nothing}
        ${props.truncated
          ? html`
              <div class="callout" style="margin-top: 10px">
                Trace history truncated to the latest chunk.
              </div>
            `
          : nothing}
        ${props.error
          ? html`<div class="callout danger" style="margin-top: 10px;">${props.error}</div>`
          : nothing}

        <div class="tracing-table" style="margin-top: 14px;">
          <div class="tracing-table__header">
            ${renderSortHeader("Time", "time", props.sortKey, props.sortDir, props.onSortChange)}
            ${renderSortHeader("Name", "name", props.sortKey, props.sortDir, props.onSortChange)}
            ${renderSortHeader(
              "Status",
              "status",
              props.sortKey,
              props.sortDir,
              props.onSortChange,
            )}
            ${renderSortHeader(
              "Tokens",
              "tokens",
              props.sortKey,
              props.sortDir,
              props.onSortChange,
            )}
            ${renderSortHeader("Cost", "cost", props.sortKey, props.sortDir, props.onSortChange)}
            ${renderSortHeader(
              "Latency",
              "latency",
              props.sortKey,
              props.sortDir,
              props.onSortChange,
            )}
            ${renderSortHeader(
              "LLM Input",
              "input",
              props.sortKey,
              props.sortDir,
              props.onSortChange,
            )}
            ${renderSortHeader(
              "LLM Output",
              "output",
              props.sortKey,
              props.sortDir,
              props.onSortChange,
            )}
          </div>
          ${filtered.length === 0
            ? html` <div class="muted tracing-table__empty">No traces yet.</div> `
            : filtered.map(
                (entry) => html`
                  <button
                    class="tracing-table__row ${props.selectedId === entry.traceId
                      ? "is-selected"
                      : ""}"
                    @click=${() => props.onSelect(entry.traceId)}
                  >
                    <div class="tracing-table__cell tracing-table__time">
                      <div>${formatCompactTime(entry.startedAt)}</div>
                      <div class="muted">${formatDateTime(entry.startedAt)}</div>
                    </div>
                    <div class="tracing-table__cell tracing-table__name">
                      <div class="list-title">
                        ${entry.provider ?? "unknown"} /
                        ${entry.modelId ?? entry.modelApi ?? "unknown"}
                      </div>
                      <div class="list-sub mono">${entry.traceId.slice(0, 8)}</div>
                    </div>
                    <div class="tracing-table__cell tracing-table__status">
                      <span class="tracing-status tracing-status--${entry.status}"
                        >${entry.status}</span
                      >
                    </div>
                    <div class="tracing-table__cell tracing-table__metric">
                      ${formatTokens(entry.usage?.total, "--")}
                    </div>
                    <div class="tracing-table__cell tracing-table__metric">
                      ${entry.costTotal == null ? "--" : formatCost(entry.costTotal)}
                    </div>
                    <div class="tracing-table__cell tracing-table__metric">
                      ${formatDuration(entry.durationMs)}
                    </div>
                    <div class="tracing-table__cell tracing-table__preview">
                      ${entry.requestPreview ?? "--"}
                    </div>
                    <div class="tracing-table__cell tracing-table__preview">
                      ${entry.responsePreview ?? entry.error ?? "--"}
                    </div>
                  </button>
                `,
              )}
        </div>
      </div>

      ${props.selected
        ? html`
            <div class="tracing-detail-overlay" @click=${() => props.onCloseDetail()}>
              <aside
                class="card tracing-detail-panel"
                @click=${(event: Event) => event.stopPropagation()}
              >
                <div class="tracing-detail-header">
                  <div>
                    <div class="card-title">Trace Detail</div>
                    <div class="card-sub">
                      Request payloads, final response snapshot, and timeline.
                    </div>
                  </div>
                  <div class="tracing-detail-header__actions">
                    <button
                      class="btn"
                      ?disabled=${!exportPayload}
                      @click=${() =>
                        exportPayload
                          ? props.onExport([exportPayload], `trace-${props.selectedId ?? "detail"}`)
                          : undefined}
                    >
                      Export JSON
                    </button>
                    <button class="btn btn--ghost" @click=${() => props.onCloseDetail()}>×</button>
                  </div>
                </div>

                <div class="tracing-detail-meta">
                  <div>
                    <div class="muted">Session</div>
                    <div class="mono">
                      ${props.selected.sessionKey ?? props.selected.sessionId ?? "--"}
                    </div>
                  </div>
                  <div>
                    <div class="muted">Duration</div>
                    <div>${formatDuration(props.selected.durationMs)}</div>
                  </div>
                  <div>
                    <div class="muted">Started</div>
                    <div>${formatDateTime(props.selected.startedAt)}</div>
                  </div>
                  <div>
                    <div class="muted">Status</div>
                    <div>${props.selected.status}</div>
                  </div>
                </div>

                <section class="tracing-detail-section">
                  <div class="card-title tracing-detail-section__title">Summary</div>
                  <div class="tracing-summary-layout">
                    ${renderSummaryJsonCard(props.selected)}
                    ${renderCompositionCard(props.selected)}
                  </div>
                </section>

                <section class="tracing-detail-section">
                  <div class="card-title tracing-detail-section__title">Timeline</div>
                  ${renderTimelineFlow(props.selected)}
                </section>

                <section class="tracing-detail-section">
                  ${renderJsonInspectorCard("Request Payloads", props.selected.requests)}
                </section>

                <section class="tracing-detail-section">
                  ${renderJsonInspectorCard("Response Snapshot", props.selected.response ?? null)}
                </section>
              </aside>
            </div>
          `
        : nothing}
    </section>
  `;
}
