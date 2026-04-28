import { Type } from "@sinclair/typebox";
import { ChatSendSessionKeyString, InputProvenanceSchema, NonEmptyString } from "./primitives.js";

export const LogsTailParamsSchema = Type.Object(
  {
    cursor: Type.Optional(Type.Integer({ minimum: 0 })),
    limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 5000 })),
    maxBytes: Type.Optional(Type.Integer({ minimum: 1, maximum: 1_000_000 })),
  },
  { additionalProperties: false },
);

export const LogsTailResultSchema = Type.Object(
  {
    file: NonEmptyString,
    cursor: Type.Integer({ minimum: 0 }),
    size: Type.Integer({ minimum: 0 }),
    lines: Type.Array(Type.String()),
    truncated: Type.Optional(Type.Boolean()),
    reset: Type.Optional(Type.Boolean()),
  },
  { additionalProperties: false },
);
export const LlmTraceUsageSchema = Type.Object(
  {
    input: Type.Optional(Type.Number()),
    output: Type.Optional(Type.Number()),
    cacheRead: Type.Optional(Type.Number()),
    cacheWrite: Type.Optional(Type.Number()),
    promptTokens: Type.Optional(Type.Number()),
    total: Type.Optional(Type.Number()),
  },
  { additionalProperties: false },
);

export const LlmTraceTimelineEntrySchema = Type.Object(
  {
    ts: Type.Integer({ minimum: 0 }),
    kind: Type.Union([Type.Literal("request"), Type.Literal("response"), Type.Literal("error")]),
    label: NonEmptyString,
    detail: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export const LlmTraceSummarySchema = Type.Object(
  {
    traceId: NonEmptyString,
    runId: Type.Optional(NonEmptyString),
    sessionId: Type.Optional(NonEmptyString),
    sessionKey: Type.Optional(NonEmptyString),
    provider: Type.Optional(NonEmptyString),
    modelId: Type.Optional(NonEmptyString),
    modelApi: Type.Optional(Type.String()),
    workspaceDir: Type.Optional(Type.String()),
    startedAt: Type.Integer({ minimum: 0 }),
    endedAt: Type.Optional(Type.Integer({ minimum: 0 })),
    updatedAt: Type.Integer({ minimum: 0 }),
    status: Type.Union([Type.Literal("ok"), Type.Literal("error"), Type.Literal("in_progress")]),
    durationMs: Type.Optional(Type.Integer({ minimum: 0 })),
    usage: Type.Optional(LlmTraceUsageSchema),
    costTotal: Type.Optional(Type.Number()),
    error: Type.Optional(Type.String()),
    requestCount: Type.Integer({ minimum: 0 }),
    requestPreview: Type.Optional(Type.String()),
    responsePreview: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);

export const TracesTailParamsSchema = Type.Object(
  {
    cursor: Type.Optional(Type.Integer({ minimum: 0 })),
    limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 1000 })),
    maxBytes: Type.Optional(Type.Integer({ minimum: 1, maximum: 2_000_000 })),
    query: Type.Optional(Type.String()),
    provider: Type.Optional(Type.String()),
    status: Type.Optional(
      Type.Union([Type.Literal("ok"), Type.Literal("error"), Type.Literal("in_progress")]),
    ),
  },
  { additionalProperties: false },
);

export const TracesTailResultSchema = Type.Object(
  {
    file: NonEmptyString,
    cursor: Type.Integer({ minimum: 0 }),
    size: Type.Integer({ minimum: 0 }),
    records: Type.Array(LlmTraceSummarySchema),
    truncated: Type.Optional(Type.Boolean()),
    reset: Type.Optional(Type.Boolean()),
  },
  { additionalProperties: false },
);

export const TracesGetParamsSchema = Type.Object(
  {
    traceId: NonEmptyString,
  },
  { additionalProperties: false },
);

export const LlmTraceRecordSchema = Type.Composite([
  LlmTraceSummarySchema,
  Type.Object(
    {
      requests: Type.Array(Type.Unknown()),
      response: Type.Optional(Type.Unknown()),
      timeline: Type.Array(LlmTraceTimelineEntrySchema),
    },
    { additionalProperties: false },
  ),
]);

// WebChat/WebSocket-native chat methods
export const ChatHistoryParamsSchema = Type.Object(
  {
    sessionKey: NonEmptyString,
    limit: Type.Optional(Type.Integer({ minimum: 1, maximum: 1000 })),
    maxChars: Type.Optional(Type.Integer({ minimum: 1, maximum: 500_000 })),
  },
  { additionalProperties: false },
);

export const ChatSendParamsSchema = Type.Object(
  {
    sessionKey: ChatSendSessionKeyString,
    message: Type.String(),
    thinking: Type.Optional(Type.String()),
    deliver: Type.Optional(Type.Boolean()),
    originatingChannel: Type.Optional(Type.String()),
    originatingTo: Type.Optional(Type.String()),
    originatingAccountId: Type.Optional(Type.String()),
    originatingThreadId: Type.Optional(Type.String()),
    attachments: Type.Optional(Type.Array(Type.Unknown())),
    timeoutMs: Type.Optional(Type.Integer({ minimum: 0 })),
    systemInputProvenance: Type.Optional(InputProvenanceSchema),
    systemProvenanceReceipt: Type.Optional(Type.String()),
    idempotencyKey: NonEmptyString,
  },
  { additionalProperties: false },
);

export const ChatAbortParamsSchema = Type.Object(
  {
    sessionKey: NonEmptyString,
    runId: Type.Optional(NonEmptyString),
  },
  { additionalProperties: false },
);

export const ChatInjectParamsSchema = Type.Object(
  {
    sessionKey: NonEmptyString,
    message: NonEmptyString,
    label: Type.Optional(Type.String({ maxLength: 100 })),
  },
  { additionalProperties: false },
);

export const ChatEventSchema = Type.Object(
  {
    runId: NonEmptyString,
    sessionKey: NonEmptyString,
    seq: Type.Integer({ minimum: 0 }),
    state: Type.Union([
      Type.Literal("delta"),
      Type.Literal("final"),
      Type.Literal("aborted"),
      Type.Literal("error"),
    ]),
    message: Type.Optional(Type.Unknown()),
    errorMessage: Type.Optional(Type.String()),
    errorKind: Type.Optional(
      Type.Union([
        Type.Literal("refusal"),
        Type.Literal("timeout"),
        Type.Literal("rate_limit"),
        Type.Literal("context_length"),
        Type.Literal("unknown"),
      ]),
    ),
    usage: Type.Optional(Type.Unknown()),
    stopReason: Type.Optional(Type.String()),
  },
  { additionalProperties: false },
);
