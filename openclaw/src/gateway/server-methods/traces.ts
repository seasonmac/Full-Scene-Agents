import { getLlmTrace, tailLlmTraces } from "../../infra/llm-traces.js";
import {
  ErrorCodes,
  errorShape,
  formatValidationErrors,
  validateTracesGetParams,
  validateTracesTailParams,
} from "../protocol/index.js";
import type { GatewayRequestHandlers } from "./types.js";

export const tracesHandlers: GatewayRequestHandlers = {
  "traces.tail": async ({ params, respond }) => {
    if (!validateTracesTailParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid traces.tail params: ${formatValidationErrors(validateTracesTailParams.errors)}`,
        ),
      );
      return;
    }

    const payload = params as {
      cursor?: number;
      limit?: number;
      maxBytes?: number;
      query?: string;
      provider?: string;
      status?: "ok" | "error" | "in_progress";
    };
    try {
      const result = await tailLlmTraces(payload);
      respond(true, result, undefined);
    } catch (error) {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.UNAVAILABLE, `trace read failed: ${String(error)}`),
      );
    }
  },

  "traces.get": async ({ params, respond }) => {
    if (!validateTracesGetParams(params)) {
      respond(
        false,
        undefined,
        errorShape(
          ErrorCodes.INVALID_REQUEST,
          `invalid traces.get params: ${formatValidationErrors(validateTracesGetParams.errors)}`,
        ),
      );
      return;
    }

    const payload = params as { traceId: string };
    try {
      const trace = await getLlmTrace({ traceId: payload.traceId });
      if (!trace) {
        respond(
          false,
          undefined,
          errorShape(ErrorCodes.INVALID_REQUEST, `trace not found: ${payload.traceId}`),
        );
        return;
      }
      respond(true, trace, undefined);
    } catch (error) {
      respond(
        false,
        undefined,
        errorShape(ErrorCodes.UNAVAILABLE, `trace lookup failed: ${String(error)}`),
      );
    }
  },
};
