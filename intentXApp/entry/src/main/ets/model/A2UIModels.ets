/**
 * A2UI message models — subset of A2UI 0.9 server→client used by intentx.app:cards@1.
 * Inline transport: the agent emits A2UI JSON line(s) inside the chat text stream
 * (fenced ```a2ui block or bare JSONL); the client parses, validates, and maps
 * them to the prebuilt UIBlock cards. No A2UI standard components are used.
 *
 * Spec: opentmp/vendor/a2ui/specification/0.9/json/server_to_client.json
 * Catalog: docs/a2ui/intentx-cards-catalog@1.json
 */

export const A2UI_CATALOG_ID: string = 'intentx.app:cards@1';
export const A2UI_CARD_CAPABILITY: string = 'a2ui:' + A2UI_CATALOG_ID;

/** The 6 custom component names = the only legal `component` values. */
export const A2UI_COMPONENTS: string[] = [
  'ClarifyQuestion',
  'ApproachProposal',
  'PlanConfirm',
  'SkillCreate',
  'ExecuteSummary',
  'AgentText',
];

/** One parsed A2UI message (createSurface / updateComponents / deleteSurface). */
export class A2UIMessage {
  kind: string = ''; // 'createSurface' | 'updateComponents' | 'deleteSurface'
  surfaceId: string = '';
  catalogId: string = '';
  /** Raw component objects for updateComponents (each a Record). */
  components: Array<Record<string, Object>> = [];
}

/** Result of validating one A2UI message against the custom catalog. */
export class A2UIValidationError {
  /** JSON-pointer-ish path to the offending field, e.g. '/updateComponents/components/0/options'. */
  path: string = '';
  message: string = '';

  constructor(path: string, message: string) {
    this.path = path;
    this.message = message;
  }
}

/** Outcome of scanning one assistant message for inline A2UI. */
export class A2UIScanResult {
  /** True if any A2UI block was detected (well-formed or not). */
  found: boolean = false;
  /** Parsed + validated messages, in order. */
  messages: A2UIMessage[] = [];
  /** Validation errors to feed back to the LLM (directive 3). */
  errors: A2UIValidationError[] = [];
  /** Assistant text with the A2UI block(s) stripped out (the human-readable remainder). */
  residualText: string = '';
}
