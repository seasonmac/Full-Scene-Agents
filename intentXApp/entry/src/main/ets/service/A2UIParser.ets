/**
 * A2UIParser — extract inline A2UI JSON from an assistant message, validate it
 * against the genskill.cards@1 catalog, and map valid messages to UIBlock.
 *
 * Directive 1: the agent emits A2UI inline in the chat stream — either a fenced
 *   ```a2ui … ``` block (preferred) or bare JSONL lines. No canvas.a2ui.push.
 * Directive 3: invalid A2UI does NOT render; instead the client returns precise
 *   errors (path + message) so the caller can ask the LLM to regenerate.
 *
 * Pure logic, no UI/network — unit-testable.
 */
import { A2UI_CATALOG_ID, A2UIMessage, A2UIScanResult, A2UIValidationError } from '../model/A2UIModels';
import { A2UIComponentValidator } from './A2UIComponentValidator';

export class A2UIParser {
  /** Matches the catalog marker `"catalogId":"genskill.cards@1"` (whitespace-tolerant). */
  private static readonly CATALOG_MARKER = new RegExp('"catalogId"\\s*:\\s*"' + A2UI_CATALOG_ID.replace('.', '\\.') + '"');

  /** Scan one assistant message for inline A2UI; validate; return messages + errors + residual text. */
  static scan(text: string): A2UIScanResult {
    const result = new A2UIScanResult();
    if (!text || text.length === 0) {
      result.residualText = text || '';
      return result;
    }

    // Gate: a message only carries A2UI worth validating if it has a fenced ```a2ui block
    // OR explicitly tags the genskill catalog. Otherwise treat as plain text — never
    // mis-fire validation on casual replies (e.g. "hello") or prose that happens to start
    // with an envelope key. The catalogId marker is the single trusted signal.
    const hasFence = text.indexOf('```a2ui') >= 0;
    const hasCatalog = A2UIParser.CATALOG_MARKER.test(text);
    if (!hasFence && !hasCatalog) {
      result.residualText = text;
      return result;
    }

    const extracted = A2UIParser.extractBlocks(text);
    if (extracted.jsonLines.length === 0) {
      result.residualText = text;
      return result;
    }
    result.found = true;
    result.residualText = extracted.residual.trim();

    for (let i = 0; i < extracted.jsonLines.length; i++) {
      const line = extracted.jsonLines[i].trim();
      if (line.length === 0) {
        continue;
      }
      let obj: Record<string, Object> | null = null;
      try {
        obj = JSON.parse(line) as Record<string, Object>;
      } catch (e) {
        result.errors.push(new A2UIValidationError('/line/' + i,
          'Line ' + i + ' is not valid JSON. Each A2UI line must be a single JSON object.'));
        continue;
      }
      A2UIParser.validateMessage(obj, i, result);
    }
    return result;
  }

  /** A2UI is present iff a fenced ```a2ui block exists, or a bare {"createSurface"|"updateComponents"...} line. */
  private static extractBlocks(text: string): ExtractResult {
    const out = new ExtractResult();
    // 1) fenced ```a2ui ... ``` (preferred, unambiguous)
    const fence = text.match(/```a2ui\s*([\s\S]*?)```/);
    if (fence && fence.index !== undefined) {
      const body = fence[1];
      const lines = body.split('\n');
      for (let i = 0; i < lines.length; i++) {
        const t = lines[i].trim();
        if (t.length > 0) {
          out.jsonLines.push(t);
        }
      }
      out.residual = text.substring(0, fence.index) + text.substring(fence.index + fence[0].length);
      return out;
    }
    // 2) bare JSONL: lines starting with an A2UI envelope key
    const lines = text.split('\n');
    const residualLines: string[] = [];
    for (let i = 0; i < lines.length; i++) {
      const t = lines[i].trim();
      if (/^\{\s*"(createSurface|updateComponents|updateDataModel|deleteSurface)"/.test(t)) {
        out.jsonLines.push(t);
      } else {
        residualLines.push(lines[i]);
      }
    }
    out.residual = residualLines.join('\n');
    return out;
  }

  /** Validate one envelope, append a mapped A2UIMessage or errors. */
  private static validateMessage(obj: Record<string, Object>, idx: number, result: A2UIScanResult): void {
    const keys = Object.keys(obj);
    if (keys.length !== 1) {
      result.errors.push(new A2UIValidationError('/line/' + idx,
        'An A2UI message must have exactly one top-level key (createSurface/updateComponents/deleteSurface).'));
      return;
    }
    const kind = keys[0];

    if (kind === 'createSurface') {
      const body = obj['createSurface'] as Record<string, Object>;
      const surfaceId = A2UIParser.readStr(body, 'surfaceId');
      const catalogId = A2UIParser.readStr(body, 'catalogId');
      if (surfaceId.length === 0) {
        result.errors.push(new A2UIValidationError('/createSurface/surfaceId', 'createSurface requires a non-empty surfaceId.'));
        return;
      }
      if (catalogId !== A2UI_CATALOG_ID) {
        result.errors.push(new A2UIValidationError('/createSurface/catalogId',
          'catalogId must be "' + A2UI_CATALOG_ID + '" (got "' + catalogId + '").'));
        return;
      }
      const msg = new A2UIMessage();
      msg.kind = 'createSurface';
      msg.surfaceId = surfaceId;
      msg.catalogId = catalogId;
      result.messages.push(msg);
      return;
    }

    if (kind === 'updateComponents') {
      const body = obj['updateComponents'] as Record<string, Object>;
      const surfaceId = A2UIParser.readStr(body, 'surfaceId');
      if (surfaceId.length === 0) {
        result.errors.push(new A2UIValidationError('/updateComponents/surfaceId', 'updateComponents requires a non-empty surfaceId.'));
        return;
      }
      const compsVal = body['components'];
      if (!compsVal || !Array.isArray(compsVal)) {
        result.errors.push(new A2UIValidationError('/updateComponents/components', 'components must be a non-empty array.'));
        return;
      }
      const comps = compsVal as Array<Object>;
      if (comps.length === 0) {
        result.errors.push(new A2UIValidationError('/updateComponents/components', 'components must contain at least one component.'));
        return;
      }
      const msg = new A2UIMessage();
      msg.kind = 'updateComponents';
      msg.surfaceId = surfaceId;
      const errCountBefore = result.errors.length;
      for (let i = 0; i < comps.length; i++) {
        const comp = comps[i] as Record<string, Object>;
        A2UIComponentValidator.validate(comp, '/updateComponents/components/' + i, result.errors);
        msg.components.push(comp);
      }
      if (result.errors.length === errCountBefore) {
        result.messages.push(msg);
      }
      return;
    }

    if (kind === 'deleteSurface') {
      const body = obj['deleteSurface'] as Record<string, Object>;
      const surfaceId = A2UIParser.readStr(body, 'surfaceId');
      if (surfaceId.length === 0) {
        result.errors.push(new A2UIValidationError('/deleteSurface/surfaceId', 'deleteSurface requires a non-empty surfaceId.'));
        return;
      }
      const msg = new A2UIMessage();
      msg.kind = 'deleteSurface';
      msg.surfaceId = surfaceId;
      result.messages.push(msg);
      return;
    }

    if (kind === 'updateDataModel') {
      // Not used by genskill.cards@1 (cards are self-contained); ignore silently.
      return;
    }

    result.errors.push(new A2UIValidationError('/line/' + idx,
      'Unknown A2UI message type "' + kind + '". Use createSurface/updateComponents/deleteSurface.'));
  }

  static readStr(obj: Record<string, Object>, key: string): string {
    const v = obj[key];
    return typeof v === 'string' ? v as string : '';
  }
}

class ExtractResult {
  jsonLines: string[] = [];
  residual: string = '';
}
