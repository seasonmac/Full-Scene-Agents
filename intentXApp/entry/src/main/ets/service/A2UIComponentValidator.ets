/**
 * A2UIComponentValidator — validate one component object from an updateComponents
 * message against the genskill.cards@1 catalog. Emits precise path+message
 * errors (directive 3) so the caller can ask the LLM to regenerate.
 *
 * Catalog: docs/a2ui/intentx-cards-catalog@1.json
 */
import { A2UI_COMPONENTS, A2UIValidationError } from '../model/A2UIModels';

export class A2UIComponentValidator {
  static validate(comp: Record<string, Object>, path: string, errors: A2UIValidationError[]): void {
    const id = A2UIComponentValidator.str(comp, 'id');
    if (id.length === 0) {
      errors.push(new A2UIValidationError(path + '/id', 'Every component requires a non-empty string id.'));
    }
    const name = A2UIComponentValidator.str(comp, 'component');
    if (name.length === 0) {
      errors.push(new A2UIValidationError(path + '/component', 'Component requires a "component" name.'));
      return;
    }
    if (A2UI_COMPONENTS.indexOf(name) < 0) {
      errors.push(new A2UIValidationError(path + '/component',
        'Unknown component "' + name + '". Allowed: ' + A2UI_COMPONENTS.join(', ') + '.'));
      return;
    }

    switch (name) {
      case 'ClarifyQuestion':
        A2UIComponentValidator.validateClarify(comp, path, errors);
        break;
      case 'ApproachProposal':
        A2UIComponentValidator.validateApproach(comp, path, errors);
        break;
      case 'PlanConfirm':
        A2UIComponentValidator.validatePlan(comp, path, errors);
        break;
      case 'SkillCreate':
        A2UIComponentValidator.validateSkillCreate(comp, path, errors);
        break;
      case 'ExecuteSummary':
        A2UIComponentValidator.validateExecSummary(comp, path, errors);
        break;
      case 'AgentText':
        if (A2UIComponentValidator.str(comp, 'markdown').length === 0) {
          errors.push(new A2UIValidationError(path + '/markdown', 'AgentText requires "markdown".'));
        }
        break;
      default:
        break;
    }
  }

  private static validateClarify(comp: Record<string, Object>, path: string, errors: A2UIValidationError[]): void {
    A2UIComponentValidator.requirePhase(comp, 'clarify', path, errors);
    if (A2UIComponentValidator.str(comp, 'question').length === 0) {
      errors.push(new A2UIValidationError(path + '/question', 'ClarifyQuestion requires "question".'));
    }
    const opts = A2UIComponentValidator.arr(comp, 'options');
    if (opts === null || opts.length < 2 || opts.length > 3) {
      errors.push(new A2UIValidationError(path + '/options',
        'ClarifyQuestion requires 2-3 options (Hard Rule: at most 3 mutually-exclusive choices).'));
    } else {
      for (let i = 0; i < opts.length; i++) {
        const o = opts[i] as Record<string, Object>;
        if (A2UIComponentValidator.str(o, 'id').length === 0) {
          errors.push(new A2UIValidationError(path + '/options/' + i + '/id', 'Each option requires "id".'));
        }
        if (A2UIComponentValidator.str(o, 'label').length === 0) {
          errors.push(new A2UIValidationError(path + '/options/' + i + '/label', 'Each option requires a short "label".'));
        }
      }
    }
    A2UIComponentValidator.requireAction(comp, 'action', path, errors);
  }

  private static validateApproach(comp: Record<string, Object>, path: string, errors: A2UIValidationError[]): void {
    // ApproachProposal is the close of the clarify phase; the emit-guide labels it
    // phase:"clarify", but models (incl. Claude) overwhelmingly write phase:"approach"
    // because the component is literally named ApproachProposal. The component name
    // already disambiguates the semantics, so accept either rather than rejecting the
    // whole card (which would drop it from the workbench and leak A/B/C as plain text).
    A2UIComponentValidator.requirePhaseOneOf(comp, ['clarify', 'approach'], path, errors);
    const apps = A2UIComponentValidator.arr(comp, 'approaches');
    if (apps === null || apps.length < 2 || apps.length > 3) {
      errors.push(new A2UIValidationError(path + '/approaches', 'ApproachProposal requires 2-3 approaches.'));
    } else {
      for (let i = 0; i < apps.length; i++) {
        const a = apps[i] as Record<string, Object>;
        if (A2UIComponentValidator.str(a, 'id').length === 0) {
          errors.push(new A2UIValidationError(path + '/approaches/' + i + '/id', 'Each approach requires "id" (A/B/C).'));
        }
        if (A2UIComponentValidator.str(a, 'title').length === 0) {
          errors.push(new A2UIValidationError(path + '/approaches/' + i + '/title', 'Each approach requires "title".'));
        }
      }
    }
    A2UIComponentValidator.requireAction(comp, 'action', path, errors);
  }

  private static validatePlan(comp: Record<string, Object>, path: string, errors: A2UIValidationError[]): void {
    A2UIComponentValidator.requirePhase(comp, 'customize', path, errors);
    if (A2UIComponentValidator.str(comp, 'taskTitle').length === 0) {
      errors.push(new A2UIValidationError(path + '/taskTitle', 'PlanConfirm requires "taskTitle".'));
    }
    const items = A2UIComponentValidator.arr(comp, 'items');
    if (items === null || items.length === 0) {
      errors.push(new A2UIValidationError(path + '/items', 'PlanConfirm requires at least one item.'));
    } else {
      for (let i = 0; i < items.length; i++) {
        const it = items[i] as Record<string, Object>;
        if (A2UIComponentValidator.str(it, 'label').length === 0) {
          errors.push(new A2UIValidationError(path + '/items/' + i + '/label', 'Each plan item requires "label".'));
        }
        if (A2UIComponentValidator.str(it, 'value').length === 0) {
          errors.push(new A2UIValidationError(path + '/items/' + i + '/value', 'Each plan item requires "value".'));
        }
      }
    }
    if (A2UIComponentValidator.str(comp, 'question').length === 0) {
      errors.push(new A2UIValidationError(path + '/question', 'PlanConfirm requires "question".'));
    }
    A2UIComponentValidator.requireAction(comp, 'confirmAction', path, errors);
  }

  private static validateSkillCreate(comp: Record<string, Object>, path: string, errors: A2UIValidationError[]): void {
    A2UIComponentValidator.requirePhase(comp, 'create', path, errors);
    const steps = A2UIComponentValidator.arr(comp, 'steps');
    if (steps === null || steps.length === 0) {
      errors.push(new A2UIValidationError(path + '/steps', 'SkillCreate requires at least one step.'));
      return;
    }
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i] as Record<string, Object>;
      if (A2UIComponentValidator.str(s, 'id').length === 0) {
        errors.push(new A2UIValidationError(path + '/steps/' + i + '/id', 'Each step requires "id".'));
      }
      const status = A2UIComponentValidator.str(s, 'status');
      if (status !== 'done' && status !== 'error') {
        errors.push(new A2UIValidationError(path + '/steps/' + i + '/status', 'Step status must be "done" or "error".'));
      }
      if (A2UIComponentValidator.str(s, 'label').length === 0) {
        errors.push(new A2UIValidationError(path + '/steps/' + i + '/label', 'Each step requires "label".'));
      }
    }
  }

  private static validateExecSummary(comp: Record<string, Object>, path: string, errors: A2UIValidationError[]): void {
    A2UIComponentValidator.requirePhase(comp, 'execute', path, errors);
    const paths = A2UIComponentValidator.arr(comp, 'skillPaths');
    if (paths === null || paths.length === 0) {
      errors.push(new A2UIValidationError(path + '/skillPaths', 'ExecuteSummary requires at least one skillPath.'));
    }
  }

  private static requirePhase(comp: Record<string, Object>, expected: string, path: string, errors: A2UIValidationError[]): void {
    const phase = A2UIComponentValidator.str(comp, 'phase');
    if (phase !== expected) {
      errors.push(new A2UIValidationError(path + '/phase',
        'phase must be "' + expected + '" for this component (got "' + phase + '").'));
    }
  }

  private static requirePhaseOneOf(comp: Record<string, Object>, accepted: string[], path: string, errors: A2UIValidationError[]): void {
    const phase = A2UIComponentValidator.str(comp, 'phase');
    if (accepted.indexOf(phase) < 0) {
      errors.push(new A2UIValidationError(path + '/phase',
        'phase must be one of "' + accepted.join('"/"') + '" for this component (got "' + phase + '").'));
    }
  }

  private static requireAction(comp: Record<string, Object>, key: string, path: string, errors: A2UIValidationError[]): void {
    const v = comp[key];
    if (!v || typeof v === 'string' || Array.isArray(v)) {
      errors.push(new A2UIValidationError(path + '/' + key, key + ' must be an object with a "name".'));
      return;
    }
    const action = v as Record<string, Object>;
    if (A2UIComponentValidator.str(action, 'name').length === 0) {
      errors.push(new A2UIValidationError(path + '/' + key + '/name', key + '.name is required.'));
    }
  }

  static str(obj: Record<string, Object>, key: string): string {
    const v = obj[key];
    return typeof v === 'string' ? v as string : '';
  }

  static arr(obj: Record<string, Object>, key: string): Array<Object> | null {
    const v = obj[key];
    if (!v || !Array.isArray(v)) {
      return null;
    }
    return v as Array<Object>;
  }
}
