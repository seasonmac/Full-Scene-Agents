/**
 * A2UIMapper — convert validated A2UI components (intentx.app:cards@1) into the
 * existing UIBlock model so the 6 prebuilt cards render unchanged.
 *
 * Only called on components that already passed A2UIComponentValidator.
 */
import { A2UIMessage } from '../model/A2UIModels';
import { ApproachItem, ClarifyOption, PlanItem, SkillStep, UIBlock } from '../model/FlowModels';

export class A2UIMapper {
  /** Map all updateComponents across messages into ordered UIBlocks. */
  static toBlocks(messages: A2UIMessage[]): UIBlock[] {
    const blocks: UIBlock[] = [];
    for (let m = 0; m < messages.length; m++) {
      if (messages[m].kind !== 'updateComponents') {
        continue;
      }
      const comps = messages[m].components;
      for (let i = 0; i < comps.length; i++) {
        const block = A2UIMapper.toBlock(comps[i]);
        if (block) {
          blocks.push(block);
        }
      }
    }
    return blocks;
  }

  static toBlock(comp: Record<string, Object>): UIBlock | null {
    const name = A2UIMapper.str(comp, 'component');
    switch (name) {
      case 'ClarifyQuestion':
        return A2UIMapper.clarify(comp);
      case 'ApproachProposal':
        return A2UIMapper.approach(comp);
      case 'PlanConfirm':
        return A2UIMapper.plan(comp);
      case 'SkillCreate':
        return A2UIMapper.skillCreate(comp);
      case 'ExecuteSummary':
        return A2UIMapper.execSummary(comp);
      case 'AgentText': {
        const b = new UIBlock();
        b.kind = 'agent-text';
        b.markdown = A2UIMapper.str(comp, 'markdown');
        return b;
      }
      default:
        return null;
    }
  }

  private static clarify(comp: Record<string, Object>): UIBlock {
    const b = new UIBlock();
    b.kind = 'clarify-question';
    b.phase = 'clarify';
    b.preamble = A2UIMapper.str(comp, 'preamble');
    b.question = A2UIMapper.str(comp, 'question');
    b.questionIndex = A2UIMapper.num(comp, 'questionIndex', 1);
    const opts = A2UIMapper.arr(comp, 'options');
    for (let i = 0; i < opts.length; i++) {
      const o = opts[i] as Record<string, Object>;
      const opt = new ClarifyOption();
      opt.id = A2UIMapper.str(o, 'id');
      opt.label = A2UIMapper.str(o, 'label');
      opt.description = A2UIMapper.str(o, 'description');
      b.options.push(opt);
    }
    return b;
  }

  private static approach(comp: Record<string, Object>): UIBlock {
    const b = new UIBlock();
    b.kind = 'approach-proposal';
    b.phase = 'clarify';
    b.preamble = A2UIMapper.str(comp, 'preamble');
    b.recommendedId = A2UIMapper.str(comp, 'recommendedId');
    const apps = A2UIMapper.arr(comp, 'approaches');
    for (let i = 0; i < apps.length; i++) {
      const a = apps[i] as Record<string, Object>;
      const item = new ApproachItem();
      item.id = A2UIMapper.str(a, 'id');
      item.title = A2UIMapper.str(a, 'title');
      item.body = A2UIMapper.str(a, 'body');
      b.approaches.push(item);
    }
    return b;
  }

  private static plan(comp: Record<string, Object>): UIBlock {
    const b = new UIBlock();
    b.kind = 'plan-confirm';
    b.phase = 'customize';
    b.taskTitle = A2UIMapper.str(comp, 'taskTitle');
    b.question = A2UIMapper.str(comp, 'question');
    const items = A2UIMapper.arr(comp, 'items');
    for (let i = 0; i < items.length; i++) {
      const it = items[i] as Record<string, Object>;
      const pi = new PlanItem();
      pi.label = A2UIMapper.str(it, 'label');
      pi.value = A2UIMapper.str(it, 'value');
      pi.icon = A2UIMapper.str(it, 'icon');
      b.items.push(pi);
    }
    return b;
  }

  private static skillCreate(comp: Record<string, Object>): UIBlock {
    const b = new UIBlock();
    b.kind = 'skill-create';
    b.phase = 'create';
    const steps = A2UIMapper.arr(comp, 'steps');
    for (let i = 0; i < steps.length; i++) {
      const s = steps[i] as Record<string, Object>;
      const step = new SkillStep();
      step.id = A2UIMapper.str(s, 'id');
      step.status = A2UIMapper.str(s, 'status');
      step.label = A2UIMapper.str(s, 'label');
      step.detail = A2UIMapper.str(s, 'detail');
      b.steps.push(step);
    }
    const anns = A2UIMapper.arr(comp, 'annotations');
    for (let i = 0; i < anns.length; i++) {
      const a = anns[i];
      if (typeof a === 'string') {
        b.annotations.push(a as string);
      }
    }
    return b;
  }

  private static execSummary(comp: Record<string, Object>): UIBlock {
    const b = new UIBlock();
    b.kind = 'exec-summary';
    b.phase = 'execute';
    b.intro = A2UIMapper.str(comp, 'intro');
    b.cta = A2UIMapper.str(comp, 'cta');
    const paths = A2UIMapper.arr(comp, 'skillPaths');
    for (let i = 0; i < paths.length; i++) {
      const p = paths[i];
      if (typeof p === 'string') {
        b.skillPaths.push(p as string);
      }
    }
    const usage = A2UIMapper.arr(comp, 'usageSteps');
    for (let i = 0; i < usage.length; i++) {
      const u = usage[i];
      if (typeof u === 'string') {
        b.usageSteps.push(u as string);
      }
    }
    return b;
  }

  static str(obj: Record<string, Object>, key: string): string {
    const v = obj[key];
    return typeof v === 'string' ? v as string : '';
  }

  static num(obj: Record<string, Object>, key: string, fallback: number): number {
    const v = obj[key];
    return typeof v === 'number' ? v as number : fallback;
  }

  static arr(obj: Record<string, Object>, key: string): Array<Object> {
    const v = obj[key];
    if (!v || !Array.isArray(v)) {
      return [];
    }
    return v as Array<Object>;
  }
}
