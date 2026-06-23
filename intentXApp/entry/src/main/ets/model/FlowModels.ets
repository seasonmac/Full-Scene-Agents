/**
 * Flow / structured-chat UI models.
 * ArkTS port of chat-ux `lib/chat-parser.ts` UIBlock types.
 * See Docs/01-ui-design-spec.md §3 and Docs/02-stream-ui-interface.md §3.
 */

/** 工作流四阶段。'' / null 表示普通聊天（无进度条）。 */
export type FlowPhase = 'clarify' | 'customize' | 'create' | 'execute';

export const FLOW_PHASE_ORDER: FlowPhase[] = ['clarify', 'customize', 'create', 'execute'];

/** 原始对话轮（与 02 文档 message.delta 整段 / tool.result 离线形态等价）。 */
export class RawTurn {
  role: string = ''; // 'user' | 'agent' | 'tool'
  content: string = '';

  constructor(role: string, content: string) {
    this.role = role;
    this.content = content;
  }
}

export class ClarifyOption {
  id: string = '';
  label: string = '';
  description: string = '';
}

export class ApproachItem {
  id: string = '';
  title: string = '';
  body: string = '';
}

export class PlanItem {
  label: string = '';
  value: string = '';
  icon: string = '';
}

export class SkillStep {
  id: string = '';
  status: string = 'done'; // 'done' | 'error'
  label: string = '';
  detail: string = '';
}

/**
 * 统一的 UIBlock —— ArkTS 没有 TS 判别联合，用单类 + kind 区分。
 * kind:
 *  user-text | user-option-answer | agent-text |
 *  clarify-question | approach-proposal | plan-confirm |
 *  skill-create | exec-summary
 */
@Observed
export class UIBlock {
  kind: string = 'agent-text';
  /** 该 block 出现时的工作流阶段；'' = 普通聊天（无进度条）。 */
  phase: string = '';

  // 通用文本
  text: string = '';
  markdown: string = '';
  index: string = ''; // user-option-answer 序号徽章

  // clarify-question
  preamble: string = '';
  question: string = '';
  options: ClarifyOption[] = [];
  selectedId: string = '';
  questionIndex: number = 0;
  totalQuestions: number = 0;

  // approach-proposal
  approaches: ApproachItem[] = [];
  recommendedId: string = '';

  // plan-confirm
  taskTitle: string = '';
  items: PlanItem[] = [];
  approved: boolean = false;

  // skill-create
  steps: SkillStep[] = [];
  annotations: string[] = [];

  // exec-summary
  intro: string = '';
  skillPaths: string[] = [];
  usageSteps: string[] = [];
  cta: string = '';
}

/** 解析产物。 */
export class ReplayResult {
  blocks: UIBlock[] = [];
  finalPhase: string = '';
  totalQuestions: number = 1;
}
