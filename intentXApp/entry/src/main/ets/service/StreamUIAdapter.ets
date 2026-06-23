/**
 * StreamUIAdapter —— openclaw 文字流 / 导出 Markdown → UIBlock 序列。
 * ArkTS 逐规则对齐 chat-ux `lib/chat-parser.ts`。
 * 见 Docs/02-stream-ui-interface.md §4 与 Docs/03-chat-log-parsing-and-replay.md。
 *
 * 纯逻辑、无 UI / 网络依赖（02 文档 §5 要求），可单测。
 */
import { FLOW_PHASE_ORDER, RawTurn, ReplayResult, UIBlock } from '../model/FlowModels';
import { FlowParsers } from './FlowParsers';

export class StreamUIAdapter {
  // ---------- 第一步：切分 Turn（03 文档 §1） ----------

  static splitTurns(markdown: string): RawTurn[] {
    const turns: RawTurn[] = [];
    const re = /^## (You|Ember|Tool) \((.+?)\)\s*$/gm;
    const headerStarts: number[] = []; // 标题行起点
    const contentStarts: number[] = []; // 标题行之后的内容起点
    const roles: string[] = [];
    let m: RegExpExecArray | null = re.exec(markdown);
    while (m !== null) {
      const role = m[1] === 'You' ? 'user' : (m[1] === 'Ember' ? 'agent' : 'tool');
      roles.push(role);
      headerStarts.push(m.index);
      contentStarts.push(m.index + m[0].length);
      m = re.exec(markdown);
    }
    for (let i = 0; i < roles.length; i++) {
      const start = contentStarts[i];
      const end = i + 1 < roles.length ? headerStarts[i + 1] : markdown.length;
      const content = markdown.substring(start, end).trim();
      turns.push(new RawTurn(roles[i], content));
    }
    return turns;
  }

  // ---------- 阶段检测（02 文档 §2） ----------

  static detectPhaseFromTool(content: string): string {
    const m = content.match(/^---\s*\n\s*name:\s*(\S+)/);
    if (!m) {
      return '';
    }
    const name = m[1];
    if (name.indexOf('writing-plans') >= 0) {
      return 'customize';
    }
    if (name.indexOf('writing-skills') >= 0) {
      return 'create';
    }
    if (name.endsWith(':execute')) {
      return 'execute';
    }
    // genSkill / skill-creator / brainstorming → clarify
    return 'clarify';
  }

  static phaseGte(a: string, b: string): boolean {
    return FLOW_PHASE_ORDER.indexOf(a as ('clarify' | 'customize' | 'create' | 'execute')) >=
      FLOW_PHASE_ORDER.indexOf(b as ('clarify' | 'customize' | 'create' | 'execute'));
  }

  // ---------- 主管线（与流式共用，03 文档 §2） ----------

  static parseChatLog(markdown: string): ReplayResult {
    return StreamUIAdapter.parseTurns(StreamUIAdapter.splitTurns(markdown));
  }

  static parseTurns(turns: RawTurn[]): ReplayResult {
    const result = new ReplayResult();
    const blocks: UIBlock[] = result.blocks;

    let phase: string = '';
    let clarifyCount = 0;
    let maxQuestions = 1;
    let currentSkillCard: UIBlock | null = null;

    for (let ti = 0; ti < turns.length; ti++) {
      const turn = turns[ti];
      if (turn.content.length === 0) {
        continue;
      }

      if (turn.role === 'tool') {
        const newPhase = StreamUIAdapter.detectPhaseFromTool(turn.content);
        if (newPhase.length > 0) {
          if (phase.length === 0 || StreamUIAdapter.phaseGte(newPhase, phase)) {
            if (newPhase === 'clarify' && phase.length === 0) {
              clarifyCount = 0;
            }
            phase = newPhase;
          }
          continue;
        }
        const step = FlowParsers.parseToolStep(turn.content);
        if (step) {
          // 工具错误卡片仅在 create/execute 阶段呈现（03 文档 #9）
          if (phase === 'create' || phase === 'execute') {
            if (!currentSkillCard) {
              currentSkillCard = new UIBlock();
              currentSkillCard.kind = 'skill-create';
              currentSkillCard.phase = phase;
              blocks.push(currentSkillCard);
            }
            step.id = 's' + currentSkillCard.steps.length;
            currentSkillCard.steps.push(step);
          }
        }
        continue;
      }

      if (turn.role === 'user') {
        const text = turn.content;
        if (phase === 'clarify') {
          const numAnswer = text.match(/^(\d)\.\s*(.+)$/);
          if (numAnswer) {
            const lastQ = StreamUIAdapter.lastBlockOfKind(blocks, 'clarify-question');
            if (lastQ && lastQ.selectedId.length === 0) {
              lastQ.selectedId = numAnswer[1];
            }
            blocks.push(StreamUIAdapter.userOption(numAnswer[1], numAnswer[2], phase));
            continue;
          }
          const approachAnswer = text.match(/^([A-C])$/);
          if (approachAnswer) {
            const lastA = StreamUIAdapter.lastBlockOfKind(blocks, 'approach-proposal');
            if (lastA) {
              lastA.selectedId = approachAnswer[1];
            }
            blocks.push(StreamUIAdapter.userOption(approachAnswer[1], '选择方案 ' + approachAnswer[1], phase));
            continue;
          }
        }
        if (phase === 'customize' && FlowParsers.isApprovalWord(text)) {
          const lastPlan = StreamUIAdapter.lastBlockOfKind(blocks, 'plan-confirm');
          if (lastPlan) {
            lastPlan.approved = true;
          }
          blocks.push(StreamUIAdapter.userOption('✓', text.trim(), phase));
          continue;
        }
        // /genskills 触发指令或普通文本
        const isGenskills = /^\/genskills?\b/i.test(text);
        if (isGenskills) {
          // 新一轮工作流触发（02 文档 §2 workflow_start）。
          // 历史回放时网关不回传 tool frontmatter，故以 /genskills 命令作为阶段起点。
          clarifyCount = 0;
          currentSkillCard = null;
          phase = 'clarify';
        }
        const ut = new UIBlock();
        ut.kind = 'user-text';
        ut.text = text;
        ut.phase = phase;
        blocks.push(ut);
        continue;
      }

      // ---- agent ----
      const text = turn.content;

      if (phase === 'clarify') {
        const clarify = FlowParsers.parseClarifyQuestion(text);
        if (clarify) {
          clarifyCount++;
          clarify.questionIndex = clarifyCount;
          clarify.phase = phase;
          if (clarifyCount > maxQuestions) {
            maxQuestions = clarifyCount;
          }
          blocks.push(clarify);
          continue;
        }
        const approach = FlowParsers.parseApproachProposal(text);
        if (approach) {
          approach.phase = phase;
          blocks.push(approach);
          continue;
        }
      }

      if (phase === 'create' || phase === 'execute') {
        const summary = FlowParsers.parseExecSummary(text);
        if (summary) {
          phase = 'execute';
          summary.phase = phase;
          blocks.push(summary);
          currentSkillCard = null;
          continue;
        }
        if (currentSkillCard && text.length < 60) {
          currentSkillCard.annotations.push(text);
          continue;
        }
      }

      // 降级：普通文字气泡（02 文档 §4.7）
      const at = new UIBlock();
      at.kind = 'agent-text';
      at.markdown = text;
      at.phase = phase;
      blocks.push(at);
      // clarify 阶段若不再含加粗问句 → 工作流静默结束（超时规则）
      if (phase === 'clarify' && !/\*\*.+?[？?]\*\*/.test(text)) {
        phase = '';
        clarifyCount = 0;
      }
    }

    result.finalPhase = phase;
    result.totalQuestions = maxQuestions;
    return result;
  }

  private static userOption(index: string, text: string, phase: string): UIBlock {
    const b = new UIBlock();
    b.kind = 'user-option-answer';
    b.index = index;
    b.text = text;
    b.phase = phase;
    return b;
  }

  private static lastBlockOfKind(blocks: UIBlock[], kind: string): UIBlock | null {
    for (let i = blocks.length - 1; i >= 0; i--) {
      if (blocks[i].kind === kind) {
        return blocks[i];
      }
    }
    return null;
  }
}
