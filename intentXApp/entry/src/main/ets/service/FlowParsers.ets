/**
 * StreamUIAdapter 文本判定规则（02 文档 §4.1–§4.4, §4.6）。
 * 纯函数集合，供 StreamUIAdapter 主管线调用。
 */
import { ApproachItem, ClarifyOption, SkillStep, UIBlock } from '../model/FlowModels';

export class FlowParsers {
  // §4.1 clarify_question（单选题）
  static parseClarifyQuestion(text: string): UIBlock | null {
    // 加粗问句：允许行内（前面可有承接语），不强制独占整行。
    const qMatch = text.match(/\*\*\s*([^*\n]+?[？?])\s*\*\*/);
    if (!qMatch || qMatch.index === undefined) {
      return null;
    }
    const after = text.substring(qMatch.index + qMatch[0].length);
    const options: ClarifyOption[] = [];
    // 编号选项：允许行首缩进；去除选项 label 内的加粗标记。
    const optRe = /^\s*(\d)[.、)]\s+(.+)$/gm;
    let o: RegExpExecArray | null = optRe.exec(after);
    while (o !== null) {
      const opt = new ClarifyOption();
      opt.id = o[1];
      const raw = o[2].trim().replace(/\*\*/g, '');
      // 设计稿：短标题 + 下方淡色解释。把括号内的说明拆为 description。
      const split = raw.match(/^(.*?)[（(](.+?)[）)]\s*$/);
      if (split && split[1].trim().length > 0) {
        opt.label = split[1].trim();
        opt.description = split[2].trim();
      } else {
        // 破折号 / 冒号分隔的说明也下沉为 description
        const dash = raw.match(/^(.+?)\s*[—–:：]\s*(.+)$/);
        if (dash && dash[1].trim().length >= 2 && dash[1].trim().length <= 12) {
          opt.label = dash[1].trim();
          opt.description = dash[2].trim();
        } else {
          opt.label = raw;
        }
      }
      options.push(opt);
      o = optRe.exec(after);
    }
    if (options.length < 2) {
      return null;
    }
    const block = new UIBlock();
    block.kind = 'clarify-question';
    block.preamble = text.substring(0, qMatch.index).trim();
    block.question = qMatch[1].trim();
    block.options = options;
    return block;
  }

  // §4.2 approach_proposal（方案选择）
  static parseApproachProposal(text: string): UIBlock | null {
    const titleRe = /^\*\*方案 ([A-C])\s*[—\-–]\s*(.+?)\*\*\s*$/gm;
    const ids: string[] = [];
    const names: string[] = [];
    const titleEnds: number[] = [];
    const titleStarts: number[] = [];
    let t: RegExpExecArray | null = titleRe.exec(text);
    while (t !== null) {
      ids.push(t[1]);
      names.push(t[2].trim());
      titleStarts.push(t.index);
      titleEnds.push(t.index + t[0].length);
      t = titleRe.exec(text);
    }
    if (ids.length < 2) {
      return null;
    }
    const recMatch = text.match(/我推荐方案\s*([A-C])/);
    const recIndex = recMatch && recMatch.index !== undefined ? recMatch.index : text.length;
    const approaches: ApproachItem[] = [];
    for (let i = 0; i < ids.length; i++) {
      const bodyStart = titleEnds[i];
      const bodyEnd = i + 1 < ids.length ? titleStarts[i + 1] : recIndex;
      let body = text.substring(bodyStart, bodyEnd).trim();
      body = body.replace(/\*+\s*$/, '').trim();
      const a = new ApproachItem();
      a.id = ids[i];
      a.title = names[i];
      a.body = body;
      approaches.push(a);
    }
    const block = new UIBlock();
    block.kind = 'approach-proposal';
    block.preamble = text.substring(0, titleStarts[0]).trim();
    block.approaches = approaches;
    block.recommendedId = recMatch ? recMatch[1] : '';
    return block;
  }

  // §4.6 exec_summary（最终汇总）
  static parseExecSummary(text: string): UIBlock | null {
    const pathRe = /^-\s+`?(~?\/[^\s`]+SKILL\.md)`?\s*$/gm;
    const paths: string[] = [];
    let firstPathIndex = -1;
    let p: RegExpExecArray | null = pathRe.exec(text);
    while (p !== null) {
      if (firstPathIndex < 0) {
        firstPathIndex = p.index;
      }
      paths.push(p[1]);
      p = pathRe.exec(text);
    }
    if (paths.length === 0) {
      return null;
    }
    const usageRe = /^(\d+)\.\s+(.+)$/gm;
    const usage: string[] = [];
    let lastUsageEnd = -1;
    let u: RegExpExecArray | null = usageRe.exec(text);
    while (u !== null) {
      usage.push(u[2].trim());
      lastUsageEnd = u.index + u[0].length;
      u = usageRe.exec(text);
    }
    if (usage.length < 1) {
      return null;
    }
    const block = new UIBlock();
    block.kind = 'exec-summary';
    block.intro = text.substring(0, firstPathIndex < 0 ? 0 : firstPathIndex).trim();
    block.skillPaths = paths;
    block.usageSteps = usage;
    block.cta = lastUsageEnd >= 0 ? text.substring(lastUsageEnd).trim() : '';
    return block;
  }

  // §4.4 tool 事件 → 步骤
  static parseToolStep(content: string): SkillStep | null {
    const wrote = content.match(/Successfully wrote (\d+) bytes to (.+)/);
    if (wrote) {
      const full = wrote[2].trim();
      const parts = full.split('/');
      const short = parts.slice(-2).join('/');
      const step = new SkillStep();
      step.status = 'done';
      step.label = '写入 ' + short;
      step.detail = wrote[1] + ' bytes → ' + full;
      return step;
    }
    if (/"status":\s*"error"/.test(content)) {
      const step = new SkillStep();
      step.status = 'error';
      step.label = '读取失败（已降级）';
      step.detail = content;
      return step;
    }
    if (/Error: Cannot find module/.test(content) || /exited with code [1-9]/.test(content)) {
      const step = new SkillStep();
      step.status = 'error';
      step.label = '脚本执行失败（已降级）';
      step.detail = content;
      return step;
    }
    return null;
  }

  static isApprovalWord(text: string): boolean {
    const t = text.trim();
    const words = ['可以保存', '确认', '保存', '好的', '可以', 'yes', 'ok'];
    return words.indexOf(t) >= 0;
  }
}
