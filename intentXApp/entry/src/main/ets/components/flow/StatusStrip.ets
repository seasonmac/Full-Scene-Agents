import { FlowColors } from '../../common/FlowColors';

/**
 * StatusStrip —— 跨相位形变状态条（04 文档 §3.4 图层 A）。
 * thinking：呼吸点 + 轮播文案 + 计时器；
 * revealing/ready：呼吸点就地形变为对勾 + readyLabel；
 * collapsed：整条高度塌缩并淡出（由父级 PhaseStage 控制）。
 *
 * 计时器与轮播仅在 thinking 态运行，状态切换由 @Watch(status) 驱动。
 */
@Component
export struct StatusStrip {
  @Prop @Watch('onStatusChange') status: string = 'thinking';
  @Prop phase: string = 'clarify';
  @Prop collapsed: boolean = false;

  @State stageIdx: number = 0;
  @State elapsed: number = 0;
  private carouselTimer: number = -1;
  private elapsedTimer: number = -1;
  private startMs: number = 0;

  aboutToAppear(): void {
    this.onStatusChange();
  }

  aboutToDisappear(): void {
    this.stopTimers();
  }

  onStatusChange(): void {
    this.stopTimers();
    if (this.status !== 'thinking') {
      return;
    }
    this.stageIdx = 0;
    this.elapsed = 0;
    this.startMs = Date.now();
    const stages = this.stages();
    if (stages.length > 1) {
      this.carouselTimer = setInterval(() => {
        if (this.stageIdx + 1 < this.stages().length) {
          this.stageIdx = this.stageIdx + 1;
        }
      }, 1100);
    }
    this.elapsedTimer = setInterval(() => {
      this.elapsed = Math.floor((Date.now() - this.startMs) / 1000);
    }, 250);
  }

  private stopTimers(): void {
    if (this.carouselTimer >= 0) {
      clearInterval(this.carouselTimer);
      this.carouselTimer = -1;
    }
    if (this.elapsedTimer >= 0) {
      clearInterval(this.elapsedTimer);
      this.elapsedTimer = -1;
    }
  }

  private isDone(): boolean {
    return this.status === 'revealing' || this.status === 'ready';
  }

  private stages(): string[] {
    switch (this.phase) {
      case 'clarify': return ['正在理解你的需求…', '整理需要澄清的问题…', '准备澄清选项…'];
      case 'customize': return ['正在汇总你的选择…', '定制专属任务流程…', '校验任务参数…'];
      case 'create': return ['正在起草技能结构…', '撰写触发规则…', '撰写数据提取规则…', '校验依赖项…'];
      case 'execute': return ['正在扫描数据…', '分析提取指标…', '生成结果…', '推送结果…'];
      default: return ['正在准备…'];
    }
  }

  private currentStage(): string {
    const s = this.stages();
    if (s.length === 0) {
      return '';
    }
    const idx = this.stageIdx < s.length ? this.stageIdx : s.length - 1;
    return s[idx];
  }

  private readyLabel(): string {
    switch (this.phase) {
      case 'clarify': return '已准备好，请选择';
      case 'customize': return '任务已定制，请确认';
      case 'create': return '技能已生成';
      case 'execute': return '正在执行任务';
      default: return '已就绪';
    }
  }

  private accentColor(): string {
    return this.status === 'ready' ? FlowColors.STEP_DONE : FlowColors.AGENT_BLUE;
  }

  build() {
    Row({ space: 10 }) {
      // 指示器：呼吸点 ↔ 对勾，同位形变
      Stack({ alignContent: Alignment.Center }) {
        if (!this.isDone()) {
          Circle()
            .width(28)
            .height(28)
            .fill(FlowColors.AGENT_BLUE)
            .opacity(0.18);
          Circle()
            .width(10)
            .height(10)
            .fill(FlowColors.AGENT_BLUE);
        } else {
          Circle()
            .width(28)
            .height(28)
            .fill(this.accentColor());
          Text('✓')
            .fontSize(15)
            .fontWeight(FontWeight.Bold)
            .fontColor('#FFFFFF');
        }
      }
      .width(28)
      .height(28);

      Text(this.isDone() ? this.readyLabel() : this.currentStage())
        .fontSize(13)
        .fontWeight(this.isDone() ? FontWeight.Bold : FontWeight.Medium)
        .fontColor(this.isDone() ? this.accentColor() : FlowColors.FOREGROUND)
        .layoutWeight(1)
        .maxLines(1)
        .textOverflow({ overflow: TextOverflow.Ellipsis });

      if (!this.isDone()) {
        Text(`已用 ${this.elapsed}s`)
          .fontSize(12)
          .fontColor(FlowColors.MUTED_FG);
      }
    }
    .width('100%')
    .height(this.collapsed ? 0 : 56)
    .padding(this.collapsed
      ? { left: 14, right: 14, top: 0, bottom: 0 }
      : { left: 14, right: 14, top: 8, bottom: 8 })
    .opacity(this.collapsed ? 0 : 1)
    .backgroundColor(this.status === 'ready' ? FlowColors.SUMMARY_BG : FlowColors.AGENT_BLUE_MUTED)
    .borderRadius(16)
    .border({ width: 1, color: this.status === 'ready' ? '#BBF7D0' : FlowColors.AGENT_BLUE_LIGHT })
    .clip(true);
  }
}
