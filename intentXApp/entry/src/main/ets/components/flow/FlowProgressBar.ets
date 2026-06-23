import { FLOW_PHASE_ORDER } from '../../model/FlowModels';
import { FlowColors } from '../../common/FlowColors';

/**
 * FlowProgressBar —— iOS 分段式进度条（01 文档 §3.1）。
 * 仅在工作流激活时插入；高度 ≈ 34vp，绝不超过两行。
 */
@Component
export struct FlowProgressBar {
  @Prop activePhase: string = 'clarify';
  @State pulseOpacity: number = 1;

  aboutToAppear(): void {
    // 当前段呼吸动画（01 文档 §3.1：当前段加 1.8s 呼吸 opacity 动画）。
    this.getUIContext().animateTo({
      duration: 1800,
      curve: Curve.EaseInOut,
      iterations: -1,
      playMode: PlayMode.Alternate
    }, () => {
      this.pulseOpacity = 0.45;
    });
  }

  private phaseLabel(phase: string): string {
    switch (phase) {
      case 'clarify': return '意图澄清';
      case 'customize': return '个性化任务';
      case 'create': return '技能创建';
      case 'execute': return '执行技能';
      default: return '';
    }
  }

  private activeIndex(): number {
    const idx = FLOW_PHASE_ORDER.indexOf(
      this.activePhase as ('clarify' | 'customize' | 'create' | 'execute'));
    return idx < 0 ? 0 : idx;
  }

  build() {
    Column({ space: 6 }) {
      Row() {
        Row({ space: 5 }) {
          Text('Agent 工作流')
            .fontSize(10)
            .fontWeight(FontWeight.Medium)
            .fontColor(FlowColors.MUTED_FG);
          Text('·')
            .fontSize(10)
            .fontColor(FlowColors.MUTED_FG);
          Text(this.phaseLabel(this.activePhase))
            .fontSize(10)
            .fontWeight(FontWeight.Bold)
            .fontColor(FlowColors.AGENT_BLUE)
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis });
        }
        .layoutWeight(1);

        Text(`${this.activeIndex() + 1}/${FLOW_PHASE_ORDER.length}`)
          .fontSize(10)
          .fontWeight(FontWeight.Medium)
          .fontColor(FlowColors.MUTED_FG);
      }
      .width('100%');

      Row({ space: 4 }) {
        ForEach(FLOW_PHASE_ORDER, (phase: string, i: number) => {
          Stack()
            .height(3)
            .layoutWeight(1)
            .borderRadius(2)
            .backgroundColor(i <= this.activeIndex() ? FlowColors.AGENT_BLUE : '#D6E0F5')
            .opacity(i === this.activeIndex() ? this.pulseOpacity : 1);
        }, (phase: string) => phase);
      }
      .width('100%');
    }
    .width('100%')
    .padding({ left: 14, right: 14, top: 8, bottom: 8 })
    .backgroundColor(FlowColors.CARD)
    .borderRadius(14)
    .border({ width: 1, color: FlowColors.BORDER_SOFT });
  }
}
