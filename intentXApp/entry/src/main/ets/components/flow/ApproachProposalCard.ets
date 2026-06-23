import { ApproachItem, UIBlock } from '../../model/FlowModels';
import { FlowColors } from '../../common/FlowColors';

/**
 * ApproachProposalCard —— 方案选择卡（意图澄清收尾，01 文档 §3.3）。
 * 确认后发送方案字母（如 "C"）作为用户消息。
 */
@Component
export struct ApproachProposalCard {
  @ObjectLink block: UIBlock;
  @Prop isLocked: boolean = false;
  @Prop canInteract: boolean = true;
  @Prop hideActions: boolean = false;
  onConfirmOption: (id: string, label: string) => void = (id: string, label: string) => {};
  onSelectOption: (id: string, label: string) => void = (id: string, label: string) => {};

  @State localSelected: string = '';

  aboutToAppear(): void {
    this.localSelected = this.block.selectedId;
  }

  private effectiveSelected(): string {
    return this.block.selectedId.length > 0 ? this.block.selectedId : this.localSelected;
  }

  build() {
    Column() {
      // 头部徽章
      Row() {
        Text('意图澄清 · 方案选择')
          .fontSize(12)
          .fontWeight(FontWeight.Medium)
          .fontColor(FlowColors.AGENT_BLUE_FG)
          .padding({ left: 10, right: 10, top: 3, bottom: 3 })
          .backgroundColor(FlowColors.AGENT_BLUE_LIGHT)
          .borderRadius(999);
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })
      .border({ width: 1, color: FlowColors.BORDER_SOFT, style: BorderStyle.Solid })
      .borderWidth({ bottom: 1 });

      if (this.block.preamble.length > 0) {
        Text(this.block.preamble)
          .fontSize(14)
          .fontColor(FlowColors.FOREGROUND)
          .lineHeight(21)
          .width('100%')
          .padding({ left: 16, right: 16, top: 10 });
      }

      Column({ space: 8 }) {
        ForEach(this.block.approaches, (a: ApproachItem) => {
          this.ApproachRow(a);
        }, (a: ApproachItem) => a.id);
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 12 });

      if (!this.isLocked && !this.hideActions) {
        Button({ type: ButtonType.Normal }) {
          Text(this.effectiveSelected().length > 0 ? `确认方案 ${this.effectiveSelected()}` : '请选择一个方案')
            .fontSize(15)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.effectiveSelected().length > 0 ? '#FFFFFF' : FlowColors.MUTED_FG);
        }
        .width('100%')
        .height(44)
        .borderRadius(12)
        .backgroundColor(this.effectiveSelected().length > 0 ? FlowColors.AGENT_BLUE : FlowColors.MUTED)
        .margin({ left: 16, right: 16 })
        .enabled(this.canInteract && this.effectiveSelected().length > 0)
        .onClick(() => {
          const sel = this.effectiveSelected();
          if (sel.length > 0) {
            this.onConfirmOption(sel, sel);
          }
        });
        Blank().height(14);
      } else if (this.effectiveSelected().length > 0) {
        Row({ space: 6 }) {
          Text('✓').fontSize(13).fontColor(FlowColors.STEP_DONE);
          Text(`已选择方案 ${this.effectiveSelected()}`).fontSize(12).fontColor(FlowColors.MUTED_FG);
        }
        .width('100%')
        .padding({ left: 16, right: 16, bottom: 12 });
      }
    }
    .width('100%')
    .backgroundColor(FlowColors.CARD)
    .borderRadius(16)
    .border({ width: 1, color: FlowColors.BORDER })
    .clip(true)
    .opacity(this.isLocked ? 0.78 : 1);
  }

  @Builder ApproachRow(a: ApproachItem) {
    Column({ space: 6 }) {
      Row({ space: 8 }) {
        Stack({ alignContent: Alignment.Center }) {
          Circle()
            .width(20)
            .height(20)
            .fill(this.effectiveSelected() === a.id ? FlowColors.AGENT_BLUE : FlowColors.MUTED);
          Text(a.id)
            .fontSize(11)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.effectiveSelected() === a.id ? '#FFFFFF' : FlowColors.MUTED_FG);
        }
        .width(20)
        .height(20);

        Text(a.title)
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor(FlowColors.FOREGROUND)
          .layoutWeight(1);

        if (this.block.recommendedId === a.id) {
          Text('推荐')
            .fontSize(10)
            .fontWeight(FontWeight.Bold)
            .fontColor(FlowColors.AGENT_BLUE_FG)
            .padding({ left: 8, right: 8, top: 2, bottom: 2 })
            .backgroundColor(FlowColors.AGENT_BLUE_LIGHT)
            .borderRadius(999);
        }
        if (this.effectiveSelected() === a.id) {
          Text('✓').fontSize(13).fontColor(FlowColors.AGENT_BLUE);
        }
      }
      .width('100%')
      .alignItems(VerticalAlign.Center);

      if (a.body.length > 0) {
        Text(a.body)
          .fontSize(12)
          .fontColor(FlowColors.MUTED_FG)
          .lineHeight(18)
          .width('100%');
      }
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start)
    .padding({ left: 14, right: 14, top: 12, bottom: 12 })
    .backgroundColor(this.effectiveSelected() === a.id ? FlowColors.AGENT_BLUE_MUTED : FlowColors.CARD)
    .borderRadius(12)
    .border({
      width: 1,
      color: this.effectiveSelected() === a.id ? FlowColors.AGENT_BLUE : FlowColors.BORDER
    })
    .onClick(() => {
      if (this.isLocked || !this.canInteract) {
        return;
      }
      this.localSelected = a.id;
      this.onSelectOption(a.id, a.id);
    });
  }
}
