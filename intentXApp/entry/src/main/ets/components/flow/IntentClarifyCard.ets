import { ClarifyOption, UIBlock } from '../../model/FlowModels';
import { FlowColors } from '../../common/FlowColors';

/**
 * IntentClarifyCard —— 意图澄清单选卡（01 文档 §3.2）。
 * isLocked=true 为历史/回放态，隐藏确认按钮、显示「已选择」。
 * 交互合同：确认后将所选选项原文（如 "1. 定时自动生成…"）作为用户消息回传。
 */
@Component
export struct IntentClarifyCard {
  @ObjectLink block: UIBlock;
  @Prop selectedIdOverride: string = '';
  @Prop useSelectedIdOverride: boolean = false;
  @Prop isLocked: boolean = false;
  @Prop canInteract: boolean = true;
  @Prop hideActions: boolean = false;
  // (optionId, optionLabel) => void
  onConfirmOption: (id: string, label: string) => void = (id: string, label: string) => {};
  onSelectOption: (id: string, label: string) => void = (id: string, label: string) => {};

  @State localSelected: string = '';

  aboutToAppear(): void {
    this.localSelected = this.useSelectedIdOverride ? this.selectedIdOverride : this.block.selectedId;
  }

  private effectiveSelected(): string {
    if (this.useSelectedIdOverride) {
      return this.selectedIdOverride;
    }
    if (this.block.selectedId.length > 0) {
      return this.block.selectedId;
    }
    return this.localSelected;
  }

  build() {
    Column() {
      // 阶段头部
      Row() {
        Row({ space: 5 }) {
          Circle().width(6).height(6).fill(FlowColors.AGENT_BLUE);
          Text('意图澄清')
            .fontSize(12)
            .fontWeight(FontWeight.Medium)
            .fontColor(FlowColors.AGENT_BLUE_FG);
        }
        .padding({ left: 10, right: 10, top: 3, bottom: 3 })
        .backgroundColor(FlowColors.AGENT_BLUE_LIGHT)
        .borderRadius(999);
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 8 });

      // 问题
      Text(this.block.question)
        .fontSize(15)
        .fontWeight(FontWeight.Medium)
        .fontColor(FlowColors.FOREGROUND)
        .lineHeight(22)
        .width('100%')
        .padding({ left: 16, right: 16, top: 12, bottom: 8 });

      // 选项列表
      Column({ space: 6 }) {
        ForEach(this.block.options, (opt: ClarifyOption) => {
          this.OptionRow(opt);
        }, (opt: ClarifyOption) => opt.id + '|' + opt.label);
      }
      .width('100%')
      .padding({ left: 16, right: 16, bottom: 12 });

      // 确认按钮 / 锁定态
      if (!this.isLocked && !this.hideActions) {
        Button({ type: ButtonType.Normal }) {
          Text('确认选择')
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
          this.confirm();
        });
        Blank().height(14);
      } else if (this.effectiveSelected().length > 0) {
        Row({ space: 6 }) {
          Text('✓')
            .fontSize(13)
            .fontColor(FlowColors.STEP_DONE);
          Text('已选择')
            .fontSize(12)
            .fontColor(FlowColors.MUTED_FG);
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

  @Builder OptionRow(opt: ClarifyOption) {
    Row({ space: 12 }) {
      // 单选圆
      Stack({ alignContent: Alignment.Center }) {
        Circle()
          .width(16)
          .height(16)
          .fill(this.effectiveSelected() === opt.id ? FlowColors.AGENT_BLUE : FlowColors.CARD)
          .stroke(this.effectiveSelected() === opt.id ? FlowColors.AGENT_BLUE : FlowColors.BORDER)
          .strokeWidth(2);
        if (this.effectiveSelected() === opt.id) {
          Text('✓').fontSize(10).fontColor('#FFFFFF').fontWeight(FontWeight.Bold);
        }
      }
      .width(16)
      .height(16)
      .margin({ top: 2 });

      Column({ space: 3 }) {
        Text(opt.label)
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor(this.effectiveSelected() === opt.id ? FlowColors.AGENT_BLUE : FlowColors.FOREGROUND)
          .lineHeight(20)
          .width('100%');
        if (opt.description.length > 0) {
          Text(opt.description)
            .fontSize(12)
            .fontColor(FlowColors.MUTED_FG)
            .lineHeight(17)
            .width('100%');
        }
      }
      .layoutWeight(1)
      .alignItems(HorizontalAlign.Start);
    }
    .width('100%')
    .alignItems(VerticalAlign.Top)
    .padding({ left: 12, right: 12, top: 9, bottom: 9 })
    .backgroundColor(this.effectiveSelected() === opt.id ? FlowColors.AGENT_BLUE_MUTED : FlowColors.CARD)
    .borderRadius(12)
    .border({
      width: 1,
      color: this.effectiveSelected() === opt.id ? FlowColors.AGENT_BLUE : FlowColors.BORDER
    })
    .onClick(() => {
      if (this.isLocked || !this.canInteract) {
        return;
      }
      this.localSelected = opt.id;
      this.onSelectOption(opt.id, opt.label);
    });
  }

  private confirm(): void {
    const sel = this.effectiveSelected();
    if (sel.length === 0 || this.isLocked) {
      return;
    }
    let label = '';
    for (let i = 0; i < this.block.options.length; i++) {
      if (this.block.options[i].id === sel) {
        label = this.block.options[i].label;
        break;
      }
    }
    this.onConfirmOption(sel, label);
  }
}
