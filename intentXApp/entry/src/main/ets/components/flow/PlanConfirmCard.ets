import { PlanItem, UIBlock } from '../../model/FlowModels';
import { FlowColors } from '../../common/FlowColors';

/**
 * PlanConfirmCard —— 个性化任务创建 / 计划确认（01 文档 §3.4）。
 * 「确认创建」发送 "可以"；「修改」聚焦输入框。
 */
@Component
export struct PlanConfirmCard {
  @ObjectLink block: UIBlock;
  @Prop isLocked: boolean = false;
  @Prop canInteract: boolean = true;
  @Prop hideActions: boolean = false;
  onConfirm: () => void = () => {};
  onModify: () => void = () => {};

  build() {
    Column() {
      // 头部徽章
      Row() {
        Text('个性化任务创建')
          .fontSize(12)
          .fontWeight(FontWeight.Medium)
          .fontColor(FlowColors.TEAL_FG)
          .padding({ left: 10, right: 10, top: 3, bottom: 3 })
          .backgroundColor(FlowColors.TEAL_BG)
          .borderRadius(999);
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12, bottom: 8 })
      .borderWidth({ bottom: 1 })
      .borderColor(FlowColors.BORDER_SOFT);

      Row({ space: 10 }) {
        this.PlanStepIcon('target', 0, true);
        Text(this.block.taskTitle)
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor(FlowColors.FOREGROUND)
          .layoutWeight(1)
          .maxLines(2)
          .textOverflow({ overflow: TextOverflow.Ellipsis });
      }
      .width('100%')
      .padding({ left: 16, right: 16, top: 12 });

      this.PlanFlowDiagram();

      // 底部确认 / 已确认态
      Column() {
        if (this.block.approved || this.isLocked) {
          Row({ space: 6 }) {
            Text('✓').fontSize(14).fontColor(FlowColors.STEP_DONE);
            Text('已确认创建')
              .fontSize(12)
              .fontWeight(FontWeight.Medium)
              .fontColor(FlowColors.STEP_DONE);
          }
          .width('100%')
          .padding({ left: 16, right: 16, top: 12, bottom: 12 });
        } else if (!this.hideActions) {
          Text(this.block.question)
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor(FlowColors.FOREGROUND)
            .width('100%')
            .padding({ left: 16, right: 16, top: 12 });

          Row({ space: 10 }) {
            Button({ type: ButtonType.Normal }) {
              Text('修改')
                .fontSize(14)
                .fontWeight(FontWeight.Medium)
                .fontColor(FlowColors.FOREGROUND);
            }
            .layoutWeight(1)
            .height(42)
            .borderRadius(12)
            .backgroundColor(FlowColors.MUTED)
            .enabled(this.canInteract)
            .onClick(() => {
              this.onModify();
            });

            Button({ type: ButtonType.Normal }) {
              Text('确认创建')
                .fontSize(14)
                .fontWeight(FontWeight.Bold)
                .fontColor('#FFFFFF');
            }
            .layoutWeight(1)
            .height(42)
            .borderRadius(12)
            .backgroundColor(FlowColors.AGENT_BLUE)
            .enabled(this.canInteract)
            .onClick(() => {
              this.onConfirm();
            });
          }
          .width('100%')
          .padding({ left: 16, right: 16, top: 12, bottom: 14 });
        }
      }
      .width('100%')
      .borderWidth({ top: 1 })
      .borderColor(FlowColors.BORDER_SOFT);
    }
    .width('100%')
    .backgroundColor(FlowColors.CARD)
    .borderRadius(16)
    .border({ width: 1, color: FlowColors.BORDER })
    .clip(true);
  }

  @Builder PlanFlowDiagram() {
    Column() {
      ForEach(this.block.items, (item: PlanItem, i: number) => {
        this.PlanFlowStep(item, i);

        if (i < this.block.items.length - 1) {
          Row() {
            Column() {
              Column()
                .width(2)
                .height(14)
                .backgroundColor(FlowColors.BORDER);
            }
            .width(34)
            .alignItems(HorizontalAlign.Center);

            Blank().layoutWeight(1);
          }
          .width('100%')
          .height(14);
        }
      }, (item: PlanItem, i: number) => `${i}|${item.label}|${item.value}`);
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 12, bottom: 12 });
  }

  @Builder PlanFlowStep(item: PlanItem, index: number) {
    Row({ space: 10 }) {
      this.PlanStepIcon(item.icon, index, false);

      Column({ space: 3 }) {
        Text(item.label)
          .fontSize(12)
          .fontWeight(FontWeight.Bold)
          .fontColor(FlowColors.FOREGROUND)
          .width('100%');

        Text(item.value)
          .fontSize(12)
          .fontColor(FlowColors.MUTED_FG)
          .lineHeight(18)
          .width('100%');
      }
      .layoutWeight(1)
      .alignItems(HorizontalAlign.Start)
      .padding({ top: 8, bottom: 8 });
    }
    .width('100%')
    .alignItems(VerticalAlign.Top)
    .backgroundColor(index % 2 === 0 ? FlowColors.AGENT_BLUE_MUTED : FlowColors.CARD)
    .borderRadius(12)
    .padding({ left: 8, right: 10 });
  }

  @Builder PlanStepIcon(icon: string, index: number, isTask: boolean) {
    Stack({ alignContent: Alignment.Center }) {
      Circle()
        .width(28)
        .height(28)
        .fill(isTask ? FlowColors.TEAL_BG : FlowColors.AGENT_BLUE_LIGHT);
      Text(this.planIconText(icon, index, isTask))
        .fontSize(14)
        .fontWeight(FontWeight.Bold)
        .fontColor(isTask ? FlowColors.TEAL_FG : FlowColors.AGENT_BLUE_FG);
    }
    .width(34)
    .height(34);
  }

  private planIconText(icon: string, index: number, isTask: boolean): string {
    if (isTask) {
      return '◎';
    }
    const normalized = icon.trim().toLowerCase();
    if (normalized === 'input') {
      return '↓';
    }
    if (normalized === 'process') {
      return '⚙';
    }
    if (normalized === 'schedule') {
      return '⏱';
    }
    if (normalized === 'output') {
      return '✓';
    }
    if (normalized === 'warning') {
      return '!';
    }
    if (normalized === 'boundary') {
      return '×';
    }
    const fallback = ['↓', '⚙', '⏱', '✓', '!', '×'];
    return fallback[index % fallback.length];
  }
}
