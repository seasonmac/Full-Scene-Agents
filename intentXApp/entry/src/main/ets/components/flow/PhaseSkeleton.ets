import { FlowColors } from '../../common/FlowColors';

@Component
export struct PhaseSkeleton {
  @Prop phase: string = 'clarify';

  private rowCount(): number {
    if (this.phase === 'create' || this.phase === 'execute') {
      return 4;
    }
    return 3;
  }

  build() {
    Column({ space: 10 }) {
      Row() {
        Column()
          .height(16)
          .width('58%')
          .backgroundColor(FlowColors.MUTED)
          .borderRadius(8);
        Blank().layoutWeight(1);
      }
      .width('100%');

      ForEach([0, 1, 2, 3], (i: number) => {
        if (i < this.rowCount()) {
          Row({ space: 10 }) {
            Circle()
              .width(18)
              .height(18)
              .fill(FlowColors.MUTED);
            Column({ space: 6 }) {
              Column()
                .height(12)
                .width(i % 2 === 0 ? '82%' : '68%')
                .backgroundColor(FlowColors.MUTED)
                .borderRadius(6);
              Column()
                .height(10)
                .width(i % 2 === 0 ? '62%' : '76%')
                .backgroundColor(FlowColors.MUTED)
                .borderRadius(5);
            }
            .layoutWeight(1)
            .alignItems(HorizontalAlign.Start);
          }
          .width('100%')
          .height(44)
          .padding({ left: 12, right: 12 })
          .backgroundColor('#F8FAFC')
          .borderRadius(12);
        }
      }, (i: number) => String(i));
    }
    .width('100%')
    .padding(14)
    .backgroundColor(FlowColors.CARD)
    .borderRadius(16)
    .border({ width: 1, color: FlowColors.BORDER });
  }
}
