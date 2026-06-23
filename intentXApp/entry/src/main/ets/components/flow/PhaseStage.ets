import { StatusStrip } from './StatusStrip';
import { PhaseSkeleton } from './PhaseSkeleton';

/**
 * PhaseStage —— 单一相位编排层（04 文档 §3）。
 *
 * 状态机： thinking → revealing → ready → (collapsed)
 * - 真实内容【始终挂载】定义容器高度；骨架用 Stack 叠加在其上 → 零 layout shift。
 * - thinking：内容 opacity 0 + 骨架 shimmer；状态条呼吸点 + 轮播 + 计时。
 * - revealing：骨架原位淡出，内容原位淡入上浮；呼吸点形变对勾。
 * - ready：仅内容可交互；停留后状态条塌缩淡出，footer 由父级淡入。
 *
 * 时序自包含（计时器驱动，不依赖文本解析）。settleToken 变化时重跑一遍编排；
 * skipThinking=true（再次打开已就绪弹层）时直接进入 ready。
 */
@Component
export struct PhaseStage {
  @Prop @Watch('onSettleChange') settleToken: string = '';
  @Prop phase: string = 'clarify';
  @Prop skipThinking: boolean = false;
  // 进入 ready 时回调，父级据此淡入 footer
  onReady: () => void = () => {};
  @BuilderParam content: () => void = this.EmptyContent;

  @State status: string = 'thinking';
  @State statusCollapsed: boolean = false;
  @State contentShift: number = 8;

  private timers: number[] = [];

  @Builder EmptyContent() {
  }

  aboutToAppear(): void {
    this.runChoreography();
  }

  aboutToDisappear(): void {
    this.clearTimers();
  }

  onSettleChange(): void {
    this.runChoreography();
  }

  private clearTimers(): void {
    for (let i = 0; i < this.timers.length; i++) {
      clearTimeout(this.timers[i]);
    }
    this.timers = [];
  }

  private thinkingDuration(): number {
    return (this.phase === 'create' || this.phase === 'execute') ? 3200 : 2600;
  }

  private runChoreography(): void {
    this.clearTimers();

    if (this.skipThinking) {
      this.status = 'ready';
      this.statusCollapsed = true;
      this.contentShift = 0;
      this.onReady();
      return;
    }

    // thinking 起始
    this.status = 'thinking';
    this.statusCollapsed = false;
    this.contentShift = 8;

    const dur = this.thinkingDuration();

    // thinking → revealing（骨架淡出 + 内容上浮淡入）
    this.timers.push(setTimeout(() => {
      this.getUIContext().animateTo({ duration: 420, curve: Curve.FastOutSlowIn }, () => {
        this.status = 'revealing';
        this.contentShift = 0;
      });
    }, dur));

    // revealing → ready（内容可交互 + 通知父级淡入 footer）
    this.timers.push(setTimeout(() => {
      this.status = 'ready';
      this.onReady();
    }, dur + 480));

    // ready → collapsed（状态条塌缩腾出空间）
    this.timers.push(setTimeout(() => {
      this.getUIContext().animateTo({ duration: 500, curve: Curve.FastOutSlowIn }, () => {
        this.statusCollapsed = true;
      });
    }, dur + 480 + 1200));
  }

  private contentVisible(): boolean {
    return this.status === 'revealing' || this.status === 'ready';
  }

  build() {
    Column({ space: 12 }) {
      StatusStrip({
        status: this.status,
        phase: this.phase,
        collapsed: this.statusCollapsed
      });

      Stack({ alignContent: Alignment.TopStart }) {
        // 内容层：始终挂载，定义容器几何高度
        Column() {
          this.content();
        }
        .width('100%')
        .opacity(this.contentVisible() ? 1 : 0)
        .translate({ y: this.contentVisible() ? 0 : this.contentShift })
        .hitTestBehavior(this.status === 'ready' ? HitTestMode.Default : HitTestMode.None);

        // 骨架层：叠加在内容 footprint 之上，ready 后移除
        if (this.status !== 'ready') {
          Column() {
            PhaseSkeleton({ phase: this.phase })
              .width('100%');
          }
          .width('100%')
          .opacity(this.status === 'revealing' ? 0 : 1);
        }
      }
      .width('100%');
    }
    .width('100%');
  }
}
