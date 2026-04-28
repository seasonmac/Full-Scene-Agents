import { MainViewModel } from '../viewmodel/MainViewModel';

@Component
export struct ScreenPage {
  @ObjectLink viewModel: MainViewModel;
  @StorageProp('localeVersion') localeVersion: number = 0;

  aboutToAppear(): void {
    this.viewModel.refreshHomeCanvasOverviewIfConnected();
  }

  build() {
    Scroll() {
      Column({ space: 14 }) {
        this.Header();
        this.StatusSection();
        this.CanvasSection();
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 16, bottom: 24 });
    }
    .width('100%')
    .height('100%');
  }

  @Builder Header() {
    Column({ space: 6 }) {
      Text(this.localizedText('screen_header_eyebrow', 'SCREEN'))
        .fontSize(12)
        .fontWeight(FontWeight.Bold)
        .fontColor('#1D5DD8');

      Text(this.localizedText('screen_header_title', 'Screen'))
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .fontColor('#17181C');

      Text(this.localizedText('screen_header_subtitle', 'Reflect current canvas availability and recovery state without pretending the HarmonyOS screen runtime is already complete.'))
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor('#4D5563');
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start);
  }

  @Builder StatusSection() {
    Column({ space: 12 }) {
      Text(this.localizedText('screen_state_label', 'SCREEN STATE'))
        .fontSize(12)
        .fontWeight(FontWeight.Bold)
        .fontColor('#1D5DD8');

      Column({ space: 12 }) {
        Row() {
          Text(this.screenStateLabel())
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.screenStateTextColor());

          Blank();

          Row() {
            Circle()
              .width(8)
              .height(8)
              .fill(this.screenStateTextColor());

            Text(this.screenStateLabel())
              .fontSize(12)
              .fontWeight(FontWeight.Bold)
              .fontColor(this.screenStateTextColor())
              .margin({ left: 6 });
          }
          .padding({ left: 10, right: 10, top: 6, bottom: 6 })
          .backgroundColor(this.screenStateBgColor())
          .borderRadius(999)
          .border({ width: 1, color: this.screenStateBorderColor() });
        }
        .width('100%');

        Text(this.screenStateDescription())
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#4D5563');

        this.InfoRow(this.localizedText('settings_connection', 'Connection'), this.viewModel.isConnected ? this.localizedText('chat_connected', 'Connected') : this.localizedText('chat_offline', 'Offline'));
        this.DividerLine();
        this.InfoRow(this.localizedText('screen_node_label', 'Node link'), this.viewModel.isNodeConnected ? this.localizedText('chat_connected', 'Connected') : this.localizedText('screen_node_unavailable_value', 'Unavailable'));
        this.DividerLine();
        this.InfoRow(this.localizedText('screen_hydrated_label', 'Canvas hydrated'), this.viewModel.canvasA2uiHydrated ? this.localizedText('common_yes', 'Yes') : this.localizedText('common_no', 'No'));
      }
      .width('100%')
      .padding({ left: 14, right: 14, top: 14, bottom: 14 })
      .backgroundColor('#FFFFFF')
      .borderRadius(14)
      .border({ width: 1, color: '#E5E7EC' });
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start);
  }

  @Builder CanvasSection() {
    Column({ space: 12 }) {
      Text(this.localizedText('screen_canvas_label', 'CANVAS'))
        .fontSize(12)
        .fontWeight(FontWeight.Bold)
        .fontColor('#1D5DD8');

      Column({ space: 12 }) {
        if ((this.viewModel.canvasCurrentUrl || '').trim().length > 0) {
          this.InfoRow(this.localizedText('screen_current_url', 'Current URL'), this.viewModel.canvasCurrentUrl || '');
          this.DividerLine();
        }

        if ((this.viewModel.canvasRehydrateErrorText || '').trim().length > 0) {
          Column({ space: 4 }) {
            Text(this.localizedText('screen_error_title', 'LAST ERROR'))
              .fontSize(11)
              .fontWeight(FontWeight.Bold)
              .fontColor('#DC2626');

            Text(this.viewModel.canvasRehydrateErrorText || '')
              .fontSize(14)
              .fontWeight(FontWeight.Medium)
              .fontColor('#17181C');
          }
          .width('100%')
          .padding({ left: 12, right: 12, top: 10, bottom: 10 })
          .backgroundColor('#FFF5F5')
          .borderRadius(12)
          .border({ width: 1, color: '#F3C8C8' });

          this.DividerLine();
        }

        if ((this.viewModel.canvasCurrentUrl || '').trim().length === 0) {
          Text(this.localizedText('screen_empty_description', 'No canvas URL is currently available on HarmonyOS. This page reports state only and avoids fake restore success.'))
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor('#8A92A2');
        } else {
          Text(this.localizedText('screen_ready_description', 'Canvas metadata is available. Refresh only re-checks controlled state and does not fabricate a successful restore.'))
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor('#4D5563');
        }

        Button({ type: ButtonType.Normal }) {
          Text(this.localizedText('screen_refresh', 'Refresh canvas state'))
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor(this.canRefreshCanvas() ? '#1D5DD8' : '#8A92A2');
        }
        .width('100%')
        .height(44)
        .backgroundColor(this.canRefreshCanvas() ? '#ECF3FF' : '#F6F7FA')
        .borderRadius(12)
        .enabled(this.canRefreshCanvas())
        .onClick(() => {
          this.viewModel.requestCanvasRehydrate('screen_page', true);
        });
      }
      .width('100%')
      .padding({ left: 14, right: 14, top: 14, bottom: 14 })
      .backgroundColor('#FFFFFF')
      .borderRadius(14)
      .border({ width: 1, color: '#E5E7EC' });
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start);
  }

  @Builder InfoRow(label: string, value: string) {
    Row() {
      Text(label)
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor('#17181C');

      Blank();

      Text(value)
        .fontSize(13)
        .fontWeight(FontWeight.Medium)
        .fontColor('#8A92A2')
        .maxLines(2)
        .textAlign(TextAlign.End);
    }
    .width('100%');
  }

  @Builder DividerLine() {
    Divider().color('#E5E7EC');
  }

  private canRefreshCanvas(): boolean {
    const state = this.screenStateCode();
    return state === 'ready' || state === 'error';
  }

  private screenStateCode(): string {
    if (!this.viewModel.isConnected) {
      return 'not_connected';
    }
    if (!this.viewModel.isNodeConnected) {
      return 'node_unavailable';
    }
    if (this.viewModel.canvasRehydratePending) {
      return 'pending';
    }
    if ((this.viewModel.canvasRehydrateErrorText || '').trim().length > 0 && (this.viewModel.canvasCurrentUrl || '').trim().length > 0) {
      return 'error';
    }
    if ((this.viewModel.canvasCurrentUrl || '').trim().length > 0) {
      return 'ready';
    }
    return 'canvas_unavailable';
  }

  private screenStateLabel(): string {
    switch (this.screenStateCode()) {
      case 'ready':
        return this.localizedText('screen_state_ready', 'Ready');
      case 'pending':
        return this.localizedText('screen_state_pending', 'Refreshing');
      case 'error':
        return this.localizedText('screen_state_error', 'Error');
      case 'node_unavailable':
        return this.localizedText('screen_state_node_unavailable', 'Node unavailable');
      case 'canvas_unavailable':
        return this.localizedText('screen_state_canvas_unavailable', 'Canvas unavailable');
      default:
        return this.localizedText('screen_state_not_connected', 'Not connected');
    }
  }

  private screenStateDescription(): string {
    switch (this.screenStateCode()) {
      case 'ready':
        return this.localizedText('screen_reason_ready', 'A canvas URL is available. Refresh remains constrained to real state updates only.');
      case 'pending':
        return this.localizedText('screen_reason_pending', 'HarmonyOS is re-checking canvas state without inventing a successful restore.');
      case 'error':
        return this.viewModel.canvasRehydrateErrorText || this.localizedText('screen_reason_error', 'Canvas state refresh failed.');
      case 'node_unavailable':
        return this.localizedText('screen_reason_node_unavailable', 'The gateway connection is up, but the node runtime is not ready for screen state yet.');
      case 'canvas_unavailable':
        return this.localizedText('screen_reason_canvas_unavailable', 'No canvas URL has been discovered yet, so this page stays in a deterministic unavailable state.');
      default:
        return this.localizedText('screen_reason_not_connected', 'Connect to a gateway before expecting HarmonyOS screen state or canvas metadata.');
    }
  }

  private screenStateTextColor(): string {
    switch (this.screenStateCode()) {
      case 'ready':
        return '#2F8C5A';
      case 'pending':
        return '#1D5DD8';
      case 'error':
        return '#DC2626';
      case 'node_unavailable':
      case 'canvas_unavailable':
        return '#C8841A';
      default:
        return '#8A92A2';
    }
  }

  private screenStateBgColor(): string {
    switch (this.screenStateCode()) {
      case 'ready':
        return '#E8F5EC';
      case 'pending':
        return '#ECF3FF';
      case 'error':
        return '#FFF5F5';
      case 'node_unavailable':
      case 'canvas_unavailable':
        return '#FFF4DE';
      default:
        return '#F6F7FA';
    }
  }

  private screenStateBorderColor(): string {
    switch (this.screenStateCode()) {
      case 'ready':
        return '#BEE0CA';
      case 'pending':
        return '#D5E2FA';
      case 'error':
        return '#F3C8C8';
      case 'node_unavailable':
      case 'canvas_unavailable':
        return '#F0D19A';
      default:
        return '#E5E7EC';
    }
  }

  private localizedText(name: string, fallback: string): string {
    try {
      void this.localeVersion;
      return getContext(this).resourceManager.getStringByNameSync(name);
    } catch (error) {
      console.warn('ScreenPage: localizedText failed for key: ' + name);
      return fallback;
    }
  }
}
