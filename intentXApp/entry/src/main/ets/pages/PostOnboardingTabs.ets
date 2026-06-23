import { MainViewModel } from '../viewmodel/MainViewModel';
import { ConnectPage } from './ConnectPage';
import { ChatPage } from './ChatPage';
// import { VoicePage } from './VoicePage';
// import { ScreenPage } from './ScreenPage';
import { SettingsPage } from './SettingsPage';
import { ImportPage } from './ImportPage';
import { DebugPage } from './DebugPage';

/**
 * Main tabs page shown after onboarding.
 */
@Component
export struct PostOnboardingTabs {
  @ObjectLink viewModel: MainViewModel;
  @StorageProp('localeVersion') localeVersion: number = 0;
  @StorageLink('pendingActiveTab') @Watch('applyPendingTab') pendingActiveTab: number = -1;
  @StorageLink('chatWorkbenchOpen') chatWorkbenchOpen: boolean = false;

  @State activeTab: number = 0;
  @State connectedState: boolean = false;
  @State statusState: string = '';
  @State screenTabPrimed: boolean = false;

  private connectionStateListenerId: number = -1;

  aboutToAppear(): void {
    this.applyPendingTab();
    this.syncTabSideEffects();
    this.syncConnectionState();
    this.connectionStateListenerId = this.viewModel.addConnectionStateListener(() => {
      this.syncConnectionState();
    });
  }

  aboutToDisappear(): void {
    if (this.connectionStateListenerId >= 0) {
      this.viewModel.removeConnectionStateListener(this.connectionStateListenerId);
      this.connectionStateListenerId = -1;
    }
  }

  private applyPendingTab(): void {
    if (typeof this.pendingActiveTab === 'number' && this.pendingActiveTab >= 0 && this.pendingActiveTab <= 6) {
      this.activeTab = this.pendingActiveTab;
      this.syncTabSideEffects();
      this.pendingActiveTab = -1;
    }
  }

  private syncTabSideEffects(): void {
    void this.viewModel.setVoiceScreenActive(this.activeTab === 2);
    if (this.activeTab === 3 && !this.screenTabPrimed) {
      this.screenTabPrimed = true;
      this.viewModel.refreshHomeCanvasOverviewIfConnected();
    }
  }

  build() {
    Column() {
      this.StatusBar();

      Column() {
        if (this.activeTab === 0) {
          ConnectPage({ viewModel: this.viewModel });
        } else if (this.activeTab === 1) {
          ChatPage({ viewModel: this.viewModel });
        // } else if (this.activeTab === 2) {
        //   VoicePage({ viewModel: this.viewModel });
        // } else if (this.activeTab === 3) {
        //   ScreenPage({ viewModel: this.viewModel });
        } else if (this.activeTab === 4) {
          SettingsPage({ viewModel: this.viewModel });
        } else if (this.activeTab === 5) {
          ImportPage({ viewModel: this.viewModel });
        // } else if (this.activeTab === 6) {
        //   DebugPage({ viewModel: this.viewModel });
        }
      }
      .width('100%')
      .layoutWeight(1);

      if (!this.chatWorkbenchOpen) {
        this.TabBar();
      }
    }
    .width('100%')
    .height('100%')
    .linearGradient({
      angle: 180,
      colors: [['#FFFFFF', 0.0], ['#F7F8FA', 1.0]]
    });
  }

  @Builder StatusBar() {
    Row() {
      Row({ space: 10 }) {
        Image($r('app.media.intentx_logo'))
          .width(32)
          .height(32)
          .borderRadius(10)
          .objectFit(ImageFit.Cover);

        Text('IntentX Agent')
          .fontSize(20)
          .fontWeight(FontWeight.Bold)
          .fontColor('#17181C');
      }

      Blank();

      Row() {
        Circle()
          .width(8)
          .height(8)
          .fill(this.getStatusColor());

        Text(this.statusState || this.localizedText('chat_offline', 'Offline'))
          .fontSize(12)
          .fontWeight(FontWeight.Medium)
          .fontColor(this.getStatusTextColor())
          .margin({ left: 6 });
      }
      .padding({ left: 10, right: 10, top: 5, bottom: 5 })
      .backgroundColor(this.getStatusBgColor())
      .borderRadius(999)
      .border({ width: 1, color: this.getStatusBorderColor() });
    }
    .width('100%')
    .padding({ left: 18, right: 18, top: 12, bottom: 12 });
  }

  @Builder TabBar() {
    Row() {
      this.TabItem(0, this.localizedText('tab_connect', 'Connect'), '🔗');
      this.TabItem(1, this.localizedText('tab_chat', 'Chat'), '💬');
      // this.TabItem(2, this.localizedText('tab_voice', 'Voice'), '🎙️');
      // this.TabItem(3, this.localizedText('tab_screen', 'Screen'), '🖥️');
      this.TabItem(4, this.localizedText('tab_settings', 'Settings'), '⚙️');
      this.TabItem(5, this.localizedText('tab_import', 'Import'), '📥');
      // this.TabItem(6, this.localizedText('tab_debug', 'Debug'), '🔧');
    }
    .width('100%')
    .padding({ left: 10, right: 10, top: 10, bottom: 10 })
    .backgroundColor('#FFFFFF')
    .borderRadius({ topLeft: 24, topRight: 24 })
    .border({ width: 1, color: '#E5E7EC' });
  }

  @Builder TabItem(index: number, label: string, icon: string) {
    Column() {
      Text(icon)
        .fontSize(20);

      Text(label)
        .fontSize(11)
        .fontWeight(this.activeTab === index ? FontWeight.Bold : FontWeight.Medium)
        .fontColor(this.activeTab === index ? '#1D5DD8' : '#8A92A2')
        .margin({ top: 2 });
    }
      .layoutWeight(1)
    .height(58)
    .justifyContent(FlexAlign.Center)
    .backgroundColor(this.activeTab === index ? '#ECF3FF' : 'transparent')
    .borderRadius(16)
      .margin({ left: 4, right: 4 })
    .border({
      width: 1,
      color: this.activeTab === index ? '#D5E2FA' : 'transparent'
    })
    .onClick(() => {
      if (this.activeTab === index) {
        return;
      }
      this.activeTab = index;
      this.syncTabSideEffects();
    });
  }

  private currentStatusVisual(): string {
    const status = (this.statusState || '').toLowerCase();
    if (this.connectedState) {
      return 'Connected';
    }
    if (status.includes('connecting') || status.includes('reconnecting')) {
      return 'Connecting';
    }
    if (status.includes('pairing') || status.includes('approval') || status.includes('auth') || status.includes('trust')) {
      return 'Warning';
    }
    if (status.includes('error') || status.includes('failed')) {
      return 'Error';
    }
    return 'Offline';
  }

  private syncConnectionState(): void {
    this.connectedState = this.viewModel.isConnected;
    this.statusState = this.viewModel.statusText || this.localizedText('chat_offline', 'Offline');
  }

  private getStatusColor(): string {
    switch (this.currentStatusVisual()) {
      case 'Connected':
        return '#2F8C5A';
      case 'Connecting':
        return '#1D5DD8';
      case 'Warning':
        return '#C8841A';
      case 'Error':
        return '#DC2626';
      default:
        return '#8A92A2';
    }
  }

  private getStatusTextColor(): string {
    switch (this.currentStatusVisual()) {
      case 'Connected':
        return '#2F8C5A';
      case 'Connecting':
        return '#1D5DD8';
      case 'Warning':
        return '#C8841A';
      case 'Error':
        return '#DC2626';
      default:
        return '#4D5563';
    }
  }

  private getStatusBgColor(): string {
    switch (this.currentStatusVisual()) {
      case 'Connected':
        return '#CFEBD8';
      case 'Connecting':
        return '#ECF3FF';
      case 'Warning':
        return '#EED8B8';
      case 'Error':
        return '#F3C8C8';
      default:
        return '#F6F7FA';
    }
  }

  private getStatusBorderColor(): string {
    switch (this.currentStatusVisual()) {
      case 'Connected':
        return '#2F8C5A';
      case 'Connecting':
        return '#D5E2FA';
      case 'Warning':
        return '#EED8B8';
      case 'Error':
        return '#F3C8C8';
      default:
        return '#E5E7EC';
    }
  }

  private localizedText(name: string, fallback: string): string {
    try {
      void this.localeVersion;
      return getContext(this).resourceManager.getStringByNameSync(name);
    } catch (error) {
      console.warn('PostOnboardingTabs: localizedText failed for key: ' + name);
      return fallback;
    }
  }
}
