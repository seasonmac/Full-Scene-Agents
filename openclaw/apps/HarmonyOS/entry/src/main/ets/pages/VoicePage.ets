import { MainViewModel } from '../viewmodel/MainViewModel';
import { VoiceConversationEntry } from '../model/GatewayModels';

@Component
export struct VoicePage {
  @ObjectLink viewModel: MainViewModel;
  @StorageProp('localeVersion') localeVersion: number = 0;

  build() {
    Scroll() {
      Column({ space: 14 }) {
        this.Header();
        this.StatusSection();
        this.PreferencesSection();
        this.LiveTranscriptSection();
        this.ConversationSection();
      }
      .width('100%')
      .padding({ left: 20, right: 20, top: 16, bottom: 24 });
    }
    .width('100%')
    .height('100%');
  }

  @Builder Header() {
    Column({ space: 6 }) {
      Text(this.localizedText('voice_header_eyebrow', 'VOICE'))
        .fontSize(12)
        .fontWeight(FontWeight.Bold)
        .fontColor('#1D5DD8');

      Text(this.localizedText('voice_header_title', 'Voice'))
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .fontColor('#17181C');

      Text(this.localizedText('voice_header_subtitle', 'Review live voice state, device preferences, and recent turns without pretending unsupported runtime features are active.'))
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor('#4D5563');
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start);
  }

  @Builder StatusSection() {
    Column({ space: 12 }) {
      Text(this.localizedText('voice_state_label', 'VOICE STATE'))
        .fontSize(12)
        .fontWeight(FontWeight.Bold)
        .fontColor('#1D5DD8');

      Column({ space: 12 }) {
        Row() {
          Text(this.voiceStateLabel())
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.voiceStateTextColor());

          Blank();

          Row() {
            Circle()
              .width(8)
              .height(8)
              .fill(this.voiceStateTextColor());

            Text(this.voiceStateLabel())
              .fontSize(12)
              .fontWeight(FontWeight.Bold)
              .fontColor(this.voiceStateTextColor())
              .margin({ left: 6 });
          }
          .padding({ left: 10, right: 10, top: 6, bottom: 6 })
          .backgroundColor(this.voiceStateBgColor())
          .borderRadius(999)
          .border({ width: 1, color: this.voiceStateBorderColor() });
        }
        .width('100%');

        Text(this.voiceStateDescription())
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#4D5563');

        this.InfoRow(this.localizedText('voice_gateway_status', 'Gateway status'), this.viewModel.statusText || this.localizedText('chat_offline', 'Offline'));
        this.DividerLine();
        this.InfoRow(this.localizedText('settings_mic_status', 'Mic status'), this.viewModel.micStatusText || this.localizedText('settings_off', 'Off'));
        this.DividerLine();
        this.InfoRow(this.localizedText('voice_queue_label', 'Queued turns'), this.viewModel.micQueuedMessages.length.toString());
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

  @Builder PreferencesSection() {
    Column({ space: 12 }) {
      Text(this.localizedText('voice_preferences_label', 'VOICE PREFERENCES'))
        .fontSize(12)
        .fontWeight(FontWeight.Bold)
        .fontColor('#1D5DD8');

      Column({ space: 12 }) {
        this.SettingToggle(
          this.localizedText('settings_enable_microphone', 'Enable microphone'),
          this.localizedText('voice_mic_preference_desc', 'Stores whether voice mode should use the microphone when runtime support is available.'),
          this.viewModel.micEnabled,
          (value: boolean) => {
            void this.viewModel.setMicEnabled(value);
          }
        );
        this.DividerLine();
        this.SettingToggle(
          this.localizedText('settings_enable_speaker', 'Enable speaker'),
          this.localizedText('voice_speaker_preference_desc', 'Stores whether spoken responses should use the speaker when audio playback support is available.'),
          this.viewModel.speakerEnabled,
          (value: boolean) => {
            void this.viewModel.setSpeakerEnabled(value);
          }
        );
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

  @Builder LiveTranscriptSection() {
    if ((this.viewModel.micLiveTranscript || '').trim().length > 0) {
      Column({ space: 12 }) {
        Text(this.localizedText('voice_live_transcript', 'LIVE TRANSCRIPT'))
          .fontSize(12)
          .fontWeight(FontWeight.Bold)
          .fontColor('#1D5DD8');

        Text((this.viewModel.micLiveTranscript || '').trim())
          .width('100%')
          .padding({ left: 14, right: 14, top: 14, bottom: 14 })
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#17181C')
          .backgroundColor('#FFFFFF')
          .borderRadius(14)
          .border({ width: 1, color: '#E5E7EC' });
      }
      .width('100%')
      .alignItems(HorizontalAlign.Start);
    }
  }

  @Builder ConversationSection() {
    Column({ space: 12 }) {
      Text(this.localizedText('voice_conversation_label', 'VOICE TURNS'))
        .fontSize(12)
        .fontWeight(FontWeight.Bold)
        .fontColor('#1D5DD8');

      if (this.viewModel.micConversation.length === 0) {
        this.EmptyConversationCard();
      } else {
        Column({ space: 10 }) {
          ForEach(this.viewModel.micConversation, (entry: VoiceConversationEntry, index: number) => {
            this.ConversationBubble(entry);
          }, (entry: VoiceConversationEntry, index: number): string => `${entry.role}-${entry.text}-${index}`);
        }
        .width('100%');
      }
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start);
  }

  @Builder EmptyConversationCard() {
    Column({ space: 8 }) {
      Text(this.localizedText('voice_empty_title', 'No voice turns yet'))
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor('#17181C');

      Text(this.localizedText('voice_empty_description', 'This page reflects current HarmonyOS voice state. When the runtime is connected and active, recent turns will appear here.'))
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor('#8A92A2')
        .textAlign(TextAlign.Center);
    }
    .width('100%')
    .padding({ top: 32, bottom: 32, left: 18, right: 18 })
    .backgroundColor('#FFFFFF')
    .borderRadius(14)
    .border({ width: 1, color: '#E5E7EC' })
    .alignItems(HorizontalAlign.Center);
  }

  @Builder ConversationBubble(entry: VoiceConversationEntry) {
    Row() {
      if (!this.isUserTurn(entry)) {
        Blank().width(24);
      }

      if (this.isUserTurn(entry)) {
        Blank().layoutWeight(1);
      }

      Column({ space: 4 }) {
        Text(this.isUserTurn(entry) ? this.localizedText('chat_user_avatar_text', 'Me') : this.localizedText('chat_assistant_avatar_text', 'OC'))
          .fontSize(11)
          .fontWeight(FontWeight.Bold)
          .fontColor(this.isUserTurn(entry) ? '#1D5DD8' : '#4D5563');

        Text(entry.text && entry.text.trim().length > 0 ? entry.text : this.localizedText('voice_empty_turn_text', 'Waiting for response…'))
          .fontSize(14)
          .fontWeight(FontWeight.Medium)
          .fontColor('#17181C');
      }
      .constraintSize({ minWidth: 0, maxWidth: '84%' })
      .padding({ left: 12, right: 12, top: 10, bottom: 10 })
      .backgroundColor(this.isUserTurn(entry) ? '#ECF3FF' : '#FFFFFF')
      .borderRadius(14)
      .border({ width: 1, color: this.isUserTurn(entry) ? '#D5E2FA' : '#E5E7EC' });

      if (!this.isUserTurn(entry)) {
        Blank().layoutWeight(1);
      }

      if (this.isUserTurn(entry)) {
        Blank().width(24);
      }
    }
    .width('100%');
  }

  @Builder SettingToggle(title: string, description: string, isOn: boolean, onChange: (value: boolean) => void) {
    Row() {
      Column({ space: 2 }) {
        Text(title)
          .fontSize(15)
          .fontWeight(FontWeight.Medium)
          .fontColor('#17181C');

        Text(description)
          .fontSize(12)
          .fontWeight(FontWeight.Medium)
          .fontColor('#8A92A2');
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1);

      Toggle({ type: ToggleType.Switch, isOn: isOn })
        .selectedColor('#1D5DD8')
        .onChange((value: boolean) => {
          onChange(value);
        });
    }
    .width('100%');
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
        .maxLines(1);
    }
    .width('100%');
  }

  @Builder DividerLine() {
    Divider().color('#E5E7EC');
  }

  private voiceStateCode(): string {
    if (!this.viewModel.isConnected || !this.viewModel.isNodeConnected) {
      return 'unavailable';
    }
    if (this.viewModel.micIsSending || this.viewModel.micCooldown) {
      return 'busy';
    }
    if (!this.viewModel.micEnabled) {
      return 'disabled';
    }
    return 'ready';
  }

  private isUserTurn(entry: VoiceConversationEntry): boolean {
    return entry.role === 'user';
  }

  private voiceStateLabel(): string {
    switch (this.voiceStateCode()) {
      case 'ready':
        return this.localizedText('voice_state_ready', 'Ready');
      case 'busy':
        return this.localizedText('voice_state_busy', 'Busy');
      case 'disabled':
        return this.localizedText('voice_state_disabled', 'Disabled');
      default:
        return this.localizedText('voice_state_unavailable', 'Unavailable');
    }
  }

  private voiceStateDescription(): string {
    if (!this.viewModel.isConnected || !this.viewModel.isNodeConnected) {
      return this.localizedText('voice_reason_not_connected', 'Connect a gateway and wait for the node session before expecting live voice activity.');
    }
    if (this.viewModel.micIsSending || this.viewModel.micCooldown) {
      return this.localizedText('voice_reason_busy', 'Voice state is active, but HarmonyOS is currently cooling down or sending queued audio turns.');
    }
    if (!this.viewModel.micEnabled) {
      return this.localizedText('voice_reason_disabled', 'Microphone preference is off. Enable it here or in Settings before expecting live capture.');
    }
    return this.localizedText('voice_reason_ready', 'Voice preferences and connection state are aligned. This page reflects runtime status without fabricating unsupported capture behavior.');
  }

  private voiceStateTextColor(): string {
    switch (this.voiceStateCode()) {
      case 'ready':
        return '#2F8C5A';
      case 'busy':
        return '#1D5DD8';
      case 'disabled':
        return '#C8841A';
      default:
        return '#8A92A2';
    }
  }

  private voiceStateBgColor(): string {
    switch (this.voiceStateCode()) {
      case 'ready':
        return '#E8F5EC';
      case 'busy':
        return '#ECF3FF';
      case 'disabled':
        return '#FFF4DE';
      default:
        return '#F6F7FA';
    }
  }

  private voiceStateBorderColor(): string {
    switch (this.voiceStateCode()) {
      case 'ready':
        return '#BEE0CA';
      case 'busy':
        return '#D5E2FA';
      case 'disabled':
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
      console.warn('VoicePage: localizedText failed for key: ' + name);
      return fallback;
    }
  }
}
