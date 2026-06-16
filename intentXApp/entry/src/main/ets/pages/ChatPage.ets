import { MainViewModel } from '../viewmodel/MainViewModel';
import { ChatChoiceOption, ChatMessage, ChatPendingInteraction, ChatPendingInteractionChoice, ChatPendingToolCall, ChatSessionEntry } from '../model/GatewayModels';
import { UIBlock } from '../model/FlowModels';
import { FlowProgressBar } from '../components/flow/FlowProgressBar';
import { IntentClarifyCard } from '../components/flow/IntentClarifyCard';
import { ApproachProposalCard } from '../components/flow/ApproachProposalCard';
import { PlanConfirmCard } from '../components/flow/PlanConfirmCard';
import { SkillCreateCard } from '../components/flow/SkillCreateCard';
import { ExecuteSummaryCard } from '../components/flow/ExecuteSummaryCard';
import { Markdown, MarkdownController } from '@luvi/lv-markdown-in';
import { pasteboard } from '@kit.BasicServicesKit';

// Design palette — chat-ux v0 design spec (Docs/01 §2, oklch -> sRGB)
const C_BG: string = '#F7F8FA';
const C_CARD: string = '#FFFFFF';
const C_PRIMARY: string = '#2563EB';
const C_PRIMARY_SOFT: string = '#6E92F4';
const C_ACCENT: string = '#DBEAFE';
const C_ACCENT_FG: string = '#1E49B7';
const C_MUTED: string = '#EFF2F5';
const C_MUTED_FG: string = '#6B7280';
const C_FOREGROUND: string = '#14181C';
const C_BORDER: string = '#E2E5EA';
const C_USER_BUBBLE: string = '#2563EB';
const C_USER_TEXT: string = '#FFFFFF';
const C_INPUT_BG: string = '#EFF4FF';

@Component
export struct ChatPage {
  @ObjectLink viewModel: MainViewModel;
  @StorageProp('localeVersion') localeVersion: number = 0;

  @State messageText: string = '';
  @State showThinkingOptions: boolean = false;
  @State showSessionList: boolean = false;
  @State showSkillLaunchers: boolean = false;
  @State selectedSkillLauncherPrefix: string = '';
  @State selectedSkillLauncherIcon: string = '';
  @State selectedSkillLauncherLabel: string = '';
  @State chatMessagesState: ChatMessage[] = [];
  @State chatStreamingTextState: string = '';
  @State chatPendingToolCallsState: ChatPendingToolCall[] = [];
  @State chatErrorState: string = '';
  @State chatHealthOkState: boolean = false;
  @State chatThinkingLevelState: string = 'off';
  @State chatSessionKeyState: string = '';
  @State chatSessionsState: ChatSessionEntry[] = [];
  @State pendingRunCountState: number = 0;
  @State composerInputVisible: boolean = true;
  @State latestAssistantChoiceOptionsState: ChatChoiceOption[] = [];
  @State pendingInteractionsState: ChatPendingInteraction[] = [];
  // chat-ux 结构化流程
  @State flowBlocksState: UIBlock[] = [];
  @State flowPhaseState: string = '';
  @State flowTotalQuestionsState: number = 1;

  private chatStateListenerId: number = -1;
  private messageScroller: Scroller = new Scroller();
  private lastScrollToken: string = '';
  private streamingMarkdownController: MarkdownController = new MarkdownController();
  private userMarkdownController: MarkdownController = new MarkdownController();
  private openclawMarkdownController: MarkdownController = new MarkdownController();

  aboutToAppear(): void {
    this.configureMarkdownControllers();

    this.syncChatState();
    this.chatStateListenerId = this.viewModel.addChatStateListener(() => {
      this.syncChatState();
    });
    this.viewModel.loadChat(this.viewModel.mainSessionKey);
    this.viewModel.refreshChatSessions(200);
    this.scheduleInitialScroll();
  }

  aboutToDisappear(): void {
    if (this.chatStateListenerId >= 0) {
      this.viewModel.removeChatStateListener(this.chatStateListenerId);
      this.chatStateListenerId = -1;
    }
  }

  build() {
    Column() {
      this.ChatHeader();
      this.HeaderSessionPanel();
      this.WorkflowProgress();

      if (this.chatErrorState.length > 0) {
        this.ChatErrorRail(this.chatErrorState);
      }

      this.MessagesList();
      this.Composer();
    }
    .width('100%')
    .height('100%')
    .padding({ left: 16, right: 16 })
    .backgroundColor(C_BG);
  }

  @Builder ChatHeader() {
    Row({ space: 12 }) {
      Column({ space: 4 }) {
        Text(this.currentSessionLabel())
          .fontSize(18)
          .fontWeight(FontWeight.Bold)
          .fontColor(C_FOREGROUND)
          .maxLines(1)
          .textOverflow({ overflow: TextOverflow.Ellipsis });

        Row({ space: 8 }) {
          Text(this.chatSessionKeyState)
            .fontSize(12)
            .fontColor(C_MUTED_FG)
            .fontFamily('monospace')
            .layoutWeight(1)
            .maxLines(1)
            .textOverflow({ overflow: TextOverflow.Ellipsis });

          Text('⧉')
            .fontSize(14)
            .fontColor(C_MUTED_FG)
            .onClick(() => {
              this.copySessionKey();
            });
        }
        .width('100%');
      }
      .layoutWeight(1)
      .alignItems(HorizontalAlign.Start);

      Button({ type: ButtonType.Normal }) {
        Row({ space: 6 }) {
          Text('☰')
            .fontSize(14)
            .fontColor(C_ACCENT_FG);
          Text($r('app.string.chat_sessions'))
            .fontSize(14)
            .fontWeight(FontWeight.Medium)
            .fontColor(C_ACCENT_FG);
        }
      }
      .height(36)
      .backgroundColor(C_ACCENT)
      .borderRadius(14)
      .padding({ left: 14, right: 14 })
      .onClick(() => {
        this.showSessionList = !this.showSessionList;
        this.showThinkingOptions = false;
        this.showSkillLaunchers = false;
      });
    }
    .width('100%')
    .padding({ left: 4, right: 4, top: 12, bottom: 8 });
  }

  @Builder HeaderSessionPanel() {
    if (this.showSessionList) {
      Column() {
        this.SessionList();
      }
      .margin({ bottom: 8 });
    }
  }

  @Builder WorkflowProgress() {
    if (this.shouldShowWorkflowProgress()) {
      FlowProgressBar({ activePhase: this.currentWorkflowPhase() })
        .margin({ bottom: 8 });
    }
  }

  private isActiveSession(session: ChatSessionEntry): boolean {
    return session.key === this.chatSessionKeyState;
  }

  @Builder ChatErrorRail(errorText: string) {
    Column({ space: 2 }) {
      Text($r('app.string.chat_error_title'))
        .fontSize(11)
        .fontWeight(FontWeight.Medium)
        .fontColor('#DC2626');

      Text(errorText)
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor('#17181C');
    }
    .width('100%')
    .padding({ left: 12, right: 12, top: 8, bottom: 8 })
    .backgroundColor('#FFFFFF')
    .borderRadius(12)
    .border({ width: 1, color: '#DC2626' });
  }

  @Builder MessagesList() {
    Scroll(this.messageScroller) {
      Column({ space: 10 }) {
        if (this.chatMessagesState.length === 0) {
          this.EmptyState();
        } else if (this.hasStructuredFlow()) {
          this.FlowTimeline();

          if (this.chatStreamingTextState.length > 0) {
            this.StreamingMessage(this.chatStreamingTextState);
          }
          if (this.chatPendingToolCallsState.length > 0) {
            this.PendingToolCallsRail();
          }
          if (this.pendingInteractionsState.length > 0) {
            this.PendingInteractionCards(this.pendingInteractionsState);
          }
        } else {
          ForEach(this.chatMessagesState, (message: ChatMessage) => {
            this.MessageItem(message);
          }, (message: ChatMessage, index: number) => this.messageListKey(message, index));

          if (this.chatStreamingTextState.length > 0) {
            this.StreamingMessage(this.chatStreamingTextState);
          }

          if (this.chatPendingToolCallsState.length > 0) {
            this.PendingToolCallsRail();
          }

          if (this.pendingInteractionsState.length > 0) {
            this.PendingInteractionCards(this.pendingInteractionsState);
          }

          if (this.latestAssistantChoiceOptionsState.length > 0) {
            this.AssistantChoiceOptions(this.latestAssistantChoiceOptionsState);
          }
        }
      }
      .width('100%')
      .padding({ top: 8, bottom: 8 });
    }
    .width('100%')
    .layoutWeight(1);
  }

  // ---------- chat-ux 结构化流程时间线 ----------

  private hasStructuredFlow(): boolean {
    for (let i = 0; i < this.flowBlocksState.length; i++) {
      const k = this.flowBlocksState[i].kind;
      if (k === 'clarify-question' || k === 'approach-proposal' || k === 'plan-confirm'
        || k === 'skill-create' || k === 'exec-summary') {
        return true;
      }
    }
    return false;
  }

  // 工作流是否仍激活（用于决定卡片可否交互）
  private flowActive(): boolean {
    return this.flowPhaseState.length > 0;
  }

  private shouldShowWorkflowProgress(): boolean {
    return this.hasStructuredFlow() && this.currentWorkflowPhase().length > 0;
  }

  private currentWorkflowPhase(): string {
    if (this.flowPhaseState.length > 0) {
      return this.flowPhaseState;
    }
    for (let i = this.flowBlocksState.length - 1; i >= 0; i--) {
      const phase = this.flowBlocksState[i].phase;
      if (phase.length > 0) {
        return phase;
      }
    }
    return '';
  }

  @Builder FlowTimeline() {
    Column({ space: 10 }) {
      ForEach(this.flowBlocksState, (block: UIBlock, index: number) => {
        this.FlowBlock(block, index);
      }, (block: UIBlock, index: number) => this.flowBlockKey(block, index));
    }
    .width('100%');
  }

  private flowBlockKey(block: UIBlock, index: number): string {
    return block.kind + '|' + String(index) + '|' + block.phase + '|' + block.selectedId
      + '|' + (block.approved ? '1' : '0') + '|' + String(block.steps.length);
  }

  // 最后一个同类卡片是否为可交互目标（仅工作流激活且不在运行中时）
  private isLiveBlock(block: UIBlock, index: number): boolean {
    if (!this.flowActive() || this.pendingRunCountState > 0 || !this.chatHealthOkState) {
      return false;
    }
    // 只有时间线中最后一个 block 才是“当前”待回答项
    return index === this.flowBlocksState.length - 1;
  }

  @Builder FlowBlock(block: UIBlock, index: number) {
    if (block.kind === 'user-text') {
      this.FlowUserBubble(block.text, '');
    } else if (block.kind === 'user-option-answer') {
      this.FlowUserBubble(block.text, block.index);
    } else if (block.kind === 'agent-text') {
      this.FlowAgentBubble(block.markdown);
    } else if (block.kind === 'clarify-question') {
      Row() {
        IntentClarifyCard({
          block: block,
          totalQuestions: this.flowTotalQuestionsState,
          isLocked: !this.isLiveBlock(block, index),
          canInteract: this.isLiveBlock(block, index),
          onConfirmOption: (id: string, label: string) => {
            this.sendFlowOption(label);
          }
        })
          .width('86%');
        Blank().layoutWeight(1);
      }
      .width('100%');
    } else if (block.kind === 'approach-proposal') {
      Row() {
        ApproachProposalCard({
          block: block,
          isLocked: !this.isLiveBlock(block, index),
          canInteract: this.isLiveBlock(block, index),
          onConfirmOption: (id: string, label: string) => {
            this.sendFlowOption(id);
          }
        })
          .width('86%');
        Blank().layoutWeight(1);
      }
      .width('100%');
    } else if (block.kind === 'plan-confirm') {
      Row() {
        PlanConfirmCard({
          block: block,
          isLocked: !this.isLiveBlock(block, index),
          canInteract: this.isLiveBlock(block, index),
          onConfirm: () => {
            this.sendFlowOption('可以');
          },
          onModify: () => {
            this.composerInputVisible = true;
          }
        })
          .width('86%');
        Blank().layoutWeight(1);
      }
      .width('100%');
    } else if (block.kind === 'skill-create') {
      Row() {
        SkillCreateCard({ block: block })
          .width('86%');
        Blank().layoutWeight(1);
      }
      .width('100%');
    } else if (block.kind === 'exec-summary') {
      Row() {
        ExecuteSummaryCard({
          block: block,
          canInteract: this.chatHealthOkState && this.pendingRunCountState === 0,
          onCta: () => {
            this.showSkillLaunchers = false;
          }
        })
          .width('86%');
        Blank().layoutWeight(1);
      }
      .width('100%');
    }
  }

  @Builder FlowUserBubble(text: string, badge: string) {
    Row({ space: 8 }) {
      Blank().layoutWeight(1);
      Row({ space: 8 }) {
        if (badge.length > 0) {
          Stack({ alignContent: Alignment.Center }) {
            Circle().width(18).height(18).fill('#FFFFFF55');
            Text(badge)
              .fontSize(11)
              .fontWeight(FontWeight.Bold)
              .fontColor('#FFFFFF');
          }
          .width(18)
          .height(18);
        }
        Text(text)
          .fontSize(15)
          .fontColor('#FFFFFF')
          .lineHeight(21);
      }
      .constraintSize({ maxWidth: '80%' })
      .padding({ left: 14, right: 14, top: 10, bottom: 10 })
      .backgroundColor(C_PRIMARY)
      .borderRadius({ topLeft: 18, topRight: 18, bottomLeft: 18, bottomRight: 6 });
    }
    .width('100%');
  }

  @Builder FlowAgentBubble(markdown: string) {
    Row() {
      Column() {
        this.MarkdownMessage(markdown, false);
      }
      .constraintSize({ maxWidth: '86%' })
      .padding({ left: 14, right: 14, top: 10, bottom: 10 })
      .backgroundColor(C_CARD)
      .borderRadius({ topLeft: 18, topRight: 18, bottomLeft: 6, bottomRight: 18 })
      .border({ width: 1, color: C_BORDER });
      Blank().layoutWeight(1);
    }
    .width('100%');
  }

  private sendFlowOption(payload: string): void {
    const text = (payload || '').trim();
    if (text.length === 0 || this.pendingRunCountState > 0 || !this.chatHealthOkState) {
      return;
    }
    this.suppressComposerInputFocusRestore();
    this.viewModel.sendChat(text, this.chatThinkingLevelState, []);
  }

  @Builder EmptyState() {
    Column({ space: 8 }) {
      Text('💬')
        .fontSize(44);

      Text($r('app.string.chat_empty_title'))
        .fontSize(18)
        .fontWeight(FontWeight.Bold)
        .fontColor(C_FOREGROUND);

      Text($r('app.string.chat_empty_description'))
        .fontSize(14)
        .fontWeight(FontWeight.Medium)
        .fontColor(C_MUTED_FG)
        .textAlign(TextAlign.Center);
    }
    .width('100%')
    .padding({ top: 48, bottom: 48, left: 24, right: 24 })
    .justifyContent(FlexAlign.Center)
    .alignItems(HorizontalAlign.Center);
  }

  @Builder MessageItem(message: ChatMessage) {
    Row({ space: 10 }) {
      if (message.role === 'user') {
        Blank().layoutWeight(1);
        this.MessageBubble(message);
        this.MessageAvatar(true);
      } else {
        this.MessageAvatar(false);
        this.MessageBubble(message);
        Blank().layoutWeight(1);
      }
    }
    .width('100%')
    .padding({ top: 6, bottom: 6 })
    .alignItems(VerticalAlign.Top);
  }

  @Builder MessageBubble(message: ChatMessage) {
      Column({ space: 4 }) {
        this.MarkdownMessage(this.messageTextValue(message), message.role === 'user');
        // 渲染消息中的图片内容块（缩略图）
        ForEach(this.messageImageUris(message), (uri: string) => {
          Image(uri)
            .width('100%')
            .aspectRatio(1)
            .objectFit(ImageFit.Cover)
            .borderRadius(12)
            .backgroundColor(C_MUTED);
        });
      }
      .width(message.role === 'user' ? '72%' : '78%')
      .constraintSize({ minWidth: 0, maxWidth: message.role === 'user' ? '72%' : '78%' })
      .flexShrink(1)
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      .backgroundColor(message.role === 'user' ? C_USER_BUBBLE : C_CARD)
      .borderRadius(22)
      .shadow(message.role === 'user'
        ? { radius: 0, color: '#00000000', offsetX: 0, offsetY: 0 }
        : { radius: 10, color: '#0F000000', offsetX: 0, offsetY: 2 });
  }

  @Builder StreamingMessage(text: string) {
    Row({ space: 10 }) {
      this.MessageAvatar(false);

      Column({ space: 6 }) {
        Row() {
          Text('IntentX Agent')
            .fontSize(12)
            .fontWeight(FontWeight.Medium)
            .fontColor(C_MUTED_FG)
            .layoutWeight(1);

          Row({ space: 4 }) {
            Circle().width(6).height(6).fill(C_PRIMARY_SOFT);
            Circle().width(6).height(6).fill(C_PRIMARY_SOFT);
            Circle().width(6).height(6).fill(C_PRIMARY_SOFT);
          }
        }
        .width('100%');

        this.StreamingMarkdownContent(text);
      }
      .width('78%')
      .constraintSize({ minWidth: 0, maxWidth: '78%' })
      .flexShrink(1)
      .padding({ left: 16, right: 16, top: 12, bottom: 12 })
      .backgroundColor(C_CARD)
      .borderRadius(22)
      .shadow({ radius: 10, color: '#0F000000', offsetX: 0, offsetY: 2 });

      Blank().layoutWeight(1);
    }
    .width('100%')
    .alignItems(VerticalAlign.Top);
  }

  @Builder MessageAvatar(isUser: boolean) {
    Stack({ alignContent: Alignment.Center }) {
      Circle()
        .width(44)
        .height(44)
        .fill(C_PRIMARY);

      if (isUser) {
        Text($r('app.string.chat_user_avatar_text'))
          .fontSize(14)
          .fontWeight(FontWeight.Bold)
          .fontColor('#FFFFFF');
      } else {
        Image($r('app.media.lucide_bot'))
          .width(24)
          .height(24)
          .objectFit(ImageFit.Contain);
      }
    }
    .width(44)
    .height(44);
  }

  @Builder MarkdownMessage(text: string, isUser: boolean) {
    if (isUser || this.shouldUsePlainTextBubble(text)) {
      this.PlainTextMessage(text, isUser);
    } else {
      Markdown({
        controller: this.getMarkdownControllerForRole(isUser),
        text: this.bubbleSafeMarkdown(text)
      })
      .width('100%')
      .constraintSize({ minWidth: 0 });
    }
  }

  @Builder PlainTextMessage(text: string, isUser: boolean) {
    Text(text || '')
      .fontSize(16)
      .fontWeight(FontWeight.Medium)
      .fontColor(isUser ? C_USER_TEXT : C_FOREGROUND)
      .lineHeight(23)
      .width('100%')
      .wordBreak(WordBreak.BREAK_ALL);
  }

  @Builder StreamingMarkdownContent(text: string) {
    Markdown({
      controller: this.streamingMarkdownController,
      text: this.bubbleSafeMarkdown(text)
    });
  }

  @Builder PendingToolCallsRail() {
    Column({ space: 8 }) {
      Text($r('app.string.chat_tools_running'))
        .fontSize(12)
        .fontWeight(FontWeight.Medium)
        .fontColor(C_MUTED_FG)
        .margin({ left: 54 });

      ForEach(this.chatPendingToolCallsState, (toolCall: ChatPendingToolCall) => {
        Row() {
          Column({ space: 2 }) {
            Text(toolCall.name)
              .fontSize(14)
              .fontWeight(FontWeight.Bold)
              .fontColor(C_FOREGROUND);

            if ((toolCall.paramsJson || '').length > 0) {
              Text(toolCall.paramsJson)
                .fontSize(12)
                .fontWeight(FontWeight.Medium)
                .fontColor(C_MUTED_FG)
                .maxLines(3);
            }
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start);

          Text('…')
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor(C_PRIMARY);
        }
        .width('78%')
        .margin({ left: 54 })
        .padding({ left: 14, right: 14, top: 12, bottom: 12 })
        .backgroundColor(C_CARD)
        .borderRadius(16)
        .shadow({ radius: 10, color: '#0F000000', offsetX: 0, offsetY: 2 });
      });
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start);
  }

  @Builder AssistantChoiceOptions(options: ChatChoiceOption[]) {
    Column({ space: 8 }) {
      ForEach(options, (option: ChatChoiceOption) => {
        Button({ type: ButtonType.Normal }) {
          Row({ space: 8 }) {
            Text(String(option.index))
              .fontSize(13)
              .fontWeight(FontWeight.Bold)
              .fontColor(C_PRIMARY);

            Text(option.label)
              .fontSize(15)
              .fontWeight(FontWeight.Medium)
              .fontColor(C_FOREGROUND)
              .layoutWeight(1)
              .textAlign(TextAlign.Start)
              .maxLines(3);
          }
          .width('100%')
          .alignItems(VerticalAlign.Center);
        }
        .width('78%')
        .constraintSize({ minHeight: 46 })
        .margin({ left: 54 })
        .padding({ left: 14, right: 14, top: 10, bottom: 10 })
        .backgroundColor(C_CARD)
        .borderRadius(16)
        .shadow({ radius: 10, color: '#0F000000', offsetX: 0, offsetY: 2 })
        .enabled(this.pendingRunCountState === 0 && this.chatHealthOkState)
        .onClick(() => {
          this.sendChoiceOption(option);
        });
      }, (option: ChatChoiceOption) => String(option.index) + '|' + option.label);
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start);
  }

  @Builder PendingInteractionCards(interactions: ChatPendingInteraction[]) {
    Column({ space: 10 }) {
      ForEach(interactions, (interaction: ChatPendingInteraction) => {
        Column({ space: 8 }) {
          Text(interaction.title)
            .fontSize(16)
            .fontWeight(FontWeight.Bold)
            .fontColor(interaction.riskLevel === 'high' ? '#B42318' : C_FOREGROUND);

          if (interaction.body.length > 0) {
            Text(interaction.body)
              .fontSize(14)
              .fontColor('#3D4452')
              .maxLines(4);
          }

          Row({ space: 8 }) {
            ForEach(interaction.choices, (choice: ChatPendingInteractionChoice) => {
              Button(choice.label)
                .fontSize(14)
                .fontWeight(FontWeight.Medium)
                .fontColor(choice.style === 'danger' ? '#B42318' : '#FFFFFF')
                .backgroundColor(choice.style === 'danger' ? '#FFF1F0' : C_PRIMARY)
                .borderRadius(12)
                .enabled(interaction.status === 'pending')
                .onClick(() => {
                  this.sendPendingInteractionChoice(interaction, choice.value);
                });
            }, (choice: ChatPendingInteractionChoice) => choice.id + '|' + choice.label);
          }
        }
        .width('78%')
        .margin({ left: 54 })
        .padding({ left: 16, right: 16, top: 12, bottom: 12 })
        .backgroundColor(C_CARD)
        .borderRadius(16)
        .shadow({ radius: 10, color: '#0F000000', offsetX: 0, offsetY: 2 });
      }, (interaction: ChatPendingInteraction) => interaction.id);
    }
    .width('100%')
    .alignItems(HorizontalAlign.Start);
  }

  @Builder Composer() {
    Column({ space: 10 }) {
      Row() {
        Button({ type: ButtonType.Normal }) {
          Text(`${this.localizedThinking(this.chatThinkingLevelState)}`)
            .fontSize(13)
            .fontWeight(FontWeight.Medium)
            .fontColor(C_ACCENT_FG);
        }
        .height(34)
        .backgroundColor(C_ACCENT)
        .borderRadius(999)
        .padding({ left: 14, right: 14 })
        .onClick(() => {
          this.showThinkingOptions = !this.showThinkingOptions;
          this.showSessionList = false;
          this.showSkillLaunchers = false;
        });

        Blank();

        Button({ type: ButtonType.Normal }) {
          Text($r('app.string.chat_refresh'))
            .fontSize(13)
            .fontWeight(FontWeight.Medium)
            .fontColor(C_MUTED_FG);
        }
        .height(34)
        .backgroundColor(C_MUTED)
        .borderRadius(999)
        .padding({ left: 14, right: 14 })
        .onClick(() => {
          this.viewModel.refreshChat();
          this.viewModel.refreshChatSessions(200);
        });
      }
      .width('100%')
      .padding({ left: 4, right: 4 });

      if (this.showThinkingOptions) {
        this.ThinkingOptions();
      }

      if (this.showSkillLaunchers) {
        this.SkillLauncherPanel();
      }

      Row({ space: 10 }) {
        Button({ type: ButtonType.Normal }) {
          Text(this.selectedSkillLauncherIcon.length > 0 ? this.selectedSkillLauncherIcon : '/')
            .fontSize(this.selectedSkillLauncherIcon.length > 0 ? 16 : 16)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.canOpenSkillLaunchers() || this.selectedSkillLauncherPrefix.length > 0 ? C_PRIMARY : C_MUTED_FG);
        }
        .width(34)
        .height(34)
        .backgroundColor(C_ACCENT)
        .borderRadius(13)
        .enabled(this.canOpenSkillLaunchers() || this.selectedSkillLauncherPrefix.length > 0)
        .onClick(() => {
          if (this.selectedSkillLauncherPrefix.length > 0) {
            this.clearSkillLauncher();
            return;
          }
          this.showSkillLaunchers = !this.showSkillLaunchers;
          this.showThinkingOptions = false;
          this.showSessionList = false;
        });

        if (this.composerInputVisible) {
          TextInput({ placeholder: this.inputPlaceholder(), text: this.messageText })
            .layoutWeight(1)
            .height(34)
            .backgroundColor(Color.Transparent)
            .borderWidth(0)
            .padding({ left: 8, right: 8 })
            .fontSize(16)
            .fontColor(C_FOREGROUND)
            .placeholderColor(C_MUTED_FG)
            .enabled(this.chatHealthOkState && this.pendingRunCountState === 0)
            .onChange((value: string) => {
              this.messageText = value;
            });
        } else {
          Blank()
            .layoutWeight(1)
            .height(34);
        }

        Button({ type: ButtonType.Circle }) {
          Text(this.pendingRunCountState > 0 ? '■' : '➤')
            .fontSize(this.pendingRunCountState > 0 ? 14 : 18)
            .fontWeight(FontWeight.Bold)
            .fontColor(this.canSendMessage() || this.canAbortMessage() ? '#FFFFFF' : C_MUTED_FG)
            .textAlign(TextAlign.Center);
        }
        .width(34)
        .height(34)
        .backgroundColor(this.canSendMessage() || this.canAbortMessage() ? C_PRIMARY : C_MUTED)
        .enabled(this.canSendMessage() || this.canAbortMessage())
        .onClick(() => {
          if (this.canAbortMessage()) {
            this.abortMessage();
          } else {
            this.sendMessage();
          }
        });
      }
      .width('100%')
      .padding({ left: 6, right: 6, top: 6, bottom: 6 })
      .backgroundColor(C_INPUT_BG)
      .borderRadius(20);
    }
    .width('100%')
    .padding({ top: 8, bottom: 10 })
    .backgroundColor(C_BG);
  }

  @Builder SkillLauncherPanel() {
    Column({ space: 8 }) {
      Row() {
        Text(this.localizedText('chat_skill_launchers', 'Shortcuts'))
          .fontSize(15)
          .fontWeight(FontWeight.Bold)
          .fontColor(C_FOREGROUND);

        Blank();

        Text($r('app.string.chat_close'))
          .fontSize(16)
          .fontColor(C_MUTED_FG)
          .onClick(() => {
            this.showSkillLaunchers = false;
          });
      }
      .width('100%');

      Row({ space: 10 }) {
        Button({ type: ButtonType.Normal }) {
          Column({ space: 4 }) {
            Text('🆕')
              .fontSize(22);
            Text('创建新会话')
              .fontSize(12)
              .fontWeight(FontWeight.Bold)
              .fontColor(C_FOREGROUND);
          }
        }
        .width(92)
        .height(72)
        .backgroundColor(C_MUTED)
        .borderRadius(14)
        .onClick(() => {
          this.startNewSession();
        });

        Button({ type: ButtonType.Normal }) {
          Column({ space: 4 }) {
            Text('🧩')
              .fontSize(22);
            Text('个性化任务')
              .fontSize(12)
              .fontWeight(FontWeight.Bold)
              .fontColor(C_FOREGROUND);
          }
        }
        .width(92)
        .height(72)
        .backgroundColor(C_MUTED)
        .borderRadius(14)
        .onClick(() => {
          this.selectedSkillLauncherPrefix = '/genskills';
          this.selectedSkillLauncherIcon = '🧩';
          this.selectedSkillLauncherLabel = 'genSkills';
          this.showSkillLaunchers = false;
        });
      }
      .width('100%');
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 14, bottom: 14 })
    .backgroundColor(C_CARD)
    .borderRadius(18)
    .shadow({ radius: 10, color: '#0F000000', offsetX: 0, offsetY: 2 });
  }

  @Builder ThinkingOptions() {
    Column({ space: 8 }) {
      Text($r('app.string.chat_thinking_level'))
        .fontSize(15)
        .fontWeight(FontWeight.Bold)
        .fontColor(C_FOREGROUND);

      ForEach(['off', 'low', 'medium', 'high'], (level: string) => {
        Row() {
          Text(this.localizedThinkingLevel(level))
            .fontSize(15)
            .fontWeight(FontWeight.Medium)
            .fontColor(C_FOREGROUND);

          Blank();

          if (this.chatThinkingLevelState === level) {
            Text('✓')
              .fontSize(15)
              .fontColor(C_PRIMARY);
          }
        }
        .width('100%')
        .padding({ top: 8, bottom: 8 })
        .onClick(() => {
          this.viewModel.setChatThinkingLevel(level);
          this.showThinkingOptions = false;
        });
      });
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 14, bottom: 14 })
    .backgroundColor(C_CARD)
    .borderRadius(18)
    .shadow({ radius: 10, color: '#0F000000', offsetX: 0, offsetY: 2 });
  }

  @Builder SessionList() {
    Column({ space: 8 }) {
      Row() {
        Text($r('app.string.chat_sessions'))
          .fontSize(15)
          .fontWeight(FontWeight.Bold)
          .fontColor(C_FOREGROUND);

        Blank();

        Text($r('app.string.chat_close'))
          .fontSize(16)
          .fontColor(C_MUTED_FG)
          .onClick(() => {
            this.showSessionList = false;
          });
      }
      .width('100%');

      ForEach(this.sessionOptions(), (session: ChatSessionEntry) => {
        Row() {
          Column({ space: 2 }) {
            Text(this.friendlySessionName(session.displayName || session.key))
              .fontSize(15)
              .fontWeight(this.isActiveSession(session) ? FontWeight.Bold : FontWeight.Medium)
              .fontColor(this.isActiveSession(session) ? C_PRIMARY : C_FOREGROUND);

            if (session.updatedAtMs > 0) {
              Text(this.sessionUpdatedText(session.updatedAtMs))
                .fontSize(12)
                .fontWeight(FontWeight.Medium)
                .fontColor(C_MUTED_FG);
            }
          }
          .layoutWeight(1)
          .alignItems(HorizontalAlign.Start);

          if (this.isActiveSession(session)) {
            Text('✓')
              .fontSize(15)
              .fontColor(C_PRIMARY);
          }
        }
        .width('100%')
        .padding({ top: 8, bottom: 8 })
        .onClick(() => {
          this.viewModel.switchChatSession(session.key);
          this.showSessionList = false;
        });
      });
    }
    .width('100%')
    .padding({ left: 16, right: 16, top: 14, bottom: 14 })
    .backgroundColor(C_CARD)
    .borderRadius(18)
    .shadow({ radius: 10, color: '#0F000000', offsetX: 0, offsetY: 2 });
  }

  private canSendMessage(): boolean {
    return this.chatHealthOkState && this.pendingRunCountState === 0 && this.messageText.trim().length > 0;
  }

  private canOpenSkillLaunchers(): boolean {
    return this.chatHealthOkState && this.pendingRunCountState === 0;
  }

  private canAbortMessage(): boolean {
    return this.pendingRunCountState > 0;
  }

  private sendMessage(): void {
    if (!this.canSendMessage()) {
      return;
    }

    const message = this.messageText.trim();
    const launcherPrefix = this.selectedSkillLauncherPrefix.trim();
    const launcherIcon = this.selectedSkillLauncherIcon.trim();
    this.messageText = '';
    this.clearSkillLauncher();
    this.suppressComposerInputFocusRestore();

    if (launcherPrefix.length > 0) {
      this.viewModel.sendChat(`${launcherPrefix} ${message}`, this.chatThinkingLevelState, [], `${launcherIcon} ${message}`.trim());
      return;
    }

    this.viewModel.sendChat(message, this.chatThinkingLevelState, []);
  }

  private startNewSession(): void {
    if (!this.canOpenSkillLaunchers()) {
      return;
    }
    this.messageText = '';
    this.clearSkillLauncher();
    this.suppressComposerInputFocusRestore();
    this.viewModel.sendChat('/new', this.chatThinkingLevelState, [], '🆕 创建新会话');
  }

  private clearSkillLauncher(): void {
    this.selectedSkillLauncherPrefix = '';
    this.selectedSkillLauncherIcon = '';
    this.selectedSkillLauncherLabel = '';
    this.showSkillLaunchers = false;
  }

  private inputPlaceholder(): string {
    if (!this.chatHealthOkState) {
      return this.localizedText('chat_connect_placeholder', 'Connect to a gateway to chat');
    }
    if (this.selectedSkillLauncherPrefix.length > 0) {
      return this.localizedText('chat_skill_launcher_placeholder', 'Type the goal to save...');
    }
    return this.localizedText('chat_input_placeholder', 'Type a message...');
  }

  private abortMessage(): void {
    if (!this.canAbortMessage()) {
      return;
    }
    this.viewModel.abortChat();
  }

  private currentSessionLabel(): string {
    return this.friendlySessionName(this.chatSessionKeyState);
  }

  private copySessionKey(): void {
    const key = this.chatSessionKeyState;
    if (!key || key.length === 0) {
      return;
    }
    try {
      const data = pasteboard.createData(pasteboard.MIMETYPE_TEXT_PLAIN, key);
      pasteboard.getSystemPasteboard().setData(data);
    } catch (error) {
      console.warn('ChatPage: copySessionKey failed');
    }
  }

  private sessionOptions(): ChatSessionEntry[] {
    const existing: ChatSessionEntry[] = [...this.chatSessionsState];
    const hasCurrent = existing.findIndex((session: ChatSessionEntry) => session.key === this.chatSessionKeyState) >= 0;
    if (hasCurrent) {
      return existing;
    }
    const current = new ChatSessionEntry();
    current.key = this.chatSessionKeyState;
    current.displayName = this.chatSessionKeyState === this.viewModel.mainSessionKey ? 'Main' : this.chatSessionKeyState;
    current.updatedAtMs = Date.now();
    existing.unshift(current);
    return existing;
  }

  private syncChatState(): void {
    const wasPending = this.pendingRunCountState > 0;
    this.chatMessagesState = [...this.viewModel.chatMessages];
    this.chatStreamingTextState = this.viewModel.chatStreamingAssistantText || '';
    this.chatPendingToolCallsState = [...this.viewModel.chatPendingToolCalls];
    this.chatErrorState = this.viewModel.chatError || '';
    this.chatHealthOkState = this.viewModel.chatHealthOk;
    this.chatThinkingLevelState = this.viewModel.chatThinkingLevel || 'off';
    this.chatSessionKeyState = this.viewModel.chatSessionKey || this.viewModel.mainSessionKey;
    this.chatSessionsState = [...this.viewModel.chatSessions];
    this.pendingRunCountState = this.viewModel.pendingRunCount;
    this.pendingInteractionsState = [...this.viewModel.chatPendingInteractions];
    this.flowBlocksState = [...this.viewModel.chatFlow.blocks];
    this.flowPhaseState = this.viewModel.chatFlow.finalPhase;
    this.flowTotalQuestionsState = this.viewModel.chatFlow.totalQuestions;
    this.latestAssistantChoiceOptionsState = this.pendingInteractionsState.length === 0 ? this.extractChoiceOptions(this.latestAssistantText()) : [];
    if (wasPending && this.pendingRunCountState === 0) {
      this.restoreComposerInputAfterRun();
    }
    this.scheduleScrollToBottom();
  }

  private suppressComposerInputFocusRestore(): void {
    this.composerInputVisible = false;
  }

  private restoreComposerInputAfterRun(): void {
    setTimeout(() => {
      if (this.pendingRunCountState === 0) {
        this.composerInputVisible = true;
      }
    }, 160);
  }

  private configureMarkdownControllers(): void {
    this.configureMarkdownController(this.streamingMarkdownController, C_FOREGROUND, C_PRIMARY, C_MUTED, C_MUTED);
    this.streamingMarkdownController.setThreadRenderEnable(true);
    this.configureMarkdownController(this.userMarkdownController, C_USER_TEXT, '#0B4F22', '#9FE0AC', '#BCEBC6');
    this.configureMarkdownController(this.openclawMarkdownController, C_FOREGROUND, C_PRIMARY, C_MUTED, C_MUTED);
    this.userMarkdownController.setThreadRenderEnable(true);
    this.openclawMarkdownController.setThreadRenderEnable(true);
  }

  private configureMarkdownController(controller: MarkdownController,
    textColor: string,
    hyperlinkColor: string,
    inlineCodeBackgroundColor: string,
    quoteBackgroundColor: string): void {
    controller
      .setTextColor(textColor)
      .setTextSize(16)
      .setTextLineHeight(23)
      .setTitleColor(C_FOREGROUND)
      .setTitleSize([22, 20, 18, 17, 16, 15])
      .setTitleLineHeight([30, 28, 26, 24, 22, 21])
      .setTitleFontWeight(FontWeight.Bold)
      .setInlineCodeColor('#C2410C')
      .setInlineCodeBackgroundColor(inlineCodeBackgroundColor)
      .setInlineCodeBackgroundRadius(6)
      .setQuoteBorderColor(C_PRIMARY)
      .setQuoteBorderWidth(4)
      .setQuoteBorderRadius(12)
      .setQuoteBackgroundColor(quoteBackgroundColor)
      .setHyperlinkTextColor(hyperlinkColor)
      .setHyperlinkUnderlineState(true)
      .setCodeBlockBorderRadius(12)
      .setCodeBlockTheme('light')
      .setCodeBlockIdxState(false)
      .setImageBorderRadius(12)
      .setImageWidth('100%')
      .setTableAlign(FlexAlign.Start)
      .setTableBorderRadius(12)
      .setTableOuterBorderColor(C_BORDER);
  }

  private getMarkdownControllerForRole(isUser: boolean): MarkdownController {
    return isUser? this.userMarkdownController : this.openclawMarkdownController;
  }

  private scheduleInitialScroll(): void {
    setTimeout(() => {
      this.messageScroller.scrollEdge(Edge.Bottom);
    }, 0);
    setTimeout(() => {
      this.messageScroller.scrollEdge(Edge.Bottom);
    }, 80);
    setTimeout(() => {
      this.messageScroller.scrollEdge(Edge.Bottom);
    }, 220);
  }

  private scheduleScrollToBottom(): void {
    const scrollToken = String(this.chatMessagesState.length) +
      '|' + this.chatStreamingTextState +
      '|' + String(this.chatPendingToolCallsState.length) +
      '|' + String(this.pendingInteractionsState.length) +
      '|' + String(this.flowBlocksState.length) +
      '|' + this.flowPhaseState;
    if (scrollToken === this.lastScrollToken) {
      return;
    }
    this.lastScrollToken = scrollToken;
    setTimeout(() => {
      this.messageScroller.scrollEdge(Edge.Bottom);
    }, 0);
    setTimeout(() => {
      this.messageScroller.scrollEdge(Edge.Bottom);
    }, 60);
  }

  private friendlySessionName(name: string):string {
    if (!name || name.trim().length === 0) {
      return this.localizedText('common_main', 'Main');
    }
    return name === this.viewModel.mainSessionKey ? this.localizedText('common_main', 'Main') : name;
  }

  private bubbleSafeMarkdown(text: string): string {
    return text || '';
  }

  private latestAssistantText(): string {
    if (this.chatStreamingTextState.trim().length > 0) {
      return this.chatStreamingTextState;
    }
    for (let i = this.chatMessagesState.length - 1; i >= 0; i--) {
      const message = this.chatMessagesState[i];
      if (message.role === 'assistant') {
        return this.messageTextValue(message);
      }
    }
    return '';
  }

  private extractChoiceOptions(text: string): ChatChoiceOption[] {
    const options: ChatChoiceOption[] = [];
    const lines = (text || '').split('\n');
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();
      const match = line.match(/^([1-9])[\.\)、)]\s*(.+)$/);
      if (!match || match.length < 3) {
        continue;
      }
      const option = new ChatChoiceOption();
      option.index = Number(match[1]);
      option.label = match[2].trim();
      option.value = String(option.index) + '. ' + option.label;
      if (option.label.length > 0) {
        options.push(option);
      }
    }
    if (options.length < 2) {
      return [];
    }
    return options;
  }

  private sendChoiceOption(option: ChatChoiceOption): void {
    if (this.pendingRunCountState > 0 || !this.chatHealthOkState || option.value.trim().length === 0) {
      return;
    }
    this.latestAssistantChoiceOptionsState = [];
    this.suppressComposerInputFocusRestore();
    this.viewModel.sendChat(option.value, this.chatThinkingLevelState, []);
  }

  private sendPendingInteractionChoice(interaction: ChatPendingInteraction, value: string): void {
    if (!this.chatHealthOkState || value.trim().length === 0) {
      return;
    }
    this.suppressComposerInputFocusRestore();
    this.viewModel.respondPendingInteraction(interaction, value);
  }

  private shouldUsePlainTextBubble(text: string): boolean {
    if (!text || text.length === 0) {
      return true;
    }
    return /[A-Za-z0-9_./:-]{24,}/.test(text);
  }

  private localizedThinkingLevel(level: string): string {
    switch (level) {
      case 'off':
        return this.localizedText('chat_thinking_level_off', 'Off');
      case 'low':
        return this.localizedText('chat_thinking_level_low', 'Low');
      case 'medium':
        return this.localizedText('chat_thinking_level_medium', 'Medium');
      case 'high':
        return this.localizedText('chat_thinking_level_high', 'High');
      default:
        return level;
    }
  }

  private localizedThinking(level: string): string {
    return this.localizedText('chat_thinking_prefix', 'Thinking') + ' : ' + this.localizedThinkingLevel(level);
  }

  private messageListKey(message: ChatMessage, index: number): string {
    if ((message.stableKey || '').length > 0) {
      return message.stableKey;
    }

    const contentText = this.messageTextValue(message);
    if (contentText.length > 0) {
      return message.role + '|' + contentText + '|' + String(index);
    }
    return message.role + '|index|' + String(index);
  }

  private titleCase(value: string): string {
    if (value.length === 0) {
      return value;
    }
    return value.charAt(0).toUpperCase() + value.slice(1);
  }

  private sessionUpdatedText(updatedAtMs: number) {
    const now = Date.now();
    const diffMs = now - updatedAtMs;
    const diffMinutes = Math.floor(diffMs / (1000 * 60));
    const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
    const justNowText = this.localizedText('chat_updated_just_now', 'Updated just now');
    const minutesAgoText = this.localizedText('chat_updated_minutes_ago', 'm ago');
    const hoursAgoText = this.localizedText('chat_updated_hours_ago', 'h ago');

    if (diffMinutes < 1) {
      return justNowText;
    } else if (diffHours < 1) {
      return `${diffMinutes}${minutesAgoText}`;
    } else {
      return `${diffHours}${hoursAgoText}`;
    }
  }

  private localizedText(name: string, fallback: string): string {
    try {
      void this.localeVersion;
      return getContext(this).resourceManager.getStringByNameSync(name);
    } catch (error) {
      console.warn('ChatPage: localizedText failed for key: ' + name);
      return fallback;
    }
  }

  private messageTextValue(message: ChatMessage): string {
    if (!message.content || message.content.length === 0) {
      return '';
    }

    const textParts: string[] = [];
    for (let i = 0; i < message.content.length; i++) {
      const block = message.content[i];
      const blockType = (block.type || '').trim();
      const blockText = (block.text || '').trim();

      if (blockType === 'text' && blockText.length > 0) {
        textParts.push(blockText);
        continue;
      }

      if (blockType.length > 0 && blockType !== 'text') {
        textParts.push(`[${blockType}]`);
      }
    }

    return textParts.join('\n\n');
  }

  private messageImageUris(message: ChatMessage): string[] {
    if (!message.content || message.content.length === 0) {
      return [];
    }
    const uris: string[] = [];
    for (let i = 0; i < message.content.length; i++) {
      const block = message.content[i];
      if (block.type === 'image' && (block.imageUri || '').length > 0) {
        uris.push(block.imageUri as string);
      }
    }
    return uris;
  }
}
