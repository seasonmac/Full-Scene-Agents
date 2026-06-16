import { GatewaySession, GatewayConnectOptions, GatewaySessionLike } from '../api/GatewaySession';
import { HermesGatewaySession, UnavailableGatewaySession } from '../api/HermesGatewaySession';
import { deviceAuthStore } from '../api/DeviceAuthStore';
import { fileIo } from '@kit.CoreFileKit';
import { deviceIdentityStore } from '../api/DeviceIdentityStore';
import { securePrefs, LocationMode, GatewayMode, HERMES_DEFAULT_HOST, HERMES_DEFAULT_PORT } from '../common/SecurePrefs';
import { HarmonyNodeRuntimeFlags, InvokeCommandRegistry } from '../node/InvokeCommandRegistry';
import { NodeInvokeDispatcher, NodeInvokeDispatcherCallbacks } from '../node/NodeInvokeDispatcher';
import {
  GatewayEndpoint,
  ChatMessage,
  ChatSessionEntry,
  ChatPendingToolCall,
  ChatPendingInteraction,
  ChatPendingInteractionBackend,
  ChatPendingInteractionChoice,
  ChatPendingInteractionInput,
  OutgoingAttachment,
  VoiceConversationEntry,
  ChatMessageContent
} from '../model/GatewayModels';
import { RawTurn, ReplayResult, UIBlock } from '../model/FlowModels';
import { StreamUIAdapter } from '../service/StreamUIAdapter';
import { A2UIParser } from '../service/A2UIParser';
import { A2UIMapper } from '../service/A2UIMapper';
import { A2UI_CARD_CAPABILITY, A2UIScanResult, A2UIValidationError } from '../model/A2UIModels';

export class CameraHudInfo {
  active: boolean = false;
}

class ConnectionStateListenerEntry {
  id: number = 0;
  listener: () => void = () => {};
}

class ChatStateListenerEntry {
  id: number = 0;
  listener: () => void = () => {};
}

class ChatHistoryRequestParams {
  sessionKey: string = '';
}

class ChatSessionsListParams {
  includeGlobal: boolean = true;
  includeUnknown: boolean = false;
  limit?: number = undefined;
}

class ChatPayloadObject {
}

class ChatSendRequestParams {
  sessionKey: string = '';
  message: string = '';
  thinking: string = 'off';
  timeoutMs: number = 30000;
  idempotencyKey: string = '';
  attachments?: OutgoingAttachment[] = undefined;
}

class ChatAbortRequestParams {
  sessionKey: string = '';
  runId: string = '';
}

class SetupCodeConnectionParams {
  host: string = '';
  port: number = 18789;
  tlsEnabled: boolean = false;
  token: string = '';
  bootstrapToken: string = '';
  password: string = '';
}

@Observed
export class MainViewModel {
  gateways: GatewayEndpoint[] = [];
  discoveryStatusText: string = 'Initializing';
  isConnected: boolean = false;
  isNodeConnected: boolean = false;
  statusText: string = 'Offline';
  serverName: string | null = null;
  remoteAddress: string | null = null;
  isForeground: boolean = true;

  mainSessionKey: string = 'main';
  chatSessionKey: string = 'main';
  chatSessionId: string | null = null;

  chatMessages: ChatMessage[] = [];
  chatError: string | null = null;
  chatHealthOk: boolean = false;
  chatThinkingLevel: string = 'medium';
  chatStreamingAssistantText: string | null = null;
  chatPendingToolCalls: ChatPendingToolCall[] = [];
  chatPendingInteractions: ChatPendingInteraction[] = [];
  chatSessions: ChatSessionEntry[] = [];
  pendingRunCount: number = 0;

  /** 有序对话轮时间线（user/agent/tool），供 StreamUIAdapter 解析结构化 UI。 */
  chatTurns: RawTurn[] = [];
  /** 解析结果：结构化 UIBlock 序列 + 当前阶段。 */
  chatFlow: ReplayResult = new ReplayResult();

  canvasCurrentUrl: string | null = null;
  canvasA2uiHydrated: boolean = false;
  canvasRehydratePending: boolean = false;
  canvasRehydrateErrorText: string | null = null;

  cameraHud: CameraHudInfo | null = null;
  cameraFlashToken: number = 0;
  screenRecordActive: boolean = false;

  micEnabled: boolean = false;
  micCooldown: boolean = false;
  micStatusText: string = 'Off';
  micLiveTranscript: string | null = null;
  micIsListening: boolean = false;
  micQueuedMessages: string[] = [];
  micConversation: VoiceConversationEntry[] = [];
  micInputLevel: number = 0;
  micIsSending: boolean = false;
  speakerEnabled: boolean = false;

  instanceId: string = '';
  displayName: string = '';
  cameraEnabled: boolean = false;
  locationMode: string = 'off';
  locationPreciseEnabled: boolean = false;
  preventSleep: boolean = false;
  manualEnabled: boolean = false;
  manualHost: string = '';
  manualPort: number = 18789;
  manualTls: boolean = false;
  gatewayMode: string = GatewayMode.Hermes;
  hermesHost: string = '';
  hermesPort: number = 0;
  hermesTls: boolean = true;
  hermesApiToken: string = '';
  gatewayToken: string = '';
  gatewayBootstrapToken: string = '';
  gatewayPassword: string = '';
  onboardingCompleted: boolean = false;
  canvasDebugStatusEnabled: boolean = false;

  private operatorSession: GatewaySessionLike | null = null;
  private nodeSession: GatewaySessionLike | null = null;
  private nodeInvokeDispatcher: NodeInvokeDispatcher;
  private connectedEndpoint: GatewayEndpoint | null = null;
  private ready: boolean = false;
  private initPromise: Promise<void>;
  private connectAttemptId: number = 0;
  private connectionStateListenerId: number = 0;
  private connectionStateListeners: ConnectionStateListenerEntry[] = [];
  private chatStateListenerId: number = 0;
  private chatStateListeners: ChatStateListenerEntry[] = [];
  private chatBootstrapId: number = 0;
  private chatSessionsRequestId: number = 0;
  private chatPendingRunIds: string[] = [];
  private chatPendingRunTimeoutMs: number = 120000;
  private statusTextResourceName: string = 'chat_offline';
  private discoveryStatusResourceName: string = '';
  private micStatusResourceName: string = 'settings_off';

  constructor() {
    this.nodeInvokeDispatcher = new NodeInvokeDispatcher(new MainViewModelNodeCallbacks(this));
    this.initPromise = this.initialize();
  }

  static waitUntilReadyTask(viewModel: MainViewModel): Promise<void> {
    return viewModel.initPromise;
  }

  static connectTask(viewModel: MainViewModel, endpoint: GatewayEndpoint): Promise<void> {
    return Promise.resolve().then((): Promise<void> => {
      if (!viewModel.operatorSession || !viewModel.nodeSession) {
        console.warn('MainViewModel: connectTask skipped because gateway session is not ready');
        viewModel.setStatusTextFromResource('common_status_gateway_session_not_ready', 'Gateway session not ready');
        return Promise.resolve();
      }
      return viewModel.connect(endpoint);
    });
  }

  static connectManualTask(viewModel: MainViewModel): Promise<void> {
    return Promise.resolve().then((): Promise<void> => {
      const trimmedHost = viewModel.manualHost.trim();
      console.info(
        'MainViewModel: connectManual host=' + trimmedHost + ', port=' + viewModel.manualPort + ', tls=' + viewModel.manualTls
      );
      if (!trimmedHost || viewModel.manualPort < 1 || viewModel.manualPort > 65535) {
        console.warn('MainViewModel: connectManual rejected due to invalid configuration');
        viewModel.setStatusTextFromResource('common_status_invalid_manual_config', 'Invalid manual gateway configuration');
        return Promise.resolve();
      }

      return viewModel.setGatewayMode(GatewayMode.OpenClaw).then((): Promise<void> => {
        return viewModel.setManualEnabled(true);
      }).then((): Promise<void> => {
        return viewModel.setManualHost(trimmedHost);
      }).then((): Promise<void> => {
        return viewModel.setManualPort(viewModel.manualPort);
      }).then((): Promise<void> => {
        return viewModel.setManualTls(viewModel.manualTls);
      }).then((): Promise<void> => {
        return viewModel.setGatewayToken(viewModel.gatewayToken.trim());
      }).then((): Promise<void> => {
        return viewModel.setGatewayBootstrapToken(viewModel.gatewayBootstrapToken.trim());
      }).then((): Promise<void> => {
        return viewModel.setGatewayPassword(viewModel.gatewayPassword.trim());
      }).then((): Promise<void> => {
        const endpoint = viewModel.createEndpoint(trimmedHost, viewModel.manualPort, viewModel.manualTls, `${trimmedHost}:${viewModel.manualPort}`);
        return MainViewModel.connectTask(viewModel, endpoint);
      });
    });
  }

  static connectHermesTask(viewModel: MainViewModel): Promise<void> {
    return Promise.resolve().then((): Promise<void> => {
      const trimmedHost = MainViewModel.resolveHermesHost(viewModel.hermesHost);
      const resolvedPort = MainViewModel.resolveHermesPort(viewModel.hermesPort);
      const trimmedToken = viewModel.hermesApiToken.trim();
      console.info(
        'MainViewModel: connectHermes host=' + trimmedHost + ', port=' + resolvedPort + ', tls=' + viewModel.hermesTls
      );
      if (resolvedPort < 1 || resolvedPort > 65535 || trimmedToken.length === 0) {
        console.warn('MainViewModel: connectHermes rejected due to invalid configuration');
        viewModel.setStatusTextFromResource('common_status_invalid_hermes_config', 'Invalid Hermes gateway configuration');
        return Promise.resolve();
      }

      return viewModel.setGatewayMode(GatewayMode.Hermes).then((): Promise<void> => {
        viewModel.hermesHost = trimmedHost;
        viewModel.hermesPort = resolvedPort;
        viewModel.hermesApiToken = trimmedToken;
        const endpoint = viewModel.createEndpoint(trimmedHost, resolvedPort, viewModel.hermesTls, `${trimmedHost}:${resolvedPort}`);
        return MainViewModel.connectTask(viewModel, endpoint);
      }).then((): Promise<void> => {
        if (!viewModel.isConnected) {
          return Promise.resolve();
        }
        return viewModel.persistHermesConnection(trimmedHost, resolvedPort, viewModel.hermesTls, trimmedToken);
      });
    });
  }

  static connectFromSetupCodeTask(viewModel: MainViewModel, setupCode: string): Promise<boolean> {
    return Promise.resolve().then((): Promise<boolean> => {
      const trimmed = setupCode.trim();
      console.info('MainViewModel: connectFromSetupCode start, length=' + trimmed.length);
      if (trimmed.length === 0) {
        console.warn('MainViewModel: connectFromSetupCode missing setup code');
        viewModel.setStatusTextFromResource('common_status_missing_setup_code', 'Missing setup code');
        return Promise.resolve(false);
      }

      const connectionParams = MainViewModel.parseSetupCodeConnection(trimmed);
      if (!connectionParams) {
        viewModel.setStatusTextFromResource('common_status_unsupported_setup_code', 'Unsupported setup code format. Use a gateway URL or host:port for now.');
        console.warn('MainViewModel: connectFromSetupCode unsupported format');
        return Promise.resolve(false);
      }

      console.info(
        'MainViewModel: connectFromSetupCode parsed host=' + connectionParams.host +
        ', port=' + connectionParams.port +
        ', tls=' + connectionParams.tlsEnabled
      );

      return MainViewModel.applySetupCodeConnection(viewModel, connectionParams).then((): Promise<boolean> => {
        viewModel.seedMockDiscovery();
        return MainViewModel.connectManualTask(viewModel).then((): boolean => {
          console.info('MainViewModel: connectFromSetupCode end, connected=' + viewModel.isConnected + ', status=' + viewModel.statusText);
          return viewModel.isConnected;
        });
      });
    });
  }

  static disconnectTask(viewModel: MainViewModel): Promise<void> {
    return viewModel.disconnect();
  }

  static refreshGatewayConnectionTask(viewModel: MainViewModel): Promise<void> {
    if (!viewModel.connectedEndpoint) {
      return Promise.resolve();
    }

    const endpoint = viewModel.connectedEndpoint;
    return MainViewModel.disconnectTask(viewModel).then((): Promise<void> => {
      return MainViewModel.connectTask(viewModel, endpoint);
    });
  }

  waitUntilReady(): Promise<void> {
    return this.initPromise;
  }

  async initialize(): Promise<void> {
    if (this.ready) {
      return;
    }

    try {
      await this.loadPreferences();

      this.configureGatewaySessions();

      this.seedMockDiscovery();
      this.startDiscovery();
      this.ready = true;
    } catch (error) {
      console.error('MainViewModel: Initialize error:', error);
      this.setStatusTextFromResource('common_status_initialization_failed', 'Initialization failed');
      this.setDiscoveryStatusFromResource('common_status_initialization_failed', 'Initialization failed');
    }
  }

  private async loadPreferences(): Promise<void> {
    this.instanceId = await securePrefs.getInstanceId();
    this.displayName = await securePrefs.getDisplayName();
    this.cameraEnabled = await securePrefs.getCameraEnabled();
    this.locationMode = await securePrefs.getLocationMode();
    this.locationPreciseEnabled = await securePrefs.getLocationPreciseEnabled();
    this.preventSleep = await securePrefs.getPreventSleep();
    this.micEnabled = await securePrefs.getTalkEnabled();
    this.speakerEnabled = await securePrefs.getSpeakerEnabled();
    this.manualEnabled = await securePrefs.getManualEnabled();
    this.manualHost = await securePrefs.getManualHost();
    this.manualPort = await securePrefs.getManualPort();
    this.manualTls = await securePrefs.getManualTls();
    this.gatewayMode = await securePrefs.getGatewayMode();
    this.hermesHost = await securePrefs.getHermesHost();
    this.hermesPort = await securePrefs.getHermesPort();
    this.hermesTls = await securePrefs.getHermesTls();
    this.hermesApiToken = await securePrefs.getHermesApiToken();
    this.gatewayToken = await securePrefs.getGatewayToken();
    this.gatewayBootstrapToken = await securePrefs.getGatewayBootstrapToken();
    this.gatewayPassword = await securePrefs.getGatewayPassword();
    this.onboardingCompleted = await securePrefs.getOnboardingCompleted();
    this.canvasDebugStatusEnabled = await securePrefs.getCanvasDebugStatusEnabled();
  }

  private configureGatewaySessions(): void {
    if (this.gatewayMode === GatewayMode.Hermes) {
      this.operatorSession = new HermesGatewaySession();
      this.nodeSession = new UnavailableGatewaySession();
    } else {
      this.operatorSession = new GatewaySession(deviceIdentityStore, deviceAuthStore);
      this.nodeSession = new GatewaySession(deviceIdentityStore, deviceAuthStore);
    }

    this.operatorSession.setHandlers({
      onConnected: (name: string, remote: string, mainSessionKey: string): void => {
        this.handleConnected(name, remote, mainSessionKey);
      },
      onDisconnected: (message: string): void => {
        this.handleDisconnected(message);
      },
      onEvent: (event: string, payloadJson: string): void => {
        this.handleEvent(event, payloadJson);
      },
    });

    this.nodeSession.setHandlers({
      onConnected: (_name: string, _remote: string, _mainSessionKey: string): void => {
        this.handleNodeConnected();
      },
      onDisconnected: (message: string): void => {
        this.handleNodeDisconnected(message);
      },
      onEvent: (_event: string, _payloadJson: string): void => {},
      onInvoke: async (request) => {
        return this.nodeInvokeDispatcher.handleInvoke(request);
      }
    });
  }

  private seedMockDiscovery(): void {
    const endpoints: GatewayEndpoint[] = [];

    if (this.gatewayMode === GatewayMode.Hermes) {
      const hermesHost = MainViewModel.resolveHermesHost(this.hermesHost);
      const hermesPort = MainViewModel.resolveHermesPort(this.hermesPort);
      if (hermesPort >= 1 && hermesPort <= 65535) {
        endpoints.push(this.createEndpoint(hermesHost, hermesPort, this.hermesTls, 'Hermes Gateway'));
      }
    }

    if (this.gatewayMode === GatewayMode.OpenClaw && this.manualHost.trim().length > 0) {
      endpoints.push(this.createEndpoint(this.manualHost, this.manualPort, this.manualTls, this.getLocalizedString('connect_saved_gateway', 'Saved Gateway')));
    }

    const deduped = new Map<string, GatewayEndpoint>();
    endpoints.forEach((endpoint: GatewayEndpoint) => {
      deduped.set(endpoint.stableId, endpoint);
    });
    this.gateways = Array.from(deduped.values());
  }

  private createEndpoint(host: string, port: number, tlsEnabled: boolean, name: string): GatewayEndpoint {
    const endpoint = new GatewayEndpoint();
    const trimmedHost = host.trim();
    endpoint.host = trimmedHost;
    endpoint.port = port;
    endpoint.tlsEnabled = tlsEnabled;
    endpoint.name = name;
    endpoint.stableId = `manual|${trimmedHost.toLowerCase()}|${port}|${tlsEnabled ? 'tls' : 'plain'}`;
    return endpoint;
  }

  private static resolveHermesHost(value: string): string {
    const trimmedHost = value.trim();
    return trimmedHost.length > 0 ? trimmedHost : HERMES_DEFAULT_HOST;
  }

  private static resolveHermesPort(value: number): number {
    return value > 0 ? value : HERMES_DEFAULT_PORT;
  }

  private async persistHermesConnection(host: string, port: number, tlsEnabled: boolean, apiToken: string): Promise<void> {
    await this.setHermesHost(host);
    await this.setHermesPort(port);
    await this.setHermesTls(tlsEnabled);
    await this.setHermesApiToken(apiToken);
  }

  private async startDiscovery(): Promise<void> {
    this.setDiscoveryStatusFromResource('connect_discovery_ready', 'Discovery ready');
  }

  private createTextContent(text: string): ChatMessageContent {
    const content = new ChatMessageContent();
    content.type = 'text';
    content.text = text;
    return content;
  }

  private createImageContent(dataUrl: string): ChatMessageContent {
    const content = new ChatMessageContent();
    content.type = 'image';
    content.imageUri = dataUrl;
    return content;
  }

  private handleConnected(name: string, remote: string, mainSessionKey: string): void {
    this.isConnected = true;
    this.serverName = name;
    this.remoteAddress = remote;
    this.mainSessionKey = mainSessionKey;
    this.chatSessionKey = mainSessionKey;
    this.chatHealthOk = true;
    this.setStatusTextFromResource('chat_connected', 'Connected');
    this.chatError = null;
    this.chatSessionId = null;
    this.chatMessages = [];
    this.chatTurns = [];
    this.chatFlow = new ReplayResult();
    this.a2uiBlocks = [];
    this.a2uiAutoRetryUsed = false;
    this.notifyChatStateChanged();

    // 将连接端点写入文件，供 ShareReceiveExtension 等扩展能力读取
    this.persistEndpointToFile(this.connectedEndpoint);

    if (this.chatSessions.findIndex((entry: ChatSessionEntry) => entry.key === mainSessionKey) === -1) {
      const session = new ChatSessionEntry();
      session.key = mainSessionKey;
      session.displayName = this.getLocalizedString('common_main', 'Main');
      session.updatedAtMs = Date.now();
      this.chatSessions = [session, ...this.chatSessions];
    }

    this.notifyConnectionStateChanged();
    this.bootstrapChat(true);
  }

  private handleDisconnected(message: string): void {
    this.isConnected = false;
    this.chatHealthOk = false;
    this.serverName = null;
    this.remoteAddress = null;
    this.statusText = this.isNodeConnected ? 'Connected (operator offline)' : message;
    this.chatStreamingAssistantText = null;
    // 清除共享的连接端点文件
    this.persistEndpointToFile(null);
    this.chatPendingToolCalls = [];
    this.clearChatPendingRuns();
    this.notifyChatStateChanged();
    this.notifyConnectionStateChanged();
  }

  /**
   * 将网关端点信息持久化到文件，供同一 app 内其他上下文（如 ShareExtension）读取。
   * @param endpoint 要保存的端点，传 null 清除文件
   */
  private persistEndpointToFile(endpoint: GatewayEndpoint | null): void {
    try {
      const ctx = AppStorage.get<Context>('abilityContext');
      if (!ctx) {
        console.warn('MainViewModel: cannot persist endpoint, no abilityContext');
        return;
      }
      const filePath = ctx.filesDir + '/gateway_endpoint.json';
      if (endpoint === null) {
        // 删除文件（文件不存在时忽略错误）
        try {
          fileIo.unlinkSync(filePath);
        } catch (_e) {
          // 文件可能不存在，忽略
        }
        return;
      }
      const json = JSON.stringify({
        host: endpoint.host,
        port: endpoint.port,
        tlsEnabled: endpoint.tlsEnabled,
        stableId: endpoint.stableId
      });
      const file = fileIo.openSync(filePath, fileIo.OpenMode.CREATE | fileIo.OpenMode.TRUNC | fileIo.OpenMode.WRITE_ONLY);
      try {
        fileIo.writeSync(file.fd, json);
      } finally {
        fileIo.closeSync(file);
      }
    } catch (err) {
      console.warn('MainViewModel: persistEndpointToFile failed: ' + (err as Error).message);
    }
  }

  private handleNodeConnected(): void {
    this.isNodeConnected = true;
    const canvasHostUrl = this.nodeSession ? this.nodeSession.currentCanvasHostUrl() : null;
    if (canvasHostUrl && canvasHostUrl.trim().length > 0) {
      this.canvasCurrentUrl = canvasHostUrl;
    }
    if (!this.isConnected) {
      this.statusText = 'Connected (operator offline)';
    } else {
      this.setStatusTextFromResource('chat_connected', 'Connected');
    }
    this.canvasRehydrateErrorText = null;
    this.notifyConnectionStateChanged();
  }

  private handleNodeDisconnected(message: string): void {
    this.isNodeConnected = false;
    this.canvasA2uiHydrated = false;
    this.canvasRehydratePending = false;
    if (this.isConnected) {
      this.statusText = 'Connected (node offline)';
    } else {
      this.statusText = message;
    }
    this.notifyConnectionStateChanged();
  }

  addConnectionStateListener(listener: () => void): number {
    const entry = new ConnectionStateListenerEntry();
    entry.id = ++this.connectionStateListenerId;
    entry.listener = listener;
    this.connectionStateListeners = [...this.connectionStateListeners, entry];
    return entry.id;
  }

  removeConnectionStateListener(listenerId: number): void {
    this.connectionStateListeners = this.connectionStateListeners.filter((entry: ConnectionStateListenerEntry): boolean => {
      return entry.id !== listenerId;
    });
  }

  private notifyConnectionStateChanged(): void {
    for (let i = 0; i < this.connectionStateListeners.length; i++) {
      const entry = this.connectionStateListeners[i];
      try {
        entry.listener();
      } catch (error) {
        console.warn('MainViewModel: connection state listener failed:', error);
      }
    }
  }

  addChatStateListener(listener: () => void): number {
    const entry = new ChatStateListenerEntry();
    entry.id = ++this.chatStateListenerId;
    entry.listener = listener;
    this.chatStateListeners = [...this.chatStateListeners, entry];
    return entry.id;
  }

  removeChatStateListener(listenerId: number): void {
    this.chatStateListeners = this.chatStateListeners.filter((entry: ChatStateListenerEntry): boolean => {
      return entry.id !== listenerId;
    });
  }

  private notifyChatStateChanged(): void {
    for (let i = 0; i < this.chatStateListeners.length; i++) {
      const entry = this.chatStateListeners[i];
      try {
        entry.listener();
      } catch (error) {
        console.warn('MainViewModel: chat state listener failed:', error);
      }
    }
  }

  private handleEvent(event: string, payloadJson: string): void {
    console.info('MainViewModel: Received event:', event, payloadJson);

    switch (event) {
      case 'tick':
        this.handleTickEvent();
        break;
      case 'health':
        this.handleHealthEvent(payloadJson);
        break;
      case 'seqGap':
        this.handleSeqGapEvent();
        break;
      case 'chat':
        this.handleChatEvent(payloadJson);
        break;
      case 'agent':
        this.handleAgentEvent(payloadJson);
        break;
      case 'chat.message':
        this.handleChatMessage(payloadJson);
        break;
      case 'chat.session':
        this.handleChatSession(payloadJson);
        break;
      case 'clarify.request':
      case 'approval.request':
      case 'sudo.request':
      case 'secret.request':
        this.handlePendingInteractionEvent(event, payloadJson);
        break;
      case 'canvas.update':
        this.handleCanvasUpdate(payloadJson);
        break;
      default:
        console.info('MainViewModel: Unknown event:', event);
        break;
    }
  }

  private handleHealthEvent(payloadJson: string): void {
    const payload = this.parseJsonObject(payloadJson);
    this.chatHealthOk = this.deriveHealthState(payload, true);
    if (this.isConnected) {
      this.setStatusTextFromResource('chat_connected', 'Connected');
    }
    this.notifyChatStateChanged();
  }

  private handleTickEvent(): void {
    if (!this.isConnected || !this.operatorSession) {
      return;
    }

    this.operatorSession.request('health').then((response) => {
      const payload = this.readPayloadObject(response.payload);
      this.chatHealthOk = this.deriveHealthState(payload, response.result.success);
      this.notifyChatStateChanged();
    }).catch(() => {
      this.chatHealthOk = false;
      this.notifyChatStateChanged();
    });
  }

  private handleSeqGapEvent(): void {
    this.chatError = this.getLocalizedString('chat_error_seq_gap', 'Event stream interrupted, please refresh and try again.');
    this.chatStreamingAssistantText = null;
    this.chatPendingToolCalls = [];
    this.chatPendingInteractions = [];
    this.clearChatPendingRuns();
    this.notifyChatStateChanged();
  }

  private async handleInvoke(command: string, paramsJson: string): Promise<void> {
    console.info('MainViewModel: Received invoke:', command, paramsJson);
  }

  private handleChatMessage(payloadJson: string): void {
    try {
      const message = JSON.parse(payloadJson) as ChatMessage;
      this.chatMessages = [...this.chatMessages, message];
      this.notifyChatStateChanged();
    } catch (error) {
      console.error('MainViewModel: Parse chat message error:', error);
    }
  }

  private handleChatSession(payloadJson: string): void {
    try {
      const session = JSON.parse(payloadJson) as ChatSessionEntry;
      this.chatSessions = [...this.chatSessions, session];
      this.notifyChatStateChanged();
    } catch (error) {
      console.error('MainViewModel: Parse chat session error:', error);
    }
  }

  private handlePendingInteractionEvent(event: string, payloadJson: string): void {
    const eventPayload = this.parseJsonObject(payloadJson);
    const nestedPayload = this.readNestedObject(eventPayload, 'payload');
    const payload = Object.keys(nestedPayload).length > 0 ? nestedPayload : eventPayload;
    const rawSessionKey =
      this.readOptionalString(eventPayload, 'sessionKey') ||
      this.readOptionalString(eventPayload, 'session_id') ||
      this.readOptionalString(payload, 'sessionKey') ||
      this.readOptionalString(payload, 'session_id');
    const sessionKey = this.normalizeChatSessionKey(rawSessionKey);
    if (rawSessionKey.length > 0 && sessionKey !== this.chatSessionKey) {
      console.info('MainViewModel: pending interaction ignored due to session mismatch, current=' + this.chatSessionKey + ', incoming=' + sessionKey);
      return;
    }

    const interaction = new ChatPendingInteraction();
    interaction.id = this.readOptionalString(payload, 'id') ||
      this.readOptionalString(payload, 'request_id') ||
      this.readOptionalString(payload, 'requestId') ||
      event + '|' + Date.now().toString();
    interaction.kind = this.mapPendingInteractionKind(event);
    interaction.title = this.pendingInteractionTitle(interaction.kind);
    interaction.body = this.pendingInteractionBody(event, payload);
    interaction.riskLevel = interaction.kind === 'action_approval' ? 'medium' : 'low';
    interaction.choices = this.pendingInteractionChoices(event, payload);
    interaction.input = this.pendingInteractionInput(event, payload);

    const backend = new ChatPendingInteractionBackend();
    backend.method = this.pendingInteractionRespondMethod(event);
    backend.requestId = this.readOptionalString(payload, 'request_id') || this.readOptionalString(payload, 'requestId');
    backend.sessionId = rawSessionKey.length > 0 ? sessionKey : this.chatSessionKey;
    backend.runId = this.readOptionalString(eventPayload, 'run_id') ||
      this.readOptionalString(eventPayload, 'runId') ||
      this.readOptionalString(payload, 'run_id') ||
      this.readOptionalString(payload, 'runId');
    interaction.backend = backend;

    this.chatPendingInteractions = [interaction];
    this.notifyChatStateChanged();
  }

  private mapPendingInteractionKind(event: string): string {
    switch (event) {
      case 'clarify.request':
        return 'clarification';
      case 'approval.request':
        return 'action_approval';
      case 'secret.request':
        return 'secret_input';
      case 'sudo.request':
        return 'handoff_required';
      default:
        return 'confirmation';
    }
  }

  private pendingInteractionTitle(kind: string): string {
    switch (kind) {
      case 'clarification':
        return this.getLocalizedString('pending_interaction_clarification_title', '需要你选择');
      case 'action_approval':
        return this.getLocalizedString('pending_interaction_action_title', '允许执行这个操作？');
      case 'secret_input':
        return this.getLocalizedString('pending_interaction_secret_title', '需要输入敏感信息');
      case 'handoff_required':
        return this.getLocalizedString('pending_interaction_handoff_title', '需要在桌面端完成');
      default:
        return this.getLocalizedString('pending_interaction_confirmation_title', '请确认');
    }
  }

  private pendingInteractionBody(event: string, payload: Record<string, Object | string>): string {
    if (event === 'clarify.request') {
      return this.readOptionalString(payload, 'question');
    }
    if (event === 'approval.request') {
      const description = this.readOptionalString(payload, 'description');
      const command = this.readOptionalString(payload, 'command');
      return description.length > 0 ? description : command;
    }
    if (event === 'secret.request') {
      return this.readOptionalString(payload, 'prompt');
    }
    if (event === 'sudo.request') {
      return this.getLocalizedString('pending_interaction_sudo_body', '这个操作需要在桌面端或开发者环境完成授权，手机端不会要求输入 sudo 密码。');
    }
    return this.readOptionalString(payload, 'message');
  }

  private pendingInteractionChoices(event: string, payload: Record<string, Object | string>): ChatPendingInteractionChoice[] {
    if (event === 'clarify.request') {
      const value = payload['choices'];
      const rawChoices = Array.isArray(value) ? value as Array<Object | string> : [];
      const choices: ChatPendingInteractionChoice[] = [];
      for (let i = 0; i < rawChoices.length; i++) {
        const label = this.pendingInteractionChoiceLabel(rawChoices[i]);
        if (label.length === 0) {
          continue;
        }
        const choice = new ChatPendingInteractionChoice();
        choice.id = this.pendingInteractionChoiceId(rawChoices[i], i);
        choice.label = label;
        choice.value = this.pendingInteractionChoiceValue(rawChoices[i], label);
        choice.style = i === 0 ? 'primary' : 'secondary';
        choices.push(choice);
      }
      return choices;
    }
    if (event === 'approval.request') {
      return this.fixedPendingChoices([
        ['once', this.getLocalizedString('pending_interaction_allow_once', '允许这一次'), 'primary'],
        ['deny', this.getLocalizedString('pending_interaction_deny', '不允许'), 'danger'],
      ]);
    }
    return this.fixedPendingChoices([
      ['cancel', this.getLocalizedString('pending_interaction_cancel', '取消'), 'secondary'],
    ]);
  }

  private fixedPendingChoices(rows: string[][]): ChatPendingInteractionChoice[] {
    const choices: ChatPendingInteractionChoice[] = [];
    for (let i = 0; i < rows.length; i++) {
      const choice = new ChatPendingInteractionChoice();
      choice.id = rows[i][0];
      choice.label = rows[i][1];
      choice.value = rows[i][0];
      choice.style = rows[i][2];
      choices.push(choice);
    }
    return choices;
  }

  private pendingInteractionChoiceLabel(rawChoice: Object | string): string {
    if (typeof rawChoice === 'string') {
      return rawChoice.trim();
    }
    const source = rawChoice as Record<string, Object | string>;
    return this.readOptionalString(source, 'label') ||
      this.readOptionalString(source, 'title') ||
      this.readOptionalString(source, 'text') ||
      this.readOptionalString(source, 'value');
  }

  private pendingInteractionChoiceValue(rawChoice: Object | string, fallback: string): string {
    if (typeof rawChoice === 'string') {
      return rawChoice.trim();
    }
    const source = rawChoice as Record<string, Object | string>;
    return this.readOptionalString(source, 'value') ||
      this.readOptionalString(source, 'id') ||
      fallback;
  }

  private pendingInteractionChoiceId(rawChoice: Object | string, index: number): string {
    if (typeof rawChoice !== 'string') {
      const source = rawChoice as Record<string, Object | string>;
      const explicitId = this.readOptionalString(source, 'id');
      if (explicitId.length > 0) {
        return explicitId;
      }
    }
    return 'choice_' + String(index + 1);
  }

  private pendingInteractionInput(event: string, payload: Record<string, Object | string>): ChatPendingInteractionInput | null {
    if (event !== 'secret.request') {
      return null;
    }
    const input = new ChatPendingInteractionInput();
    input.type = 'secure';
    input.placeholder = this.readOptionalString(payload, 'env_var');
    return input;
  }

  private pendingInteractionRespondMethod(event: string): string {
    switch (event) {
      case 'clarify.request':
        return 'clarify.respond';
      case 'approval.request':
        return 'approval.respond';
      case 'secret.request':
        return 'secret.respond';
      case 'sudo.request':
        return 'handoff_required';
      default:
        return 'confirm.respond';
    }
  }

  respondPendingInteraction(interaction: ChatPendingInteraction, value: string): void {
    if (!this.operatorSession || interaction.status !== 'pending') {
      return;
    }
    if (interaction.kind === 'handoff_required') {
      this.chatPendingInteractions = this.chatPendingInteractions.filter((item: ChatPendingInteraction): boolean => item.id !== interaction.id);
      this.notifyChatStateChanged();
      return;
    }
    const params: Record<string, string> = {};
    if (interaction.backend.requestId.length > 0) {
      params['request_id'] = interaction.backend.requestId;
    }
    if (interaction.backend.sessionId.length > 0) {
      params['session_id'] = interaction.backend.sessionId;
    }
    if (interaction.backend.runId.length > 0) {
      params['run_id'] = interaction.backend.runId;
    }
    if (interaction.kind === 'clarification') {
      params['answer'] = value;
    } else if (interaction.kind === 'action_approval') {
      params['choice'] = value;
    } else {
      params['value'] = value;
    }
    interaction.status = 'submitting';
    this.notifyChatStateChanged();
    this.operatorSession.request(interaction.backend.method, JSON.stringify(params)).then((response) => {
      if (!response.result.success) {
        throw new Error(response.error?.message || 'Interaction response failed.');
      }
      this.chatPendingInteractions = this.chatPendingInteractions.filter((item: ChatPendingInteraction): boolean => item.id !== interaction.id);
      this.notifyChatStateChanged();
    }).catch((error: Error) => {
      interaction.status = 'pending';
      this.chatError = error instanceof Error ? error.message : 'Interaction response failed.';
      this.notifyChatStateChanged();
    });
  }

  private handleChatEvent(payloadJson: string): void {
    const payload = this.parseJsonObject(payloadJson);
    const sessionKey = this.normalizeChatSessionKey(this.readOptionalString(payload, 'sessionKey'));
    const runId = this.readOptionalString(payload, 'runId');
    const state = this.readOptionalString(payload, 'state');

    this.logChatProtocolEvent('chat', payload, state, runId, sessionKey);

    if (sessionKey.length > 0 && sessionKey !== this.chatSessionKey) {
      console.info('MainViewModel: chat event ignored due to session mismatch, current=' + this.chatSessionKey + ', incoming=' + sessionKey);
      return;
    }

    if (state === 'delta') {
      if (runId.length > 0 && !this.isChatRunPending(runId)) {
        console.info('MainViewModel: chat delta ignored because run is not pending, runId=' + runId);
        return;
      }
      const deltaText = this.parseAssistantDeltaText(payload);
      if (deltaText.length > 0) {
        this.chatStreamingAssistantText = deltaText;
        this.notifyChatStateChanged();
      } else {
        console.info('MainViewModel: chat delta received without assistant text payload');
      }
      return;
    }

    if (state === 'final' || state === 'aborted' || state === 'error') {
      const finalMessage = this.parseChatEventMessage(payload);
      if (state === 'final' && finalMessage !== null) {
        this.commitFinalChatMessage(finalMessage);
      }

      if (state === 'error') {
        const errorMessage = this.readChatErrorMessage(payload);
        this.chatError = errorMessage.length > 0 ? errorMessage : this.getLocalizedString('chat_error_request_failed', 'Chat request failed.');
      }

      if (runId.length > 0) {
        this.clearChatPendingRun(runId);
      } else {
        this.clearChatPendingRuns();
      }

      this.chatStreamingAssistantText = null;
      this.chatPendingToolCalls = [];
      this.chatPendingInteractions = [];
      this.fetchChatSessions(50);
      this.notifyChatStateChanged();
      return;
    }

    console.warn('MainViewModel: unhandled chat event state=' + state + ', keys=' + this.describeObjectKeys(payload));
  }

  private handleAgentEvent(payloadJson: string): void {
    const payload = this.parseJsonObject(payloadJson);
    const sessionKey = this.normalizeChatSessionKey(this.readOptionalString(payload, 'sessionKey'));
    const stream = this.readOptionalString(payload, 'stream');
    const data = this.readNestedObject(payload, 'data');

    this.logChatProtocolEvent('agent', payload, stream, this.readOptionalString(payload, 'runId'), sessionKey);

    if (sessionKey.length > 0 && sessionKey !== this.chatSessionKey) {
      console.info('MainViewModel: agent event ignored due to session mismatch, current=' + this.chatSessionKey + ', incoming=' + sessionKey);
      return;
    }

    if (stream === 'assistant') {
      const text = this.readAgentAssistantText(data);
      if (text.length > 0) {
        this.chatStreamingAssistantText = text;
        this.notifyChatStateChanged();
      } else {
        console.info('MainViewModel: agent assistant event received without text, dataKeys=' + this.describeObjectKeys(data));
      }
      return;
    }

    if (stream === 'lifecycle' || stream === 'compaction') {
      return;
    }

    if (stream === 'tool') {
      const phase = this.readOptionalString(data, 'phase');
      const name = this.readOptionalString(data, 'name');
      const toolCallId = this.readOptionalString(data, 'toolCallId');
      if (phase.length === 0 || name.length === 0 || toolCallId.length === 0) {
        return;
      }

      if (phase === 'start') {
        const toolCall = new ChatPendingToolCall();
        toolCall.id = toolCallId;
        toolCall.name = name;
        const args = this.readNestedObject(data, 'args');
        toolCall.paramsJson = JSON.stringify(args);
        this.upsertPendingToolCall(toolCall);
      } else if (phase === 'result') {
        this.removePendingToolCall(toolCallId);
        // 捕获工具结果内容供 StreamUIAdapter 阶段检测 / 步骤识别（02 文档 §2/§4.4）
        const resultContent = this.readToolResultContent(data);
        if (resultContent.length > 0) {
          this.appendChatTurn('tool', resultContent);
        }
      }
      return;
    }

    if (stream === 'error') {
      const errorData = this.readNestedObject(payload, 'data');
      const errorReason = this.readOptionalString(errorData, 'reason');
      if (errorReason === 'seq gap' && this.pendingRunCount === 0) {
        console.info('MainViewModel: ignoring post-final agent seq gap signal');
        return;
      }

      const errorMessage = this.readChatErrorMessage(payload);
      this.chatError = errorMessage.length > 0 ? errorMessage : this.getLocalizedString('chat_error_seq_gap', 'Event stream interrupted, please refresh and try again.');
      this.chatStreamingAssistantText = null;
      this.chatPendingToolCalls = [];
      this.chatPendingInteractions = [];
      this.clearChatPendingRuns();
      this.notifyChatStateChanged();
      return;
    }

    console.warn('MainViewModel: unhandled agent event stream=' + stream + ', payloadKeys=' + this.describeObjectKeys(payload) + ', dataKeys=' + this.describeObjectKeys(data));
  }

  private handleCanvasUpdate(payloadJson: string): void {
    try {
      const data = JSON.parse(payloadJson) as Record<string, string>;
      if (data['url']) {
        this.canvasCurrentUrl = data['url'];
        this.canvasA2uiHydrated = true;
        this.canvasRehydratePending = false;
      }
    } catch (error) {
      console.error('MainViewModel: Parse canvas update error:', error);
    }
  }

  private createOperatorConnectOptions(): GatewayConnectOptions {
    const options = GatewaySession.defaultOperatorOptions();
    options.client.displayName = this.displayName || 'HarmonyOS Device';
    options.client.instanceId = this.instanceId;
    options.caps = [A2UI_CARD_CAPABILITY];
    return options;
  }

  private createNodeConnectOptions(): GatewayConnectOptions {
    const flags = new HarmonyNodeRuntimeFlags();
    const options = GatewaySession.defaultNodeOptions(
      InvokeCommandRegistry.advertisedCapabilities(flags),
      InvokeCommandRegistry.advertisedCommands(flags)
    );
    options.client.displayName = this.displayName || 'HarmonyOS Device';
    options.client.instanceId = this.instanceId;
    return options;
  }

  connect(endpoint: GatewayEndpoint): Promise<void> {
    return this.waitUntilReady().then((): Promise<void> => {
      const attemptId = ++this.connectAttemptId;
      if (!this.operatorSession || !this.nodeSession) {
        this.setStatusTextFromResource('common_status_gateway_session_not_ready', 'Gateway session not ready');
        return Promise.resolve();
      }

      console.info(
        'MainViewModel: Connect attempt #' + attemptId + ' to ' + endpoint.host + ':' + endpoint.port + ', tls=' + endpoint.tlsEnabled
      );
      this.connectedEndpoint = endpoint;
      this.isConnected = false;
      this.isNodeConnected = false;
      this.setStatusTextFromResource('common_status_connecting', 'Connecting…');
      this.notifyConnectionStateChanged();

      const authToken = this.gatewayMode === GatewayMode.Hermes ? this.hermesApiToken : this.gatewayToken;
      const bootstrapToken = this.gatewayMode === GatewayMode.Hermes ? '' : this.gatewayBootstrapToken;
      const password = this.gatewayMode === GatewayMode.Hermes ? '' : this.gatewayPassword;
      const operatorConnect = this.operatorSession.connect(
        endpoint,
        authToken,
        bootstrapToken,
        password,
        this.createOperatorConnectOptions()
      );
      const nodeConnect = this.nodeSession.connect(
        endpoint,
        authToken,
        bootstrapToken,
        password,
        this.createNodeConnectOptions()
      );
      return Promise.all([operatorConnect, nodeConnect]).then((): void => {
        if (attemptId !== this.connectAttemptId) {
          return;
        }
        console.info(
          'MainViewModel: Connect attempt #' + attemptId + ' finished, operator=' + this.isConnected + ', node=' + this.isNodeConnected + ', status=' + this.statusText
        );
      }).catch((error: Error): void => {
        if (attemptId !== this.connectAttemptId) {
          return;
        }
        console.error('MainViewModel: Connect error:', error);
        if (this.operatorSession) {
          void this.operatorSession.disconnect();
        }
        if (this.nodeSession) {
          void this.nodeSession.disconnect();
        }
        const errorMsg = error instanceof Error
          ? error.message
          : this.getLocalizedString('common_error_unable_to_reach_gateway', 'Unable to reach gateway.');
        this.isConnected = false;
        this.isNodeConnected = false;
        this.chatHealthOk = false;
        this.setStatusTextFromResource('common_status_connection_failed', 'Connection failed');
        this.chatError = errorMsg;
        this.notifyConnectionStateChanged();
      });
    });
  }

  connectManual(): Promise<void> {
    return this.waitUntilReady().then((): Promise<void> => {
      const trimmedHost = this.manualHost.trim();
      console.info(
        'MainViewModel: connectManual host=' + trimmedHost + ', port=' + this.manualPort + ', tls=' + this.manualTls
      );
      if (!trimmedHost || this.manualPort < 1 || this.manualPort > 65535) {
        console.warn('MainViewModel: connectManual rejected due to invalid configuration');
        this.setStatusTextFromResource('common_status_invalid_manual_config', 'Invalid manual gateway configuration');
        return Promise.resolve();
      }

      const endpoint = this.createEndpoint(trimmedHost, this.manualPort, this.manualTls, `${trimmedHost}:${this.manualPort}`);
      return this.setGatewayMode(GatewayMode.OpenClaw).then((): Promise<void> => {
        return this.connect(endpoint);
      });
    });
  }

  connectHermes(): Promise<void> {
    return this.waitUntilReady().then((): Promise<void> => {
      const trimmedHost = MainViewModel.resolveHermesHost(this.hermesHost);
      const resolvedPort = MainViewModel.resolveHermesPort(this.hermesPort);
      const trimmedToken = this.hermesApiToken.trim();
      console.info(
        'MainViewModel: connectHermes host=' + trimmedHost + ', port=' + resolvedPort + ', tls=' + this.hermesTls
      );
      if (resolvedPort < 1 || resolvedPort > 65535 || trimmedToken.length === 0) {
        console.warn('MainViewModel: connectHermes rejected due to invalid configuration');
        this.setStatusTextFromResource('common_status_invalid_hermes_config', 'Invalid Hermes gateway configuration');
        return Promise.resolve();
      }

      this.hermesHost = trimmedHost;
      this.hermesPort = resolvedPort;
      this.hermesApiToken = trimmedToken;
      const endpoint = this.createEndpoint(trimmedHost, resolvedPort, this.hermesTls, `${trimmedHost}:${resolvedPort}`);
      return this.setGatewayMode(GatewayMode.Hermes).then((): Promise<void> => {
        return this.connect(endpoint);
      }).then((): Promise<void> => {
        if (!this.isConnected) {
          return Promise.resolve();
        }
        return this.persistHermesConnection(trimmedHost, resolvedPort, this.hermesTls, trimmedToken);
      });
    });
  }

  connectFromSetupCode(setupCode: string): Promise<boolean> {
    return this.waitUntilReady().then((): Promise<boolean> => {
      const trimmed = setupCode.trim();
      console.info('MainViewModel: connectFromSetupCode start, length=' + trimmed.length);
      if (trimmed.length === 0) {
        console.warn('MainViewModel: connectFromSetupCode missing setup code');
        this.setStatusTextFromResource('common_status_missing_setup_code', 'Missing setup code');
        return Promise.resolve(false);
      }

      const connectionParams = MainViewModel.parseSetupCodeConnection(trimmed);
      if (!connectionParams) {
        this.setStatusTextFromResource('common_status_unsupported_setup_code', 'Unsupported setup code format. Use a gateway URL or host:port for now.');
        console.warn('MainViewModel: connectFromSetupCode unsupported format');
        return Promise.resolve(false);
      }

      console.info(
        'MainViewModel: connectFromSetupCode parsed host=' + connectionParams.host +
        ', port=' + connectionParams.port +
        ', tls=' + connectionParams.tlsEnabled
      );

      return MainViewModel.applySetupCodeConnection(this, connectionParams).then((): Promise<boolean> => {
        this.seedMockDiscovery();
        return this.connectManual().then((): boolean => {
          console.info('MainViewModel: connectFromSetupCode end, connected=' + this.isConnected + ', status=' + this.statusText);
          return this.isConnected;
        });
      });
    });
  }

  private deriveHealthState(source: Record<string, Object | string>, fallback: boolean): boolean {
    const directHealth = this.readKnownHealthState(source);
    if (directHealth !== null) {
      return directHealth;
    }

    const resultHealth = this.readKnownHealthState(this.readNestedObject(source, 'result'));
    if (resultHealth !== null) {
      return resultHealth;
    }

    const dataHealth = this.readKnownHealthState(this.readNestedObject(source, 'data'));
    if (dataHealth !== null) {
      return dataHealth;
    }

    return fallback;
  }

  private readKnownHealthState(source: Record<string, Object | string>): boolean | null {
    const healthKeys = ['healthy', 'ok', 'success'];
    for (let i = 0; i < healthKeys.length; i++) {
      const value = this.readOptionalBoolean(source, healthKeys[i]);
      if (value !== null) {
        return value;
      }
    }

    const status = this.readOptionalString(source, 'status').trim().toLowerCase();
    if (status === 'ok' || status === 'healthy' || status === 'ready') {
      return true;
    }
    if (status === 'error' || status === 'unhealthy' || status === 'not_ready') {
      return false;
    }

    return null;
  }

  private static applySetupCodeConnection(
    viewModel: MainViewModel,
    connectionParams: SetupCodeConnectionParams
  ): Promise<void> {
    return viewModel.setManualHost(connectionParams.host).then((): Promise<void> => {
      return viewModel.setManualPort(connectionParams.port);
    }).then((): Promise<void> => {
      return viewModel.setManualTls(connectionParams.tlsEnabled);
    }).then((): Promise<void> => {
      return viewModel.setGatewayToken(connectionParams.token);
    }).then((): Promise<void> => {
      return viewModel.setGatewayBootstrapToken(connectionParams.bootstrapToken);
    }).then((): Promise<void> => {
      return viewModel.setGatewayPassword(connectionParams.password);
    }).then((): Promise<void> => {
      return viewModel.setManualEnabled(true);
    });
  }

  private static parseSetupCodeConnection(setupCode: string): SetupCodeConnectionParams | null {
    const directConnection = MainViewModel.parseConnectionTarget(setupCode);
    if (directConnection) {
      return directConnection;
    }

    const payload = MainViewModel.decodeSetupCodePayload(setupCode);
    if (!payload) {
      return null;
    }

    const nestedAuth = MainViewModel.readSetupCodeObject(payload, 'auth');
    const targetUrl = MainViewModel.readSetupCodeString(payload, ['url', 'gatewayUrl', 'endpoint']);
    const connection = targetUrl.length > 0
      ? MainViewModel.parseConnectionTarget(targetUrl)
      : MainViewModel.parseConnectionFields(payload);
    if (!connection) {
      return null;
    }

    connection.token = MainViewModel.readSetupCodeString(payload, ['token', 'gatewayToken']);
    if (connection.token.length === 0 && nestedAuth) {
      connection.token = MainViewModel.readSetupCodeString(nestedAuth, ['token', 'gatewayToken']);
    }

    connection.bootstrapToken = MainViewModel.readSetupCodeString(payload, ['bootstrapToken', 'gatewayBootstrapToken']);
    if (connection.bootstrapToken.length === 0 && nestedAuth) {
      connection.bootstrapToken = MainViewModel.readSetupCodeString(nestedAuth, ['bootstrapToken', 'gatewayBootstrapToken']);
    }

    connection.password = MainViewModel.readSetupCodeString(payload, ['password', 'gatewayPassword']);
    if (connection.password.length === 0 && nestedAuth) {
      connection.password = MainViewModel.readSetupCodeString(nestedAuth, ['password', 'gatewayPassword']);
    }

    return connection;
  }

  private static parseConnectionFields(payload: Record<string, Object | string>): SetupCodeConnectionParams | null {
    const host = MainViewModel.readSetupCodeString(payload, ['host', 'hostname']);
    const port = MainViewModel.readSetupCodeNumber(payload, ['port']);
    const tlsEnabled = MainViewModel.readSetupCodeBoolean(payload, ['tls', 'tlsEnabled', 'secure']);
    if (host.length === 0) {
      return null;
    }

    const connection = new SetupCodeConnectionParams();
    connection.host = host;
    connection.port = port > 0 ? port : 18789;
    connection.tlsEnabled = tlsEnabled;
    if (connection.port < 1 || connection.port > 65535) {
      return null;
    }
    if (!connection.tlsEnabled && !MainViewModel.isPrivateOrLoopbackHost(connection.host)) {
      return null;
    }
    return connection;
  }

  private static parseConnectionTarget(rawValue: string): SetupCodeConnectionParams | null {
    const trimmed = rawValue.trim();
    if (trimmed.length === 0) {
      return null;
    }

    const looksLikeUrl = /^(openclaw:\/\/|https?:\/\/|wss?:\/\/)/i.test(trimmed);
    const looksLikeHostPort = /^[a-z0-9.-]+(:\d+)?(\/.*)?$/i.test(trimmed);
    if (!looksLikeUrl && !looksLikeHostPort) {
      return null;
    }

    const normalized = trimmed
      .replace(/^openclaw:\/\//i, '')
      .replace(/^https?:\/\//i, '')
      .replace(/^wss?:\/\//i, '');

    const slashIndex = normalized.indexOf('/');
    const hostPort = slashIndex >= 0 ? normalized.substring(0, slashIndex) : normalized;
    const parts = hostPort.split(':');
    const host = (parts[0] || '').trim();
    const tlsEnabled = trimmed.startsWith('wss://') || trimmed.startsWith('https://');
    const defaultPort = tlsEnabled ? 443 : 18789;
    const parsedPort = parts.length > 1 ? parseInt(parts[1], 10) : defaultPort;
    if (!host || Number.isNaN(parsedPort) || parsedPort < 1 || parsedPort > 65535) {
      return null;
    }
    if (!tlsEnabled && !MainViewModel.isPrivateOrLoopbackHost(host)) {
      return null;
    }

    const connection = new SetupCodeConnectionParams();
    connection.host = host;
    connection.port = parsedPort;
    connection.tlsEnabled = tlsEnabled;
    return connection;
  }

  private static isPrivateOrLoopbackHost(host: string): boolean {
    const normalized = host.trim().toLowerCase().replace(/^\[/, '').replace(/\]$/, '');
    if (normalized === 'localhost' || normalized === '::1') {
      return true;
    }
    const parts = normalized.split('.');
    if (parts.length !== 4) {
      return normalized.endsWith('.local');
    }
    const octets = parts.map((part: string): number => parseInt(part, 10));
    if (octets.some((value: number): boolean => Number.isNaN(value) || value < 0 || value > 255)) {
      return false;
    }
    if (octets[0] === 10 || octets[0] === 127) {
      return true;
    }
    if (octets[0] === 192 && octets[1] === 168) {
      return true;
    }
    if (octets[0] === 172 && octets[1] >= 16 && octets[1] <= 31) {
      return true;
    }
    return false;
  }

  private static decodeSetupCodePayload(setupCode: string): Record<string, Object | string> | null {
    try {
      const normalized = setupCode.trim().replace(/-/g, '+').replace(/_/g, '/');
      const paddingLength = normalized.length % 4;
      const padded = paddingLength === 0 ? normalized : normalized + '='.repeat(4 - paddingLength);
      const decodedBytes = MainViewModel.decodeBase64(padded);
      if (decodedBytes.length === 0) {
        return null;
      }
      const decodedText = MainViewModel.bytesToString(decodedBytes);
      const parsed = JSON.parse(decodedText) as Object;
      return parsed as Record<string, Object | string>;
    } catch (error) {
      console.warn('MainViewModel: failed to decode setup code payload:', error);
      return null;
    }
  }

  private static decodeBase64(input: string): number[] {
    const alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    const output: number[] = [];
    let buffer = 0;
    let bits = 0;

    for (let i = 0; i < input.length; i++) {
      const ch = input.charAt(i);
      if (ch === '=') {
        break;
      }
      const value = alphabet.indexOf(ch);
      if (value < 0) {
        continue;
      }
      buffer = (buffer << 6) | value;
      bits += 6;
      while (bits >= 8) {
        bits -= 8;
        output.push((buffer >> bits) & 0xff);
      }
    }

    return output;
  }

  private static bytesToString(bytes: number[]): string {
    let output = '';
    for (let i = 0; i < bytes.length; i++) {
      output += String.fromCharCode(bytes[i]);
    }
    return output;
  }

  private static readSetupCodeObject(
    payload: Record<string, Object | string>,
    key: string
  ): Record<string, Object | string> | null {
    const value = payload[key];
    if (!value || typeof value === 'string') {
      return null;
    }
    return value as Record<string, Object | string>;
  }

  private static readSetupCodeString(payload: Record<string, Object | string>, keys: string[]): string {
    for (let i = 0; i < keys.length; i++) {
      const value = payload[keys[i]];
      if (typeof value === 'string') {
        const trimmed = value.trim();
        if (trimmed.length > 0) {
          return trimmed;
        }
      }
    }
    return '';
  }

  private static readSetupCodeNumber(payload: Record<string, Object | string>, keys: string[]): number {
    for (let i = 0; i < keys.length; i++) {
      const value = payload[keys[i]];
      if (typeof value === 'number') {
        return value;
      }
      if (typeof value === 'string') {
        const parsed = parseInt(value.trim(), 10);
        if (!Number.isNaN(parsed)) {
          return parsed;
        }
      }
    }
    return 0;
  }

  private static readSetupCodeBoolean(payload: Record<string, Object | string>, keys: string[]): boolean {
    for (let i = 0; i < keys.length; i++) {
      const value = payload[keys[i]];
      if (typeof value === 'boolean') {
        return value;
      }
      if (typeof value === 'string') {
        const normalized = value.trim().toLowerCase();
        if (normalized === 'true') {
          return true;
        }
        if (normalized === 'false') {
          return false;
        }
      }
    }
    return false;
  }

  disconnect(): Promise<void> {
    ++this.connectAttemptId;
    const operatorDisconnect = this.operatorSession ? this.operatorSession.disconnect() : Promise.resolve();
    const nodeDisconnect = this.nodeSession ? this.nodeSession.disconnect() : Promise.resolve();
    return Promise.all([operatorDisconnect, nodeDisconnect]).then((): void => {
      this.connectedEndpoint = null;
      this.isConnected = false;
      this.isNodeConnected = false;
      this.chatHealthOk = false;
      this.setStatusTextFromResource('common_status_disconnected', 'Disconnected');
      this.notifyConnectionStateChanged();
    });
  }

  refreshGatewayConnection(): Promise<void> {
    if (!this.connectedEndpoint) {
      return Promise.resolve();
    }

    const endpoint = this.connectedEndpoint;
    return this.disconnect().then((): Promise<void> => {
      return this.connect(endpoint);
    });
  }

  private findEndpointByStableId(stableId: string): GatewayEndpoint | null {
    const match = this.gateways.find((endpoint: GatewayEndpoint) => endpoint.stableId === stableId);
    return match ?? null;
  }

  async setDisplayName(value: string): Promise<void> {
    this.displayName = value;
    await securePrefs.setDisplayName(value);
  }

  async setCameraEnabled(value: boolean): Promise<void> {
    this.cameraEnabled = value;
    await securePrefs.setCameraEnabled(value);
  }

  async setLocationMode(mode: LocationMode): Promise<void> {
    this.locationMode = mode;
    await securePrefs.setLocationMode(mode);
  }

  async setLocationPreciseEnabled(value: boolean): Promise<void> {
    this.locationPreciseEnabled = value;
    await securePrefs.setLocationPreciseEnabled(value);
  }

  async setPreventSleep(value: boolean): Promise<void> {
    this.preventSleep = value;
    await securePrefs.setPreventSleep(value);
  }

  async setManualEnabled(value: boolean): Promise<void> {
    this.manualEnabled = value;
    await securePrefs.setManualEnabled(value);
  }

  async setManualHost(value: string): Promise<void> {
    this.manualHost = value;
    await securePrefs.setManualHost(value);
    this.seedMockDiscovery();
  }

  async setManualPort(value: number): Promise<void> {
    this.manualPort = value;
    await securePrefs.setManualPort(value);
    this.seedMockDiscovery();
  }

  async setManualTls(value: boolean): Promise<void> {
    this.manualTls = value;
    await securePrefs.setManualTls(value);
    this.seedMockDiscovery();
  }

  async setGatewayMode(value: string): Promise<void> {
    const nextMode = value === GatewayMode.OpenClaw ? GatewayMode.OpenClaw : GatewayMode.Hermes;
    const changed = this.gatewayMode !== nextMode;
    if (changed) {
      if (this.operatorSession) {
        await this.operatorSession.disconnect();
      }
      if (this.nodeSession) {
        await this.nodeSession.disconnect();
      }
      this.isConnected = false;
      this.isNodeConnected = false;
      this.connectedEndpoint = null;
      this.chatHealthOk = false;
      this.chatStreamingAssistantText = null;
      this.chatPendingToolCalls = [];
      this.chatPendingInteractions = [];
      this.clearChatPendingRuns();
    }
    this.gatewayMode = nextMode;
    await securePrefs.setGatewayMode(nextMode);
    if (changed) {
      this.configureGatewaySessions();
      this.setStatusTextFromResource('common_status_disconnected', 'Disconnected');
      this.notifyConnectionStateChanged();
      this.notifyChatStateChanged();
    }
    this.seedMockDiscovery();
  }

  async setHermesHost(value: string): Promise<void> {
    this.hermesHost = value;
    await securePrefs.setHermesHost(value);
    this.seedMockDiscovery();
  }

  async setHermesPort(value: number): Promise<void> {
    this.hermesPort = value;
    await securePrefs.setHermesPort(value);
    this.seedMockDiscovery();
  }

  async setHermesTls(value: boolean): Promise<void> {
    this.hermesTls = value;
    await securePrefs.setHermesTls(value);
    this.seedMockDiscovery();
  }

  async setHermesApiToken(value: string): Promise<void> {
    this.hermesApiToken = value;
    await securePrefs.setHermesApiToken(value);
  }

  async setGatewayToken(value: string): Promise<void> {
    this.gatewayToken = value;
    await securePrefs.setGatewayToken(value);
  }

  async setGatewayBootstrapToken(value: string): Promise<void> {
    this.gatewayBootstrapToken = value;
    await securePrefs.setGatewayBootstrapToken(value);
  }

  async setGatewayPassword(value: string): Promise<void> {
    this.gatewayPassword = value;
    await securePrefs.setGatewayPassword(value);
  }

  async setOnboardingCompleted(value: boolean): Promise<void> {
    this.onboardingCompleted = value;
    await securePrefs.setOnboardingCompleted(value);
  }

  async setCanvasDebugStatusEnabled(value: boolean): Promise<void> {
    this.canvasDebugStatusEnabled = value;
    await securePrefs.setCanvasDebugStatusEnabled(value);
  }

  async setVoiceScreenActive(active: boolean): Promise<void> {
    if (!active) {
      this.micIsListening = false;
      this.setMicStatusFromResource(this.micEnabled ? 'settings_ready' : 'settings_off', this.micEnabled ? 'Ready' : 'Off');
    }
  }

  async setMicEnabled(enabled: boolean): Promise<void> {
    this.micEnabled = enabled;
    this.setMicStatusFromResource(enabled ? 'settings_ready' : 'settings_off', enabled ? 'Ready' : 'Off');
    await securePrefs.setTalkEnabled(enabled);
  }

  async setSpeakerEnabled(enabled: boolean): Promise<void> {
    this.speakerEnabled = enabled;
    await securePrefs.setSpeakerEnabled(enabled);
  }

  async setVoiceScreenActiveAs(active: boolean): Promise<void> {
    await this.setVoiceScreenActive(active);
  }

  refreshHomeCanvasOverviewIfConnected(): void {
    const currentUrl = (this.canvasCurrentUrl || '').trim();
    if (!this.isConnected || !this.isNodeConnected || currentUrl.length === 0) {
      return;
    }
    this.requestCanvasRehydrate('screen_tab', false);
  }

  loadChat(sessionKey: string): void {
    const normalizedSessionKey = this.normalizeChatSessionKey(sessionKey);
    this.chatSessionKey = normalizedSessionKey;
    this.bootstrapChat(true);
    this.notifyChatStateChanged();
  }

  refreshChat(): void {
    this.bootstrapChat(true);
  }

  refreshChatSessions(limit?: number): void {
    this.fetchChatSessions(limit);
  }

  setChatThinkingLevel(level: string): void {
    this.chatThinkingLevel = this.normalizeChatThinkingLevel(level);
    this.notifyChatStateChanged();
  }

  switchChatSession(sessionKey: string): void {
    const normalizedSessionKey = this.normalizeChatSessionKey(sessionKey);
    if (normalizedSessionKey === this.chatSessionKey) {
      return;
    }

    this.chatSessionKey = normalizedSessionKey;
    this.bootstrapChat(true);
    this.notifyChatStateChanged();
  }

  abortChat(): void {
    if (!this.operatorSession || this.chatPendingRunIds.length === 0) {
      this.chatStreamingAssistantText = null;
      this.chatPendingInteractions = [];
      this.clearChatPendingRuns();
      this.notifyChatStateChanged();
      return;
    }

    const pendingRuns = [...this.chatPendingRunIds];
    for (let i = 0; i < pendingRuns.length; i++) {
      const params = new ChatAbortRequestParams();
      params.sessionKey = this.chatSessionKey;
      params.runId = pendingRuns[i];
      this.operatorSession.request('chat.abort', JSON.stringify(params)).catch(() => {
        // best-effort
      });
    }
  }

  sendChat(message: string, thinking: string, attachments: OutgoingAttachment[], displayMessage?: string, sessionKey?: string): void {
    const trimmedMessage = message.trim();
    if (trimmedMessage.length === 0 && attachments.length === 0) {
      return;
    }

    if (!this.isConnected || !this.operatorSession) {
      this.chatError = this.getLocalizedString('chat_error_not_connected_send', 'Connect to a gateway before sending messages.');
      return;
    }

    this.chatError = null;
    this.chatThinkingLevel = this.normalizeChatThinkingLevel(thinking);

    const now = Date.now();
    const userMessage = new ChatMessage();
    userMessage.stableKey = 'local|' + String(now) + '|' + this.createChatRunId();
    userMessage.role = 'user';
    const outgoingText = trimmedMessage.length > 0 ? trimmedMessage : this.getLocalizedString('chat_see_attached', 'See attached.');
    const visibleText = displayMessage && displayMessage.trim().length > 0 ? displayMessage.trim() : outgoingText;
    userMessage.content = [this.createTextContent(visibleText)];
    // 带附件时，为每张图片构建本地缩略图 content block（data URL）
    if (attachments.length > 0) {
      for (let attachment of attachments) {
        if (attachment.type === 'image' && attachment.content.length > 0) {
          const dataUrl = 'data:' + attachment.mimeType + ';base64,' + attachment.content;
          userMessage.content.push(this.createImageContent(dataUrl));
        }
      }
    }

    this.chatMessages = [...this.chatMessages, userMessage];
    this.appendChatTurn('user', outgoingText);
    this.chatStreamingAssistantText = null;
    this.chatPendingToolCalls = [];
    this.chatPendingInteractions = [];

    const runId = this.createChatRunId();
    this.addChatPendingRun(runId);

    const requestParams = new ChatSendRequestParams();
    requestParams.sessionKey = sessionKey ?? this.chatSessionKey;
    requestParams.message = outgoingText;
    requestParams.thinking = this.chatThinkingLevel;
    requestParams.idempotencyKey = runId;
    if (attachments.length > 0) {
      requestParams.attachments = attachments;
      console.info(
        'MainViewModel.sendChat: attaching ' + attachments.length +
        ' attachment(s), first mimeType=' + attachments[0].mimeType +
        ', contentLen=' + attachments[0].content.length
      );
    } else {
      console.info('MainViewModel.sendChat: no attachments');
    }

    const paramsJson = JSON.stringify(requestParams);
    console.info(
      'MainViewModel.sendChat: paramsJson length=' + paramsJson.length +
      ', hasAttachments=' + (paramsJson.indexOf('"attachments"') >= 0)
    );

    this.operatorSession.request('chat.send', paramsJson).then((response) => {
      console.info(
        'MainViewModel.sendChat: chat.send response received, success=' + response.result.success +
        ', error=' + (response.error ? JSON.stringify(response.error) : 'none') +
        ', payload=' + (response.payload ? JSON.stringify(response.payload) : 'none')
      );
      if (!response.result.success) {
        console.error(
          'MainViewModel.sendChat: chat.send FAILED, error=' +
          JSON.stringify(response.error)
        );
        throw new Error(response.error?.message || this.getLocalizedString('chat_error_send_failed', 'Failed to send chat message.'));
      }

      const actualRunId = this.readChatRunIdFromPayload(response.payload);
      console.info(
        'MainViewModel.sendChat: chat.send succeeded, actualRunId=' + actualRunId +
        ', requestedRunId=' + runId
      );
      if (actualRunId.length > 0 && actualRunId !== runId) {
        this.clearChatPendingRun(runId);
        this.addChatPendingRun(actualRunId);
      }
    }).catch((error: Error) => {
      console.error(
        'MainViewModel.sendChat: chat.send catch, error=' + (error instanceof Error ? error.message : String(error))
      );
      this.clearChatPendingRun(runId);
      this.chatError = error instanceof Error
        ? error.message
        : this.getLocalizedString('chat_error_send_failed', 'Failed to send chat message.');
    });

    const sessionIndex = this.chatSessions.findIndex((entry: ChatSessionEntry) => entry.key === this.chatSessionKey);
    if (sessionIndex >= 0) {
      const updated = [...this.chatSessions];
      updated[sessionIndex].updatedAtMs = now;
      this.chatSessions = updated;
    } else {
      const session = new ChatSessionEntry();
      session.key = this.chatSessionKey;
      session.displayName = this.chatSessionKey === this.mainSessionKey ? this.getLocalizedString('common_main', 'Main') : this.chatSessionKey;
      session.updatedAtMs = now;
      this.chatSessions = [session, ...this.chatSessions];
    }
    this.notifyChatStateChanged();
  }

  private bootstrapChat(forceHealth: boolean): void {
    const session = this.operatorSession;
    if (!session || !this.isConnected) {
      this.chatHealthOk = false;
      this.chatError = this.isConnected
        ? this.getLocalizedString('chat_error_session_not_ready', 'Gateway session not ready.')
        : this.getLocalizedString('chat_error_not_connected_chat', 'Connect to a gateway before chatting.');
      return;
    }

    const bootstrapId = ++this.chatBootstrapId;
    const sessionKey = this.normalizeChatSessionKey(this.chatSessionKey);
    this.chatSessionKey = sessionKey;
    this.chatError = null;
    this.chatHealthOk = false;
    this.chatStreamingAssistantText = null;
    this.chatPendingToolCalls = [];
    this.chatPendingInteractions = [];
    this.chatPendingRunIds = [];
    this.pendingRunCount = 0;
    this.chatSessionId = null;

    const historyParams = new ChatHistoryRequestParams();
    historyParams.sessionKey = sessionKey;
    session.request('chat.history', JSON.stringify(historyParams)).then((response) => {
      if (bootstrapId !== this.chatBootstrapId) {
        return;
      }

      if (!response.result.success) {
        throw new Error(
          response.error?.message || this.getLocalizedString('chat_error_history_load_failed', 'Unable to load chat history.')
        );
      }

      const payload = this.readPayloadObject(response.payload);
      const historyMessages = this.parseChatHistoryMessages(payload);
      this.chatMessages = historyMessages;
      this.rebuildChatTurnsFromHistory(historyMessages);
      this.chatSessionId = this.readOptionalString(payload, 'sessionId');

      const thinkingLevel = this.normalizeChatThinkingLevel(this.readOptionalString(payload, 'thinkingLevel'));
      if (thinkingLevel.length > 0) {
        this.chatThinkingLevel = thinkingLevel;
      }

      if (forceHealth) {
        return session.request('health').then((healthResponse) => {
          if (bootstrapId !== this.chatBootstrapId) {
            return;
          }
          const payload = this.readPayloadObject(healthResponse.payload);
          this.chatHealthOk = this.deriveHealthState(payload, healthResponse.result.success);
          this.notifyChatStateChanged();
        });
      }

      this.chatHealthOk = true;
      this.notifyChatStateChanged();
      return Promise.resolve();
    }).then(() => {
      if (bootstrapId !== this.chatBootstrapId) {
        return;
      }
      this.fetchChatSessions(50, bootstrapId);
    }).catch((error: Error) => {
      if (bootstrapId !== this.chatBootstrapId) {
        return;
      }
      this.chatError = error instanceof Error
        ? error.message
        : this.getLocalizedString('chat_error_load_failed', 'Unable to load chat.');
      this.chatHealthOk = false;
      this.notifyChatStateChanged();
    });
  }

  private fetchChatSessions(limit?: number, bootstrapId?: number): void {
    const session = this.operatorSession;
    if (!session || !this.isConnected) {
      this.chatSessions = this.ensureMainChatSession([], limit);
      return;
    }

    const requestId = ++this.chatSessionsRequestId;
    const requestParams = new ChatSessionsListParams();
    if (limit !== undefined && limit > 0) {
      requestParams.limit = limit;
    }

    session.request('sessions.list', JSON.stringify(requestParams)).then((response) => {
      if (requestId !== this.chatSessionsRequestId) {
        return;
      }
      if (bootstrapId !== undefined && bootstrapId !== this.chatBootstrapId) {
        return;
      }
      if (!response.result.success) {
        throw new Error(
          response.error?.message || this.getLocalizedString('chat_error_sessions_load_failed', 'Unable to load chat sessions.')
        );
      }

      const payload = this.readPayloadObject(response.payload);
      const sessions = this.parseChatSessions(payload);
      this.chatSessions = this.ensureMainChatSession(sessions, limit);
      this.notifyChatStateChanged();
    }).catch((error: Error) => {
      if (requestId !== this.chatSessionsRequestId) {
        return;
      }
      if (bootstrapId !== undefined && bootstrapId !== this.chatBootstrapId) {
        return;
      }
      console.warn('MainViewModel: fetchChatSessions failed:', error);
      this.chatSessions = this.ensureMainChatSession(this.chatSessions, limit);
      this.notifyChatStateChanged();
    });
  }

  private ensureMainChatSession(sessions: ChatSessionEntry[], limit?: number): ChatSessionEntry[] {
    const main = new ChatSessionEntry();
    main.key = this.mainSessionKey;
    main.displayName = this.getLocalizedString('common_main', 'Main');
    main.updatedAtMs = Date.now();

    const merged = [main, ...sessions.filter((entry: ChatSessionEntry) => entry.key !== main.key)];
    if (limit !== undefined && limit > 0) {
      return merged.slice(0, limit);
    }
    return merged;
  }

  private parseChatHistoryMessages(payload: Record<string, Object | string>): ChatMessage[] {
    const messagesValue = payload['messages'];
    if (!messagesValue || typeof messagesValue === 'string' || !Array.isArray(messagesValue)) {
      return [];
    }

    const sourceMessages = messagesValue as Array<Object>;
    const parsedMessages: ChatMessage[] = [];
    for (let i = 0; i < sourceMessages.length; i++) {
      const entry = sourceMessages[i] as Record<string, Object | string>;
      const role = this.readOptionalString(entry, 'role');
      if (role.length === 0) {
        continue;
      }

      const message = new ChatMessage();
      message.stableKey = this.buildChatMessageStableKey(entry, role, i);
      message.role = role;
      message.content = this.parseChatMessageContent(entry['content']);
      parsedMessages.push(message);
    }
    return parsedMessages;
  }

  private parseAssistantDeltaText(payload: Record<string, Object | string>): string {
    const message = this.readNestedObject(payload, 'message');
    const role = this.readOptionalString(message, 'role');
    if (role !== 'assistant') {
      return '';
    }

    const contentValue = message['content'];
    if (!contentValue || typeof contentValue === 'string' || !Array.isArray(contentValue)) {
      return '';
    }

    const items = contentValue as Array<Object>;
    for (let i = 0; i < items.length; i++) {
      const item = items[i] as Record<string, Object | string>;
      const type = this.readOptionalString(item, 'type');
      const text = this.readOptionalString(item, 'text');
      if (type === 'text' && text.length > 0) {
        return text;
      }
    }
    return '';
  }

  private parseChatEventMessage(payload: Record<string, Object | string>): ChatMessage | null {
    const messageValue = payload['message'];
    if (!messageValue || typeof messageValue === 'string') {
      return null;
    }

    const source = messageValue as Record<string, Object | string>;
    const role = this.readOptionalString(source, 'role');
    if (role.length === 0) {
      return null;
    }

    const message = new ChatMessage();
    message.stableKey = this.buildChatMessageStableKey(source, role, this.chatMessages.length);
    message.role = role;
    message.content = this.parseChatMessageContent(source['content']);
    return message;
  }

  private buildChatMessageStableKey(source: Record<string, Object | string>, role: string, indexHint: number): string {
    const explicitId = this.readOptionalString(source, 'id');
    if (explicitId.length > 0) {
      return explicitId;
    }

    const createdAt = this.readOptionalNumber(source, 'createdAt');
    if (createdAt > 0) {
      return role + '|' + String(createdAt);
    }

    const text = this.stringifyChatMessageContentFromValue(source['content']);
    if (text.length > 0) {
      return role + '|' + text + '|' + String(indexHint);
    }

    return role + '|index|' + String(indexHint);
  }

  private stringifyChatMessageContentFromValue(contentValue: Object | string | undefined): string {
    const content = this.parseChatMessageContent(contentValue);
    if (content.length === 0) {
      return '';
    }

    const parts: string[] = [];
    for (let i = 0; i < content.length; i++) {
      const block = content[i];
      const text = (block.text || '').trim();
      if (text.length > 0) {
        parts.push(text);
      } else if ((block.type || '').trim().length > 0) {
        parts.push('[' + block.type + ']');
      }
    }
    return parts.join('\n\n');
  }

  private commitFinalChatMessage(message: ChatMessage): void {
    const nextText = this.stringifyChatMessageContent(message);
    if (nextText.length === 0) {
      return;
    }

    if (this.chatMessages.length > 0) {
      const lastMessage = this.chatMessages[this.chatMessages.length - 1];
      if (lastMessage.role === message.role && this.stringifyChatMessageContent(lastMessage) === nextText) {
        return;
      }
    }

    this.chatMessages = [...this.chatMessages, message];
    if (message.role === 'assistant') {
      this.appendChatTurn('agent', nextText);
    }
    this.notifyChatStateChanged();
  }

  // ---------- 结构化流程时间线（A2UI 优先，文本解析兜底） ----------

  /** A2UI 校验失败时的反馈，等待回灌给 LLM（每会话最多自动回灌一次，避免死循环）。 */
  private pendingA2UIErrorFeedback: string = '';
  private a2uiAutoRetryUsed: boolean = false;
  /** 已渲染的 A2UI 结构化 block（持久，不被 reparse 清除）。按时间顺序追加。 */
  private a2uiBlocks: UIBlock[] = [];

  /**
   * 追加一轮到时间线并重解析。仅 final 文本入时间线。
   * agent 轮先扫描内联 A2UI：
   *  - 有且全部有效 → 用映射出的结构化 block，文本残留作普通气泡；
   *  - 有但校验失败 → 不渲染，记录错误反馈（directive 3），后续回灌 LLM；
   *  - 无 → 维持原样（文本进 StreamUIAdapter 正则兜底）。
   */
  private appendChatTurn(role: string, content: string): void {
    const trimmed = (content || '').trim();
    if (trimmed.length === 0) {
      return;
    }

    if (role === 'agent') {
      const scan = A2UIParser.scan(trimmed);
      if (scan.found) {
        this.handleA2UIAgentTurn(scan);
        return;
      }
    }

    this.chatTurns = [...this.chatTurns, new RawTurn(role, trimmed)];
    this.reparseChatFlow();
  }

  /** Agent 轮含内联 A2UI 的处理。 */
  private handleA2UIAgentTurn(scan: A2UIScanResult): void {
    if (scan.errors.length > 0) {
      // 校验失败：不渲染卡片，把精确错误回灌给 LLM 让其重生成（directive 3）。
      this.requestA2UIRegeneration(scan.errors);
      // 残留文本（若有）仍按普通气泡呈现，保证不空屏。
      if (scan.residualText.length > 0) {
        this.chatTurns = [...this.chatTurns, new RawTurn('agent', scan.residualText)];
        this.reparseChatFlow();
      }
      return;
    }

    // 校验通过：把 A2UI 映射成 UIBlock，持久保存（reparse 不清除）。
    const blocks = A2UIMapper.toBlocks(scan.messages);
    if (scan.residualText.length > 0) {
      // 残留承接语作为前置气泡，进文本时间线。
      this.chatTurns = [...this.chatTurns, new RawTurn('agent', scan.residualText)];
    }
    for (let i = 0; i < blocks.length; i++) {
      this.a2uiBlocks.push(blocks[i]);
    }
    this.reparseChatFlow();
    this.a2uiAutoRetryUsed = false;
    this.notifyChatStateChanged();
  }

  /** 组织校验错误并自动回灌 LLM 一次。 */
  private requestA2UIRegeneration(errors: A2UIValidationError[]): void {
    const lines: string[] = [];
    for (let i = 0; i < errors.length && i < 8; i++) {
      lines.push('- ' + errors[i].path + ': ' + errors[i].message);
    }
    const feedback = 'A2UI 校验未通过，请仅修正以下问题并重新输出单条 ```a2ui 代码块（catalogId 必须是 '
      + 'intentx.app:cards@1，组件只能用 ClarifyQuestion/ApproachProposal/PlanConfirm/SkillCreate/ExecuteSummary/AgentText）：\n'
      + lines.join('\n');
    this.pendingA2UIErrorFeedback = feedback;
    console.warn('MainViewModel: A2UI validation failed:\n' + feedback);

    // 自动回灌一次（避免无限循环）。仅在已连接且空闲时。
    if (!this.a2uiAutoRetryUsed && this.isConnected && this.pendingRunCount === 0) {
      this.a2uiAutoRetryUsed = true;
      this.sendChat(feedback, this.chatThinkingLevel, []);
    }
  }

  private reparseChatFlow(): void {
    const parsed = StreamUIAdapter.parseTurns(this.chatTurns);
    // 持久的 A2UI 结构化 block 追加在文本解析结果之后（结构化优先，不被正则覆盖）。
    if (this.a2uiBlocks.length > 0) {
      const merged: UIBlock[] = [...parsed.blocks, ...this.a2uiBlocks];
      parsed.blocks = merged;
      const lastPhase = this.a2uiBlocks[this.a2uiBlocks.length - 1].phase;
      if (lastPhase.length > 0) {
        parsed.finalPhase = lastPhase;
      }
    }
    this.chatFlow = parsed;
  }

  /** 用整段历史重建时间线（loadChat 后调用）。 */
  private rebuildChatTurnsFromHistory(messages: ChatMessage[]): void {
    const turns: RawTurn[] = [];
    this.a2uiBlocks = [];
    this.a2uiAutoRetryUsed = false;
    for (let i = 0; i < messages.length; i++) {
      const msg = messages[i];
      const text = this.stringifyChatMessageContent(msg);
      if (text.length === 0) {
        continue;
      }
      const role = msg.role === 'user' ? 'user' : (msg.role === 'tool' ? 'tool' : 'agent');
      // 历史里的 agent 文本同样可能内联 A2UI：扫描并映射，残留文本入时间线。
      if (role === 'agent') {
        const scan = A2UIParser.scan(text);
        if (scan.found && scan.errors.length === 0) {
          if (scan.residualText.length > 0) {
            turns.push(new RawTurn('agent', scan.residualText));
          }
          const blocks = A2UIMapper.toBlocks(scan.messages);
          for (let b = 0; b < blocks.length; b++) {
            this.a2uiBlocks.push(blocks[b]);
          }
          continue;
        }
      }
      turns.push(new RawTurn(role, text));
    }
    this.chatTurns = turns;
    this.reparseChatFlow();
  }

  private stringifyChatMessageContent(message: ChatMessage): string {
    if (!message.content || message.content.length === 0) {
      return '';
    }

    const parts: string[] = [];
    for (let i = 0; i < message.content.length; i++) {
      const block = message.content[i];
      const text = (block.text || '').trim();
      if (text.length > 0) {
        parts.push(text);
      } else if ((block.type || '').trim().length > 0) {
        parts.push('[' + block.type + ']');
      }
    }
    return parts.join('\n\n');
  }

  private readAgentAssistantText(data: Record<string, Object | string>): string {
    const directText = this.readOptionalString(data, 'text');
    if (directText.length > 0) {
      return directText;
    }

    const delta = this.readNestedObject(data, 'delta');
    const deltaText = this.readOptionalString(delta, 'text');
    if (deltaText.length > 0) {
      return deltaText;
    }

    const message = this.readNestedObject(data, 'message');
    return this.parseAssistantDeltaText(message);
  }

  /** 从 tool result 事件 data 中提取文本内容（skill frontmatter / 写入回执 / 报错）。 */
  private readToolResultContent(data: Record<string, Object | string>): string {
    // 常见字段优先：result.content / result / content / output / text
    const result = this.readNestedObject(data, 'result');
    const fromResult = this.extractTextFromValue(result['content']) || this.readOptionalString(result, 'text')
      || this.readOptionalString(result, 'output');
    if (fromResult.length > 0) {
      return fromResult;
    }
    const direct = this.extractTextFromValue(data['content']) || this.readOptionalString(data, 'output')
      || this.readOptionalString(data, 'text');
    if (direct.length > 0) {
      return direct;
    }
    // 兜底：序列化 result 对象（用于 JSON 工具错误 §4.4）
    if (Object.keys(result).length > 0) {
      return JSON.stringify(result);
    }
    return '';
  }

  /** content 可能是 string，或 [{type,text}] / {text} 结构。 */
  private extractTextFromValue(value: Object | string | undefined): string {
    if (!value) {
      return '';
    }
    if (typeof value === 'string') {
      return value;
    }
    if (Array.isArray(value)) {
      const items = value as Array<Object>;
      const parts: string[] = [];
      for (let i = 0; i < items.length; i++) {
        const item = items[i] as Record<string, Object | string>;
        const t = this.readOptionalString(item, 'text');
        if (t.length > 0) {
          parts.push(t);
        }
      }
      return parts.join('\n');
    }
    const obj = value as Record<string, Object | string>;
    return this.readOptionalString(obj, 'text');
  }

  private parseChatMessageContent(contentValue: Object | string | undefined): ChatMessageContent[] {
    if (!contentValue) {
      return [];
    }

    if (typeof contentValue === 'string') {
      return this.createTextMessageContent(contentValue);
    }

    if (!Array.isArray(contentValue)) {
      const source = contentValue as Record<string, Object | string>;
      const directText = this.readOptionalString(source, 'text');
      if (directText.length > 0) {
        return this.createTextMessageContent(directText);
      }
      return [];
    }

    const sourceItems = contentValue as Array<Object>;
    const content: ChatMessageContent[] = [];
    for (let i = 0; i < sourceItems.length; i++) {
      const item = sourceItems[i] as Record<string, Object | string>;
      const type = this.readOptionalString(item, 'type');
      const text = this.readOptionalString(item, 'text');
      if (type !== 'text' && text.length === 0) {
        continue;
      }

      const block = new ChatMessageContent();
      block.type = type.length > 0 ? type : 'text';
      block.text = text;
      content.push(block);
    }
    return content;
  }

  private createTextMessageContent(text: string): ChatMessageContent[] {
    const trimmed = text.trim();
    if (trimmed.length === 0) {
      return [];
    }

    const block = new ChatMessageContent();
    block.type = 'text';
    block.text = trimmed;
    return [block];
  }

  private parseChatSessions(payload: Record<string, Object | string>): ChatSessionEntry[] {
    const sessionsValue = payload['sessions'];
    if (!sessionsValue || typeof sessionsValue === 'string' || !Array.isArray(sessionsValue)) {
      return [];
    }

    const sourceSessions = sessionsValue as Array<Object>;
    const parsedSessions: ChatSessionEntry[] = [];
    for (let i = 0; i < sourceSessions.length; i++) {
      const source = sourceSessions[i] as Record<string, Object | string>;
      const key = this.normalizeChatSessionKey(this.readOptionalString(source, 'key'));
      if (key.length === 0) {
        continue;
      }

      const entry = new ChatSessionEntry();
      entry.key = key;
      entry.displayName = this.readOptionalString(source, 'displayName');
      entry.updatedAtMs = this.readOptionalNumber(source, 'updatedAt');
      parsedSessions.push(entry);
    }
    return parsedSessions;
  }

  private readPayloadObject(payload: Object | string | undefined): Record<string, Object | string> {
    if (!payload || typeof payload === 'string') {
      return new ChatPayloadObject() as Record<string, Object | string>;
    }
    return payload as Record<string, Object | string>;
  }

  private parseJsonObject(payloadJson: string): Record<string, Object | string> {
    try {
      const parsed = JSON.parse(payloadJson) as Object;
      return parsed as Record<string, Object | string>;
    } catch (error) {
      console.warn('MainViewModel: parseJsonObject failed:', error);
      return new ChatPayloadObject() as Record<string, Object | string>;
    }
  }

  private readChatErrorMessage(source: Record<string, Object | string>): string {
    const directError = this.readOptionalString(source, 'errorMessage');
    if (directError.length > 0) {
      return directError;
    }

    const errorData = this.readNestedObject(source, 'data');
    const dataReason = this.readOptionalString(errorData, 'reason');
    if (dataReason.length > 0) {
      return dataReason;
    }

    const nestedError = this.readNestedObject(source, 'error');
    return this.readOptionalString(nestedError, 'message');
  }

  private describeObjectKeys(source: Record<string, Object | string>): string {
    const keys = Object.keys(source);
    if (keys.length === 0) {
      return '-';
    }
    return keys.join(',');
  }

  private logChatProtocolEvent(kind: string, payload: Record<string, Object | string>, marker: string, runId: string, sessionKey: string): void {
    console.info(
      'MainViewModel: protocol[' + kind + '] marker=' + marker +
      ', runId=' + runId +
      ', sessionKey=' + sessionKey +
      ', keys=' + this.describeObjectKeys(payload)
    );
  }

  private readNestedObject(source: Record<string, Object | string>, key: string): Record<string, Object | string> {
    const value = source[key];
    if (!value || typeof value === 'string') {
      return new ChatPayloadObject() as Record<string, Object | string>;
    }
    return value as Record<string, Object | string>;
  }

  private readOptionalString(source: Record<string, Object | string>, key: string): string {
    const value = source[key];
    return typeof value === 'string' ? value : '';
  }

  private readOptionalBoolean(source: Record<string, Object | string>, key: string): boolean | null {
    const value = source[key];
    if (typeof value === 'boolean') {
      return value;
    }
    if (typeof value === 'string') {
      const normalized = value.trim().toLowerCase();
      if (normalized === 'true') {
        return true;
      }
      if (normalized === 'false') {
        return false;
      }
    }
    return null;
  }

  private readOptionalNumber(source: Record<string, Object | string>, key: string): number {
    const value = source[key];
    if (typeof value === 'number') {
      return value;
    }
    if (typeof value === 'string') {
      const parsed = parseInt(value, 10);
      return Number.isNaN(parsed) ? 0 : parsed;
    }
    return 0;
  }

  private normalizeChatSessionKey(sessionKey: string | null | undefined): string {
    const trimmed = (sessionKey || '').trim();
    if (trimmed.length > 0) {
      return trimmed;
    }
    const mainSessionKey = (this.mainSessionKey || '').trim();
    return mainSessionKey.length > 0 ? mainSessionKey : 'main';
  }

  private normalizeChatThinkingLevel(level: string | null | undefined): string {
    const normalized = (level || '').trim().toLowerCase();
    switch (normalized) {
      case 'off':
      case 'low':
      case 'medium':
      case 'high':
        return normalized;
      default:
        return 'off';
    }
  }

  private createChatRunId(): string {
    const now = Date.now();
    const suffix = Math.floor(Math.random() * 1000000000);
    return 'run-' + String(now) + '-' + String(suffix);
  }

  private addChatPendingRun(runId: string): void {
    if (this.chatPendingRunIds.findIndex((value: string) => value === runId) >= 0) {
      return;
    }
    this.chatPendingRunIds = [...this.chatPendingRunIds, runId];
    this.pendingRunCount = this.chatPendingRunIds.length;
    this.scheduleChatPendingRunTimeout(runId);
    this.notifyChatStateChanged();
  }

  private clearChatPendingRun(runId: string): void {
    this.chatPendingRunIds = this.chatPendingRunIds.filter((value: string) => value !== runId);
    this.pendingRunCount = this.chatPendingRunIds.length;
    this.notifyChatStateChanged();
  }

  private clearChatPendingRuns(): void {
    this.chatPendingRunIds = [];
    this.pendingRunCount = 0;
    this.notifyChatStateChanged();
  }

  private isChatRunPending(runId: string): boolean {
    return this.chatPendingRunIds.findIndex((value: string) => value === runId) >= 0;
  }

  private scheduleChatPendingRunTimeout(runId: string): void {
    setTimeout(() => {
      if (!this.isChatRunPending(runId)) {
        return;
      }
      this.clearChatPendingRun(runId);
      this.chatStreamingAssistantText = null;
      this.chatPendingToolCalls = [];
      this.chatPendingInteractions = [];
      this.chatError = this.getLocalizedString('chat_error_pending_timeout', 'Reply timed out, please retry or refresh.');
      this.notifyChatStateChanged();
    }, this.chatPendingRunTimeoutMs);
  }

  private readChatRunIdFromPayload(payloadValue: Object | string | undefined): string {
    const payload = this.readPayloadObject(payloadValue);
    return this.readOptionalString(payload, 'runId');
  }

  private upsertPendingToolCall(toolCall: ChatPendingToolCall): void {
    const existingIndex = this.chatPendingToolCalls.findIndex((entry: ChatPendingToolCall) => entry.id === toolCall.id);
    if (existingIndex >= 0) {
      const updated = [...this.chatPendingToolCalls];
      updated[existingIndex] = toolCall;
      this.chatPendingToolCalls = updated;
      this.notifyChatStateChanged();
      return;
    }
    this.chatPendingToolCalls = [...this.chatPendingToolCalls, toolCall];
    this.notifyChatStateChanged();
  }

  private removePendingToolCall(toolCallId: string): void {
    this.chatPendingToolCalls = this.chatPendingToolCalls.filter((entry: ChatPendingToolCall) => entry.id !== toolCallId);
    this.notifyChatStateChanged();
  }

  requestCanvasRehydrate(source: string = 'manual', force: boolean = true): void {
    void source;
    void force;
    if (!this.isConnected) {
      this.canvasRehydrateErrorText = this.getLocalizedString('screen_reason_not_connected', 'Connect to a gateway before expecting HarmonyOS screen state or canvas metadata.');
      this.canvasRehydratePending = false;
      return;
    }
    if (!this.isNodeConnected) {
      this.canvasRehydrateErrorText = this.getLocalizedString('screen_reason_node_unavailable', 'The gateway connection is up, but the node runtime is not ready for screen state yet.');
      this.canvasRehydratePending = false;
      return;
    }

    const currentUrl = (this.canvasCurrentUrl || '').trim();
    if (currentUrl.length === 0) {
      this.canvasRehydrateErrorText = this.getLocalizedString('screen_reason_canvas_unavailable', 'No canvas URL has been discovered yet, so this page stays in a deterministic unavailable state.');
      this.canvasRehydratePending = false;
      return;
    }

    this.canvasRehydratePending = true;
    this.canvasRehydrateErrorText = null;
    const session = this.nodeSession;
    if (!session) {
      this.canvasRehydrateErrorText = this.getLocalizedString('common_status_gateway_session_not_ready', 'Gateway session not ready');
      this.canvasRehydratePending = false;
      return;
    }

    session.refreshNodeCanvasCapability().then((ok: boolean): void => {
      if (!ok) {
        this.canvasRehydratePending = false;
        this.canvasRehydrateErrorText = this.getLocalizedString('screen_reason_error', 'Canvas state refresh failed.');
        return;
      }
      const refreshedUrl = session.currentCanvasHostUrl();
      if (refreshedUrl && refreshedUrl.trim().length > 0) {
        this.canvasCurrentUrl = refreshedUrl;
        this.canvasA2uiHydrated = true;
      }
      setTimeout(() => {
        this.canvasRehydratePending = false;
      }, 300);
    }).catch((error: Error): void => {
      this.canvasRehydratePending = false;
      this.canvasRehydrateErrorText = error instanceof Error
        ? error.message
        : this.getLocalizedString('screen_reason_error', 'Canvas state refresh failed.');
    });
  }

  handleCanvasA2UIActionFromWebView(payloadJson: string): void {
    console.info('MainViewModel: Canvas A2UI action:', payloadJson);
  }

  setForeground(value: boolean): void {
    this.isForeground = value;
  }

  refreshLocalizedUiState(): void {
    this.refreshResourceBackedStatusText();
    this.refreshResourceBackedDiscoveryText();
    this.refreshResourceBackedMicText();
    this.seedMockDiscovery();
    this.chatSessions = this.chatSessions.map((entry: ChatSessionEntry): ChatSessionEntry => {
      if (entry.key !== this.mainSessionKey) {
        return entry;
      }
      const updated = new ChatSessionEntry();
      updated.key = entry.key;
      updated.displayName = this.getLocalizedString('common_main', 'Main');
      updated.updatedAtMs = entry.updatedAtMs;
      return updated;
    });
    this.notifyConnectionStateChanged();
    this.notifyChatStateChanged();
  }

  private setStatusTextFromResource(name: string, fallback: string): void {
    this.statusTextResourceName = name;
    this.statusText = this.getLocalizedString(name, fallback);
  }

  private setDiscoveryStatusFromResource(name: string, fallback: string): void {
    this.discoveryStatusResourceName = name;
    this.discoveryStatusText = this.getLocalizedString(name, fallback);
  }

  private setMicStatusFromResource(name: string, fallback: string): void {
    this.micStatusResourceName = name;
    this.micStatusText = this.getLocalizedString(name, fallback);
  }

  private refreshResourceBackedStatusText(): void {
    switch (this.statusTextResourceName) {
      case 'chat_connected':
        this.statusText = this.getLocalizedString('chat_connected', 'Connected');
        break;
      case 'chat_offline':
        this.statusText = this.getLocalizedString('chat_offline', 'Offline');
        break;
      case 'common_status_gateway_session_not_ready':
        this.statusText = this.getLocalizedString('common_status_gateway_session_not_ready', 'Gateway session not ready');
        break;
      case 'common_status_connecting':
        this.statusText = this.getLocalizedString('common_status_connecting', 'Connecting…');
        break;
      case 'common_status_connection_failed':
        this.statusText = this.getLocalizedString('common_status_connection_failed', 'Connection failed');
        break;
      case 'common_status_invalid_manual_config':
        this.statusText = this.getLocalizedString('common_status_invalid_manual_config', 'Invalid manual gateway configuration');
        break;
      case 'common_status_invalid_hermes_config':
        this.statusText = this.getLocalizedString('common_status_invalid_hermes_config', 'Invalid Hermes gateway configuration');
        break;
      case 'common_status_missing_setup_code':
        this.statusText = this.getLocalizedString('common_status_missing_setup_code', 'Missing setup code');
        break;
      case 'common_status_unsupported_setup_code':
        this.statusText = this.getLocalizedString('common_status_unsupported_setup_code', 'Unsupported setup code format. Use a gateway URL or host:port for now.');
        break;
      case 'common_status_disconnected':
        this.statusText = this.getLocalizedString('common_status_disconnected', 'Disconnected');
        break;
      case 'common_status_initialization_failed':
        this.statusText = this.getLocalizedString('common_status_initialization_failed', 'Initialization failed');
        break;
      default:
        break;
    }
  }

  private refreshResourceBackedDiscoveryText(): void {
    switch (this.discoveryStatusResourceName) {
      case 'connect_discovery_ready':
        this.discoveryStatusText = this.getLocalizedString('connect_discovery_ready', 'Discovery ready');
        break;
      case 'common_status_initialization_failed':
        this.discoveryStatusText = this.getLocalizedString('common_status_initialization_failed', 'Initialization failed');
        break;
      default:
        break;
    }
  }

  private refreshResourceBackedMicText(): void {
    switch (this.micStatusResourceName) {
      case 'settings_ready':
        this.micStatusText = this.getLocalizedString('settings_ready', 'Ready');
        break;
      case 'settings_off':
        this.micStatusText = this.getLocalizedString('settings_off', 'Off');
        break;
      default:
        break;
    }
  }

  /**
   * Retrieve a localized string by resource name via the UIAbilityContext stored in AppStorage.
   * Falls back to the provided English string when the context or key is unavailable.
   */
  private getLocalizedString(name: string, fallback: string): string {
    try {
      const ctx = AppStorage.get<Context>('abilityContext');
      if (ctx) {
        return ctx.resourceManager.getStringByNameSync(name);
      }
    } catch (e) {
      console.warn('MainViewModel: getLocalizedString failed for key: ' + name);
    }
    return fallback;
  }
}

class MainViewModelNodeCallbacks implements NodeInvokeDispatcherCallbacks {
  private viewModel: MainViewModel;

  constructor(viewModel: MainViewModel) {
    this.viewModel = viewModel;
  }

  isForeground(): boolean {
    return this.viewModel.isForeground;
  }

  getStatusText(): string {
    return this.viewModel.statusText;
  }

  isOperatorConnected(): boolean {
    return this.viewModel.isConnected;
  }

  isNodeConnected(): boolean {
    return this.viewModel.isNodeConnected;
  }

  getCanvasUrl(): string {
    return this.viewModel.canvasCurrentUrl || '';
  }

  setCanvasUrl(url: string): void {
    this.viewModel.canvasCurrentUrl = url;
    this.viewModel.canvasA2uiHydrated = url.trim().length > 0;
    this.viewModel.canvasRehydratePending = false;
    this.viewModel.canvasRehydrateErrorText = null;
  }

  clearCanvasUrl(): void {
    this.viewModel.canvasCurrentUrl = null;
    this.viewModel.canvasA2uiHydrated = false;
    this.viewModel.canvasRehydratePending = false;
  }
}

export const mainViewModel = new MainViewModel();
