/**
 * Gateway endpoint model
 */
export class GatewayEndpoint {
  stableId: string = '';
  name: string = '';
  host: string = '';
  port: number = 0;
  tlsEnabled: boolean = false;
}

/**
 * Gateway authentication response
 */
export interface GatewayAuthResponse {
  serverName: string;
  mainSessionKey: string;
  protocolVersion?: string;
  serverAddress?: string;
}

/**
 * Chat message content block.
 * type='text' → text 字段存文本
 * type='image' → imageUri 字段存本地 URI（用于缩略图展示）
 */
export class ChatMessageContent {
  type: string = 'text';
  text: string = '';
  imageUri: string = '';
}

/**
 * Chat message
 */
export class ChatMessage {
  stableKey: string = '';
  role: string = '';
  content: ChatMessageContent[] = [];
}

/**
 * Chat session entry
 */
export class ChatSessionEntry {
  key: string = '';
  displayName: string = '';
  updatedAtMs: number = 0;
}

/**
 * Chat pending tool call
 */
export class ChatPendingToolCall {
  id: string = '';
  name: string = '';
  paramsJson: string = '';
}

/**
 * Assistant-rendered choice option parsed from a chat message
 */
export class ChatChoiceOption {
  index: number = 0;
  label: string = '';
  value: string = '';
}

/**
 * User-facing pending interaction choice
 */
export class ChatPendingInteractionChoice {
  id: string = '';
  label: string = '';
  value: string = '';
  style: string = 'secondary';
}

/**
 * Optional input metadata for a pending interaction
 */
export class ChatPendingInteractionInput {
  type: string = 'text';
  placeholder: string = '';
}

/**
 * Backend response target for a pending interaction
 */
export class ChatPendingInteractionBackend {
  method: string = '';
  requestId: string = '';
  sessionId: string = '';
  runId: string = '';
}

/**
 * Ordinary-user interaction request rendered in chat
 */
export class ChatPendingInteraction {
  id: string = '';
  kind: string = 'clarification';
  title: string = '';
  body: string = '';
  riskLevel: string = 'low';
  choices: ChatPendingInteractionChoice[] = [];
  input: ChatPendingInteractionInput | null = null;
  backend: ChatPendingInteractionBackend = new ChatPendingInteractionBackend();
  status: string = 'pending';
}

/**
 * Outgoing attachment
 */
export class OutgoingAttachment {
  type: string = '';
  mimeType: string = '';
  fileName: string = '';
  content: string = '';
    data: string = '';
}

/**
 * Voice conversation entry
 */
export class VoiceConversationEntry {
  role: string = '';
  text: string = '';
}

/**
 * Gateway protocol version constant
 * 老版本的Gateway服务请使用3，新版本的Gateway服务请改成4。新版本(如2026.6.5)的Gateway报错样式：
 * 18:39:55 [ws] protocol mismatch conn=e2ceb847-da86-4d09-a9a5-1cbee8c5ec52 peer=192.168.3.2:35646->192.168.3.5:18789 remote=192.168.3.2 remotePort=35646 client=HarmonyOS Device ui v2026.3.2 min=3 max=3 expected=4 probeMin=4 instance=a38ad1cd-3dfc-49e9-b419-f514424233b9           
 * 18:39:55 [ws] protocol mismatch conn=c485067b-65ff-4669-a41b-653e815a8194 peer=192.168.3.2:35647->192.168.3.5:18789 remote=192.168.3.2 remotePort=35647 client=HarmonyOS Device node v2026.3.2 min=3 max=3 expected=4 probeMin=4 instance=a38ad1cd-3dfc-49e9-b419-f514424233b9
 */
export const GATEWAY_PROTOCOL_VERSION = 3;
//export const GATEWAY_PROTOCOL_VERSION = 4;
