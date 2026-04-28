import { deviceInfo } from '@kit.BasicServicesKit';
import { InvokeRequest, InvokeResult } from '../api/GatewaySession';
import {
  OpenClawCanvasCommand,
  OpenClawDeviceCommand
} from '../protocol/OpenClawProtocolConstants';
import { InvokeCommandRegistry } from './InvokeCommandRegistry';

export interface NodeInvokeDispatcherCallbacks {
  isForeground(): boolean;
  getStatusText(): string;
  isOperatorConnected(): boolean;
  isNodeConnected(): boolean;
  getCanvasUrl(): string;
  setCanvasUrl(url: string): void;
  clearCanvasUrl(): void;
}

export class NodeInvokeDispatcher {
  private callbacks: NodeInvokeDispatcherCallbacks;

  constructor(callbacks: NodeInvokeDispatcherCallbacks) {
    this.callbacks = callbacks;
  }

  async handleInvoke(request: InvokeRequest): Promise<InvokeResult> {
    const spec = InvokeCommandRegistry.find(request.command);
    if (!spec) {
      return InvokeResult.error('INVALID_REQUEST', 'INVALID_REQUEST: unknown command');
    }
    if (spec.requiresForeground && !this.callbacks.isForeground()) {
      return InvokeResult.error(
        'NODE_BACKGROUND_UNAVAILABLE',
        'NODE_BACKGROUND_UNAVAILABLE: canvas commands require foreground'
      );
    }

    switch (request.command) {
      case OpenClawCanvasCommand.Present:
      case OpenClawCanvasCommand.Navigate:
        return this.handleCanvasNavigate(request.paramsJson);
      case OpenClawCanvasCommand.Hide:
        this.callbacks.clearCanvasUrl();
        return InvokeResult.ok('{}');
      case OpenClawDeviceCommand.Status:
        return InvokeResult.ok(this.buildDeviceStatusPayload());
      case OpenClawDeviceCommand.Info:
        return InvokeResult.ok(this.buildDeviceInfoPayload());
      case OpenClawDeviceCommand.Permissions:
        return InvokeResult.ok(this.buildDevicePermissionsPayload());
      case OpenClawDeviceCommand.Health:
        return InvokeResult.ok(this.buildDeviceHealthPayload());
      default:
        return InvokeResult.error('INVALID_REQUEST', 'INVALID_REQUEST: unknown command');
    }
  }

  private handleCanvasNavigate(paramsJson: string): InvokeResult {
    const url = this.parseCanvasUrl(paramsJson);
    this.callbacks.setCanvasUrl(url);
    return InvokeResult.ok('{}');
  }

  private parseCanvasUrl(paramsJson: string): string {
    const trimmed = paramsJson.trim();
    if (trimmed.length === 0) {
      return '';
    }
    try {
      const parsed = JSON.parse(trimmed) as Record<string, Object | string>;
      const directUrl = parsed['url'];
      if (typeof directUrl === 'string') {
        return directUrl.trim();
      }
      const href = parsed['href'];
      if (typeof href === 'string') {
        return href.trim();
      }
    } catch (_) {
    }
    return '';
  }

  private buildDeviceStatusPayload(): string {
    return JSON.stringify({
      platform: 'harmonyos',
      status: this.callbacks.getStatusText(),
      connected: this.callbacks.isOperatorConnected(),
      nodeConnected: this.callbacks.isNodeConnected(),
      foreground: this.callbacks.isForeground(),
      canvasUrl: this.callbacks.getCanvasUrl()
    });
  }

  private buildDeviceInfoPayload(): string {
    return JSON.stringify({
      platform: 'harmonyos',
      manufacturer: deviceInfo.manufacture,
      brand: deviceInfo.brand,
      productModel: deviceInfo.productModel,
      osFullName: deviceInfo.osFullName,
      sdkApiVersion: deviceInfo.sdkApiVersion
    });
  }

  private buildDevicePermissionsPayload(): string {
    return JSON.stringify({
      camera: 'not_requested',
      location: 'not_requested',
      microphone: 'not_requested',
      notifications: 'not_implemented',
      sms: 'not_implemented',
      contacts: 'not_implemented',
      calendar: 'not_implemented',
      callLog: 'not_implemented'
    });
  }

  private buildDeviceHealthPayload(): string {
    return JSON.stringify({
      ok: true,
      platform: 'harmonyos',
      operatorConnected: this.callbacks.isOperatorConnected(),
      nodeConnected: this.callbacks.isNodeConnected()
    });
  }
}
