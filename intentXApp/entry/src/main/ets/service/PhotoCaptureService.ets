import { cameraPicker } from '@kit.CameraKit';
import { photoAccessHelper } from '@kit.MediaLibraryKit';
import { fileIo } from '@kit.CoreFileKit';
import { image } from '@kit.ImageKit';
import { util } from '@kit.ArkTS';
import { BusinessError } from '@kit.BasicServicesKit';
import { mainViewModel } from '../viewmodel/MainViewModel';
import { OutgoingAttachment } from '../model/GatewayModels';

/** 客户端侧安全上限（原始文件大小），超过拒绝以防止 OOM。 */
const MAX_FILE_SIZE_BYTES: number = 20 * 1024 * 1024;
/** 服务端硬限制（解码后大小），客户端应通过压缩确保低于此值。 */
const SERVER_MAX_SIZE_BYTES: number = 5 * 1024 * 1024;
/**
 * 服务端 parseMessageWithAttachments 的 offload 阈值为 2MB 解码后文件大小。
 * 图片压缩目标：压缩后 JPEG ≤ 1.5MB，保留足够余量确保内联传输，避免 offload。
 */
const MAX_INLINE_BYTES: number = 1.5 * 1024 * 1024;
/** 压缩时 JPEG 质量 (0-100)，取值越高文件越大画质越好 */
const JPEG_QUALITY: number = 80;
/** 迭代压缩时初始长边最大像素数 */
const INITIAL_MAX_DIMENSION: number = 1920;
/** 迭代压缩时最小边长兜底，防止无限循环 */
const MIN_DIMENSION: number = 256;

export class PhotoResult {
  uri: string = '';
  displayName: string = '';
}

export class PhotoCaptureService {
  private base64Helper: util.Base64Helper = new util.Base64Helper();

  /**
   * 从相册选取照片，使用 PhotoViewPicker 安全控件模式。
   * 用户取消选取时返回 null（不抛异常）。
   */
  async pickFromAlbum(): Promise<PhotoResult | null> {
    try {
      let picker: photoAccessHelper.PhotoViewPicker = new photoAccessHelper.PhotoViewPicker();
      let options: photoAccessHelper.PhotoSelectOptions = new photoAccessHelper.PhotoSelectOptions();
      options.MIMEType = photoAccessHelper.PhotoViewMIMETypes.IMAGE_TYPE;
      options.maxSelectNumber = 1;

      let result: photoAccessHelper.PhotoSelectResult = await picker.select(options);
      if (!result || !result.photoUris || result.photoUris.length === 0) {
        console.info('PhotoCaptureService: album pick cancelled or returned no URI');
        return null;
      }

      let uri: string = result.photoUris[0];
      let photoResult: PhotoResult = new PhotoResult();
      photoResult.uri = uri;
      photoResult.displayName = this.extractDisplayName(uri);
      console.info('PhotoCaptureService: album pick success, uri=' + uri);
      return photoResult;
    } catch (error) {
      let bizErr: BusinessError = error as BusinessError;
      console.error('PhotoCaptureService: pickFromAlbum error, code=' + bizErr.code + ', msg=' + bizErr.message);
      return null;
    }
  }

  /**
   * 使用相机拍照，使用 cameraPicker 安全控件模式。
   * 拍照后自动存入系统相册，返回存入后的 URI。
   * 用户取消拍照时返回 null（不抛异常）。
   */
  async capturePhoto(): Promise<PhotoResult | null> {
    try {
      let ctx: Context | undefined = AppStorage.get<Context>('abilityContext');
      if (!ctx) {
        console.error('PhotoCaptureService: abilityContext not found in AppStorage');
        return null;
      }

      let pickerProfile: cameraPicker.PickerProfile = new cameraPicker.PickerProfile();
      let result: cameraPicker.PickerResult = await cameraPicker.pick(
        ctx,
        [cameraPicker.PickerMediaType.PHOTO],
        pickerProfile
      );

      if (!result || !result.resultUri || result.resultUri.length === 0) {
        console.info('PhotoCaptureService: camera capture cancelled or returned no URI');
        return null;
      }

      let uri: string = result.resultUri;
      let photoResult: PhotoResult = new PhotoResult();
      photoResult.uri = uri;
      photoResult.displayName = this.extractDisplayName(uri);
      console.info('PhotoCaptureService: camera capture success, uri=' + uri);
      return photoResult;
    } catch (error) {
      let bizErr: BusinessError = error as BusinessError;
      console.error('PhotoCaptureService: capturePhoto error, code=' + bizErr.code + ', msg=' + bizErr.message);
      return null;
    }
  }

  /**
   * 照片后处理入口。读取照片并作为附件以聊天消息形式发送给 OpenClaw 服务端。
   * 消息内容为 "/ImageData"。
   *
   * @param photoResult 拍摄或选取得到的照片信息（uri + displayName）
   */
  async obtainPhotoToProcess(photoResult: PhotoResult): Promise<void> {
    console.info(
      'PhotoCaptureService: obtainPhotoToProcess called, uri=' + photoResult.uri +
      ', displayName=' + photoResult.displayName
    );

    try {
      // 打开文件 — fd 必须在 finally 中关闭
      let file = fileIo.openSync(photoResult.uri, fileIo.OpenMode.READ_ONLY);
      try {
        let stat = fileIo.statSync(file.fd);

        // P0-02: 前置校验文件大小
        if (stat.size <= 0) {
          console.error('PhotoCaptureService: file is empty, uri=' + photoResult.uri);
          AppStorage.setOrCreate<string>('shareReceiveStatus', 'fail');
          AppStorage.setOrCreate<string>('shareReceiveError', 'Image file is empty.');
          return;
        }
        // P1-01: 安全上限检查，防止 OOM（仅拒绝 > 20MB 的极端文件）
        if (stat.size > MAX_FILE_SIZE_BYTES) {
          console.error(
            'PhotoCaptureService: file too large for safety (' + stat.size + ' bytes), max=' + MAX_FILE_SIZE_BYTES
          );
          AppStorage.setOrCreate<string>('shareReceiveStatus', 'fail');
          AppStorage.setOrCreate<string>(
            'shareReceiveError',
            'Image is too large (' + Math.round(stat.size / 1048576) +
            ' MB). Maximum is 20 MB.'
          );
          return;
        }

        let buffer = new ArrayBuffer(stat.size);
        fileIo.readSync(file.fd, buffer);

        // 迭代压缩：反复将宽高减半，直到压缩后 JPEG ≤ 1.5MB
        let finalBuffer = await this.compressToTarget(buffer, MAX_INLINE_BYTES);
        let wasCompressed = finalBuffer !== buffer;

        // 压缩后仍超 5MB（服务端硬限制），拒绝发送
        if (finalBuffer.byteLength > SERVER_MAX_SIZE_BYTES) {
          console.error(
            'PhotoCaptureService: compressed file still too large (' + finalBuffer.byteLength +
            ' bytes), cannot send'
          );
          AppStorage.setOrCreate<string>('shareReceiveStatus', 'fail');
          AppStorage.setOrCreate<string>(
            'shareReceiveError',
            'Image could not be compressed enough. Please try a smaller image.'
          );
          return;
        }

        // 使用系统 Base64Helper 编码
        let base64 = await this.base64Helper.encodeToString(new Uint8Array(finalBuffer));

        // 通过文件后缀推断 mimeType；压缩后统一输出 JPEG
        let mimeType = wasCompressed ? 'image/jpeg' : this.inferMimeType(photoResult.displayName || photoResult.uri);
        let fileName = photoResult.displayName || ('photo.' + this.extensionFromMimeType(mimeType));

        // 构造附件
        let attachment = new OutgoingAttachment();
        attachment.type = 'image';
        attachment.mimeType = mimeType;
        attachment.fileName = fileName;
        attachment.content = base64;

        // 发送到专用的 ImportImage 会话，避免主对话被心跳刷屏
        const SHARE_TAG = '[Share2Agent]';
        console.info(
          SHARE_TAG + ' about to sendChat: isConnected=' + mainViewModel.isConnected +
          ', chatSessionKey=' + mainViewModel.chatSessionKey +
          ', prevError=' + (mainViewModel.chatError || '(none)')
        );
        const prevError = mainViewModel.chatError;
        // 确定发送文本：skill 非空时用 /skill <name>，否则默认 /ImageData
        const skill = AppStorage.get<string>('cardSkill');
        const msgText = (skill && skill.trim().length > 0)
          ? ('/skill ' + skill.trim())
          : '/ImageData';
        console.info(SHARE_TAG + ' sendChat msgText=' + msgText + ' skill=' + (skill || '(none)'));
        // sessionKey='ImportImage' — 发到专用会话，避免主对话被心跳刷屏
        mainViewModel.sendChat(msgText, 'off', [attachment], undefined, 'main');
        // sendChat 内部若检测到 isConnected==false，会设置 chatError 并 return
        const sendFailed = (mainViewModel.chatError !== null && mainViewModel.chatError !== prevError);
        if (sendFailed) {
          AppStorage.setOrCreate<string>('shareReceiveStatus', 'fail');
          AppStorage.setOrCreate<string>(
            'shareReceiveError',
            'Not connected to Gateway. Please open IntentX App, go to Connect tab, and ensure it is connected before sharing.'
          );
          console.warn(
            SHARE_TAG + ' sendChat FAILED: isConnected=' + mainViewModel.isConnected +
            ', chatError=' + mainViewModel.chatError
          );
        } else {
          AppStorage.setOrCreate<string>('shareReceiveStatus', 'success');
        }
        console.info(
          SHARE_TAG + ' done: finalSize=' + finalBuffer.byteLength +
          ' bytes, wasCompressed=' + wasCompressed + ', sendOk=' + !sendFailed
        );
      } finally {
        // P0-01: 无论 read/encode 是否异常，fd 必须关闭
        fileIo.closeSync(file);
      }
    } catch (error) {
      let bizErr: BusinessError = error as BusinessError;
      console.error(
        'PhotoCaptureService: obtainPhotoToProcess error, code=' + bizErr.code +
        ', msg=' + bizErr.message
      );
      AppStorage.setOrCreate<string>('shareReceiveStatus', 'fail');
      AppStorage.setOrCreate<string>(
        'shareReceiveError',
        'Failed to process image: ' + (bizErr.message || 'Unknown error.')
      );
    }
  }

  /**
   * 从外部 URI 处理图片（系统分享等入口）。
   * 构造 PhotoResult 后委托给 obtainPhotoToProcess 完成后续管线。
   *
   * @param uri 图片文件 URI
   */
  processSharedUri(uri: string): void {
    const photoResult = new PhotoResult();
    photoResult.uri = uri;
    photoResult.displayName = this.extractDisplayName(uri);
    this.obtainPhotoToProcess(photoResult);
  }

  /**
   * 迭代压缩图片到目标文件大小。
   * 策略：反复将宽高各减半，直到 JPEG 编码后 ≤ targetBytes。
   * 最大迭代从 INITIAL_MAX_DIMENSION 开始，最细到 MIN_DIMENSION 兜底。
   *
   * @param buffer 原始图片字节
   * @param targetBytes 目标最大字节数
   * @returns 压缩后的 ArrayBuffer（如果原图已足够小则返回原 buffer）
   */
  private async compressToTarget(buffer: ArrayBuffer, targetBytes: number): Promise<ArrayBuffer> {
    if (buffer.byteLength <= targetBytes) {
      return buffer;
    }

    console.info(
      'PhotoCaptureService: compressToTarget start, original=' + buffer.byteLength +
      ' bytes, target≤' + targetBytes
    );

    let source: image.ImageSource | null = null;
    try {
      source = image.createImageSource(buffer);
      let info = await source.getImageInfo(0);
      let ow = info.size.width;
      let oh = info.size.height;

      // 初始缩放比例：让长边不超过 INITIAL_MAX_DIMENSION
      let scale = Math.min(1.0, INITIAL_MAX_DIMENSION / Math.max(ow, oh));
      let iteration = 0;

      while (iteration < 10) {  // 安全上限，最多 10 轮
        let tw = Math.max(1, Math.floor(ow * scale));
        let th = Math.max(1, Math.floor(oh * scale));

        console.info(
          'PhotoCaptureService: compressToTarget iteration ' + iteration +
          ', scale=' + Number(scale).toFixed(3) + ', target=' + tw + 'x' + th
        );

        let decodeOptions: image.DecodingOptions = {
          desiredSize: { width: tw, height: th }
        };
        let pixelMap = await source.createPixelMap(decodeOptions);

        let packer = image.createImagePacker();
        let packOptions: image.PackingOption = {
          format: 'image/jpeg',
          quality: JPEG_QUALITY
        };
        let compressed = await packer.packing(pixelMap, packOptions);
        pixelMap.release();

        if (compressed.byteLength <= targetBytes) {
          console.info(
            'PhotoCaptureService: compressToTarget success, ' + buffer.byteLength +
            '→' + compressed.byteLength + ' bytes (' + tw + 'x' + th + ') in ' + iteration + ' iteration(s)'
          );
          return compressed;
        }

        // 兜底：不再继续缩小
        if (Math.min(tw, th) <= MIN_DIMENSION) {
          console.info(
            'PhotoCaptureService: compressToTarget floor hit (' + tw + 'x' + th +
            '), returning best-effort ' + compressed.byteLength + ' bytes'
          );
          return compressed;
        }

        // 长宽各减半，面积约 1/4
        scale = scale / 2;
        iteration++;
      }

      // 不应到达此处（有 MIN_DIMENSION 兜底），但保留保底返回
      console.warn('PhotoCaptureService: compressToTarget max iterations reached');
      return buffer;
    } catch (error) {
      console.warn(
        'PhotoCaptureService: compressToTarget failed, using original buffer: ' +
        (error as Error).message
      );
      return buffer;
    } finally {
      if (source !== null) {
        source.release();
      }
    }
  }

  /**
   * 根据文件名后缀推断图片 MIME 类型。
   */
  private inferMimeType(fileName: string): string {
    if (!fileName || fileName.length === 0) {
      return 'image/jpeg';
    }
    let lower = fileName.toLowerCase();
    if (lower.endsWith('.png')) {
      return 'image/png';
    }
    if (lower.endsWith('.webp')) {
      return 'image/webp';
    }
    if (lower.endsWith('.gif')) {
      return 'image/gif';
    }
    if (lower.endsWith('.heic') || lower.endsWith('.heif')) {
      return 'image/heic';
    }
    if (lower.endsWith('.bmp')) {
      return 'image/bmp';
    }
    return 'image/jpeg';
  }

  /**
   * 根据 MIME 类型返回文件扩展名（不含点号）。
   */
  private extensionFromMimeType(mime: string): string {
    if (mime === 'image/png') {
      return 'png';
    }
    if (mime === 'image/webp') {
      return 'webp';
    }
    if (mime === 'image/gif') {
      return 'gif';
    }
    if (mime === 'image/heic') {
      return 'heic';
    }
    return 'jpg';
  }

  private extractDisplayName(uri: string): string {
    if (!uri || uri.length === 0) {
      return '';
    }
    let lastSlash: number = uri.lastIndexOf('/');
    if (lastSlash >= 0 && lastSlash < uri.length - 1) {
      return uri.substring(lastSlash + 1);
    }
    return uri;
  }
}

export const photoCaptureService = new PhotoCaptureService();
