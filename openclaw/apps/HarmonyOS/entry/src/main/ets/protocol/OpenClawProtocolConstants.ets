export enum OpenClawCapability {
  Canvas = 'canvas',
  Camera = 'camera',
  Sms = 'sms',
  VoiceWake = 'voiceWake',
  Location = 'location',
  Device = 'device',
  Notifications = 'notifications',
  System = 'system',
  Photos = 'photos',
  Contacts = 'contacts',
  Calendar = 'calendar',
  Motion = 'motion',
  CallLog = 'callLog'
}

export enum OpenClawCanvasCommand {
  Present = 'canvas.present',
  Hide = 'canvas.hide',
  Navigate = 'canvas.navigate',
  Eval = 'canvas.eval',
  Snapshot = 'canvas.snapshot'
}

export enum OpenClawCanvasA2UICommand {
  Push = 'canvas.a2ui.push',
  PushJSONL = 'canvas.a2ui.pushJSONL',
  Reset = 'canvas.a2ui.reset'
}

export enum OpenClawCameraCommand {
  List = 'camera.list',
  Snap = 'camera.snap',
  Clip = 'camera.clip'
}

export enum OpenClawSmsCommand {
  Send = 'sms.send',
  Search = 'sms.search'
}

export enum OpenClawLocationCommand {
  Get = 'location.get'
}

export enum OpenClawDeviceCommand {
  Status = 'device.status',
  Info = 'device.info',
  Permissions = 'device.permissions',
  Health = 'device.health'
}

export enum OpenClawNotificationsCommand {
  List = 'notifications.list',
  Actions = 'notifications.actions'
}

export enum OpenClawSystemCommand {
  Notify = 'system.notify'
}

export enum OpenClawPhotosCommand {
  Latest = 'photos.latest'
}

export enum OpenClawContactsCommand {
  Search = 'contacts.search',
  Add = 'contacts.add'
}

export enum OpenClawCalendarCommand {
  Events = 'calendar.events',
  Add = 'calendar.add'
}

export enum OpenClawMotionCommand {
  Activity = 'motion.activity',
  Pedometer = 'motion.pedometer'
}

export enum OpenClawCallLogCommand {
  Search = 'callLog.search'
}
