import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const root = process.cwd();
const chatPagePath = path.join(root, 'entry/src/main/ets/pages/ChatPage.ets');
const botIconPath = path.join(root, 'entry/src/main/resources/base/media/lucide_bot.svg');

const source = fs.readFileSync(chatPagePath, 'utf8');

function assert(condition, message) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    process.exitCode = 1;
  }
}

function builderBody(name) {
  const marker = `@Builder ${name}(`;
  const start = source.indexOf(marker);
  assert(start >= 0, `${name} builder exists`);
  if (start < 0) {
    return '';
  }
  const next = source.indexOf('\n  @Builder ', start + marker.length);
  return source.slice(start, next < 0 ? source.length : next);
}

assert(fs.existsSync(botIconPath), 'lucide Bot icon is copied into app media resources');
assert(source.includes("$r('app.media.lucide_bot')"), 'assistant avatar references local lucide Bot resource');
assert(source.includes('this.ChatHeader();'), 'build renders compact ChatHeader');
assert(source.includes('this.WorkflowProgress();'), 'build renders workflow progress outside message timeline');
assert(!builderBody('FlowTimeline').includes('FlowProgressBar'), 'FlowTimeline no longer inserts progress bars inside messages');
assert(/this\.SessionList\(\)/.test(builderBody('HeaderSessionPanel')), 'session list is attached to the header area');
assert(!builderBody('Composer').includes('this.SessionList();'), 'composer does not render the session list panel');
assert(builderBody('Composer').includes('.backgroundColor(C_INPUT_BG)'), 'composer keeps the translucent input container color');
assert(builderBody('Composer').includes('.backgroundColor(Color.Transparent)'), 'TextInput stays transparent inside the composer');

if (process.exitCode && process.exitCode !== 0) {
  process.exit(process.exitCode);
}

console.log('ChatPage UI guard passed');
