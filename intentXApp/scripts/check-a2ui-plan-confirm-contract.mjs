import fs from 'node:fs';
import path from 'node:path';
import process from 'node:process';

const root = process.cwd();

const files = {
  flowParsers: path.join(root, 'entry/src/main/ets/service/FlowParsers.ets'),
  streamAdapter: path.join(root, 'entry/src/main/ets/service/StreamUIAdapter.ets'),
  a2uiModels: path.join(root, 'entry/src/main/ets/model/A2UIModels.ets'),
  mainViewModel: path.join(root, 'entry/src/main/ets/viewmodel/MainViewModel.ets'),
  mapper: path.join(root, 'entry/src/main/ets/service/A2UIMapper.ets'),
  catalog: path.join(root, 'docs/a2ui/intentx-cards-catalog@1.json'),
  guide: path.join(root, '../genSkill-main/references/a2ui-emit-guide.md'),
  planCard: path.join(root, 'entry/src/main/ets/components/flow/PlanConfirmCard.ets'),
};

function read(file) {
  return fs.readFileSync(file, 'utf8');
}

function assert(condition, message) {
  if (!condition) {
    console.error(`FAIL: ${message}`);
    process.exitCode = 1;
  }
}

const flowParsers = read(files.flowParsers);
const streamAdapter = read(files.streamAdapter);
const a2uiModels = read(files.a2uiModels);
const mainViewModel = read(files.mainViewModel);
const mapper = read(files.mapper);
const catalog = read(files.catalog);
const guide = read(files.guide);
const planCard = read(files.planCard);

assert(!flowParsers.includes('parsePlanConfirm'), 'FlowParsers no longer parses PlanConfirm from prose');
assert(!streamAdapter.includes('FlowParsers.parsePlanConfirm'), 'StreamUIAdapter does not infer PlanConfirm via keywords');
assert(
  a2uiModels.includes("A2UI_CARD_CAPABILITY: string = 'a2ui:' + A2UI_CATALOG_ID") &&
    mainViewModel.includes('options.caps = [A2UI_CARD_CAPABILITY]'),
  'Harmony operator connection advertises the intentx card A2UI capability'
);
assert(mapper.includes("A2UIMapper.str(it, 'icon')"), 'A2UIMapper maps optional PlanItem.icon from A2UI');
assert(catalog.includes('"icon"'), 'A2UI catalog documents optional PlanItem.icon');
assert(
  guide.includes('"component":"PlanConfirm"') && guide.includes('"icon":"'),
  'genSkill A2UI guide includes an explicit PlanConfirm JSONL example with icons'
);
assert(planCard.includes('@Builder PlanFlowDiagram'), 'PlanConfirmCard renders a flow diagram section');
assert(planCard.includes('@Builder PlanStepIcon'), 'PlanConfirmCard uses icon elements for plan steps');

if (process.exitCode && process.exitCode !== 0) {
  process.exit(process.exitCode);
}

console.log('A2UI PlanConfirm contract guard passed');
