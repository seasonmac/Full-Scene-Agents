# EBM OpenClaw 插件第二轮修复与集成验证记录

日期：2026-04-18

## 背景

本轮工作目标是把 `ebm-openclaw-plugin` 从“结构完整但语义未闭合”的插件骨架，修补到可以在 OpenClaw 中稳定安装、启动、ingest、recall 的状态。

参考对象是 `OpenViking-main/examples/openclaw-plugin`，但本轮没有复制 OpenViking 的 hooks/tools 业务层，只对齐以下关键能力：

- `local` / `remote` 两种运行模式
- OpenClaw context-engine API 形状
- token 估算与 `compact` 返回结构
- 通过 skill/helper 安装插件
- local sidecar 生命周期管理
- session 内与跨 session 记忆验证

## 主要修复

### 1. Context Engine 协议闭环

修复前，TS 插件虽然声明了 `sessionKey`、`sessionFile`、`runtimeContext` 等字段，但实际没有完整传到 Python sidecar。

已修复为端到端透传：

- `bootstrap`: `sessionId`、`sessionKey`、`sessionFile`
- `ingest`: `sessionId`、`sessionKey`、`sessionFile`、`message`
- `ingestBatch`: `sessionId`、`sessionKey`、`sessionFile`、`messages`
- `assemble`: `sessionId`、`sessionKey`、`messages`、`tokenBudget`、`prompt`、`runtimeContext`
- `afterTurn`: `sessionId`、`sessionKey`、`sessionFile`、`messages`、`prePromptMessageCount`、`tokenBudget`、`runtimeContext`
- `compact`: `sessionId`、`sessionKey`、`sessionFile`、`tokenBudget`、`force`、`currentTokenCount`、`compactionTarget`、`customInstructions`、`runtimeContext`

其中 `prompt` 是后续 review 时发现的遗漏字段，已补齐。

### 2. `compact` 结构化返回

Python 端 `compact()` 仍然是功能性 stub，但返回结构已稳定为：

```json
{
  "ok": true,
  "compacted": false,
  "reason": "delegateCompactionToRuntime not implemented in Python translation",
  "result": {
    "summary": "",
    "firstKeptEntryId": "",
    "tokensBefore": 0,
    "details": {}
  }
}
```

`tokensBefore` 优先来自 `currentTokenCount`，缺省为 `0`。这样 OpenClaw 侧可以稳定依赖返回结构，即使真正压缩逻辑尚未实现。

### 3. `slowPathEnabled` 生效

修复前，`slowPathEnabled` 只存在于 manifest 和 TS 配置里，Python sidecar 实际没有接收到。

现在 local 模式启动 sidecar 时会设置：

```bash
EBM_PY_SLOWPATH_ENABLED=1
```

Python server 会读取该环境变量并覆盖最终 `EbmConfig.slowPathEnabled`。

### 4. Local / Remote 模式

插件现在支持：

- `local`: OpenClaw 插件启动并管理 Python sidecar
- `remote`: 插件连接已有的 EBM HTTP server

remote 模式下，健康检查失败只记录 warning，不阻止插件加载。这是故障容忍策略。

### 5. 本地 sidecar 生命周期修复

本轮修复了多个 local sidecar 生命周期问题：

- 支持 context engine 在非 gateway service 路径下懒启动 local sidecar。
- sidecar 运行时状态按配置 key 隔离，不再是单一全局缓存。
- 共享运行时包含 `client`、`process`、`startupPromise`、`refCount`、`startupOwners`。
- 多个相同配置的 plugin registration 会复用同一个 sidecar。
- 不同 `port` / `ebmPyPath` / `configJsonPath` / `dbPath` 的实例不会错误复用。
- `stop()` 在 startup 期间会释放 startup owner，避免 orphan sidecar。
- 如果目标端口已有健康 EBM sidecar，插件应复用它，而不是强行杀掉。

### 6. 安装 helper 与 self-contained local install

新增：

- `ebm-openclaw-plugin/setup-helper/install.js`
- `ebm-openclaw-plugin/install-manifest.json`

helper 会：

- 复制插件到 `~/.openclaw/extensions/ebm-context-engine`
- 写入或更新 `~/.openclaw/openclaw.json`
- 设置 `plugins.slots.contextEngine = "ebm-context-engine"`
- 生成 local / remote 配置模板
- 为 local 模式 vendoring 最小运行时子集到 `~/.openclaw/vendor/noteLM`

vendored 子集目前包括：

- `ebm_context_engine`
- `ebm`
- `cram`

不会复制：

- `.git`
- `node_modules`
- `coverage`
- `__pycache__`

## 环境隔离修复

在集成测试中发现，用户机器上同时存在旧插件和新插件：

- `~/.openclaw/extensions/ebm-context-engine`
- `~/.openclaw/extensions/ebm-py-engine`

虽然 `plugins.slots.contextEngine` 指向 `ebm-context-engine`，但因为 `plugins.allow` 为空，OpenClaw 会发现并 auto-load 非 bundled 插件，造成日志混乱和 sidecar 竞争风险。

本轮测试采用的干净配置：

```json
{
  "plugins": {
    "allow": ["duckduckgo", "ebm-context-engine"],
    "slots": {
      "contextEngine": "ebm-context-engine"
    },
    "entries": {
      "ebm-context-engine": {
        "enabled": true,
        "config": {
          "mode": "local",
          "pythonCommand": "/usr/local/opt/python@3.12/bin/python3",
          "port": 18791,
          "ebmPyPath": "/Users/season/.openclaw/vendor/noteLM",
          "configJsonPath": "/Users/season/.openclaw/vendor/noteLM/ebm/config.json",
          "dbPath": "~/.openclaw/memory/ebm_context_engine_cross.sqlite",
          "timeoutMs": 120000,
          "healthTimeoutMs": 30000,
          "slowPathEnabled": true,
          "portScanRange": 10
        }
      }
    }
  }
}
```

另外将旧插件目录临时移走：

```bash
mv ~/.openclaw/extensions/ebm-context-engine ~/.openclaw/extensions/ebm-context-engine.disabled
```

## 验证记录

### 静态与单元测试

在 `ebm-openclaw-plugin` 下执行：

```bash
npm run build
npm test
```

结果：

- TypeScript build 通过
- Vitest 通过，测试数最终达到 77 个

Python 合约测试：

```bash
python3 -m unittest ebm_context_engine.tests.test_openclaw_plugin_contracts
```

结果：通过。

### 单 session ingest / recall 验证

使用：

```bash
cd openclaw
pnpm run openclaw agent --local \
  --session-id ebm-integration-test-2 \
  --message "Remember this exact fact: my favorite dessert is mango sticky rice." \
  --thinking minimal \
  --json
```

返回：

```text
Got it — your favorite dessert is mango sticky rice.
```

同一 session 再问：

```bash
pnpm run openclaw agent --local \
  --session-id ebm-integration-test-2 \
  --message "What is my favorite dessert? Answer with only the dessert name." \
  --thinking minimal \
  --json
```

返回：

```text
mango sticky rice
```

日志确认：

- `[EBM-PY] bootstrap`
- `[EBM-PY] assemble`
- `[EBM-PY] afterTurn`

结论：单 session ingest 和 recall 通过。

### 跨 session 记忆验证

测试前做了环境隔离：

- 禁用/移走旧 `ebm-context-engine`
- `plugins.allow` 只允许 `duckduckgo` 和 `ebm-context-engine`
- 使用独立端口 `18791`
- 使用独立数据库 `~/.openclaw/memory/ebm_context_engine_cross.sqlite`

Session A 写入：

```bash
pnpm run openclaw agent --local \
  --session-id ebm-cross-clean-a \
  --message "Remember this exact user preference: I prefer black sesame ice cream over all other desserts." \
  --thinking minimal \
  --json
```

返回：

```text
Remembered: you prefer black sesame ice cream over all other desserts.
```

数据库确认：

```sql
select count(*) from transcript_entries where session_id='ebm-cross-clean-a';
```

返回：

```text
8
```

Session B 提问：

```bash
pnpm run openclaw agent --local \
  --session-id ebm-cross-clean-b \
  --message "Across our previous chats, what dessert do I prefer most? Answer with only the dessert name." \
  --thinking minimal \
  --json
```

返回：

```text
black sesame ice cream
```

日志确认：

- `[EBM-PY] reusing healthy sidecar at http://127.0.0.1:18791`
- `[EBM-PY] assemble: session=ebm-cross-clean-b ...`
- `[EBM-PY] afterTurn: session=ebm-cross-clean-b ...`

结论：跨 session 记忆效果通过。

## 当前已知限制

### 1. 旧插件配置仍可能产生 stale warning

如果 `openclaw.json` 里还保留 `plugins.entries.ebm-context-engine`，即使扩展目录已移走，也会出现：

```text
plugins.entries.ebm-context-engine: plugin not found: ebm-context-engine
```

建议后续清理该配置项。

### 2. slow path 结构化抽取未在本轮完全证明

跨 session 测试通过时，独立测试库里：

```sql
select count(*) from facts;
select count(*) from unified_facts;
select count(*) from session_summaries;
```

曾返回：

```text
0
0
0
```

说明本轮跨 session 命中更可能来自 transcript/session memory 注入路径，而不是 slow path 完整结构化事实抽取。

因此当前结论是：

- transcript 级跨 session recall：通过
- slow path 结构化事实 recall：仍需单独测试

### 3. OpenClaw runtime build artifact 偶发异常

测试中曾遇到：

```text
Failed to write runtime build artifacts: ENOTEMPTY ... openclaw/dist/extensions/diffs/node_modules
```

这是 OpenClaw runtime build artifacts 的环境问题，不是 EBM 插件本身问题，但会干扰 `openclaw config` 等命令。

## 建议后续工作

1. 清理 `openclaw.json` 中旧的 `plugins.entries.ebm-context-engine`。
2. 将 `plugins.allow` 写入安装 helper，安装时默认只 allow `ebm-context-engine` 和必要内置插件。
3. 增加可重复执行的跨 session smoke test 脚本。
4. 单独测试 slow path：
   - `afterTurn` 入队
   - `/flush` 完成
   - `facts` / `unified_facts` / `session_summaries` 增长
   - 新 session 通过结构化事实召回命中
5. 优化 vendored runtime 内容，进一步减少 `setup-helper/install.js` 的复制耗时。
