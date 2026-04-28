# EBM Python Memory Engine — OpenClaw 插件

这是一个面向 OpenClaw 的 Evidence-Backed Memory 插件。插件本身是 TypeScript 实现的轻量代理，负责注册 OpenClaw context-engine、工具、hooks 和服务生命周期；实际记忆检索、写入、慢路径处理由 Python sidecar `ebm_context_engine.server` 执行。

## 架构

```text
OpenClaw Gateway (Node.js)
    ↕ Plugin SDK
TS Plugin (ebm-context-engine)
    ↕ HTTP (localhost:18790)
Python Sidecar (ebm_context_engine.server)
    ├── Plane A: Task-Frontier Workspace
    ├── Plane B: Structured Skill Graph
    ├── Plane C: Temporal Semantic Ledger
    └── SQLite + FTS5
```

## 文件说明

| 文件 | 作用 |
|------|------|
| `src/index.ts` | 插件入口：注册 context-engine、工具、hooks、服务生命周期和 HTTP 路由 |
| `src/config.ts` | 插件配置解析和校验，支持 local/remote 两种模式 |
| `src/client.ts` | Python sidecar HTTP 客户端，封装所有 `ebm_context_engine.server` 端点 |
| `src/process-manager.ts` | 本地 sidecar 生命周期管理、端口检测、健康检查 |
| `src/context-engine.ts` | OpenClaw ContextEngine 契约实现：`bootstrap`、`ingest`、`assemble`、`afterTurn`、`compact` 等 |
| `src/shims.d.ts` | OpenClaw Plugin SDK 的本地类型声明 |
| `openclaw.plugin.json` | 插件 manifest 和配置 schema |
| `setup-helper/install.js` | 安装助手：复制插件、复制 EBM 运行依赖、更新 `openclaw.json` |

## 运行模式

- **Local，本地模式，默认**：插件启动并管理 Python sidecar，自动处理端口、健康检查和关闭。
- **Remote，远程模式**：插件连接已有的 EBM HTTP 服务，只在启动时做健康检查。

## 关键配置

完整配置见 `openclaw.plugin.json`。

| 配置项 | 说明 |
|--------|------|
| `mode` | `"local"` 或 `"remote"` |
| `pythonCommand` | Python 3.12+ 可执行文件路径，默认 `python3` |
| `port` | 本地 sidecar 端口，默认 `18790` |
| `baseUrl` | remote 模式下的 EBM 服务地址 |
| `ebmPyPath` | 包含 `ebm_context_engine/` 和 `ebm/config.json` 的项目根目录 |
| `configJsonPath` | `ebm/config.json` 路径 |
| `dbPath` | SQLite 数据库路径 |
| `slowPathEnabled` | 是否启用慢路径蒸馏 |
| `timeoutMs` | 单次 HTTP 请求超时，默认 120 秒 |
| `healthTimeoutMs` | sidecar 启动健康检查超时 |

## 快速开始

本地安装：

```bash
cd ebm-openclaw-plugin
pnpm install
pnpm build
pnpm test
node ./setup-helper/install.js --mode local
```

远程服务安装：

```bash
node ./setup-helper/install.js --mode remote --base-url http://127.0.0.1:18790
```

验证插件配置：

```bash
openclaw config get plugins.entries.ebm-context-engine
openclaw config get plugins.slots.contextEngine
```

期望结果：`plugins.slots.contextEngine` 指向 `ebm-context-engine`。

## Context Engine 能力

插件注册的 context-engine id 是 `ebm-context-engine`，核心方法全部代理到 Python sidecar：

| 方法 | Python 端点 | 说明 |
|------|-------------|------|
| `bootstrap` | `POST /bootstrap` | 从 session 文件导入历史消息 |
| `ingest` | `POST /ingest` | 写入单条消息 |
| `ingestBatch` | `POST /ingest-batch` | 批量写入消息 |
| `assemble` | `POST /assemble` | 召回 EBM 记忆并构造上下文 |
| `afterTurn` | `POST /after-turn` | 轮次结束后写入 transcript 并入队 slow path |
| `compact` | `POST /compact` | 返回结构化 compaction 结果；当前 Python 实现仍是委托 runtime 的 stub |
| `dispose` | `POST /dispose` | 释放 sidecar 资源，best-effort |

## EBM Agent 工具

插件注册 4 个 agent tools，内部语义映射到 EBM 的 memory/search/transcript 模型。

### `memory_recall`

搜索 EBM 记忆、结构化事实和 transcript 证据。

参数：

```json
{
  "query": "用户喜欢什么茶？",
  "limit": 6
}
```

行为：

- 调用 Python sidecar `POST /memory-search`。
- 返回命中的 `id`、`title`、`content`、`source`、`score` 和 `evidence`。
- 如果没有命中，返回 `No relevant EBM memories found.`。

适用场景：

- 查询用户偏好。
- 查询历史项目决策。
- 查询之前对话中的事实或证据。

### `memory_store`

手动写入一段记忆，并立即 flush slow path，使后续 `memory_recall` 更容易命中。

参数：

```json
{
  "text": "用户偏好：默认使用中文回复。",
  "role": "user",
  "sessionId": "optional-session-id",
  "sessionKey": "optional-session-key"
}
```

行为：

- 调用 `POST /ingest` 写入文本。
- 随后调用 `POST /flush` 排空 slow path。
- 返回写入 session、`ingested` 状态和 flush 后的队列状态。

说明：

- 如果未传 `sessionId`，工具会使用当前 OpenClaw session id。
- 如果当前上下文也没有 session id，会生成 `memory-store-<timestamp>` 临时 session。

### `memory_forget`

忘记 EBM 中可安全删除或失效的记忆对象。

按 id 删除：

```json
{
  "id": "fact:example-id"
}
```

先搜索再删除：

```json
{
  "query": "用户喜欢 jasmine tea",
  "limit": 5
}
```

行为：

- 如果传入 `id`，调用 `POST /memory-forget`。
- 如果传入 `query`，先调用 `POST /memory-search` 找候选。
- 当只有一个高置信候选时自动 forget。
- 如果有多个候选，会返回候选 id 列表，要求再次指定 `id`。

删除策略：

- `FACT`、`UNIFIED_FACT` 采用软删除：设置为非 active，并记录失效时间。
- `HM_FACT` 可删除派生 fact，并从 episode 引用中移除。
- transcript、session summary、entity、event、community 等证据链对象不会硬删；后端会返回 `unsupported_type`，避免破坏可追溯性。

### `ebm_archive_expand`

展开 EBM session summary 或 transcript 引用，返回原始消息。

参数：

```json
{
  "archiveId": "session-or-summary-id",
  "limit": 200
}
```

行为：

- 调用 `POST /archive-expand`。
- `archiveId` 可以是 EBM summary id、session id、session key 或 transcript session ref。
- 返回 summary 信息和 messages 列表。

适用场景：

- `assemble` 给出的摘要不够详细，需要回看原始对话。
- 需要精确命令、文件路径、配置值或代码片段。

## Hooks

插件注册 EBM hooks：

| Hook | 行为 |
|------|------|
| `session_start` | 保持兼容注册形状，目前不做重型处理 |
| `session_end` | 保持兼容注册形状，目前不做重型处理 |
| `before_prompt_build` | 根据当前 prompt/最后一条用户消息调用 `memorySearch`，把相关记忆注入 `<relevant-memories>`；如果输入像多说话人 transcript，则注入 ingest reply assist |
| `agent_end` | 保持兼容注册形状，目前不做重型处理 |
| `before_reset` | reset 前尽量补写当前 session 消息并调用 `flush`，降低记忆丢失概率 |
| `after_compaction` | 预留扩展点，目前不做重型处理 |

`before_prompt_build` 注入示例：

```text
<relevant-memories>
The following EBM memories may be relevant:
1. Preference (fact:1 source=ledger score=0.910)
prefers jasmine tea
</relevant-memories>
```

## TS 客户端方法

`src/client.ts` 中的 `EbmPyClient` 封装了 context-engine、运维和兼容工具所需端点。

| 方法 | 端点 | 用法 |
|------|------|------|
| `healthCheck()` | `GET /health` | 检查 sidecar 是否可用 |
| `bootstrap(params)` | `POST /bootstrap` | 导入 session 文件 |
| `ingest(params)` | `POST /ingest` | 写入单条消息 |
| `ingestBatch(params)` | `POST /ingest-batch` | 批量写入消息 |
| `assemble(params)` | `POST /assemble` | 召回并组装上下文 |
| `afterTurn(params)` | `POST /after-turn` | 轮次结束处理 |
| `compact(params)` | `POST /compact` | 获取结构化 compaction 返回 |
| `status()` | `GET /status` | 查看 slow path 队列状态 |
| `flush()` | `POST /flush` | 排空 slow path 队列 |
| `retryFailed()` | `POST /retry-failed` | 重试失败 slow path 作业 |
| `query(question, useAaak)` | `POST /query` | 独立问答调试接口 |
| `memorySearch(query, limit)` | `POST /memory-search` | 记忆搜索 |
| `memoryGet(id)` | `POST /memory-get` | 读取单个记忆对象 |
| `memoryForget(id)` | `POST /memory-forget` | 忘记可支持的记忆对象 |
| `archiveExpand(params)` | `POST /archive-expand` | 展开 summary/transcript |
| `dispose()` | `POST /dispose` | 释放 sidecar 资源 |

示例：

```ts
const client = new EbmPyClient("http://127.0.0.1:18790", 120_000);

const hits = await client.memorySearch("用户喜欢什么茶？", 6);
const item = await client.memoryGet(hits[0].id);
const forgotten = await client.memoryForget(hits[0].id);
const archive = await client.archiveExpand({
  archiveId: "session-1",
  sessionId: "session-1",
  limit: 200,
});
```

## Python Sidecar HTTP 端点

这些端点由 `python -m ebm_context_engine.server` 提供，TS 插件通过 `EbmPyClient` 调用。

### 核心端点

| 路径 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/bootstrap` | POST | 导入 session 文件 |
| `/ingest` | POST | 写入单条消息 |
| `/ingest-batch` | POST | 批量写入消息 |
| `/assemble` | POST | 召回并组装上下文 |
| `/after-turn` | POST | 轮次结束处理 |
| `/compact` | POST | compaction 契约返回 |
| `/dispose` | POST | 释放资源 |

### 记忆工具端点

| 路径 | 方法 | 说明 |
|------|------|------|
| `/memory-search` | POST | 搜索 EBM 记忆 |
| `/memory-get` | POST | 按 id 读取记忆对象 |
| `/memory-forget` | POST | 按 id 忘记可支持的记忆对象 |
| `/archive-expand` | POST | 展开 session summary 或 transcript |
| `/query` | POST | 独立问答调试接口 |

### 运维端点

| 路径 | 方法 | 说明 |
|------|------|------|
| `/status` | GET | slow path 队列状态 |
| `/flush` | POST | 排空 slow path 队列 |
| `/retry-failed` | POST | 重试失败 slow path 作业 |

## OpenClaw Gateway 调试路由

插件还在 OpenClaw Gateway 侧注册了 3 个调试路由：

| 路径 | 方法 | 说明 |
|------|------|------|
| `/v1/extensions/ebm-py/status` | GET | 查看 slow path 队列状态 |
| `/v1/extensions/ebm-py/flush` | POST | 排空 slow path 队列 |
| `/v1/extensions/ebm-py/retry-failed` | POST | 重试失败 slow path 作业 |

这些路由会转发到 Python sidecar 的 `/status`、`/flush`、`/retry-failed`。

## 测试

```bash
cd ebm-openclaw-plugin
pnpm run build
pnpm test

cd ..
python3 -m unittest ebm_context_engine.tests.test_openclaw_plugin_contracts
python3 -m unittest discover ebm_context_engine/tests
```

当前验证覆盖：

- TypeScript 类型检查。
- TS client、context-engine、process-manager、index 注册、安装助手测试。
- Python OpenClaw 插件契约测试。
- `memory_get`、`memory_forget`、`archive_expand` 后端行为。
