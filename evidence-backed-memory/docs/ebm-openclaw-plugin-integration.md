# EBM Python — OpenClaw 记忆插件集成文档

## 1. 架构概览

```
OpenClaw Gateway (Node.js, port 18789)
    ↕ Plugin SDK
TS 插件 (ebm-context-engine)
    ↕ HTTP (localhost:18790)
Python Sidecar (ebm_context_engine.server)
    ↕
EvidenceBackedMemoryEngine
    ├── Plane A: TaskFrontierWorkspace (工作记忆)
    ├── Plane B: StructuredSalientMemoryGraph (图/社区)
    ├── Plane C: TemporalSemanticLedger (事实/账本)
    ├── HyperGraph: Episode/Topic/Fact/C2F
    ├── SlowPath: 后台 LLM 提取管线
    └── SQLite + FTS5 持久化
```

**核心思路**: TS 插件是一个薄代理层，所有记忆逻辑在 Python 端完成。TS 端仅负责：
1. 启动/管理 Python sidecar 进程
2. 将 OpenClaw Context Engine 接口调用转发为 HTTP 请求
3. 注册 Gateway HTTP 路由供外部调试

## 2. 文件清单（v0.3.0 重构后）

| 文件 | 角色 |
|------|------|
| `ebm_context_engine/server.py` | Python HTTP sidecar 服务器 |
| `ebm-openclaw-plugin/src/config.ts` | 配置模型（local/remote 模式，校验+默认值） |
| `ebm-openclaw-plugin/src/client.ts` | HTTP 客户端（封装所有 sidecar 端点） |
| `ebm-openclaw-plugin/src/process-manager.ts` | 进程管理（端口、健康检查、zombie 清理） |
| `ebm-openclaw-plugin/src/context-engine.ts` | ContextEngine 接口实现（代理到 client） |
| `ebm-openclaw-plugin/src/index.ts` | 插件入口（definePluginEntry, service, HTTP routes） |
| `ebm-openclaw-plugin/src/shims.d.ts` | OpenClaw SDK 类型声明 |
| `ebm-openclaw-plugin/openclaw.plugin.json` | 插件元数据 + configSchema |
| `ebm-openclaw-plugin/skills/install-ebm-py-memory/SKILL.md` | Agent 安装技能文档 |
| `~/.openclaw/openclaw.json` | 插件配置（端口、路径等） |
| `ebm/config.json` | EBM 引擎配置（模型端点、调优参数） |

## 3. Python Sidecar 接口

### 启动方式
```bash
python3 -m ebm_context_engine.server --port 18790 --config /path/to/config.json --db /path/to/ebm.sqlite
```

### HTTP 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查，返回 `{"ok": true}` |
| GET | `/status` | slow path 状态 |
| POST | `/bootstrap` | 初始化 session |
| POST | `/ingest` | 写入单条消息到工作记忆 |
| POST | `/ingest-batch` | 批量写入 |
| POST | `/assemble` | 组装上下文（召回 + systemPrompt） |
| POST | `/after-turn` | 轮次结束处理（slow path 入队） |
| POST | `/compact` | 压缩（当前委托 runtime） |
| POST | `/dispose` | 释放资源 |
| POST | `/query` | 独立查询（测试用） |
| POST | `/memory-search` | 记忆搜索 |
| POST | `/flush` | 排空 slow path 队列 |
| POST | `/retry-failed` | 重试失败的 slow path 作业 |

## 4. TS 插件 Context Engine 接口映射

```
bootstrap()  → POST /bootstrap  {sessionId}
ingest()     → POST /ingest     {sessionId, message}
assemble()   → POST /assemble   {sessionId, messages, tokenBudget, prompt}
afterTurn()  → POST /after-turn {sessionId, messages, prePromptMessageCount, sessionFile}
compact()    → 本地返回 (委托 Python)
dispose()    → POST /dispose + kill sidecar
```

## 5. 配置说明（v0.3.0）

`~/.openclaw/openclaw.json` 中 `plugins.entries.ebm-context-engine.config`:

```json
{
  "mode": "local",
  "pythonCommand": "python3",
  "port": 18790,
  "ebmPyPath": "/Users/season/workspace/github/noteLM",
  "configJsonPath": "/Users/season/workspace/github/noteLM/ebm/config.json",
  "dbPath": "/Users/season/.openclaw/memory/ebm_context_engine.sqlite",
  "slowPathEnabled": true,
  "timeoutMs": 120000,
  "healthTimeoutMs": 30000,
  "portScanRange": 10
}
```

关键参数:
- `mode`: `"local"` 自动启动 sidecar; `"remote"` 连接已有服务器
- `pythonCommand`: Python 3.12+ 路径，默认 "python3"
- `port`: sidecar HTTP 端口，默认 18790
- `baseUrl`: remote 模式下的服务器 URL（local 模式自动从 port 推导）
- `ebmPyPath`: ebm_context_engine 包所在的项目根目录
- `dbPath`: SQLite 数据库路径
- `timeoutMs`: 每次 HTTP 请求超时（ms），最小 1000
- `healthTimeoutMs`: 健康检查超时（ms），最小 1000
- `portScanRange`: 端口被占时扫描范围，默认 10

## 6. 数据流

### 写入流（ingest → afterTurn）
```
用户消息 → ingest() → Plane A 工作记忆写入
                     → transcript_entries 表
轮次结束 → afterTurn() → slow path 入队
                       → 后台 LLM 提取实体/事实/关系
                       → Plane B 图更新
                       → Plane C 账本更新
                       → Episode/Topic 聚合
```

### 读取流（assemble）
```
assemble() → 意图分类 (intent_router)
           → C2F 粗到细检索 (topic → episode → fact)
           → PPR 图传播
           → 混合排序 (BM25 + vector + graph + rerank)
           → 构建 systemPromptAddition
           → 返回增强后的 messages
```

## 7. 集成测试结论

### 测试环境
- macOS Darwin 23.6.0, Node 25.1.0, Python 3.12
- OpenClaw gateway port 18789, sidecar port 18790
- SQLite: `~/.openclaw/memory/ebm_context_engine.sqlite`

### 测试结果

| 测试项 | 结果 | 说明 |
|--------|------|------|
| sidecar 启动 | 通过 | `/health` 返回 `{"ok": true}` |
| bootstrap | 通过 | session 初始化成功 |
| ingest (3条消息) | 通过 | transcript_entries 写入 5 条 |
| assemble | 通过 | 返回 systemPromptAddition |
| after-turn | 通过 | slow path 作业入队 |
| flush (slow path) | 通过 | 后台 LLM 提取完成 |
| query 验证 | 通过 | 正确回答"张三是软件工程师，住在北京" |
| 数据持久化 | 通过 | SQLite 包含 5 transcript, 3 facts, 5 entities |
| status 查询 | 通过 | slow path 状态正常返回 |

### 已知限制
- sidecar 使用 Python `http.server`（单线程），高并发场景需替换为 uvicorn/gunicorn
- `compact()` 当前为空实现，压缩由 OpenClaw runtime 处理
- Python 3.9 不兼容（类型注解语法），需 3.12+

## 8. 冗余代码清理记录

本次清理删除的冗余代码：

| 项目 | 文件 | 说明 |
|------|------|------|
| 重复 `_stable_id` | engine.py | 改用 `core/hash.stableId` |
| 废弃 `_relationship_chain_hits` | planes/plane_b.py | Legacy 方法，已被 `_build_relationship_chains` 替代 |
| 未使用 `_build_topic_section` | planes/plane_a.py | `buildPacket()` 从未调用 |
| 未使用 `_build_scratchpad_section` | planes/plane_a.py | 同上 |
| 未调用 `_classify_with_llm` | retrieval/intent_router.py | 仅保留规则分类 |
| 死导入 `vectorCosineSimilarity` | planes/plane_b.py | 导入但从未使用 |
| 孤立模块 `core/openai_chat_response.py` | core/ | 全项目无引用 |
| 死重导出 `core/types.py` | core/ | 仅 `from ebm_context_engine.types import *` |
