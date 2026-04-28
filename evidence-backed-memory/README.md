# EBM 接入 OpenClaw 操作指南

> 本文档一步步指导你将 ebm_py（Evidence-Backed Memory）记忆引擎接入 OpenClaw，
> 使其作为 context engine 插件运行，为对话提供长期记忆能力。

---

## 前置条件

在开始之前，确认你的环境满足以下要求：

| 依赖 | 最低版本 | 检查命令 |
|------|---------|---------|
| Python | 3.12+ | `python3 --version` |
| Node.js | 22.16+ | `node --version` |
| OpenClaw | 已安装并可运行 | `cd openclaw && pnpm run openclaw --version` |
| numpy | — | `python3 -c "import numpy"` |
| httpx | — | `python3 -c "import httpx"` |
| openai | — | `python3 -c "import openai"` |

> **注意**: macOS 自带的 Python 3.9 (Xcode) 不兼容，因为 ebm_py 使用了 `X | None` 类型语法。
> 请使用 Homebrew 安装的 Python 3.12：`/usr/local/opt/python@3.12/bin/python3`

如果缺少 Python 依赖：
```bash
pip3 install numpy httpx openai
```

---

## 第一步：确认 ebm_py 项目结构

确保你的 ebm_py 项目目录结构如下：

```
noteLM/                          ← 项目根目录（后面称为 EBM_PY_PATH）
├── ebm_py/                      ← Python 引擎包
│   ├── __init__.py
│   ├── engine.py                ← 核心引擎 EvidenceBackedMemoryEngine
│   ├── server.py                ← HTTP sidecar 服务器（关键文件）
│   ├── client.py
│   ├── types.py
│   ├── core/
│   ├── db/
│   ├── planes/
│   ├── retrieval/
│   ├── slowpath/
│   └── hypergraph/
├── config.json              ← 模型端点 & 调优参数配置
└── ebm-py-plugin/               ← OpenClaw TS 插件源码
    ├── package.json
    ├── openclaw.plugin.json
    └── src/
        └── index.ts
```

验证 server.py 存在且可运行：
```bash
cd /path/to/noteLM
python3 -c "from ebm_py.server import main; print('server 模块加载成功')"
```

---

## 第二步：配置 ebm/config.json

这个文件定义了 EBM 引擎使用的模型端点。你需要配置 4 个服务：

```bash
vim ebm/config.json
```

最小可用配置模板：

```json
{
  "embedding": {
    "enabled": true,
    "baseUrl": "http://127.0.0.1:8085/v1",
    "apiKey": "your-api-key",
    "model": "你的embedding模型名",
    "dimension": 1024
  },
  "rerank": {
    "enabled": true,
    "baseUrl": "http://127.0.0.1:8086/v1",
    "apiKey": "your-api-key",
    "model": "你的rerank模型名"
  },
  "llm": {
    "enabled": true,
    "baseUrl": "https://api.example.com/v1",
    "apiKey": "your-api-key",
    "model": "你的LLM模型名"
  },
  "memllm": {
    "enabled": true,
    "baseUrl": "https://api.example.com/v1",
    "apiKey": "your-api-key",
    "model": "你的LLM模型名"
  }
}
```

各服务用途：
- `embedding` — 文本向量化，用于语义检索（推荐本地部署小模型）
- `rerank` — 重排序，提升检索精度（可选，推荐本地部署）
- `llm` — 主 LLM，用于意图分类、问答生成
- `memllm` — 记忆专用 LLM，用于 slow path 实体/事实提取（可与 llm 相同）

> 每个服务都支持 `fallback` 字段配置备用端点，格式与主端点相同。

---

## 第三步：手动测试 Python sidecar

在接入 OpenClaw 之前，先单独验证 sidecar 能正常启动：

```bash
cd /path/to/noteLM

# 启动 sidecar（前台运行，方便观察日志）
python3 -m ebm_py.server \
  --port 18790 \
  --config ./ebm/config.json \
  --db /tmp/test_ebm.sqlite \
  --log-level INFO
```

你应该看到类似输出：
```
2026-04-17 10:00:00 [ebm_py.server] INFO 正在初始化 EBM 引擎: db_path=/tmp/test_ebm.sqlite ...
2026-04-17 10:00:01 [ebm_py.server] INFO EBM 引擎初始化完成: db_path=/tmp/test_ebm.sqlite 状态已加载
2026-04-17 10:00:01 [ebm_py.server] INFO EBM Python HTTP 服务器启动: http://127.0.0.1:18790
EBM_PY_READY port=18790
```

在另一个终端验证健康检查：
```bash
curl http://127.0.0.1:18790/health
# 期望输出: {"ok": true, "engine": "ebm_py"}
```

验证完毕后 `Ctrl+C` 停止 sidecar。测试数据库可删除：
```bash
rm /tmp/test_ebm.sqlite
```

---

## 第四步：安装插件到 OpenClaw

有两种方式安装插件，选择其一即可。

### 方式 A：直接复制到 extensions 目录（推荐）

```bash
# 创建插件目录
mkdir -p ~/.openclaw/extensions/ebm-context-engine/src

# 复制插件源码
cp ebm-py-plugin/src/index.ts ~/.openclaw/extensions/ebm-context-engine/src/index.ts
cp ebm-py-plugin/package.json ~/.openclaw/extensions/ebm-context-engine/package.json
cp ebm-py-plugin/openclaw.plugin.json ~/.openclaw/extensions/ebm-context-engine/openclaw.plugin.json
```

### 方式 B：通过 openclaw.json 指定本地路径

在 `~/.openclaw/openclaw.json` 的 `plugins.installs` 中添加：

```json
{
  "plugins": {
    "installs": {
      "ebm-context-engine": {
        "source": "path",
        "sourcePath": "/path/to/noteLM/ebm-py-plugin",
        "installPath": "/Users/你的用户名/.openclaw/extensions/ebm-context-engine",
        "version": "0.2.0"
      }
    }
  }
}
```

---

## 第五步：配置 openclaw.json

编辑 `~/.openclaw/openclaw.json`，添加插件配置。

### 5.1 注册插件为 context engine

在 `plugins.slots` 中指定 context engine：

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "ebm-context-engine"
    }
  }
}
```

### 5.2 配置插件参数

在 `plugins.entries` 中添加 ebm-context-engine 配置：

```json
{
  "plugins": {
    "entries": {
      "ebm-context-engine": {
        "enabled": true,
        "config": {
          "pythonSidecar": true,
          "pythonCommand": "/usr/local/opt/python@3.12/bin/python3",
          "sidecarPort": 18790,
          "ebmPyPath": "/path/to/noteLM",
          "configJsonPath": "/path/to/noteLM/ebm/config.json",
          "dbPath": "/Users/你的用户名/.openclaw/memory/ebm_py.sqlite"
        }
      }
    }
  }
}
```

各参数说明：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `pythonCommand` | 否 | `python3` | Python 可执行文件的完整路径 |
| `sidecarPort` | 否 | `18790` | sidecar HTTP 监听端口 |
| `ebmPyPath` | 是 | — | ebm_py 项目根目录（包含 `ebm_py/` 包的目录） |
| `configJsonPath` | 否 | `{ebmPyPath}/ebm/config.json` | 模型配置文件路径 |
| `dbPath` | 否 | `~/.openclaw/memory/ebm_py.sqlite` | SQLite 数据库存储路径 |
| `slowPathEnabled` | 否 | `true` | 是否启用后台 LLM 蒸馏 |

> **关键**: `pythonCommand` 必须指向 Python 3.12+。如果你用 Homebrew 安装，
> 路径通常是 `/usr/local/opt/python@3.12/bin/python3` 或 `/opt/homebrew/opt/python@3.12/bin/python3`。

### 5.3 完整配置示例

以下是一个完整的 `openclaw.json` 片段（只展示插件相关部分）：

```json
{
  "plugins": {
    "slots": {
      "contextEngine": "ebm-context-engine"
    },
    "entries": {
      "ebm-context-engine": {
        "enabled": true,
        "config": {
          "pythonSidecar": true,
          "pythonCommand": "/usr/local/opt/python@3.12/bin/python3",
          "sidecarPort": 18790,
          "ebmPyPath": "/Users/season/workspace/github/noteLM",
          "configJsonPath": "/Users/season/workspace/github/noteLM/ebm/config.json",
          "dbPath": "/Users/season/.openclaw/memory/ebm_py.sqlite"
        }
      }
    },
    "installs": {
      "ebm-context-engine": {
        "source": "path",
        "sourcePath": "/Users/season/workspace/github/noteLM/ebm-py-plugin",
        "installPath": "/Users/season/.openclaw/extensions/ebm-context-engine",
        "version": "0.2.0"
      }
    }
  }
}
```

---

## 第六步：启动 OpenClaw Gateway

```bash
cd /path/to/openclaw
pnpm run openclaw gateway
```

启动后观察日志，你应该看到类似输出：

```
[EBM-PY] 配置: python=/usr/local/opt/python@3.12/bin/python3 port=18790 ...
[EBM-PY] 启动 sidecar: /usr/local/opt/python@3.12/bin/python3 -m ebm_py.server --port 18790 ...
[EBM-PY:stdout] EBM_PY_READY port=18790
[EBM-PY] 插件注册完成: context engine + 3 HTTP routes
```

> 插件会自动启动 Python sidecar 进程，你不需要手动启动它。

---

## 第七步：验证集成

### 7.1 检查 sidecar 健康状态

```bash
# 通过 sidecar 直接检查
curl http://127.0.0.1:18790/health
# 期望: {"ok": true, "engine": "ebm_py"}

# 通过 OpenClaw gateway 代理检查（需要 gateway auth token）
curl -H "Authorization: Bearer 你的gateway-token" \
  http://127.0.0.1:18789/v1/extensions/ebm-py/status
```

### 7.2 手动测试写入和查询

```bash
BASE=http://127.0.0.1:18790

# 1. Bootstrap
curl -X POST $BASE/bootstrap \
  -H "Content-Type: application/json" \
  -d '{"sessionId": "test-001"}'

# 2. 写入消息
curl -X POST $BASE/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test-001",
    "message": {
      "role": "user",
      "content": "我叫张三，是一名软件工程师，住在北京海淀区"
    }
  }'

curl -X POST $BASE/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test-001",
    "message": {
      "role": "assistant",
      "content": "你好张三！很高兴认识你。作为软件工程师住在海淀区，离中关村很近呢。"
    }
  }'

# 3. 触发 after-turn（让 slow path 处理）
curl -X POST $BASE/after-turn \
  -H "Content-Type: application/json" \
  -d '{
    "sessionId": "test-001",
    "sessionFile": "",
    "messages": [],
    "prePromptMessageCount": 0
  }'

# 4. 等待 slow path 完成
curl -X POST $BASE/flush
# 等待几秒后再次 flush，直到 pending=0

# 5. 查询验证
curl -X POST $BASE/query \
  -H "Content-Type: application/json" \
  -d '{"question": "张三是做什么的？住在哪里？"}'
```

期望 query 返回的 answer 中包含"软件工程师"和"北京"。

### 7.3 检查数据持久化

```bash
# 查看 SQLite 数据库中的表
sqlite3 ~/.openclaw/memory/ebm_py.sqlite ".tables"

# 查看写入的对话记录
sqlite3 ~/.openclaw/memory/ebm_py.sqlite \
  "SELECT message_index, role, substr(text, 1, 60) FROM transcript_entries LIMIT 10;"

# 查看提取的事实
sqlite3 ~/.openclaw/memory/ebm_py.sqlite \
  "SELECT key, substr(value, 1, 80), confidence FROM facts LIMIT 10;"

# 查看提取的实体
sqlite3 ~/.openclaw/memory/ebm_py.sqlite \
  "SELECT name, entity_type FROM entities LIMIT 10;"
```

---

## 第八步：正式使用

验证通过后，正常使用 OpenClaw 对话即可。EBM 引擎会：

1. **自动写入** — 每条对话消息通过 `ingest()` 写入工作记忆
2. **自动召回** — 每次生成回复前通过 `assemble()` 召回相关记忆，注入 system prompt
3. **后台蒸馏** — 每轮对话结束后通过 `afterTurn()` 触发 slow path，用 LLM 提取实体、事实、关系
4. **持久存储** — 所有数据保存在 SQLite 中，重启后自动恢复

---

## 常见问题排查

### Q: sidecar 启动失败，报 `TypeError: unsupported operand type(s) for |`

Python 版本太低。ebm_py 使用了 `X | None` 语法，需要 Python 3.12+。

```bash
# 检查实际使用的 Python 版本
/usr/local/opt/python@3.12/bin/python3 --version

# 在 openclaw.json 中指定完整路径
"pythonCommand": "/usr/local/opt/python@3.12/bin/python3"
```

### Q: Gateway 启动后看不到 `[EBM-PY]` 日志

检查 `openclaw.json` 中：
1. `plugins.slots.contextEngine` 是否设为 `"ebm-context-engine"`
2. `plugins.entries.ebm-context-engine.enabled` 是否为 `true`
3. 插件文件是否存在于 `~/.openclaw/extensions/ebm-context-engine/src/index.ts`

### Q: sidecar 启动了但 `/health` 返回连接拒绝

端口可能被占用。检查并更换端口：
```bash
lsof -i :18790
# 如果被占用，在 openclaw.json 中改为其他端口，如 18791
```

### Q: assemble 返回空的 systemPromptAddition

这是正常的 — 如果还没有通过 slow path 提取出实体和事实，召回结果为空。
需要先 ingest 几条消息，触发 after-turn，然后 flush 等待 slow path 完成。

### Q: slow path 作业一直 pending

检查 `ebm/config.json` 中的 `llm` 和 `memllm` 配置是否正确，API key 是否有效：
```bash
# 查看 slow path 状态
curl http://127.0.0.1:18790/status

# 重试失败的作业
curl -X POST http://127.0.0.1:18790/retry-failed
```

### Q: 想清空所有记忆数据重新开始

```bash
# 停止 gateway
# 删除数据库文件
rm ~/.openclaw/memory/ebm_py.sqlite
# 重新启动 gateway
```

---

## 架构图

```
┌─────────────────────────────────────────────────────┐
│                  OpenClaw Gateway                    │
│                  (Node.js, :18789)                   │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │         ebm-context-engine (TS 插件)           │  │
│  │                                               │  │
│  │  bootstrap() ──→ POST /bootstrap              │  │
│  │  ingest()    ──→ POST /ingest                 │  │
│  │  assemble()  ──→ POST /assemble               │  │
│  │  afterTurn() ──→ POST /after-turn             │  │
│  │  dispose()   ──→ POST /dispose + kill process │  │
│  └───────────────────┬───────────────────────────┘  │
└──────────────────────┼──────────────────────────────┘
                       │ HTTP (localhost:18790)
┌──────────────────────┼──────────────────────────────┐
│  ebm_py.server       │    Python Sidecar             │
│  (http.server, :18790)                               │
│                                                     │
│  ┌───────────────────────────────────────────────┐  │
│  │      EvidenceBackedMemoryEngine               │  │
│  │                                               │  │
│  │  Plane A ── 工作记忆 (对话上下文)              │  │
│  │  Plane B ── 结构化图 (实体/关系/社区)          │  │
│  │  Plane C ── 时序账本 (事实/置信度/衰减)        │  │
│  │                                               │  │
│  │  HyperGraph ── Episode/Topic/Fact/C2F         │  │
│  │  SlowPath  ── 后台 LLM 蒸馏管线              │  │
│  │  Retrieval ── 混合检索 (BM25+Vec+Graph+PPR)   │  │
│  └───────────────────────────────────────────────┘  │
│                       │                             │
│              SQLite + FTS5                           │
│        (~/.openclaw/memory/ebm_py.sqlite)           │
└─────────────────────────────────────────────────────┘
```
