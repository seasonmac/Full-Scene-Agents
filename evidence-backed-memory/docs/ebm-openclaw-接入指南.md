# EBM 接入 OpenClaw 操作指南

> 本文档一步步指导你将 ebm_context_engine（Evidence-Backed Memory）记忆引擎接入 OpenClaw，
> 使其作为 context engine 插件运行，为对话提供长期记忆能力。
>
> 插件统一 ID：`ebm-context-engine`
>
> 支持两种运行模式：
> - **local** — 插件自动启动 Python sidecar 子进程（默认，适合开发和单机部署）
> - **remote** — 连接已运行的 ebm_context_engine 服务器（适合多实例共享、服务器部署）

---

## 前置条件

| 依赖 | 最低版本 | 检查命令 |
|------|---------|---------|
| Python | 3.12+ | `python3 --version` |
| Node.js | 22.16+ | `node --version` |
| OpenClaw | 已安装 | `cd openclaw && pnpm run openclaw --version` |
| numpy | — | `python3 -c "import numpy"` |
| httpx | — | `python3 -c "import httpx"` |
| openai | — | `python3 -c "import openai"` |

> **注意**: macOS 自带的 Python 3.9 (Xcode) 不兼容，ebm_context_engine 使用了 `X | None` 类型语法。
> 请使用 Homebrew 安装的 Python 3.12：`/usr/local/opt/python@3.12/bin/python3`

缺少依赖时：
```bash
pip3 install numpy httpx openai
```

---

## 第一步：确认 ebm_context_engine 项目结构

# EBM 代码目录
```
ROOT/                              ← 项目根目录（总代码：python 10K 行，TS：5.8K 行）
├── ebm_context_engine/                        ← Python 引擎包
│   ├── __init__.py                ← 包初始化，导出版本号与公共 API
│   ├── engine.py                  ← 核心引擎 EvidenceBackedMemoryEngine：ingest / recall / forget 全流程
│   ├── server.py                  ← HTTP sidecar 服务器，暴露 REST 端点供 TS 插件调用
│   ├── client.py                  ← Python 侧 HTTP 客户端，封装对 server.py 的请求
│   ├── types.py                   ← 公共数据类型定义
│   ├── index.py                   ← 索引管理，维护向量 & BM25 索引的构建与更新
│   ├── slowpath_processor.py      ← 慢路径入口 shim，委托到 slowpath/ 子包
│   ├── core/                      ← 基础工具层
│   ├── db/                        ← 持久化层
│   ├── planes/                    ← 三平面记忆架构
│   ├── retrieval/                 ← 检索策略层
│   ├── slowpath/                  ← 异步慢路径处理
│   └── hypergraph/                ← 超图记忆编码
├── config.json                    ← 模型端点 & 调优参数配置（JSON）
└── ebm-openclaw-plugin/           ← OpenClaw TS 插件源码
    ├── package.json
    ├── openclaw.plugin.json       ← OpenClaw 插件清单：声明 hooks / capabilities / 权限
    └── src/
        ├── index.ts               ← 插件入口：注册 hooks（before_prompt_build / ingest 等）
        ├── client.ts              ← HTTP 客户端：封装对 Python sidecar 的 REST 调用
        ├── process-manager.ts     ← sidecar 进程管理：启动 / 健康检查 / 自动重启
        ├── context-engine.ts      ← ContextEngine 接口实现：recall 结果 → prompt 注入
        ├── config.ts              ← 配置解析
        ├── shims.d.ts             ← TypeScript 类型垫片：补充第三方库缺失的类型声明（184 行）
        └── memory-ranking.ts      ← 记忆排序 & token 预算：estimateTokenCount + buildMemoryLinesWithBudget
```
# locomo10 benchmark测试
* 测试集：基于 LoCoMo10(https://github.com/snap-research/locomo) 的长程对话进行效果测试（去除无真值的 category5 后，共 1540 条 case）
* 记忆理解模型：gpt-5.4-mini
* 裁判模型: kimi-k2-turbo-preview
* embedding模型: qwen3-embedding-0.6b-Q4_K_M.gguf

EBM性能指标
```json
  平均得分(1分制): 0.633,
  平均 QA token 消耗: 765.592,
  查询时延(s): 2.695,
```
openviking性能指标
```json
  平均得分(1分制): 0.512,
  平均 QA token 消耗: 1363.39,
  查询时延(s): 3.1165,
```



验证 server.py 可加载：
```bash
cd /path/to/noteLM
python3 -c "from ebm_context_engine.server import main; print('OK')"
```

---

## 第二步：配置 ebm/config.json

定义 EBM 引擎使用的模型端点，需要配置 4 个服务：

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

python3 -m ebm_context_engine.server \
  --port 18790 \
  --config ./ebm/config.json \
  --db /tmp/test_ebm.sqlite \
  --log-level INFO
```

期望输出：
```
2026-04-17 10:00:00 [ebm_context_engine.server] INFO EBM 引擎初始化完成 ...
EBM_PY_READY port=18790
```

在另一个终端验证健康检查：
```bash
curl http://127.0.0.1:18790/health
# 期望输出: {"ok": true, "engine": "ebm_context_engine"}
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
cp ebm-openclaw-plugin/src/*.ts ~/.openclaw/extensions/ebm-context-engine/src/
cp ebm-openclaw-plugin/package.json ~/.openclaw/extensions/ebm-context-engine/
cp ebm-openclaw-plugin/openclaw.plugin.json ~/.openclaw/extensions/ebm-context-engine/
```

### 方式 B：通过 openclaw.json 指定本地路径

在 `~/.openclaw/openclaw.json` 的 `plugins.installs` 中添加：

```json
{
  "plugins": {
    "installs": {
      "ebm-context-engine": {
        "source": "path",
        "sourcePath": "/path/to/noteLM/ebm-openclaw-plugin",
        "installPath": "/Users/你的用户名/.openclaw/extensions/ebm-context-engine",
        "version": "0.3.0"
      }
    }
  }
}
```

---

## 第五步：配置 openclaw.json

编辑 `~/.openclaw/openclaw.json`，根据你选择的模式配置插件。

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

### 5.2 选择运行模式

插件支持 `local` 和 `remote` 两种模式，通过 `config.mode` 字段切换。

---

### 模式一：Local 模式（默认）

插件自动 spawn Python sidecar 子进程，管理其生命周期。适合开发和单机部署。

```json
{
  "plugins": {
    "entries": {
      "ebm-context-engine": {
        "enabled": true,
        "config": {
          "mode": "local",
          "pythonCommand": "/usr/local/opt/python@3.12/bin/python3",
          "port": 18790,
          "ebmPyPath": "/path/to/noteLM",
          "configJsonPath": "/path/to/noteLM/ebm/config.json",
          "dbPath": "~/.openclaw/memory/ebm_context_engine.sqlite",
          "slowPathEnabled": true
        }
      }
    }
  }
}
```

Local 模式特性：
- 插件启动时自动 spawn `python3 -m ebm_context_engine.server`
- 如果指定端口被占用，自动扫描 `port` ~ `port + portScanRange` 范围内的可用端口
- 如果检测到端口上已有健康的 sidecar，直接复用而不重复启动
- 插件停止时自动 SIGTERM 终止 sidecar
- 多个 OpenClaw 实例可共享同一个 sidecar（引用计数管理）

Local 模式参数：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | 否 | `"local"` | 设为 `"local"` 或省略 |
| `pythonCommand` | 否 | `"python3"` | Python 3.12+ 可执行文件路径 |
| `port` | 否 | `18790` | sidecar HTTP 监听端口 |
| `ebmPyPath` | 是 | — | ebm_context_engine 项目根目录（包含 `ebm_context_engine/` 包），支持 `~` 和 `${ENV}` |
| `configJsonPath` | 否 | `{ebmPyPath}/ebm/config.json` | 模型配置文件路径 |
| `dbPath` | 否 | `~/.openclaw/memory/ebm_context_engine.sqlite` | SQLite 数据库路径 |
| `slowPathEnabled` | 否 | `true` | 是否启用后台 LLM 蒸馏 |
| `portScanRange` | 否 | `10` | 端口被占用时的扫描范围 |
| `healthTimeoutMs` | 否 | `30000` | 启动健康检查超时（ms） |

---

### 模式二：Remote 模式

连接已运行的 ebm_context_engine 服务器，不管理进程生命周期。适合：
- 多个 OpenClaw 实例共享同一个 EBM 服务
- EBM 服务部署在远程服务器
- 需要独立管理 EBM 服务的生命周期
- Docker / K8s 等容器化部署

```json
{
  "plugins": {
    "entries": {
      "ebm-context-engine": {
        "enabled": true,
        "config": {
          "mode": "remote",
          "baseUrl": "http://127.0.0.1:18790"
        }
      }
    }
  }
}
```

Remote 模式参数：

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `mode` | 是 | — | 必须设为 `"remote"` |
| `baseUrl` | 否 | `http://127.0.0.1:18790` 或 `$EBM_PY_BASE_URL` | EBM 服务器地址，支持 `${ENV}` 展开 |
| `timeoutMs` | 否 | `120000` | 每次 HTTP 请求超时（ms） |

> Remote 模式下 `pythonCommand`、`ebmPyPath`、`port`、`portScanRange` 等 local 参数会被忽略。

#### Remote 模式部署步骤

1. 在目标机器上启动 EBM 服务：

```bash
cd /path/to/noteLM

python3 -m ebm_context_engine.server \
  --port 18790 \
  --config ./ebm/config.json \
  --db /path/to/ebm_context_engine.sqlite \
  --log-level INFO
```

2. 确认服务健康：

```bash
curl http://<服务器IP>:18790/health
# {"ok": true, "engine": "ebm_context_engine"}
```

3. 在 OpenClaw 配置中使用 remote 模式：

```json
{
  "config": {
    "mode": "remote",
    "baseUrl": "http://<服务器IP>:18790"
  }
}
```

> 也可以通过环境变量 `EBM_PY_BASE_URL` 设置，省略 `baseUrl` 字段。

#### 用 systemd 管理 Remote 服务（可选）

```ini
# /etc/systemd/system/ebm-py.service
[Unit]
Description=EBM Python Memory Engine
After=network.target

[Service]
Type=simple
User=your-user
WorkingDirectory=/path/to/noteLM
ExecStart=/usr/bin/python3 -m ebm_context_engine.server --port 18790 --config ./ebm/config.json --db /var/lib/ebm/ebm_context_engine.sqlite
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable ebm-py
sudo systemctl start ebm-py
```

---

### 5.3 通用参数（两种模式都适用）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `timeoutMs` | `120000` | 每次 HTTP 请求超时（ms） |
| `autoRecall` | `true` | 在 `before_prompt_build` 时自动召回记忆注入 system prompt |
| `recallLimit` | `6` | 自动召回最大条数 |
| `recallScoreThreshold` | `0.15` | 自动召回最低分数阈值 |
| `recallTokenBudget` | `2000` | 自动召回 token 预算上限 |
| `recallMaxContentChars` | `500` | 每条记忆内容最大字符数 |
| `recallTimeoutMs` | `5000` | 自动召回超时（ms） |
| `ingestReplyAssist` | `true` | 检测 transcript 输入时注入回复辅助提示 |

### 5.4 完整配置示例

#### Local 模式完整示例

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
          "mode": "local",
          "pythonCommand": "/usr/local/opt/python@3.12/bin/python3",
          "port": 18790,
          "ebmPyPath": "/Users/season/workspace/github/noteLM",
          "configJsonPath": "/Users/season/workspace/github/noteLM/ebm/config.json",
          "dbPath": "/Users/season/.openclaw/memory/ebm_context_engine.sqlite",
          "slowPathEnabled": true,
          "autoRecall": true,
          "recallTokenBudget": 2000
        }
      }
    },
    "installs": {
      "ebm-context-engine": {
        "source": "path",
        "sourcePath": "/Users/season/workspace/github/noteLM/ebm-openclaw-plugin",
        "installPath": "/Users/season/.openclaw/extensions/ebm-context-engine",
        "version": "0.3.0"
      }
    }
  }
}
```

#### Remote 模式完整示例

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
          "mode": "remote",
          "baseUrl": "http://192.168.1.100:18790",
          "timeoutMs": 120000,
          "autoRecall": true,
          "recallTokenBudget": 2000
        }
      }
    },
    "installs": {
      "ebm-context-engine": {
        "source": "path",
        "sourcePath": "/Users/season/workspace/github/noteLM/ebm-openclaw-plugin",
        "installPath": "/Users/season/.openclaw/extensions/ebm-context-engine",
        "version": "0.3.0"
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

### Local 模式启动日志

```
[EBM-PY] config: mode=local port=18790 timeout=120000ms python=python3 ebmPyPath=/path/to/noteLM
[EBM-PY] spawning: python3 -m ebm_context_engine.server --port 18790 --config ... --db ...
[EBM-PY:stdout] EBM_PY_READY port=18790
[EBM-PY] local sidecar started at http://127.0.0.1:18790 (port=18790)
[EBM-PY] plugin registered: context-engine + service + 3 HTTP routes
```

### Remote 模式启动日志

```
[EBM-PY] config: mode=remote port=18790 timeout=120000ms baseUrl=http://192.168.1.100:18790
[EBM-PY] remote server healthy at http://192.168.1.100:18790
[EBM-PY] plugin registered: context-engine + service + 3 HTTP routes
```

> Local 模式下插件自动管理 sidecar 生命周期；Remote 模式下需要你自己确保服务已启动。

---

## 第七步：验证集成

### 7.1 检查健康状态

```bash
# 直接检查 sidecar
curl http://127.0.0.1:18790/health
# 期望: {"ok": true, "engine": "ebm_context_engine"}

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
sqlite3 ~/.openclaw/memory/ebm_context_engine.sqlite ".tables"

# 查看写入的对话记录
sqlite3 ~/.openclaw/memory/ebm_context_engine.sqlite \
  "SELECT message_index, role, substr(text, 1, 60) FROM transcript_entries LIMIT 10;"

# 查看提取的事实
sqlite3 ~/.openclaw/memory/ebm_context_engine.sqlite \
  "SELECT key, substr(value, 1, 80), confidence FROM facts LIMIT 10;"

# 查看提取的实体
sqlite3 ~/.openclaw/memory/ebm_context_engine.sqlite \
  "SELECT name, entity_type FROM entities LIMIT 10;"
```

---

## 第八步：正式使用

验证通过后，正常使用 OpenClaw 对话即可。EBM 引擎会：

1. **自动写入** — 每条对话消息通过 `ingest()` 写入工作记忆
2. **自动召回** — 每次生成回复前通过 `assemble()` 召回相关记忆，注入 system prompt
3. **后台蒸馏** — 每轮对话结束后通过 `afterTurn()` 触发 slow path，用 LLM 提取实体、事实、关系
4. **持久存储** — 所有数据保存在 SQLite 中，重启后自动恢复

插件还注册了以下工具供 LLM 主动调用：
- `memory_recall` — 搜索长期记忆
- `memory_store` — 手动存储记忆
- `memory_forget` — 遗忘指定记忆
- `ebm_archive_expand` — 展开会话摘要为原始消息

---

## 常见问题排查

### Q: sidecar 启动失败，报 `TypeError: unsupported operand type(s) for |`

Python 版本太低。ebm_context_engine 使用了 `X | None` 语法，需要 Python 3.12+。

```bash
# 检查实际使用的 Python 版本
/usr/local/opt/python@3.12/bin/python3 --version
# 在 config 中指定完整路径
"pythonCommand": "/usr/local/opt/python@3.12/bin/python3"
```

### Q: Gateway 启动后看不到 `[EBM-PY]` 日志

检查 `openclaw.json` 中：
1. `plugins.slots.contextEngine` 是否设为 `"ebm-context-engine"`
2. `plugins.entries.ebm-context-engine.enabled` 是否为 `true`
3. 插件文件是否存在于 `~/.openclaw/extensions/ebm-context-engine/src/index.ts`

### Q: sidecar 启动了但 `/health` 返回连接拒绝

插件会自动扫描 `port` ~ `port + portScanRange` 范围。如果仍然失败：
```bash
lsof -i :18790
# 手动终止占用进程，或在 config 中改为其他端口
```

### Q: Remote 模式连接失败

```bash
# 确认远程服务可达
curl http://<服务器IP>:18790/health

# 检查防火墙
# 检查 baseUrl 是否正确（注意不要有尾部斜杠）
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
rm ~/.openclaw/memory/ebm_context_engine.sqlite
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
│  │       ebm-context-engine (TS 插件)             │  │
│  │                                               │  │
│  │  mode=local:  spawn + manage sidecar          │  │
│  │  mode=remote: connect to existing server      │  │
│  │                                               │  │
│  │  bootstrap() ──→ POST /bootstrap              │  │
│  │  ingest()    ──→ POST /ingest                 │  │
│  │  assemble()  ──→ POST /assemble               │  │
│  │  afterTurn() ──→ POST /after-turn             │  │
│  │  dispose()   ──→ POST /dispose + kill(local)  │  │
│  └───────────────────┬───────────────────────────┘  │
└──────────────────────┼──────────────────────────────┘
                       │ HTTP (localhost:18790 or remote)
┌──────────────────────┼──────────────────────────────┐
│  ebm_context_engine.server       │    Python HTTP Server         │
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
│        (~/.openclaw/memory/ebm_context_engine.sqlite)           │
└─────────────────────────────────────────────────────┘
```
