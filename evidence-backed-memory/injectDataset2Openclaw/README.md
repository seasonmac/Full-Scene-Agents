# Inject Dataset To OpenClaw

本目录用于把 `dataset/*.jsonl` 会话写入 OpenClaw transcript，并可选桥接到 `ebm-context-engine`。

默认行为是：

- 写入 OpenClaw session transcript。
- 调用 EBM `/bootstrap` 导入 transcript rows。
- 不主动触发 slow path 提取，避免 EBM Python sidecar 长时间占用导致 OpenClaw `/health`、`assemble`、`afterTurn` 超时。

## ingest2openclaw.py

建议在本目录执行：

```bash
cd injectDataset2Openclaw
mkdir -p openclaw_ingest
```

### A. 导入单个文件

只导入一个 JSONL 文件，默认只做 transcript 写入和 EBM bootstrap：

```bash
python3 ingest2openclaw.py \
  --dataset-dir ./dataset \
  --glob daily_report_0412.jsonl \
  --session-layout per-file \
  --mode replay \
  --session-prefix agent:main:news/dataset \
  --user-prefix news:dataset \
  --request-timeout 600
```

### B. 导入所有文件

导入 `dataset` 下所有 `*.jsonl` 文件，仍然只做 bootstrap，不触发 slow path：

```bash
python3 ingest2openclaw.py \
  --dataset-dir ./dataset \
  --glob '*.jsonl' \
  --session-layout per-file \
  --mode replay \
  --session-prefix agent:main:news/dataset \
  --user-prefix news:dataset \
  --request-timeout 600
```

### C. 导入所有文件并完成 ingest

如果需要导入后立即创建 slow path job 并触发处理，使用 `both` 和 `--ebm-flush`：

```bash
python3 ingest2openclaw.py \
  --dataset-dir ./dataset \
  --glob '*.jsonl' \
  --session-layout per-file \
  --mode replay \
  --session-prefix agent:main:news/dataset \
  --user-prefix news:dataset \
  --ebm-bridge-mode both \
  --ebm-flush \
  --request-timeout 600
```

注意：`--ebm-flush` 会触发 EBM slow path 处理，可能需要较长时间，并会占用本地 EBM sidecar。处理进度可用：

```bash
curl -s http://127.0.0.1:18790/status | jq .
```

如果通过 OpenClaw gateway 查看状态：

```bash
TOKEN=$(jq -r '.gateway.auth.token' ~/.openclaw/openclaw.json)
curl -s \
  -H "Authorization: Bearer $TOKEN" \
  http://127.0.0.1:18789/v1/extensions/ebm-py/status | jq .
```

常用参数说明：

| 参数 | 说明 |
| --- | --- |
| `--dataset-dir` | JSONL 数据目录，默认 `./dataset`。 |
| `--glob` | 要导入的文件匹配规则，例如 `daily_report_0412.jsonl` 或 `*.jsonl`。 |
| `--session-layout` | `per-file` 表示每个文件一个 OpenClaw session。 |
| `--mode` | `replay` 表示逐条消息写入；`bundle` 表示把整段会话合成单条消息。 |
| `--session-prefix` | OpenClaw/EBM session key 前缀。 |
| `--user-prefix` | session store 中使用的 user 前缀。 |
| `--ebm-bridge-mode` | `bootstrap` 只导入 transcript rows；`after-turn` 只入队 slow path；`both` 两者都做；`off` 不调用 EBM。 |
| `--ebm-flush` | 导入后触发 EBM slow path 队列处理。 |
| `--skip-ebm-bridge` | 只写 OpenClaw transcript，不调用 EBM。 |
| `--dry-run` | 只解析和生成计划，不实际写入。 |

## dumpDB.py

`dumpDB.py` 用于只读分析 EBM SQLite，输出 Markdown 格式的导入覆盖情况。

默认读取：

- dataset: `injectDataset2Openclaw/dataset/*.jsonl`
- DB: `~/.openclaw/memory/ebm_context_engine.sqlite`
- session prefix: `agent:main:news/dataset`

### 输出到终端

```bash
python3 dumpDB.py
```

输出格式包含逐文件汇总表：

```markdown
| 文件 | 源消息数 | DB transcript | summaries | facts |
| --- | ---: | ---: | ---: | ---: |
| daily_report_0412.jsonl | 41 | 82 | 0 | 0 |
```

以及整体状态：

```markdown
## 整体状态

dataset 源消息总数: 589
DB 中 agent:main:news/dataset/* transcript 总数: 685
session_summaries: 0
facts: 0
```

### 输出到 Markdown 文件

```bash
python3 dumpDB.py \
  --output ebm-db-report.md
```

### 指定数据目录、DB 或 session prefix

```bash
python3 dumpDB.py \
  --dataset-dir ./dataset \
  --glob '*.jsonl' \
  --db-path ~/.openclaw/memory/ebm_context_engine.sqlite \
  --session-prefix agent:main:news/dataset
```
