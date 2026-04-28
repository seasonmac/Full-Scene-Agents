# EBM 查询管线重构：嵌入分类 + Rerank

**日期**: 2025-06-11

## 背景

EBM 查询管线中，LLM 意图分类占总延迟的 68%（1.1-5.3s/次）。参考 OpenViking 的分层检索与 rerank-with-fallback 模式进行优化。

## 变更概要

### 1. 意图分类: LLM → 嵌入余弦相似度 (`intent_router.py`)

- 移除 LLM 调用，改用预计算的意图原型向量（5 类：temporal/causal/multi_hop/entity/generic）
- 查询向量与原型向量做余弦相似度，取最高分
- 实体提取从 `MemoryState` 已知实体名匹配，不再依赖 LLM
- 时间标记检测保留 Unicode 启发式方法
- 原型缓存按 `embed_fn` id 分离，支持多引擎实例

### 2. Rerank 集成 (`progressive.py`, `client.py`, `engine.py`)

- `OpenAICompatClient.rerank()` 方法：POST `/rerank` 端点
- `ProgressiveRecaller` 新增 `rerank_fn` 参数
- Layer2 事实检索后，可选 rerank 重排序，失败自动回退 RRF 融合分数
- 修复了 rerank 索引映射 bug（过滤缺失 ID 后的偏移问题）

### 3. 上下文渲染优化 (`engine.py`)

- `_build_ranked_context` 去除重复的 title+content，精简格式
- `transcriptContextMaxChars` 从 2200 降至 1500

## 性能结果（Sample 0, 6 QA）

| 指标 | 重构前 | 重构后 | 变化 |
|------|--------|--------|------|
| 平均延迟 | 4.11s | 1.55s | -62% |
| 平均分数 | 0.767 | 0.767 | 持平 |
| 平均 tokens | 990 | 708 | -28% |

## 修改文件

- `ebm_context_engine/retrieval/intent_router.py` — 完全重写
- `ebm_context_engine/retrieval/progressive.py` — 新增 rerank_fn + rerank 集成
- `ebm_context_engine/engine.py` — 调用顺序、上下文渲染、_reranker 客户端
- `ebm_context_engine/client.py` — rerank() 方法
- `ebm_context_engine/core/config.py` — transcriptContextMaxChars 调整
