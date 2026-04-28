# ebm_context_engine 架构深度解析与重构方案

> 日期: 2026-04-16
> 对比系统: graph-memory (TypeScript), MAMGA (Python), OpenViking (Python)

---

## 1. ebm_context_engine 的图结构

ebm_context_engine 采用**多平面 + 超图**的混合架构，共有 6 层数据结构：

### 1.1 三个平面 (Plane)

#### Plane A — TaskFrontierWorkspace
- **用途**: 工作区，存储当前会话的 pinned facts 和 scratchpad
- **数据**: `PinnedEntry`（高优先级事实）+ `ScratchpadEntry`（临时笔记）
- **特点**: 短期记忆，不参与图检索

#### Plane B — StructuredSalientMemoryGraph
- **用途**: 核心知识图谱，负责图检索和排序
- **节点类型**: 4 种
  - `ENTITY` — 命名实体（人、地点、组织），存储在 `state.entities`
  - `EVENT` — 对话事件（每条消息），存储在 `state.events`
  - `FACT` — 提取的原子事实，存储在 `state.facts`
  - `TASK` / `SALIENT_MEMORY` — distillTurn 产生的任务节点和显著记忆节点，最终也存入 `state.facts`
- **边类型**: 7 种
  - `triggers` — TASK → EVENT（错误触发）
  - `supports` — TASK → EVENT（正常支持）
  - `temporal` — EVENT → EVENT（时间顺序链）
  - `solves` — EVENT → SALIENT_MEMORY（错误解决方案）
  - `related_to` — ENTITY ↔ ENTITY（实体关系）
  - `has_attribute` — ENTITY → FACT（实体属性）
  - `participates_in` — ENTITY → EVENT（实体参与事件）
  - `causes` / `prevents` / `enables` — 因果边
- **检索算法**: entity seed → multi-hop BFS expansion → PPR (Personalized PageRank) → RRF 融合
- **社区检测**: Louvain-like 贪心模块度优化，生成 `CommunitySummaryRecord`

#### Plane C — TemporalSemanticLedger
- **用途**: 事实账本，按时间衰减管理所有 `LedgerFact`
- **检索**: 向量相似度 + 时间衰减 + confidence 加权
- **特点**: 每个 fact 有 `validFrom` / `validTo` / `invalidAt` / `expiresAt` 生命周期字段

### 1.2 HyperMem 三级超图

在三个平面之上，还有一个独立的三级超图结构：

```
Topic (主题层)
  ├── Episode (片段层) — 语义连贯的对话片段
  │     ├── HmFact (事实层) — 原子事实 + query anticipation
  │     ├── HmFact
  │     └── ...
  ├── Episode
  └── ...
```

- `HmTopic`: 主题聚类，包含 title/summary/keywords + episode_ids
- `HmEpisode`: 对话片段，包含 session_key/dialogue/turn_start~turn_end + fact_ids + topic_ids
- `HmFact`: 原子事实，包含 content/potential(预期查询)/importance(high/mid/low)

**HyperMem 的检索路径**: Topic 向量匹配 → Episode 展开 → Fact 精确匹配（Coarse-to-Fine）

### 1.3 数据存储

所有数据存储在内存中的 `MemoryState` 对象：
```python
state.entities: dict[str, EntityNode]
state.events: dict[str, EventNode]
state.facts: dict[str, LedgerFact]
state.graph_edges: dict[str, GraphEdgeRecord]
state.adjacency: dict[str, set[str]]
state.communities: dict[str, CommunitySummaryRecord]
state.summaries: dict[str, SessionSummary]
state.transcripts: dict[str, TranscriptEntry]
state.hm_topics: dict[str, HmTopic]
state.hm_episodes: dict[str, HmEpisode]
state.hm_facts: dict[str, HmFact]
```

持久化通过 `db/store.py` 的 SQLite 存储。

---

## 2. 与 graph-memory 和 MAMGA 的图结构对比

### 2.1 graph-memory

| 维度 | graph-memory | ebm_context_engine |
|------|-------------|--------|
| **节点类型** | 3 种: TASK / SKILL / EVENT | 4+3 种: ENTITY / EVENT / FACT / TASK / SALIENT_MEMORY + HmTopic / HmEpisode / HmFact |
| **边类型** | 5 种: USED_SKILL / SOLVED_BY / REQUIRES / PATCHES / CONFLICTS_WITH | 7+ 种（见上） |
| **图引擎** | SQLite + FTS5 + 向量索引 | 内存 dict + numpy 向量 |
| **社区检测** | 有，基于 Louvain | 有，基于贪心模块度 |
| **PPR** | 有，可配置 damping/iterations，有 dangling node 处理 | 有，固定 damping=0.85，无 dangling 处理（已补齐） |
| **层级结构** | 单层图（nodes + edges + communities） | 多层（3 planes + 3-level hypergraph） |
| **recall 时 LLM 调用** | 0 次 | 3-5 次（classify + structured + rerank + temporal + answer） |
| **面向场景** | Agent 技能记忆（工具使用、错误修复） | 通用对话记忆 |

**核心差异**: graph-memory 是单层扁平图，recall 完全本地化（向量+FTS5+PPR），不调 LLM。ebm_context_engine 是多层嵌套结构，recall 过程中大量调用 LLM。

### 2.2 MAMGA (TRG Memory)

| 维度 | MAMGA | ebm_context_engine |
|------|-------|--------|
| **节点类型** | 5 种: EVENT / EPISODE / NARRATIVE / ENTITY / SESSION | 4+3 种（同上） |
| **边类型** | 4 大类 × 多子类: TEMPORAL(PRECEDES/SUCCEEDS/CONCURRENT) / SEMANTIC(RELATED_TO/SIMILAR_TO/PART_OF/CONTAINS) / CAUSAL(LEADS_TO/BECAUSE_OF/ENABLES/PREVENTS) / ENTITY(REFERS_TO/MENTIONED_IN) | 7+ 种 |
| **图引擎** | NetworkX MultiDiGraph | 内存 dict |
| **层级结构** | 3 层: EVENT → EPISODE → SESSION（NARRATIVE 独立） | 3 planes + 3-level hypergraph |
| **检索** | 向量搜索 + 关键词 + RRF 融合 + BFS 图遍历 | 向量 + PPR + multi-hop BFS + RRF |
| **Episode 分割** | 基于语义相似度的动态分割 | 基于 LLM 的 episode detection |

**核心差异**: MAMGA 的 EVENT → EPISODE → SESSION 三层结构与 ebm_context_engine 的 HmFact → HmEpisode → HmTopic 非常相似，但 MAMGA 用 NetworkX 做图计算，ebm_context_engine 用自己的 PPR 实现。MAMGA 的边类型体系更完整（有子类型），ebm_context_engine 的边类型更扁平。

### 2.3 三者图结构对比总结

```
graph-memory:  [TASK/SKILL/EVENT] ──edges──> [TASK/SKILL/EVENT]  (单层扁平图)
                        ↓ community detection
                   [Community]

MAMGA:         [SESSION] ──contains──> [EPISODE] ──contains──> [EVENT]
                                                        ↕ temporal/causal/semantic
                                                      [ENTITY]
               (3 层层级图 + 4 类边)

ebm_context_engine:        Plane A: [PinnedEntry] + [ScratchpadEntry]  (工作区)
               Plane B: [ENTITY/EVENT/FACT/TASK/SALIENT_MEMORY] ──edges──> [...]  (知识图谱)
                              ↓ community detection
                         [Community]
               Plane C: [LedgerFact] with time decay  (事实账本)
               HyperMem: [HmTopic] → [HmEpisode] → [HmFact]  (超图)
               (3 planes + 3-level hypergraph = 6 层)
```

ebm_context_engine 的复杂度最高，但很多层之间存在数据冗余（LedgerFact 和 HmFact 存储了相似的信息，Plane B 的 FACT 节点和 Plane C 的 LedgerFact 是同一份数据的不同视图）。

---

## 3. Evidence 的作用机制

### 3.1 Evidence 是什么

`EvidenceRef` 是 ebm_context_engine 的核心设计之一，它为每个提取的事实提供**溯源链接**：

```python
@dataclass
class EvidenceRef:
    sessionFile: str          # 来源 session 文件名
    messageIndex: int | None  # 来源消息在 session 中的索引
    startLine: int | None     # 来源文本的起始行
    endLine: int | None       # 来源文本的结束行
    snippet: str | None       # 原始文本片段（通常 < 200 字符）
    dateTime: str             # 来源消息的时间戳
    speaker: str              # 说话人
```

### 3.2 Evidence 存储在哪

Evidence 附着在以下数据结构上：
- `LedgerFact.evidence` — 每个事实的来源
- `EventNode.evidence` — 每个事件的来源
- `GraphEdgeRecord.evidence` — 每条边的来源
- `PinnedEntry.evidence` — 每个 pinned fact 的来源
- `ScratchpadEntry.evidence` — 每个 scratchpad 条目的来源
- `RecallHit.evidence` — 检索结果中携带的来源

### 3.3 Evidence 什么时候需要召回

**需要召回 evidence 的场景**:

1. **Temporal 类问题** — "When did X happen?"
   - evidence.dateTime 提供时间锚点
   - evidence.snippet 提供原始上下文用于时间推理
   - `_answer_from_temporal_evidence` 和 `_fast_temporal_grounding` 都依赖 evidence

2. **Verification 验证** — 当 fact 的 confidence < 阈值时
   - evidence.snippet 用于 LLM 验证 fact 的准确性
   - 验证后 fact.source 从 `llm-slot-extraction` 变为 `llm-slot-extraction-verified`

3. **Transcript recall** — 当需要原始对话上下文时
   - 根据 top hits 的 evidence.sessionFile + evidence.messageIndex 定位原始 transcript
   - 提取 ±2 条消息作为上下文窗口

4. **Answer generation** — 最终回答生成时
   - evidence.snippet 作为 "supporting evidence" 拼入 context
   - 帮助 LLM 生成更准确的回答

**不需要召回 evidence 的场景**:

1. **Graph recall** — PPR 排序阶段
   - 只用节点的 vector 和 keywords 做匹配，不需要 evidence
   - evidence 只在最终构建 RecallHit 时附带

2. **Community recall** — 社区级检索
   - 用社区的 summary 和 keywords 做匹配
   - 社区本身没有 evidence

3. **简单属性查询** — 当 fact 的 confidence 足够高时
   - 直接用 fact.value 回答，不需要回溯 evidence

### 3.4 Evidence 在 query 流程中的体现

```
query() → classify → embed → structured_path (evidence 用于验证)
                            → graph_recall (evidence 附带在 RecallHit 上)
                            → ledger_recall (evidence 附带在 RecallHit 上)
                            → transcript_recall (根据 evidence 定位原始对话)
                            → render_context (evidence.snippet 拼入 context)
                            → temporal_grounding (evidence.dateTime 做时间推理)
                            → answer_generation (evidence 作为 supporting context)
```

---

## 4. 置信区间的格式解读

### 4.1 LedgerFact.confidence

```python
confidence: float = 0.7  # 默认值
```

取值范围 `[0.0, 1.0]`，含义：

| 范围 | 含义 | 来源 |
|------|------|------|
| 0.95 | 明确陈述的事实 | LLM 提取时标注 "explicitly stated" |
| 0.90 | 高置信度推断 | LLM 提取时标注 "strongly implied" |
| 0.80-0.85 | 中等置信度 | LLM 提取时标注 "implied" 或 "inferred" |
| 0.70 | 默认值 | 未经 LLM 标注的本地提取 |
| 0.60 | 低置信度推断 | LLM 提取时标注 "weakly inferred" |
| < 0.60 | 不可靠 | 通常不会被检索返回 |

### 4.2 confidence 在检索中的作用

1. **Plane C (Ledger) 检索**: `effective_score = base_score * time_decay * confidence_factor`
2. **Structured path**: 只有 `confidence >= 0.85` 且 source 以 `-verified` 结尾的 fact 才进入候选
3. **Verification 触发**: `confidence >= structuredVerificationConfidence`（默认 0.85）的 fact 会被验证

### 4.3 其他 confidence 值

- `EntityNode`: `min(0.99, 0.6 + mention_count * 0.05)` — 提及次数越多越可信
- `EventNode`: `0.92`（错误事件）或 `0.75`（普通事件）
- `TASK` 节点: `0.9`
- `SALIENT_MEMORY` 节点: `0.81`
- `ClassificationResult.confidence`: intent 分类的置信度，规则引擎固定 `0.7`

---

## 5. ebm_context_engine Query 的详细流程

### 5.1 完整流程图

```
query(question)
│
├── 1. ensure_loaded()                          # 加载 SQLite → 内存 state
│
├── 2. classify_query(question)                 # Intent 分类
│      → 规则引擎: temporal/causal/multi_hop/entity/generic
│      → 输出: ClassificationResult + QueryPlan
│
├── 3. embed_query(question)                    # 向量化查询
│      → OpenAI embedding API → 1024-dim vector
│
├── 4. structured_path                          # 高置信度 fact 本地匹配
│      → _retrieve_structured_slot_hits()
│      → 从 verified facts 中找 confidence >= 0.85 的候选
│      → 结果作为 extra_hits 合并到后续排序
│
├── 5. graph_recall (Plane B)                   # 图检索
│      ├── _match_entities()                    # 实体名匹配 → entity seeds
│      ├── _find_entity_seeds()                 # 向量相似度 → 更多 seeds
│      ├── _expand_multi_hop()                  # BFS 2-hop 扩展
│      ├── rank_graph_nodes()                   # RRF(关键词 + 向量) 排序
│      ├── personalized_page_rank()             # PPR 全局排序
│      ├── merge multi-hop + PPR scores         # 分数融合
│      ├── _build_entity_summary()              # 实体摘要构建
│      ├── _recall_generalized()                # 社区级 fallback
│      └── _build_relationship_chains()         # 2-hop 关系链
│
├── 6. ledger_recall (Plane C)                  # 事实账本检索
│      → 向量相似度 + 时间衰减 + confidence 加权
│      → 按 subject 过滤（speaker names + 查询实体）
│
├── 7. summary_recall                           # Session 摘要检索
│      → 向量相似度匹配 SessionSummary
│
├── 8. transcript_recall                        # 原始对话片段召回
│      → 根据 top hits 的 evidence 定位原始 transcript
│      → 提取 ±2 条消息作为上下文窗口
│
├── 9. hypermem_recall (可选)                   # HyperMem C2F 检索
│      → Topic 向量匹配 → Episode 展开 → Fact 精确匹配
│
├── 10. _rank_combined_hits()                   # 多源结果融合排序
│       → graph + ledger + summary + community + structured + hypermem
│       → 按 source priority + score 排序
│
├── 11. _filtered_context_hits()                # 过滤不相关结果
│       → 移除 relationship chain hits
│       → temporal 类问题过滤 summary 和非 event graph hits
│       → 取 top 6
│
├── 12. _rerank_hits()                          # 重排序
│       → 按现有 PPR/confidence score 排序（不调 LLM）
│       → 取 top 5
│
├── 13. render_context()                        # 构建 LLM context
│       → 拼接 reranked hits 的 title + content + evidence.snippet
│       → 添加 transcript snippets
│       → 添加 intent 提示
│
├── 14. temporal_grounding (temporal 类问题)    # 时间推理
│       ├── 快速路径: evidence.dateTime 直接提取
│       └── LLM 路径: 调 LLM 做时间推理
│
└── 15. answer_generation                       # 最终回答
        → 调 LLM 基于 context 生成回答
        → 空答案重试（无 sleep）
```

### 5.2 各阶段的数据流

```
                    ┌─────────────┐
                    │   question   │
                    └──────┬──────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        classify()    embed()    structured_path()
              │            │            │
              ▼            ▼            ▼
        Classification  query_vector  structured_hits
              │            │            │
              ├────────────┼────────────┤
              ▼            ▼            ▼
        graph_recall()  ledger_recall()  summary_recall()
              │            │            │
              └────────────┼────────────┘
                           ▼
                  _rank_combined_hits()
                           │
                           ▼
                  _filtered_context_hits()
                           │
                           ▼
                    _rerank_hits()
                           │
                           ▼
                   render_context()
                           │
                    ┌──────┴──────┐
                    ▼             ▼
            temporal?        answer()
            grounding          │
                    │          │
                    └────┬─────┘
                         ▼
                PythonEbmQueryResult
```

---

## 6. 三层图结构渐进式披露重构方案

### 6.1 问题分析

当前 ebm_context_engine 的 query 流程一次性召回所有层级的数据（graph + ledger + summary + transcript + hypermem），然后在 `_rank_combined_hits` 中做扁平排序。这导致：

1. **token 浪费**: 大量低相关性的 hits 被拼入 context，浪费 LLM token
2. **噪声干扰**: 低质量 hits 稀释了高质量 evidence 的信号
3. **无法自适应**: 简单问题和复杂问题使用相同的召回深度

### 6.2 OpenViking 的三层渐进式披露

OpenViking 的 `HierarchicalRetriever` 实现了经典的三层渐进式披露：

```
Level 0: Abstract (目录级摘要，~50 tokens)
    ↓ rerank 筛选
Level 1: Overview (资源级概述，~200 tokens)
    ↓ rerank 筛选
Level 2: Full Content (完整内容，~1000 tokens)
```

关键设计:
- 每层都有独立的向量索引
- 每层都经过 rerank 筛选，只有得分超过阈值的才进入下一层
- 收敛检测: 如果连续 N 轮 top-k 不变，停止递归
- 全局搜索 + 递归搜索并行: 全局向量搜索找到起始点，递归搜索沿目录树深入

### 6.3 ebm_context_engine 的三层正交架构（已实现）

消除原有 6 层数据冗余，合并为 3 层正交结构：

**核心合并**:
- `UnifiedFact` = `LedgerFact` + `HmFact`（合并去重，保留两者优势字段）
- `HmTopic` 吸收 `CommunitySummaryRecord`（社区检测结果写入 HmTopic，source="community_detection"）
- `HmEpisode` 吸收 `SessionSummary`（session 摘要作为 is_session_summary=True 的 episode）

```
Layer 0: HmTopic（唯一数据源，粗粒度索引，~50 tokens/hit）
    ├── LLM 聚合生成的主题（source="llm_aggregation"）
    └── 社区检测生成的主题（source="community_detection"，含 member_entity_ids）
    → BM25 + Vector RRF → top-K topics

Layer 1: HmEpisode（唯一数据源，中粒度上下文，~200 tokens/hit）
    ├── 对话片段 episode（is_session_summary=False）
    └── Session 级摘要 episode（is_session_summary=True）
    → 只展开 Layer 0 命中 topic 的 episode_ids
    → BM25 + Vector RRF → top-K episodes

Layer 2: UnifiedFact（唯一数据源，细粒度事实，~100 tokens/hit）
    ├── slowpath 提取的事实（含 evidence, confidence, lifecycle）
    └── HyperMem 提取的事实（含 potential, importance）
    → 只展开 Layer 1 命中 episode 的 fact_ids
    → BM25 + Vector RRF + confidence × importance × temporal_decay
    → entity seed 加分 → top-K facts
```

**正交性保证**:
- 每层只有一种数据源，不存在同一数据的多个视图
- 层间关系通过 ID 引用（topic.episode_ids → episode.fact_ids）
- ingestion 时 LedgerFact 和 HmFact 自动合并去重（cosine > 0.9 的同 episode 内 fact 合并）

### 6.4 实现文件清单

| 文件 | 改动 |
|------|------|
| `ebm_context_engine/types.py` | 新增 `UnifiedFact` 数据类；扩展 `HmTopic`（+member_entity_ids, source）和 `HmEpisode`（+entity_ids, source_event_ids, is_session_summary）；`MemoryState` 新增 `unified_facts` 字段 |
| `ebm_context_engine/engine.py` | 新增 `_ledger_fact_to_unified` / `_hm_fact_to_unified` / `_merge_unified_facts` / `_upsert_unified_fact` 转换函数；ingestion 时同步写入 `unified_facts`；`_rebuild_communities` 同步写入 `HmTopic`；SessionSummary 同步生成 `HmEpisode(is_session_summary=True)`；`query()` 中当 `unified_facts` 存在时使用 `ProgressiveRecaller` 替代 5 路并行召回 |
| `ebm_context_engine/retrieval/progressive.py` | 新增 `ProgressiveRecaller` 类，三层渐进式检索（Topic → Episode → UnifiedFact），复用 c2f_retriever 的 BM25+Vector+RRF 评分函数 |
| `ebm_context_engine/core/config.py` | 新增 `layer0TopK=5` / `layer1TopK=8` / `layer2TopK=6` 配置 |
| `ebm_context_engine/db/store.py` | 新增 `unified_facts` 表 + FTS5 索引 + CRUD 方法 |

### 6.5 数据流对比

**重构前（6 层，冗余）**:
```
ingestion:
  对话 ──slowpath──→ LedgerFact ──→ state.facts ──→ Plane B (PPR) + Plane C (decay)
       ──HyperMem──→ HmFact ──→ state.hm_facts ──→ C2F retrieval
       ──community─→ CommunitySummaryRecord ──→ state.communities
       ──summary───→ SessionSummary ──→ state.session_summaries

query: 5 路并行 → _rank_combined_hits 合并 → _rerank_hits → answer
```

**重构后（3 层，正交）**:
```
ingestion:
  对话 ──slowpath──→ UnifiedFact ──→ state.unified_facts（合并去重）
       ──HyperMem──→ UnifiedFact ──→ state.unified_facts（合并去重）
       ──community─→ HmTopic(source="community") ──→ state.hm_topics
       ──summary───→ HmEpisode(is_session_summary=True) ──→ state.hm_episodes

query: ProgressiveRecaller(Topic → Episode → UnifiedFact) → answer
```
