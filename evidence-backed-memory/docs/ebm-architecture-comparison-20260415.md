# EBM 架构对比说明（2026-04-15）

## 1. 这份文档回答什么问题

这份文档专门回答 4 个问题：

1. 旧版 EBM 里 `fastpath / slowpath` 各自负责什么。
2. 为什么当前重构过程中会出现“`fast ingest` 已开启，但 `fast query` 暂时关闭”的分离状态。
3. 新架构与旧架构相比，变化到底有多大。
4. 当前已经落地到哪一步，还有哪些模块仍停留在旧路径。

这份文档不记录 benchmark 实验细节；实验与日志对照请看：

- [evermemos-ebm-refactor-20260414.md](/Users/season/workspace/github/noteLM/docs/evermemos-ebm-refactor-20260414.md)

## 2. 旧版 EBM 的真实架构

### 2.1 写入侧：有 fast/slow 分层

旧版 EBM 的 `fastpath / slowpath` 主要是 **写入侧概念**。

数据流：

```text
afterTurn / ingestBatch
  -> 先写 transcript / workspace / turn payload
  -> slowPath.enqueue(...)
  -> 后台或延迟执行：
       - graph distill
       - ledger facts
       - entity graph
       - HyperMem episode/fact/topic
       - session summary
```

也就是说，旧版的 `slowpath` 是一个 **ingest 补全器**，负责把原始对话慢慢加工成更重的记忆结构。

### 2.2 查询侧：实际上只有一条重路径

旧版 query 没有真正意义上的 fast path。

旧版 query 的实际流程更接近：

```text
query
  -> classify
  -> graph_recall
  -> ledger_recall
  -> summary_recall
  -> c2f_retrieval
  -> transcript_recall
  -> merge / rerank / render_context
  -> LLM answer
```

这意味着：

- 写入有快慢之分
- 查询没有快慢之分
- 查询一上来就是“全量重路径”

### 2.3 旧架构的问题

从运行日志看，旧 query 路径有两个结构性问题：

1. **所有重型召回默认并联执行**
   - graph
   - ledger
   - summary
   - c2f
   - transcript
   每题都跑

2. **context 组装没有候选收敛层**
   - 先堆很多候选
   - 再做合并 / 去重 / 拼上下文
   - 最终才喂给 LLM

所以旧版 EBM 的本质不是“检索慢”，而是：

- **候选过多**
- **表示重复**
- **render_context 太重**

## 3. 新版 EBM 的目标架构

### 3.1 设计目标

新版目标不是简单调参，而是把 query 也做成真正的两层架构：

```text
query
  -> classify
  -> route
     -> fast query path
     -> deep query path
```

这里的核心变化是：

- 旧版只有 ingest 有 fast/slow
- 新版希望 ingest 和 query 都有 fast/slow

### 3.2 新版数据面：增加 `memory_units`

新架构的第一层不是 graph，也不是 HyperMem，而是：

- `memory_units`

它们是从已有结构里再抽一层出来的轻量通用记忆单元，目标是更接近 EverMemOS 的 `memcell`。

当前已实现的数据结构：

- `MemoryUnit`
- `MemoryState.memory_units`
- `MemoryState.memory_unit_index`

设计上，`memory_units` 用来承接：

- 原子事实
- 轻量摘要
- 高价值 transcript 片段

它们的职责不是替代 graph，而是做 query 第一层候选收敛。

## 4. 为什么现在是“分开的”

当前代码状态里，确实存在：

- `benchmarkFastIngest = true`
- `fastQueryEnabled = false`

这不是设计矛盾，而是**有意分阶段上线**。

### 4.1 为什么 ingest 可以先开

因为 ingest 侧删除的是非常明确的冗余环节：

- slow-path LLM 抽取
- 同步 community rebuild
- 同步 HyperMem propagation
- benchmark 下 HyperMem LLM 路径

这些改动不改变 query 的判断逻辑，只是去掉重型同步后处理。

收益已经明确验证：

- `sample0 ingest`
  - 旧：`3620.8s`
  - 新：`202.7s`

所以 ingest fast path 已经是“稳定收益”。

### 4.2 为什么 query 不能一起全开

因为 query fast path 当前还在收敛阶段。

它已经具备骨架：

- `memory_units`
- candidate retrieval
- rerank
- direct answer / structured answer

但它还存在两个现实问题：

1. `memory_units` 质量还不够干净
   - 混入了伪事实和系统内部中间表示

2. 命中条件还不够保守
   - 某些题被错误地提前命中 fast path
   - 导致速度更快，但精度明显回退

所以当前策略是：

- 先保留已经验证有效的 `fast ingest`
- 暂时关闭不稳定的 `fast query`
- 继续只在局部回归里收敛 query fast path

### 4.3 本质上现在是“架构迁移中的双轨期”

可以把当前状态理解成：

```text
写入侧：新路径已接管
查询侧：新骨架已接入，但默认仍由旧重路径接管
```

这是一种典型的迁移方式：

- 先接骨架
- 再局部放流量
- 最后才默认接管

## 5. 旧架构 vs 新架构：差异到底有多大

### 5.1 控制面的变化很大

旧版 query 控制面：

```text
classify
-> 全量召回
-> 合并
-> render_context
-> LLM
```

新版目标 query 控制面：

```text
classify
-> route
   -> fast query
   -> deep query
-> answer strategy
-> fallback / escalation
```

这说明变化不是“优化某个函数”，而是 **query orchestration 模式变化**。

### 5.2 数据面的变化中等偏大

旧版 query 直接消费：

- graph hits
- ledger hits
- summary hits
- HyperMem hits
- transcript windows

新版 query 目标先消费：

- `memory_units`

只有在候选不足时，才升级到：

- graph
- ledger
- HyperMem
- transcript 扩展

所以数据面变化可以总结为：

- **从多结构并联召回**
- 变成
- **先轻量候选收敛，再重型扩展**

### 5.3 渲染面的变化也很大

旧版：

- 最终 prompt 来自大而重复的 evidence block

新版目标：

- 最终 prompt 来自少量、压缩、去重后的 evidence
- 单值题/时间题优先本地组装答案
- LLM 不再负责“从杂乱大上下文里自己找答案”

## 6. 当前已经落地的模块

### 6.1 已接入

1. `memory_units` 数据结构
2. benchmark 专用 config
3. `benchmarkFastIngest`
4. query fast-path 骨架
5. fast-path 候选检索 / rerank / structured answer 骨架
6. 结构化阶段日志

### 6.2 当前默认启用

默认启用的是：

- `benchmarkFastIngest`

### 6.3 当前默认关闭

默认关闭的是：

- `fastQueryEnabled`

原因不是功能不存在，而是质量尚未完全收敛。

## 7. 当前没有切过去的边界

现在仍然保留旧 deep query 路径的模块：

1. `graph_recall`
2. `ledger_recall`
3. `summary_recall`
4. `transcript_recall`
5. `render_context`
6. `LLM answer`

也就是说，当前 query 默认还是：

```text
旧 deep path
```

只是：

- 有了新骨架
- 有了新日志
- 有了新数据层
- 还没让它默认接管

## 8. 为什么这次重构比旧 fastpath/slowpath 的区别大

因为旧 `fastpath/slowpath` 主要是 **写入层分层**。  
而这次重构开始改的是 **查询层的控制面**。

这不是在 slowPath 上加几个 if，而是在改变：

- 查询该先做什么
- 候选该先收敛还是先扩展
- 哪些题不该走重路径
- 哪些题必须升级到重路径

所以它和旧架构的区别，不是“局部优化”，而是“路径治理方式变了”。

## 9. 推荐如何理解当前系统

当前 EBM 最准确的描述是：

### 写入侧

- 已经进入新架构
- benchmark 下已经大幅去冗余

### 查询侧

- 新架构骨架已经接进来
- 但默认还没有完全接管
- 正处于“旧路径保底，新路径收敛”的阶段

## 10. 后续最合理的推进顺序

如果继续推进，不建议再做大范围重写，而是按下面顺序：

1. 先把 fast query 的候选质量收敛
   - 清洗 `memory_units`
   - 去掉伪事实

2. 再把 fast query 只放给高置信题型
   - temporal exact
   - single-slot fact
   - small list aggregation

3. 最后才逐步收缩 deep path
   - 不再默认并联所有重召回

这个顺序的好处是：

- ingest 的收益已经锁住
- query 的质量不会因为激进切换而回退
- 架构迁移风险最低

## 11. 架构图（ASCII 数据流）

### 11.1 旧版架构

```text
                +----------------------+
                |   afterTurn/ingest   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | transcript/workspace |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  slowPath enqueue    |
                +----------+-----------+
                           |
                           v
      +--------------------------------------------------+
      | slowPath drain / synchronous rebuild chain       |
      | - graph distill                                  |
      | - ledger facts                                   |
      | - entity graph                                   |
      | - HyperMem episode/fact/topic                    |
      | - session summary                                |
      | - community rebuild / propagation                |
      +--------------------------------------------------+


                +----------------------+
                |       query()        |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |      classify         |
                +----------+-----------+
                           |
                           v
      +--------------------------------------------------+
      | default heavy path (every query)                 |
      | - graph_recall                                   |
      | - ledger_recall                                  |
      | - summary_recall                                 |
      | - c2f_retrieval                                  |
      | - transcript_recall                              |
      +--------------------------------------------------+
                           |
                           v
                +----------------------+
                |   merge / rerank     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |   render_context     |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |    LLM answer()      |
                +----------------------+
```

### 11.2 当前重构中的目标架构

```text
                +----------------------+
                |   afterTurn/ingest   |
                +----------+-----------+
                           |
                           v
                +----------------------+
                | transcript/workspace |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |  compact units build  |
                |   (memory_units)      |
                +----------+-----------+
                           |
                           v
      +--------------------------------------------------+
      | optional heavy enrichers                         |
      | - graph distill                                  |
      | - ledger facts                                   |
      | - HyperMem heuristic/LLM path                    |
      | - community rebuild / propagation                |
      +--------------------------------------------------+


                +----------------------+
                |       query()        |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |      classify         |
                +----------+-----------+
                           |
                           v
                +----------------------+
                |       route           |
                +----+-------------+----+
                     |             |
          fastQueryEnabled      fallback
                     |             |
                     v             v
        +-------------------+   +----------------------+
        |   fast query path |   |   deep query path    |
        | - memory_units    |   | - graph_recall       |
        | - light retrieval |   | - ledger_recall      |
        | - rerank(optional)|   | - summary_recall     |
        | - direct answer   |   | - transcript_recall  |
        +---------+---------+   | - optional HyperMem  |
                  |             +----------+-----------+
                  |                        |
                  +-----------+------------+
                              |
                              v
                    +----------------------+
                    |   answer strategy    |
                    | direct / small LLM / |
                    | deep-context LLM     |
                    +----------------------+
```

### 11.3 当前实际落地状态

```text
写入侧:
  compact units build         -> 已落地
  benchmarkFastIngest         -> 已落地
  heavy enrichers             -> 部分保留，部分裁剪

查询侧:
  route/fast skeleton         -> 已接入
  memory_units retrieval      -> 已接入
  fast query default takeover -> 未开启（默认关闭）
  deep path fallback          -> 当前默认主路径
```

## 12. 代码映射（架构到函数/模块）

### 12.1 写入侧映射

#### Transcript / workspace 写入

- `ebm_context_engine/engine.py`
  - `bootstrap()`
  - `ingestBatch()`
  - `after_turn()`
  - `_store_transcript_rows()`

职责：

- 把对话行写入 transcript 存储
- 刷新运行时基础状态

#### Slow path 入口

- `ebm_context_engine/engine.py`
  - `_build_slow_path_job_payload()`
  - `_execute_slow_path_job()`
  - `flush_slow_path()`
  - `retry_failed()`

职责：

- slow path job 构建与消费
- ingest 后台补全

#### Graph / Ledger / Summary / HyperMem 构建

- `ebm_context_engine/engine.py`
  - `_apply_slow_path_payload()`
  - `_apply_hypermem_pipeline()`
  - `_rebuild_communities()`
  - `_embed_state()`
  - `_propagate_hm_embeddings()`
  - `_rebuild_indices()`

- 关联模块
  - `ebm_context_engine/planes/plane_b.py`
  - `ebm_context_engine/planes/plane_c.py`
  - `ebm_context_engine/hypergraph/*`

#### 新增 compact units 层

- `ebm_context_engine/engine.py`
  - `_rebuild_memory_units()`
  - `_infer_value_type()`
  - `_extract_time_text()`
  - `_normalize_time_text()`

职责：

- 从已有 `facts / summaries / transcripts` 中抽出轻量记忆单元
- 为后续 fast query 提供统一候选层

### 12.2 查询侧映射

#### Query 总控入口

- `ebm_context_engine/engine.py`
  - `query()`

职责：

- classify
- route
- 执行 fast 或 deep path
- 记录 query phase timing

#### Fast query path

- `ebm_context_engine/engine.py`
  - `_should_use_fast_path()`
  - `_fast_path_reason()`
  - `_candidate_memory_units_for_fast_path()`
  - `_retrieve_memory_units()`
  - `_rerank_memory_units()`
  - `_compose_temporal_answer()`
  - `_compose_slot_answer()`
  - `_fast_context()`
  - `_structured_answer()`
  - `_try_fast_answer()`

职责：

- 只依赖轻量候选层
- 为 temporal / single-slot / list 类问题提供快速答案路径

当前状态：

- 骨架已实现
- 默认配置中暂未开启全面接管

#### Deep query path

- `ebm_context_engine/engine.py`
  - `_match_entities()`
  - `_recall_summaries()`
  - `_recall_transcript_context()`
  - `_rank_combined_hits()`
  - `_rerank_hits()`
  - `_build_ranked_context()`
  - `_answer()`

- 关联模块
  - `ebm_context_engine/planes/plane_b.py` -> graph recall
  - `ebm_context_engine/planes/plane_c.py` -> ledger recall
  - `ebm_context_engine/hypergraph/c2f_retriever.py` -> HyperMem C2F

职责：

- 旧版全量重路径
- 当前仍是默认主路径

### 12.3 benchmark 配置映射

- `benchmark/adapters/ebm_adapter.py`
  - `_ensure_engine()`

职责：

- 将 benchmark 模式下的 `benchmarkFastIngest / fastQueryEnabled ...`
  透传给 `EbmConfig`

## 13. 为什么 Query 现在仍然是 deep path 主导

虽然 query fast skeleton 已经接入，但从代码和日志看，它还没有成为默认主路径，原因是：

1. `memory_units` 还没有彻底清洗干净
2. fast query 对 temporal / single-slot 的命中条件还不够保守
3. 一旦误命中，错误答案会直接跳过 deep fallback，造成明显掉分

因此，当前架构选择是：

- 让新 query 架构先“并存”
- 不让它直接“接管”

这和写入侧已经接管成功的 `benchmarkFastIngest` 是完全不同的成熟度阶段。
