# EBM Benchmark 对照组设计：OpenClaw vs OpenClaw+EBM

## 一、需求说明书 4 个核心痛点与评价指标映射

| 痛点 | 评价指标 | 说明 |
|------|---------|------|
| 1. 线索寻找与多跳推理（大海捞针） | **检索命中率 + 多跳召回率** | 不只是 precision，还包括跨文档/跨 session 的 recall |
| 2. 上下文瓶颈与算力成本（脑容量有限） | **Context Token 消耗量** | tokens per successful answer |
| 3. 跨会话遗忘与状态维持（睡一觉就断片） | **跨 Session 长期记忆一致性** | 长期事实留存率 + 事实演化正确性 |
| 4. 经验孤岛与自我进化（无法越用越聪明） | **跨任务经验复用率 + Skill 级 bugfix 复用成功率** | 新闻分析/SOP 复用与代码/工具/Skill 的 bugfix 复用需要拆开评测 |

与需求说明书第七页收敛的 4 个技术目标对齐：Token 效率、跨任务局部复用率、长期连续性、任务成功率。

---

## 二、实验设计

### 场景 A："每日财经与科技新闻"（当前已落地）

模拟用户每天和 AI 讨论财经/科技新闻的长期伴随场景。天然覆盖：
- 实体密集（公司名、人名、股票代码、产品名）
- 时间敏感（日期、季度、财报周期）
- 跨 session 关联（同一公司连续多天的新闻发展）
- 事实演化（股价变动、CEO 更替、产品迭代）

### 当前 `news_report/dataset` 实际规模（2026-04-21 审计）

- 实际为 **9 个 session**，时间范围为 `2026-04-12 ~ 2026-04-20`
- 共 **589 条合法 JSONL 消息**：`user=290`，`assistant=299`
- 单个 session 规模在 **41~83 条消息**
- 当前数据集足以覆盖痛点 1、2、3，以及痛点 4 中的 **新闻分析框架复用 / 观察清单更新 / 输出 SOP 复用**
- 当前数据集**不覆盖**“代码 / 工具 / Skill 解 bug 成功经验复用”，这部分需要单独的 coding/bugfix 数据集

### 场景 B："Coding / Skill Bugfix"（当前未落地）

若要完整覆盖痛点 4，还需要补充单独的代码任务数据集，至少应包含：
- 明确的报错/失败 trace（如 `Traceback`、测试失败、安装失败）
- 至少一次成功修复路径
- 可复用的工具调用顺序 / 操作 SOP
- 后续新任务中对同类问题的经验迁移

### 实验结构

```
┌─────────────────────────────────────────────────────────┐
│  数据集构造：当前可用数据为 9 个 session                    │
│  每 session 41-83 条消息，含 1 条日报 + 多轮用户追问       │
│  总计 589 条消息，跨越 9 天                                │
├─────────────────────────────────────────────────────────┤
│  对照组 A: OpenClaw (原生 memory)                        │
│  实验组 B: OpenClaw + EBM                               │
├─────────────────────────────────────────────────────────┤
│  评测阶段：Ingest 全部 session 后，执行 50 道评测问题      │
│  问题类型分布：                                           │
│    - 单跳事实 / 概念解释 (20%)                            │
│    - 时间推理 / 事实演化 (20%)                            │
│    - 多跳推理 / 跨主题归因 (25%)                          │
│    - 跨 session 观察清单 / 策略更新 (20%)                 │
│    - 输出 SOP / 风格复用 (15%)                            │
└─────────────────────────────────────────────────────────┘
```

---

## 三、4 个指标计算公式

### 指标 1：检索精准度（Recall Precision）— 对应痛点 1

```python
# 每道题有人工标注的 gold_evidence_set（应该被召回的关键信息片段）
# 系统实际召回的 context_hits 与 gold 对比

def recall_precision(questions: list[Question]) -> dict:
    """衡量系统能否精准找到相关线索，而非返回噪声"""
    
    precisions = []
    recalls = []
    multi_hop_scores = []
    
    for q in questions:
        retrieved_ids = set(q.system_retrieved_evidence_ids)
        gold_ids = set(q.gold_evidence_ids)
        
        # Precision: 召回的内容中有多少是真正相关的
        precision = len(retrieved_ids & gold_ids) / max(len(retrieved_ids), 1)
        # Recall: 应该被找到的内容中有多少被找到了
        recall = len(retrieved_ids & gold_ids) / max(len(gold_ids), 1)
        
        precisions.append(precision)
        recalls.append(recall)
        
        # 多跳专项：需要 >= 2 个不同 session 的证据才能回答的题
        if q.required_hops >= 2:
            sessions_covered = len(set(
                e.session_key for e in q.retrieved if e.id in gold_ids
            ))
            multi_hop_scores.append(sessions_covered / q.required_hops)
    
    return {
        "precision_mean": mean(precisions),
        "recall_mean": mean(recalls),
        "f1": 2 * mean(precisions) * mean(recalls) / max(
            mean(precisions) + mean(recalls), 1e-9
        ),
        "multi_hop_coverage": mean(multi_hop_scores),  # 越高说明多跳能力越强
    }
```

**显著性对比设计**：
- 多跳题（需跨 session 关联）是拉开差距的关键 — 原生 OpenClaw 的扁平检索在这类题上 recall 会显著低于 EBM 的图检索 + PPR
- 预期效果：EBM 在多跳题上 F1 提升 30%+

---

### 指标 2：Context Token 效率 — 对应痛点 2

```python
def token_efficiency(questions: list[Question]) -> dict:
    """衡量回答同样问题所消耗的 context token 量"""
    
    results = []
    for q in questions:
        results.append({
            "context_tokens": q.context_token_count,      # 注入到 prompt 的记忆 token 数
            "total_tokens": q.total_prompt_tokens,        # 完整 prompt token 数
            "answer_correct": q.judge_score >= 0.8,       # 回答是否正确
        })
    
    correct_only = [r for r in results if r["answer_correct"]]
    all_results = results
    
    return {
        # 核心指标：每个正确回答平均消耗多少 context token
        "tokens_per_correct_answer": mean(
            [r["context_tokens"] for r in correct_only]
        ),
        # 总 token 消耗（含错误回答）
        "total_context_tokens": sum(
            r["context_tokens"] for r in all_results
        ),
        # 压缩率：相对于把全部历史塞进去的 baseline
        "compression_ratio": 1 - (
            sum(r["context_tokens"] for r in all_results) / FULL_HISTORY_TOKEN_COUNT
        ),
        # 信息密度：正确率 / 平均 context token（越高越好）
        "info_density": len(correct_only) / max(
            sum(r["context_tokens"] for r in all_results) / 1000, 1
        ),
    }
```

**显著性对比设计**：
- 原生 OpenClaw 倾向于塞入大量 raw transcript 作为 context
- EBM 通过三层结构化压缩（Topic→Episode→Fact）+ token budget 控制，预期在同等正确率下 token 消耗降低 50-75%
- 关键对比图：**正确率 vs context token 散点图** — EBM 应该在左上角（低 token、高正确率）

---

### 指标 3：跨 Session 长期记忆一致性 — 对应痛点 3

```python
def long_term_consistency(questions: list[Question]) -> dict:
    """衡量跨 session 的事实留存与演化能力"""
    
    # 子指标 A：事实留存率 — 早期 session 提到的事实，后期能否正确回忆
    retention_scores = []
    for q in questions:
        if q.category == "cross_session":
            # 该问题的答案来源于 session_distance 天前的对话
            session_distance = q.answer_source_session_distance  # 1-9
            correct = q.judge_score >= 0.8
            retention_scores.append({
                "distance": session_distance,
                "correct": correct,
            })
    
    # 按距离分桶统计留存率
    retention_by_distance = {}
    for d in range(1, 10):
        bucket = [r for r in retention_scores if r["distance"] == d]
        if bucket:
            retention_by_distance[f"day_{d}"] = mean(
                [r["correct"] for r in bucket]
            )
    
    # 子指标 B：事实演化正确性 — 同一实体的状态更新后，系统是否返回最新值
    evolution_questions = [q for q in questions if q.has_fact_evolution]
    evolution_accuracy = mean(
        [q.judge_score for q in evolution_questions]
    ) if evolution_questions else 0
    
    # 子指标 C：抗污染率 — 旧事实不应覆盖新事实
    contradiction_questions = [q for q in questions if q.has_contradiction]
    anti_pollution = mean(
        [q.returns_latest_fact for q in contradiction_questions]
    ) if contradiction_questions else 0
    
    return {
        "retention_curve": retention_by_distance,       # 随时间的留存曲线
        "avg_retention": mean(retention_by_distance.values()),
        "evolution_accuracy": evolution_accuracy,       # 事实更新后的正确率
        "anti_pollution_rate": anti_pollution,          # 不被旧事实污染的比率
    }
```

**显著性对比设计**：
- 构造"事实演化"题：Day 3 说「特斯拉股价 180」，Day 7 说「特斯拉跌到 160」，问「特斯拉当前股价？」
- 构造"远距离留存"题：Day 1 讨论的公司细节，Day 10 才问
- 原生 OpenClaw 没有时序账本，容易返回过时信息或完全遗忘
- 预期：EBM 的 retention curve 显著平坦（不随距离衰减），evolution_accuracy 高 40%+

---

### 指标 4A：新闻分析 / SOP 经验复用 — 对应痛点 4（当前 `news_report/dataset` 覆盖）

```python
def experience_reuse(questions: list[Question]) -> dict:
    """衡量系统能否将历史讨论中的分析模式复用到新问题"""
    
    # 设计：前几个 session 中用户和 AI 讨论过某些分析框架 / 观察清单 / 输出模板
    # 例如：Day 2 建立了一个观察清单，Day 4 询问“你前天让我盯的三条线现在怎么样了”
    # 或者 Day 1 确立了“睡前一句提醒”的输出风格，Day 3 要求沿用前天风格
    
    reuse_questions = [q for q in questions if q.category == "experience_reuse"]
    
    # 子指标 A：模式召回率 — 历史讨论过的分析框架/结论能否被召回
    pattern_recalled = mean(
        [q.historical_pattern_in_context for q in reuse_questions]
    )
    
    # 子指标 B：应用正确性 — 召回后能否正确应用到新场景
    application_correct = mean(
        [q.judge_score for q in reuse_questions]
    )
    
    # 子指标 C：证据可追溯性 — 回答中是否能指向原始讨论来源
    evidence_traceable = mean([
        1.0 if q.answer_cites_source_session else 0.0 
        for q in reuse_questions
    ])
    
    return {
        "pattern_recall_rate": pattern_recalled,     # 历史模式被召回的比率
        "application_accuracy": application_correct, # 正确应用到新场景的比率
        "evidence_traceability": evidence_traceable, # 证据可追溯率
        "reuse_f1": 2 * pattern_recalled * application_correct / max(
            pattern_recalled + application_correct, 1e-9
        ),
    }
```

**显著性对比设计**：
- 自然样例 1：Day `2026-04-13` 建立观察清单（碳酸锂 / 4000 点 / AI 服务器），Day `2026-04-15` 问「你前天让我盯的三条线，现在怎么样了」
- 自然样例 2：Day `2026-04-12` 给出一句“明早开盘前提醒自己的话”，Day `2026-04-15` 问「再给我一句睡前提醒，跟前天那种风格的」
- 自然样例 3：多个 session 中反复要求“晨会一句话 / 盘中观察 / 复盘压缩”，检验系统能否复用既有输出 SOP
- 原生 OpenClaw 对这类“跨日观察清单更新”和“输出风格复用”容易遗漏来源
- EBM 可通过图谱中的 `FACT/EVENT` 与 evidence 回链提升 pattern recall

### 指标 4B：Skill 级 bugfix 复用成功率 — 对应痛点 4（当前 `news_report/dataset` 不覆盖）

```python
def skill_bugfix_reuse(questions: list[Question]) -> dict:
    """衡量系统能否把历史 bug 修复经验迁移到新任务"""

    bugfix_questions = [q for q in questions if q.category == "skill_bugfix_reuse"]

    bug_pattern_recall = mean(
        [q.historical_fix_pattern_in_context for q in bugfix_questions]
    ) if bugfix_questions else 0

    fix_transfer_success = mean(
        [q.judge_score for q in bugfix_questions]
    ) if bugfix_questions else 0

    tool_sop_faithfulness = mean([
        1.0 if q.answer_follows_known_tool_sop else 0.0
        for q in bugfix_questions
    ]) if bugfix_questions else 0

    return {
        "bug_pattern_recall": bug_pattern_recall,
        "fix_transfer_success": fix_transfer_success,
        "tool_sop_faithfulness": tool_sop_faithfulness,
    }
```

**说明**：
- 该子指标需要单独的 coding/bugfix 数据集支持
- 当前 `news_report/dataset` 中没有 `Traceback`、测试失败、安装失败、脚本报错、修复前后对照等显式样本
- 因此不能把当前新闻数据集直接拿来宣称覆盖 “Skill 解 bug 成功经验复用”

### Day2→Day8 可执行测试用例

脚本：`benchmark/news_recall/day2_framework_recall.py`

目标：验证当前 EBM 在 8 个新闻 session 后，Day8 查询「这家新上市的 AI 公司是否值得关注」时，是否能召回 Day2 用户给出的科技股分析框架。

说明：
- 这个用例是 **synthetic overlay benchmark**，不是 `news_report/dataset` 中天然存在的原始对话
- 它用于验证“显式分析框架回忆”能力，是对当前新闻数据集的补充增强，而不是原始数据本身的直接覆盖

数据构造：
- 新闻来源使用 `/Users/season/workspace/github/noteLM/news_report` skill。该 skill 当前只支持抓取当日新闻，不支持历史日期参数；脚本默认优先使用已缓存新闻，`--refresh-news` 可调用 `node scrape_news.js` 重新抓取。
- 脚本将新闻语料重排成 8 个本地 JSONL session，每天一个 session。
- Day2 注入用户框架：「我判断科技股主要看三点：研发投入占比、用户增长率、现金流」。
- Day8 注入新上市 AI 公司 SynapseNova AI：研发费用率 38%、月活用户同比增长 72%、自由现金流为负、12 个月烧钱 1.8 亿美元。
- Day8 查询：「你觉得这家新上市的 AI 公司值得关注吗？请按我之前判断科技股的分析框架来评估。」

运行：

```bash
python3 benchmark/news_recall/day2_framework_recall.py --output-dir benchmark/results/news_recall_native
```

可选对照：

```bash
python3 benchmark/news_recall/day2_framework_recall.py --seed-skill-node --output-dir benchmark/results/news_recall_seeded
```

输出：
- `day2_framework_recall_report.md`：打印 `Structured Memory Graph` / `Memory Facts` 命中、分数和 evidence 回链。
- `day2_framework_recall_result.json`：保留完整 trace、`graphHits`、`ledgerHits`、`systemPromptAddition`。
- `sessions/day02_2026-04-14.jsonl`：Day2 原始证据文件。
- `sessions/day08_2026-04-20.jsonl`：Day8 查询前上下文。

当前观察结论：
- 原生模式下，当前 EBM 能召回 Day2 框架，主要通过 `graphHits` 中的 Day2 `EVENT` 命中进入 `Structured Memory Graph`，并注入 `systemPromptAddition`。
- 当前 Python EBM 没有独立 `SKILL` 节点类型；可复用技能/经验在实现中主要表现为 `FACT`，例如 `distilled.salient_memory`，或由 `--seed-skill-node` 模拟的 `distilled.skill.tech_stock_analysis_framework`。
- 证据回链可定位到 Day2 JSONL session 和 `messageIndex=2`；但 `distillTurn` 生成的 `graphHits` evidence 目前丢失 `startLine/endLine`，脚本在报告中额外打印 `resolvedLine=messageIndex+1` 作为定位辅助。这说明 recall 能力已具备，但“精确行号回链”仍需在 EBM 写入证据时补齐。

---

## 四、集成方案

```python
# benchmark/eval_openclaw_ebm.py

class OpenClawEbmBenchmark:
    """OpenClaw vs OpenClaw+EBM 对照评测"""
    
    def __init__(self, data_dir: str, config_path: str):
        self.sessions = self._load_news_sessions(data_dir)  # 10 天新闻对话
        self.questions = self._load_eval_questions(data_dir)  # 50 道评测题
        self.judge_llm = OpenAICompatClient(...)  # 用于自动评分
        
    def run_group(self, group: str) -> dict:
        """运行一组实验：'openclaw' 或 'openclaw_ebm'"""
        if group == "openclaw":
            engine = OpenClawNativeMemory(self.sessions)
        else:
            engine = OpenClawWithEbm(self.sessions)
        
        # Ingest 阶段
        ingest_stats = engine.ingest_all(self.sessions)
        
        # Query 阶段
        results = []
        for q in self.questions:
            result = engine.query(q.question)
            judge_score = self._judge_answer(q, result)
            results.append(EvalResult(
                question=q,
                answer=result.answer,
                context=result.context,
                context_tokens=result.prompt_tokens,
                judge_score=judge_score,
                retrieved_evidence_ids=result.debug.get("context_hits", []),
                latency_ms=result.debug.get("timings_ms", {}).get("total", 0),
            ))
        
        return {
            "recall_precision": recall_precision(results),
            "token_efficiency": token_efficiency(results),
            "long_term_consistency": long_term_consistency(results),
            "experience_reuse": experience_reuse(results),
        }
    
    def run_comparison(self) -> str:
        """运行对照实验并生成报告"""
        baseline = self.run_group("openclaw")
        experiment = self.run_group("openclaw_ebm")
        return self._generate_report(baseline, experiment)
```

---

## 五、预期效果对比

| 指标 | OpenClaw (baseline) | OpenClaw + EBM | 预期提升 |
|------|-------------------|----------------|---------|
| 检索 F1（全部题） | ~0.45 | ~0.72 | +60% |
| 多跳题 F1 | ~0.25 | ~0.65 | +160% |
| Context tokens/正确答案 | ~8000 | ~2500 | -69% |
| 压缩率 | 0% (raw) | 70%+ | — |
| 跨 session 留存率 | ~0.40 | ~0.85 | +112% |
| 事实演化正确率 | ~0.30 | ~0.80 | +167% |
| 经验复用 F1 | ~0.15 | ~0.60 | +300% |

多跳推理和经验复用是差距最大的两个维度 — 这正是 EBM 的图结构 + 证据回链相对于扁平检索的核心优势所在。

---

## 六、数据集构造要点

### 新闻话题设计原则

1. **实体重叠**：多天讨论同一公司（如 Apple、Tesla、NVIDIA），制造跨 session 关联
2. **事实演化**：同一指标在不同天有不同值（股价、市值、用户数）
3. **因果链条**：Day 3 的政策导致 Day 5 的市场反应，Day 7 的公司应对
4. **用户偏好沉淀**：用户在讨论中流露投资偏好、关注领域、分析习惯
5. **分析框架复用**：早期建立的分析方法论，后期应被系统主动调用

### 当前数据集与设计文档的对齐结论

| 检查项 | 当前 `news_report/dataset` 是否满足 | 结论 |
|------|------------------------------|------|
| 多跳问题推演（10+） | **是** | 可稳定构造 10+ 题，尤其适合“趋势更新、跨主题归因、跨日观察清单更新” |
| 跨任务经验复用（新闻分析 / 观察清单 / 风格复用） | **部分满足** | 可做，但应把问题定义收敛到新闻分析框架、观察清单与输出 SOP，不宜泛化为任意任务经验 |
| 跨任务 Skill 解 bug 成功经验复用（10+） | **否** | 当前 0 个显式 coding/bugfix 样本，需单独补充 coding/bugfix 数据集 |
| 跨任务 Skill 操作 SOP 化经验复用（3+） | **是** | 当前可稳定构造 3+ 题，且实际可扩展到 10+ 题 |

### 基于当前 `news_report/dataset` 的最小可执行题库建议

1. **多跳题库**：建议至少构造 12 题
   当前天然样例包括：
   - `2026-04-13 -> 2026-04-15`：碳酸锂“日内反弹”被后续“年内新低”证伪
   - `2026-04-13 -> 2026-04-15`：前天观察清单在后续 session 中被显式更新
   - `2026-04-17 -> 2026-04-20`：DeepSeek 融资压力与光纤/算力基础设施需求的跨天关联
   - `2026-04-18 -> 2026-04-20`：利率/地缘压制风险偏好，与硬科技/机器人情绪的对照
   - `2026-04-19 -> 2026-04-20`：机器人半马热度与资金风格切换/业绩分化的组合判断

2. **跨任务经验复用题库**：建议至少构造 10 题
   适合围绕以下可复用模式出题：
   - 观察清单更新
   - 同一风格的“睡前一句提醒”
   - 晨会一句话压缩
   - 盘中观察压缩
   - “情绪层 / 基本面层 / 生活层”分层归纳
   - “更像情绪催化 / 更像业绩兑现”二分框架

3. **Skill 操作 SOP 复用题库**：建议至少构造 3~8 题
   当前天然存在的 SOP 型输出包括：
   - 开盘前三项先看什么
   - 中午十分钟先翻哪三条
   - 晨会如何一句话表达
   - 收盘后两句复盘怎么写
   - 如何把公司/机构按类别梳理

4. **Skill bugfix 复用题库**：当前不应从本数据集硬构造
   原因：
   - 没有显式 bug/报错/修复轨迹
   - 没有工具调用序列
   - 没有 “相同 bug 在新任务里被修复复用” 的成对样本
   正确做法是新增 coding 数据集，而不是在新闻问答里勉强映射

### 文档需要更新的点（基于当前数据集）

1. **数据规模要更新**：当前是 9 个 session、589 条消息，不是 10 个 session、约 200 轮。
2. **问题类型要更新**：当前更适合“时间演化 / 多跳归因 / 观察清单更新 / SOP 复用”，不适合用“用户对加密货币的态度”这类低覆盖问题做主样例。
3. **痛点 4 的指标要拆分**：新闻分析/SOP 复用可以继续用当前数据集；Skill 级 bugfix 复用必须单独建集。
4. **事实演化样例要更新**：当前更适合使用“碳酸锂反弹→年内新低”“黄金创新高→次日回调”“观察清单更新”这类跨日消息链，而不是虚构的股价账本型样例。

### 评测题设计原则

1. 每道题标注 `gold_evidence_ids`（正确答案所需的证据来源）
2. 每道题标注 `required_hops`（需要跨越几个信息源）
3. 每道题标注 `category`（单跳/时间/多跳/跨session/经验复用）
4. 事实演化题标注 `has_fact_evolution` 和正确的最新值
5. 使用 LLM-as-Judge 自动评分，辅以人工抽检校准
