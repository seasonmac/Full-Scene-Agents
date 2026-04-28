# Coding / Skill Bugfix 经验复用数据集规范

目标：
- 补齐 `news_report/dataset` 无法覆盖的痛点 4 子项
- 专门评测 “跨任务 Skill 解 bug 成功经验复用”
- 为 `skill_bugfix_reuse` 指标提供可执行数据源

---

## 一、为什么需要单独数据集

当前 `news_report/dataset` 适合：
- 多跳推理
- 跨 session 事实演化
- 新闻分析框架复用
- 输出 SOP 复用

但它不适合：
- bug 修复经验复用
- 工具调用顺序复用
- 代码 / 环境 / 安装 / 测试失败的迁移修复

根本原因是新闻问答里没有显式的：
- `Traceback`
- 测试失败日志
- 安装失败日志
- 工具执行顺序
- 修复前后状态

---

## 二、数据集目标覆盖

新数据集应至少支持以下 3 类问题：

1. **同类报错复用**
   例：Day 2 修过 `ModuleNotFoundError: xxx`，Day 8 遇到类似依赖问题，系统能否复用修复路径

2. **同类工具 SOP 复用**
   例：此前已形成 “先看日志 -> 复现 -> 定位失败步骤 -> 最小修复 -> 跑测试” 的 SOP，后续是否能复用

3. **跨项目经验迁移**
   例：A 项目的前端构建问题修过一次，B 项目出现相同 bundler / tsconfig / env 问题时能否迁移

---

## 三、最小数据结构建议

每个 session 推荐使用 JSONL，沿用当前对话型格式：

```json
{"type":"message","timestamp":1777000000000,"message":{"role":"assistant","content":"任务背景：某 Python 项目执行测试时报错 ...","timestamp":1777000000000}}
{"type":"message","timestamp":1777000060000,"message":{"role":"user","content":"我跑 pytest 时报错了，帮我看看。","timestamp":1777000060000}}
{"type":"message","timestamp":1777000120000,"message":{"role":"assistant","content":"先贴出 traceback。","timestamp":1777000120000}}
```

每个 session 至少包含：
- `task_summary`
- `error_trace`
- `environment_info`
- `attempted_fix`
- `final_fix`
- `verification_result`

建议在消息外再配一个 metadata 文件，标注：
- `task_type`: `python_import` / `frontend_build` / `dependency_conflict` / `tool_config` / `runtime_bug`
- `root_cause`
- `successful_fix_pattern`
- `tool_sop`
- `reusable_tags`

---

## 四、建议的 10+ bugfix 题型

至少构造以下 12 类：

1. Python 依赖缺失
2. Node/npm 安装冲突
3. `pytest` 测试收集失败
4. TypeScript 类型报错
5. 前端构建失败
6. 环境变量缺失
7. 数据库连接失败
8. API 认证失败
9. 脚本路径错误
10. 配置文件版本不兼容
11. Docker / 容器运行失败
12. CI only 问题复现与修复

每类至少 2 个 session：
- 一个作为“历史成功经验”
- 一个作为“新任务迁移测试”

这样总量建议至少：
- **24 个 session**
- **12 对可迁移样本**

---

## 五、建议的 3+ SOP 经验题型

至少构造这 3 组：

1. **报错排查 SOP**
   - 先看错误日志
   - 复现
   - 缩小范围
   - 最小修复
   - 回归验证

2. **依赖安装 SOP**
   - 确认语言版本
   - 确认 package manager
   - 清理缓存/锁文件
   - 重新安装
   - 验证版本树

3. **构建失败 SOP**
   - 定位失败阶段
   - 区分代码问题/环境问题/配置问题
   - 最小修改
   - 重跑构建
   - 记录复用结论

扩展可加：
- 测试失败 SOP
- 发布回滚 SOP
- 数据迁移 SOP

---

## 六、建议的 benchmark 标注字段

每道题建议标：

```python
{
  "question_id": "...",
  "category": "skill_bugfix_reuse",
  "gold_evidence_ids": ["..."],
  "required_hops": 2,
  "historical_fix_pattern_in_context": 0/1,
  "answer_follows_known_tool_sop": 0/1,
  "judge_score": 0.0-1.0,
  "root_cause_match": 0/1,
  "final_fix_match": 0/1,
  "verification_present": 0/1
}
```

---

## 七、建议的验收标准

### 对“Skill 解 bug 成功经验复用”

- 至少 **10 个问题**
- 每题都必须有：
  - 历史成功修复 session
  - 新任务迁移 session
  - 明确可追溯的 `gold_evidence_ids`

### 对“Skill 操作 SOP 化经验复用”

- 至少 **3 组 SOP**
- 每组 SOP 至少 **2 次复用测试**

---

## 八、推荐目录结构

```text
benchmark/
  coding_bugfix_reuse/
    dataset/
      session01_python_import_fix.jsonl
      session02_python_import_reuse.jsonl
      session03_frontend_build_fix.jsonl
      session04_frontend_build_reuse.jsonl
    questions/
      bugfix_reuse_questions.json
    README.md
```

---

## 九、与现有新闻 benchmark 的关系

- `news_report/dataset`：负责痛点 1/2/3，以及痛点 4 中的新闻分析和 SOP 复用
- `coding_bugfix_reuse/dataset`：负责痛点 4 中的 Skill 级 bugfix 复用

两者不应混为一个数据集，否则会导致：
- 问题定义不清
- 指标含义失真
- EBM 的优势点被错误评估

---

## 十、下一步建议

如果要真正落地：

1. 先建 6 对最小样本
   - Python import
   - npm install
   - pytest
   - frontend build
   - env var
   - API auth

2. 再扩成 12 对正式样本

3. 最后补：
   - `benchmark/coding_bugfix_reuse/README.md`
   - `questions.json`
   - 自动 judge 规则
