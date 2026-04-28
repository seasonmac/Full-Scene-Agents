# 多跳 Benchmark

基于数据集：`news_report/dataset`

审计时间：`2026-04-21`

适用目标：
- 痛点 1：线索寻找与多跳推理
- 痛点 3：跨 session 长期记忆一致性
- 痛点 4A：新闻分析框架 / 观察清单 / 输出 SOP 复用

说明：
- 本题库优先使用当前数据集中**天然存在**的跨日事实演化、跨主题关联和跨任务输出复用样本。
- 每题都可以扩展为 `gold_evidence_ids` + `required_hops` + `category` 标注。
- 这里先给出题目草案与证据来源建议，便于后续转成结构化 benchmark。

---

## 一、多跳推理题（建议首批 12 题）

### 1. 碳酸锂的“反弹”后来为什么没有走成趋势反转？
- 类型：多跳 / 事实演化
- 需要证据：
  - [daily_report_0413.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0413.jsonl) 中“碳酸锂日内反弹超 3%”与后续解释
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl) 中“碳酸锂跌至年内新低”和对 4 月 13 日判断的修正
- 目标答案：说明 4 月 13 日只是技术性修复，4 月 15 日事实演化证伪了趋势反转判断

### 2. 为什么 4 月 13 日建议盯的三条线里，最后只剩 AI 服务器 / 半导体设备值得继续重点跟踪？
- 类型：多跳 / 观察清单更新
- 需要证据：
  - [daily_report_0413.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0413.jsonl) 中“下周观察清单”
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl) 中“你前天让我盯的三条线”
- 目标答案：对比碳酸锂、4000 点、AI 服务器三条线的后续验证结果

### 3. 为什么 4 月 19 日机器人半马很热，但 4 月 20 日更值得跟的主线反而是 AI 硬件基础设施？
- 类型：多跳 / 跨主题归因
- 需要证据：
  - [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl) 中机器人半马、北向减持高位 AI/机器人、硅光子产线
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl) 中光纤量价齐升、DeepSeek 融资、AI 计费复杂化
- 目标答案：说明机器人更偏情绪，光纤/算力链更偏业绩与长期资本开支逻辑

### 4. 为什么 4 月 18 日市场不爱做梦，只认硬货和避风港，而到 4 月 20 日又变成“热闹很多，真金白银只会流向最硬的逻辑”？
- 类型：多跳 / 风格连续性
- 需要证据：
  - [daily_report_0418.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0418.jsonl)
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)
- 目标答案：解释两天虽然触发因素不同，但底层都是资金在压缩风险偏好、只保留高确定性线索

### 5. 为什么 4 月 17 日 DeepSeek 融资压力是模型行业的“成本决战期”，到 4 月 20 日又能进一步映射到光纤和算力基础设施？
- 类型：多跳 / 跨日因果链
- 需要证据：
  - [daily_report_0417.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0417.jsonl)
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)
- 目标答案：说明模型公司融资 -> 算力支出约束显性化 -> AI 基础设施仍要持续扩容

### 6. 4 月 15 日为什么说“AI 主线从算力切换到应用”，但 4 月 19-20 日又重新强调硅光子、光纤和 H200 这类硬件线？
- 类型：多跳 / 主线切换
- 需要证据：
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl)
  - [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl)
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)
- 目标答案：说明“应用”和“基础设施”不是互斥，而是市场在不同风格环境下偏好不同兑现环节

### 7. 为什么 4 月 16 日更适合核心仓高股息、观察仓科技硬件，而 4 月 19 日又可以开始重新关注港股科技和面板？
- 类型：多跳 / 风格迁移
- 需要证据：
  - [daily_report_0416.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0416.jsonl)
  - [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl)
- 目标答案：说明 4 月 16 日是缩量避险，4 月 19 日出现了港股科技估值修复和面板涨价的经营改善线索

### 8. 为什么 4 月 13 日的 4000 点更像“蓄力”，但 4 月 15 日却说今天没单独提大盘大概率是因为指数波动不大、需要新催化？
- 类型：多跳 / 时间演化
- 需要证据：
  - [daily_report_0413.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0413.jsonl)
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl)
- 目标答案：说明市场对 4000 点的突破预期没有被连续成交量和新催化确认

### 9. 从 4 月 17 日到 4 月 19 日，为什么“监管打压题材炒作”与“北向减持高位 AI / 机器人股”可以被看成同一条风格切换链？
- 类型：多跳 / 因果链
- 需要证据：
  - [daily_report_0417.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0417.jsonl)
  - [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl)
- 目标答案：说明监管信号与资金行为在时间上形成一致指向，均推动“伪成长”退潮

### 10. 为什么 4 月 18 日最像“板块温度计”的是 3200 点失守和题材退潮，而 4 月 20 日最该先看的却是原油、美股期货、机器人板块分化和光纤链强度？
- 类型：多跳 / 市场观察 SOP
- 需要证据：
  - [daily_report_0418.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0418.jsonl)
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)
- 目标答案：说明不同市场状态下，“先看什么”的 SOP 会随着主导风险因子变化而切换

### 11. 为什么 4 月 15 日说“高股息和半导体设备可以重点跟踪”，到 4 月 19 日面板和硅光子又成了更偏业绩线的代表？
- 类型：多跳 / 跨行业比较
- 需要证据：
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl)
  - [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl)
- 目标答案：说明“业绩线”不是固定某一个行业，而是顺着经营改善与资金验证不断迁移

### 12. 为什么 4 月 20 日要把油价和机器人拆开看，而不能混成一条主线？
- 类型：多跳 / 归因拆分
- 需要证据：
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)
- 目标答案：说明油价来自风险事件驱动，机器人来自成长想象力驱动，两者风险属性完全不同

---

## 二、跨 Session 经验复用题（建议首批 10 题）

### 1. 你前天让我盯的三条线，现在应该怎么更新观察清单？
- 证据：
  - [daily_report_0413.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0413.jsonl)
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl)

### 2. 还按前天那种风格，给我一句今晚睡前提醒。
- 证据：
  - [daily_report_0413.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0413.jsonl)
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl)

### 3. 用之前那套“情绪催化 / 业绩兑现”二分法，把今天这些新闻再分一次。
- 证据：
  - [daily_report_0412.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0412.jsonl)
  - [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl)

### 4. 继续按之前晨会的一句话风格，把今天最重要的市场背景压成一句。
- 证据：
  - [daily_report_0414.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0414.jsonl)
  - [daily_report_0418.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0418.jsonl)

### 5. 还按你之前给我的盘中观察写法，帮我把今天这堆消息压成一句。
- 证据：
  - [daily_report_0414.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0414.jsonl)
  - [daily_report_0416.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0416.jsonl)

### 6. 继续按“情绪层 / 基本面层 / 生活层”的分法，把这天新闻拆一下。
- 证据：
  - [daily_report_0418.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0418.jsonl)
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)

### 7. 结合前几天你的建议，现在如果只能看一个偏业绩方向，你会选谁？
- 证据：
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl)
  - [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl)

### 8. 按之前那种“先看三样”的 SOP，明早我应该先看什么？
- 证据：
  - [daily_report_0412.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0412.jsonl)
  - [daily_report_0418.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0418.jsonl)
  - [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl)
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)

### 9. 还是按你之前的写法，帮我做一个三句话版本的结论。
- 证据：
  - [daily_report_0412.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0412.jsonl)
  - [daily_report_0415.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0415.jsonl)

### 10. 继续按“公司 / 机构分类梳理”的方式，把今天提到的主体归一下类。
- 证据：
  - [daily_report_0412.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0412.jsonl)
  - [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)

---

## 三、输出 SOP 复用题（建议首批 6 题）

### 1. 开盘前三项先看什么？
### 2. 中午十分钟优先翻哪三条？
### 3. 晨会一句话怎么说？
### 4. 收盘后两句复盘怎么写？
### 5. 睡前提醒一句怎么写？
### 6. 盘中观察怎么压成一句？

这些题在以下文件里都有天然样本：
- [daily_report_0412.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0412.jsonl)
- [daily_report_0414.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0414.jsonl)
- [daily_report_0416.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0416.jsonl)
- [daily_report_0418.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0418.jsonl)
- [daily_report_0419.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0419.jsonl)
- [daily_report_0420.jsonl](/Users/season/workspace/github/noteLM/news_report/dataset/daily_report_0420.jsonl)

---
