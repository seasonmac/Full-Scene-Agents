---
name: genSkill
description: Use when the user invokes /genSkill or wants to turn a recurring task into a saved skill. Routes phases and runs the first (intent-clarifying) phase inline.
version: 2.1.0
---

# genSkill

Entry point for turning one repeatable task into a saved skill. This file also
runs the **first phase (intent clarification) inline** — no separate skill load
is needed before the first question, so the conversation starts immediately.

## Start

- Treat `/genSkill <goal>` as a request to create a reusable workflow for `<goal>`.
- If `<goal>` is present, skip the goal question and begin narrowing questions (below) with that goal.
- If no goal is present, ask exactly: `你想把哪件常做的事保存成以后能直接用的做法？`
- Keep the conversation in the user's language throughout.
- Never say "Agent", "Skill", "Planner", "GenSkill", "schema", "runtime", "tool registry", or "frontmatter" to the user.
- **Structured UI (optional):** if trusted inbound metadata says the client supports capability `a2ui:genskill.cards@1` or catalog `genskill.cards@1`, emit a structured A2UI card for each structured moment (clarify question, approach choice, plan confirm, skill-create progress, execute summary) instead of the prose form. If it does not, behave exactly as written below (plain text). The card carries the same content and sends the same answer back — never change the questioning logic, only its rendering.
  - **Only structured moments get a card.** A card is *only* for the five moments above. Greetings, acknowledgments, plain explanations, error/clarification asides, and any other non-structured reply MUST be plain text — never wrap them in an ```a2ui``` block. Emitting a card for a casual turn (e.g. replying to "hello") is a failure: the client validates any A2UI it detects and will reject a malformed or out-of-place card.

**Do not read any business capability reference file to ask the first question.**
Goal clarification
and the first narrowing questions are about the user's own behavior, not system
capabilities. Only load `references/capabilities-summary.md` when you are about
to offer **option-based** choices that could touch a disabled capability — see
"Capability Boundary (lazy)" below. This keeps the first reply fast.

A2UI rendering instructions are the only exception because they define output
shape, not business capability. When `a2ui:genskill.cards@1` / `genskill.cards@1`
support is present on the first question, use this minimal shape without loading
any reference file. The block must be strict JSON: every line must parse with
`JSON.parse`, and any embedded double quote inside a string value must be escaped
as `\"` or replaced with Chinese quotation marks:

```a2ui
{"createSurface":{"surfaceId":"clarify-1","catalogId":"genskill.cards@1"}}
{"updateComponents":{"surfaceId":"clarify-1","components":[{"id":"root","component":"ClarifyQuestion","phase":"clarify","question":"<one question in the user's language>","options":[{"id":"1","label":"<short option>"},{"id":"2","label":"<short option>"}],"action":{"name":"clarifyAnswer"}}]}}
```

For later structured moments, read `references/a2ui-emit-guide.md` before
emitting the next A2UI card.

## Flow

```
/genSkill <goal>
  → [inline] clarify intent          (this file — Phase 1)
    → genSkill:writing-plans         (draft + confirm plan)
      → genSkill:writing-skills      (save skill file)
        → genSkill:execute           (optional first-run)
```

Each phase has a hard gate: do not proceed to the next phase until the current
phase's exit condition is met.

## Phase Routing

1. **Always run Phase 1 (intent clarification) inline first.** Even if the goal seems simple. Do NOT load a separate brainstorming skill — the rules are in this file.
2. After Phase 1 completes (exit condition below), invoke `genSkill:writing-plans`.
3. After the plan is confirmed, invoke `genSkill:writing-skills`.
4. After generation succeeds, invoke `genSkill:execute`.

> The standalone `genSkill:brainstorming` skill remains on disk as the canonical
> reference for these rules, but the normal flow does NOT load it — Phase 1 runs
> from this file to avoid an extra skill load before the first question.

---

# Phase 1 — Clarify Intent (inline)

Help the user clarify what they actually want to save as a reusable workflow. Do
not design the solution — just understand the intent.

## Hard Rules

1. One question per turn. No exceptions.
2. Each question has at most 3 choices, numbered. Choices must be mutually exclusive — if a user could reasonably want two options at once, reframe the question.
3. **Ask about exactly one dimension per question.** Never combine trigger + input, or input + output, into a single question. (See the worked counter-example below.)
4. Questions must address concrete user behavior ("你会怎么告诉我信息？" / "你想先看到什么结果？"), not system internals ("能力边界"/"权限"/"schema").
5. Do not show capabilities the system cannot do. Only when offering option-based choices, check `references/capabilities-summary.md` (lazy — see below). If the user asks for a disabled one, explain which part is not possible and offer a workable alternative.
6. If the user gives a free-text answer, the next question must build on that answer. Never fall back to a generic template questionnaire.
7. Do not give a premature summary. Only present a "用户交互流程描述" after all necessary questions are answered.
8. Follow the rhythm: **understand context → ask one narrowing question → (repeat until converged) → propose 2-3 approaches → user picks one → exit**.
9. **Never end a turn with an acknowledgment alone.** Every Phase 1 turn must close with a concrete forward action: either the next gap question, or — once all five gaps are filled — the 2-3 approaches. "好的，明白了。" / "收到。" / "这个需求我清楚了。" as the *last thing you say* is a failure: it ends your turn without driving the flow, and in runtimes that only inject this skill on the triggering turn, the structured flow dies there. After every user answer, before replying, silently tally which of the five gaps (goal, trigger, inputs, output, failure) are still unanswered; if any remain, your reply MUST contain the next one. You are responsible for driving the questioning to completion across turns — the user will not prompt you to continue.

## The Five Gaps (one dimension each — never merge two into one question)

Ask about these separately. Each is its own question:

- **Goal**: What does the user want to accomplish?
- **Trigger** (怎么开始): "你希望什么时候用这个？" — 定时 / 手动 / 以后再说. Manual start means the user **says it in plain language** ("出一份周报") — the system does **not** register a dedicated slash command (`/healthreport` etc.) for a generated skill, so never promise one. Scheduled start is a real openclaw cron job (see `trigger.scheduled_reminder`). If the user asks for a custom command, explain it works by what they say or on a schedule, not a special command.
- **Inputs** (怎么给信息): "你会怎么把信息给我？" — 截图 / 文件 / 文字
- **Output** (怎么拿结果): "做好后你想在哪看、什么时候看？" — the delivery options are **whatever the system actually supports**, not a fixed list, but they are still a **finite, enumerable set**, so present them as **numbered choices** like every other question (Hard Rule 2), never as open free text. When you reach this question, read `references/capabilities-summary.md` and offer only the `interaction.*` channels marked Supported (e.g. `1. 应用内聊天  2. 微信`). Never invent a channel that isn't listed — if the user names one that isn't supported (飞书 / QQ / 邮件 / 短信…), say it's not available and offer the supported ones. The only delivery dimension that may be free text is the *timing* ("什么时候看"), not the *channel*.
- **Failure**: What should happen if something is missing?

Trigger, Inputs, and Output are three different dimensions. A question like
"周报怎么开始？1.定时提醒我发截图 2.我想看时自己发截图" is **wrong** — it fuses
trigger (定时 vs 手动) with input (发截图), so options 1 and 2 look identical on
the input axis and the user can't tell them apart. Split it: ask trigger alone,
then input alone, then output alone.

## Question Design Principles

- Ask about the user's concrete actions first: "你会怎么把信息给我？" not "输入模态是什么？"
- Prefer "你想先得到什么" over "能力边界是什么"
- Choices must be things a normal person can immediately understand and distinguish
- For timing, ask "你希望什么时候用这个？" not "触发条件是什么？"
- If the user answers vaguely, ask for one concrete example instead of repeating the question

## Process

```
User states goal
  → Is the goal clear enough to start asking details?
    → No: ask one question to sharpen the goal
    → Yes: run Step 0 (reuse check) below, then begin narrowing questions

For each turn:
```

### Step 0 — Check for an Existing 做法 (reuse before creating)

Run this **once**, after the goal is clear enough to describe and **before** the
first narrowing question. It stops the flow from rebuilding a workflow the user
already saved.

1. List what's already saved:
   `node scripts/generate-skill.cjs --list --target <运行时平台>`
   (platform = the runtime this is running in; default `codex`). This is internal
   plumbing — never mention the command or platform to the user.
2. Judge whether any returned skill is **semantically close** to the user's goal.
   Prefer each entry's `flow` (the plain-language 做法流程); if `flow` is null
   (older skills), fall back to its `description`. Matching is your judgment — the
   script does not decide it.
3. **No close match → say nothing about this.** Go straight to the narrowing
   questions as if Step 0 never ran. Do not announce "没找到类似的做法".
4. **Close match → show it and ask one question** (obeys Hard Rule 2: one question,
   ≤3 mutually-exclusive numbered choices). Present the matched skill's `flow`
   verbatim (or a one-line summary built from `description` if `flow` is null),
   introduced only as "我之前好像存过一个类似的做法":

   ```
   我之前好像存过一个类似的做法：

   <flow.md 内容，或由 description 概括的一句话>

   1. 就用这个
   2. 不太一样，重新做一个
   ```
5. User picks **1 (就用这个)** → do **not** continue to writing-plans. Phase 1 ends
   here: route directly to `genSkill:execute` for an optional first-run of that
   existing workflow (its plan-slug directory + `orchestration.json` are already on
   disk).
6. User picks **2 (重新做一个)** → drop the reuse path and begin the normal
   narrowing questions to create a new workflow.

```
For each turn:
  1. Identify the single biggest gap in your understanding
  2. Formulate one question closing that gap — ONE dimension only
  3. Provide 2-3 mutually exclusive **numbered** choices. Default to numbered choices — only fall back to free text when the answer space is genuinely open (e.g. the goal itself, or a concrete example). If the dimension has a finite set of answers (trigger: 定时/手动; channel: the Supported `interaction.*` list), it MUST be a numbered choice, never free text.
  4. Wait for the user's answer
  5. If the answer opens a new direction, follow it — do NOT return to a pre-planned sequence

When all gaps are filled:
  → Propose 2-3 approaches with trade-offs and your recommendation
  → User picks one
  → Exit to writing-plans phase
```

## Capability Boundary (lazy)

You do **not** need any business capability reference file to clarify the goal or
to ask trigger / input / output / failure questions in the user's own words.
Read `references/capabilities-summary.md` **only at the moment** you are about
to present option choices that might include a disabled capability (e.g.
anything touching payment, sending to other people, auto-reading health/calendar
data).

- **Supported** capabilities may appear as options.
- **Disabled** capabilities must never appear as an option. If the user asks for one, explain which part is impossible and offer the alternative the summary gives.

`writing-plans` re-reads the summary for feasibility; `writing-skills` reads the
per-capability `references/capabilities/*.md` files for exact boundary wording.
Keep the summary as the single source of truth — do not copy it inline anywhere.

## Anti-Patterns

| If you catch yourself doing this... | Stop and do this instead |
|------|------|
| Combining trigger + input (or any two dimensions) in one question | Split into one question per dimension |
| Asking a question with non-mutually-exclusive options | Reframe so options don't overlap |
| Showing more than 3 options | Pick the 3 most likely, or ask a narrower question |
| Reading a business capability reference file before the first question | Ask the goal/trigger/input question directly; load summary only before option choices |
| Asking about "schema" / "能力" / "触发条件" | Rephrase in the user's language |
| Presenting a summary before all gaps are filled | Ask the next narrowing question first |
| Offering a template question after user free-text input | Build directly on what they said |
| Following a pre-planned question sequence | Each question must respond to the previous answer |
| Ending a turn with "好的/收到/我明白了" and nothing else | Close every turn with the next gap question (or the approaches once all gaps are filled) |
| Asking "还有别的需求吗？" as a catch-all | Ask about the specific gap you still see |

## Phase 1 Exit Condition

Phase 1 is complete via **one** of two paths:

**Reuse path** (Step 0 found a close match and the user chose 就用这个):
1. The matched existing workflow's 做法流程 was shown and the user confirmed reuse

**Next step**: Invoke `genSkill:execute` directly against that existing workflow — skip writing-plans and writing-skills.

**Create path** (no match, or the user chose 重新做一个):
1. All five gaps (goal, trigger, inputs, output, failure) are answered — each asked as its own question
2. You have proposed 2-3 approaches
3. The user has picked one approach

**Next step**: Invoke `genSkill:writing-plans` with the Phase 1 outcome.

---

## Negative Examples (capability boundary)

- `/genSkill 每月自动转账` → "自动转账是我做不到的，因为我不能替你操作银行或支付。可以改成：每月提醒你该转账了，并帮你整理好转账信息。"
- `/genSkill 自动读取健康数据做报告` → "我不能自动获取你的健康数据，但你可以把截图或数据发给我，我来帮你整理成报告。"
- `/genSkill 帮我发消息给同事` → "我不能替你发消息给别人。可以改成：帮你拟好消息内容，你确认后自己发送。"
