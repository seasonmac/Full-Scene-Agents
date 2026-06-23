---
name: genSkill:writing-plans
description: Use after brainstorming is complete. Produces a structured plan and translates it to plain language for the user to confirm before generation.
---

# Writing Plans

Turn the brainstorming outcome into a confirmed plan. The plan breaks one user intent into one or more concrete, completable sub-tasks (which the next phase renders as callable workflow skills), and translates the whole thing to plain language for the user to approve.

Two representations:
- A **structured plan** (for the generator script)
- A **plain-language description** (for the user to read and approve)

## Step 1: Draft the Structured Plan

Top-level fields (the overview, for humans and review):

| Field | Description |
|-------|-------------|
| `user_goal` | One sentence describing what this does |
| `inputs_needed` | List of what the user provides each time |
| `schedule_or_trigger` | When/how the task starts |
| `memory_scope` | What is remembered between runs vs. what is not |
| `output_or_delivery` | What the user sees at the end |
| `confirmation_boundary` | What requires user approval before acting |
| `failure_handling` | What happens when inputs are missing or incomplete |
| `acceptance_criteria` | How the user knows it worked |
| `capabilities_used` | Union of all sub-skill capability IDs, for fast validation |

### Decompose into Sub-Skills

When the intent involves more than one kind of action (import, process, generate, render…), split it into a `skills[]` array. Each entry is one completable sub-task:

| Sub-skill field | Description |
|-----------------|-------------|
| `id` | Verb-first kebab id, unique within the plan (`import-data`) |
| `role` | Short human label (`导入数据`) |
| `user_goal` | What this one sub-skill accomplishes |
| `inputs_needed` | What it takes in (often an upstream product) |
| `capabilities_used` | Capability IDs this sub-skill uses |
| `produces` | Product id this sub-skill outputs |
| `produces_shape` | *(optional)* The field shape of `produces`, as a compact example record. See "Data Contracts" below |
| `consumes` | Product ids it depends on (upstream `produces`) |

### Data Contracts (produces_shape)

`produces`/`consumes` alone are opaque ids — they say two skills hand off a
product, but not what that product *looks like*. When a downstream skill must
read specific fields from an upstream product (the common case: a processing
skill feeds a report skill), add an optional `produces_shape` to the producer so
the contract is explicit:

```json
{ "id": "process-data", "produces": "structured-records",
  "produces_shape": "{ date, resting_hr, sleep_hours, steps }  // one record per day" }
```

- `produces_shape` is a **hint string**, not enforced by the script — it carries the
  field names/shape so `writing-skills` writes the producer's output and the
  consumer's `## Inputs` against the *same* fields, and so the plan reviewer can
  check that every field a consumer reads is actually present in what its upstream
  produces.
- Omit it for products with no structured payload (e.g. a rendered widget). Include
  it whenever a consumer reads named fields off the product.
- The consumer does not repeat the shape; it references the upstream product by id
  and relies on the producer's `produces_shape` as the source of truth.

**Do not write detailed `steps` here.** This phase decomposes intent and decides feasibility — *what* each sub-skill is for, whether it *can* be done, and how the sub-skills schedule together. The detailed workflow ("how to actually do it") is authored in `writing-skills`. Keep each sub-skill at the level of goal + inputs + capabilities + products. Your job is to be sure each sub-task is doable with supported capabilities; not to spell out its steps.

`produces`/`consumes` must form a DAG — no cycles. The generator orders sub-skills by this graph and skips any whose capability is unsupported (cascading the skip downstream).

Split guidance: one responsibility per sub-skill; split at product hand-offs; separate scheduled work from immediate work; keep same-domain capabilities together. If the intent is genuinely a single action, omit `skills[]` and use a flat `steps[]` instead (legacy single-skill shape — the only place `steps` belongs in a plan).

### Add Orchestration

If the sub-skills need wiring (timing, bindings), add an `orchestration` block:

| Key | Meaning |
|-----|---------|
| `scheduled[]` | `{ skill, when, after[] }` — run on a schedule after its deps |
| `immediate[]` | `{ skill, action, bindings[]? }` — run at once; bind triggers |

`bindings` entries map a trigger source to an Ability (`{ trigger, ability }`) or to a target skill (`{ trigger, target_skill }`).

Every capability in any `capabilities_used` must be listed under **Supported** in `../genSkill/references/capabilities-summary.md`. Do not include disabled capabilities.

## Step 2: Translate to Plain Language

Present the plan to the user as a "用户交互流程描述" using everyday words:

```
我来确认一下这个做法的流程：

任务：[goal in user's words]
1. 你给我：[inputs in plain terms]
2. 我会：[processing steps in plain terms]
3. 开始方式：[trigger in plain terms]
4. 完成后：[output in plain terms]
5. 如果缺东西：[failure handling in plain terms]
6. 我不会做的事：[boundaries in plain terms]

这样创建可以吗？
```

> **Structured UI (optional).** If trusted inbound metadata says the client
> supports capability `a2ui:genskill.cards@1` or catalog `genskill.cards@1`,
> render this "用户交互流程描述 + 这样创建可以吗？" as a `PlanConfirm` card
> **instead of** the plain-text block above — read
> `../genSkill/references/a2ui-emit-guide.md` for the exact `PlanConfirm` fields
> (`taskTitle` = the goal; `items[]` = the 6 plain-language lines 你给我 / 我会 /
> 开始方式 / 完成后 / 如果缺东西 / 我不会做的事; `question` = "这样创建可以吗？";
> `confirmAction`). Emit the card only; do not also output the same numbered prose
> outside the block. If the client does not advertise that capability, keep the
> plain-text form exactly as above. Same content, same approval answer — only the
> rendering changes.

### Plain-Language Rules

- No system terms. "记忆边界" → "我会记住做法本身，但不会保存你每次给我的具体内容"
- No capability IDs. "health.weekly_report" → never shown to the user
- No English technical terms unless the user is writing in English
- Each line must be understandable by someone who has never used a terminal
- "我不会做的事" must list real boundaries, not generic disclaimers

## Step 3: Review Before Showing the User

Before presenting the plan for approval, review it against the checklist in
`plan-reviewer-prompt.md`. The structured plan is the only input the `generate`
script gets, so a missing field, an unsupported capability, or a leaked system
term in the plain-language text will produce a wrong Skill or mislead the user.

**Do this review inline yourself** — read `plan-reviewer-prompt.md`, walk its
"What to Check" table against your plan JSON and plain-language text, and fix
anything it flags. This is the default and works in every runtime.

> **Do not block on dispatching a subagent.** Some runtimes (e.g. Claude Code)
> have a Task tool you *may* use to get an independent reviewer; many runtimes
> (e.g. openclaw) do not. Never wait for a subagent result that may never come —
> if you cannot dispatch one, or are unsure, just do the review inline and move
> on. The review is a self-check, not a hand-off.

Check, at minimum: every top-level field filled (no empty/"TBD"); every
capability ID is Supported in `../genSkill/references/capabilities-summary.md`; each
`consumes` edge's upstream `produces_shape` carries the fields the consumer
needs; the plain-language text leaks no system terms or capability IDs; the two
representations match.

If the review surfaces issues, fix them in the structured plan and/or the
plain-language text, then re-check. Only proceed to Step 4 once it is clean.

## Step 4: Wait for Explicit Approval

Approval must be one of: `可以保存`, `确认`, `保存`, `好的`, `yes`, `ok`.

If the user wants changes:
1. Ask one question about which part to change
2. Revise that part in the structured plan
3. Re-present the full plain-language version
4. Wait for approval again

Do NOT proceed without explicit approval. "嗯" or "看看" is not approval — ask: "这样可以保存了吗？"

## Anti-Patterns

| Wrong | Right |
|-------|-------|
| Showing capability IDs to the user | Only use them internally in the plan JSON |
| Saying "记忆边界：只保存流程" | Say "我会记住怎么做，但不会保存你每次给我的内容" |
| Saying "能力边界：不操作外部账号" | Say "我不会替你操作任何外部账号" |
| Skipping re-confirmation after revision | Always re-present the full plan after changes |
| Treating "嗯" as approval | Ask explicitly "这样可以保存了吗？" |
| Showing the plan to the user before review | Run the review checklist (inline) first, then present |
| Waiting on a reviewer subagent that never returns | Do the review inline; a subagent is optional, not a gate |

## Exit Condition

The user has explicitly approved the plain-language plan.

**Next step**: Invoke `genSkill:writing-skills` with the structured plan JSON.
