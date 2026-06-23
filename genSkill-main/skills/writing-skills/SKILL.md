---
name: genSkill:writing-skills
description: Use after the user approves the plan. Expands each approved sub-task into a complete, callable workflow skill — you author the SKILL.md bodies; a helper script handles capability gating, dependency ordering, and the orchestration manifest.
---

# Writing Skills

Take the approved plan and write the actual workflow skills an Agent will call. The plan (from `writing-plans`) has already decomposed the user's intent into sub-tasks and decided how they schedule together. Your job here is the implementation layer: turn each sub-task into a complete SKILL.md that an Agent can follow without you present.

**You write the skill bodies. You do not template them.** A helper script validates the plan, gates capabilities, orders dependencies, and emits the orchestration manifest — but the workflow steps, the `description`, the naming, and the boundaries inside each SKILL.md are judgment calls only you can make well.

## Precondition

Only run this after:
- The user explicitly approved the plan in `writing-plans`
- The structured plan JSON is complete

If not, go back to `genSkill:writing-plans`.

## Division of Labor

| Concern | Owner |
|---------|-------|
| Decomposing intent into sub-tasks | `writing-plans` (already done) |
| Scheduling/binding between sub-tasks | `writing-plans` → `orchestration` |
| Capability gating, dependency order, skip cascade | the script |
| **Detailed workflow, description, naming, body** | **you, here** |

The plan tells you *what* each sub-skill is for and *whether it can be done*. You decide *how* it reads as a runnable skill.

## Execute Without Pausing (read first)

This phase is **all action, no narration**. Steps 1-5 below — save the plan, run
the script, read the result, write every SKILL.md, save the flow.md — are a single
continuous unit of work. Do them back-to-back in one go, each as an actual tool call.

**Never end a turn with an announcement of intent.** A reply like "先把方案落成可用做法，我这就开始写" / "接下来我来生成技能" / "现在写每个 skill" — when it is the *last thing in your turn and no tool call follows it* — is a hard failure. In a turn-based runtime, emitting text ends your turn and hands control back to the user; but the user has nothing to say, they are waiting for the files. The flow dies right there, exactly where it has died before.

The rule: **if your next step is to run the script or write a file, do that — do not say you are about to.** Reading a reference file (`authoring-good-skills.md` etc.) is part of the work, not a stopping point: read it, then immediately continue to the next tool call in the same turn. The only legitimate reasons to stop and address the user mid-phase are (a) the script returns `status: blocked`, (b) `skipped[]` is non-empty and you must explain a dropped sub-skill, or (c) you need a genuine decision from the user (replace-or-rename on `skill_already_exists`). Absent one of those, keep executing until every surviving skill's SKILL.md is written.

### 1. Save the Approved Plan

Write the approved plan JSON to a temp file. Shape (sub-skills are lightweight — no detailed `steps`; that's what you're about to author):

```json
{
  "approved": true,
  "user_goal": "...",
  "inputs_needed": ["..."],
  "schedule_or_trigger": "...",
  "memory_scope": "...",
  "output_or_delivery": "...",
  "confirmation_boundary": "...",
  "failure_handling": "...",
  "acceptance_criteria": ["..."],
  "skills": [
    { "id": "import-data", "role": "导入数据", "user_goal": "...",
      "inputs_needed": ["..."], "capabilities_used": ["..."],
      "produces": "raw-input", "consumes": [] }
  ],
  "orchestration": {
    "scheduled": [{ "skill": "generate-report", "when": "weekly:SAT 08:00", "after": ["process-data"] }],
    "immediate": [{ "skill": "render-ui", "action": "render-and-update-widget" }]
  },
  "capabilities_used": ["..."]
}
```

### 2. Run the Planning Script

```bash
node ../genSkill/scripts/generate-skill.cjs --plan <path-to-plan.json> --target <platform> --write
```

Targets: `codex` (`~/.codex/skills/`), `openclaw` (`~/.openclaw/skills/`), `hermes` (`~/.hermes/skills/productivity/`). Use the platform matching the runtime; default to `codex`.

The script does NOT write any SKILL.md. With `--write` it creates the skill directories and writes `orchestration.json`. It returns a work-list for you to fill in.

### 3. Read the Result

**`status: ready_to_write`** — proceed. The result gives you:
- `context` — plan-level fields shared by every skill (trigger, memory scope, delivery, confirmation boundary, failure handling, acceptance criteria).
- `skills[]` — the work-list, in dependency order. Each item: `id`, `role`, `user_goal`, `slug`, `output_path`, `inputs_needed`, `capabilities_used`, `produces`, `produces_shape` (the field shape of the product, when the plan set it), `consumes`.
- `skipped[]` — sub-skills dropped because a capability is unsupported, or because an upstream they depend on was dropped. Each has a user-facing `message`.
- `orchestration` / `orchestration_path` — already written by the script.

**`status: blocked`** — do not write anything. Resolve by `diagnostics[].code`:
- `unsupported_capability` → an always-forbidden capability (payment, sending to other people, deleting/publishing, account changes). Explain to the user which part is impossible and offer a manual-confirmation alternative.
- `dependency_cycle` → sub-skills depend on each other in a loop. Back to `writing-plans` to fix `produces`/`consumes`.
- `missing_skill_fields` / `duplicate_skill_id` / `missing_required_fields` → malformed plan. Back to `writing-plans`.
- `skill_already_exists` → ask the user whether to replace or rename.

### 4. Write Each Skill

For every item in `skills[]`, write a complete SKILL.md to its `output_path`. This is the real work — do it one skill at a time, in the order given (producers before consumers).

**Before writing your first skill of a workflow, read `../genSkill/references/authoring-good-skills.md` (the craft layer) and `../genSkill/references/authoring-good-skills-example.md` (a complete worked skill to match).** They are how you turn a high-level sub-task into a runbook an Agent can follow with no other context.

Required frontmatter:

```markdown
---
name: <the item's slug>
description: Use when <triggering conditions only>
---
```

Required structure for the body:

- `# <role>` — the human label.
- `## When to Use` — concrete triggers.
- `## Inputs` — from `inputs_needed`; for a consumer, the input is the upstream product. When the upstream carries a `produces_shape`, describe the input using those exact field names, so the consumer reads the same shape the producer writes.
- `## Workflow` — **the part only you can write.** Expand `user_goal` into ordered, runnable steps. Be specific enough that an Agent with no other context can follow it. This is where "how to actually do it" lives — the plan deliberately left it out.
- `## Confirmation and Boundaries` — fill from `context`: trigger/schedule, memory scope, delivery, confirmation boundary. State that requests outside this scope get one clarifying question or a plain refusal.
- `## Failure Handling` — from `context.failure_handling`.
- `## Acceptance Criteria` — from `context.acceptance_criteria`.
- `## Produces` / `## Consumes` — when the item has them, so the execute phase can chain skills. If the item has a `produces_shape`, state the exact output field shape under `## Produces`; the downstream consumer relies on it.

### 5. Save the Plain-Language Flow (`flow.md`)

After every surviving SKILL.md is written, save one `flow.md` describing the
whole workflow in the **user's own language** — this is what the next `/genSkill`
run shows when it finds a similar saved做法 and asks whether to reuse it. Without
it, the reuse check can only fall back to a terse `description`.

Write it to the **plan-slug directory** — the same directory as `orchestration.json`
(`path.dirname(orchestration_path)`). For a single-skill workflow with no
orchestration manifest, write it beside that skill's `SKILL.md`.

Use a numbered list, no jargon (never "skill"/"Agent"/"capability"). Fill every
line from the `context` you already have — do not invent new facts:

```markdown
# 做法的流程

任务：<context.user_goal>

1. **你给我**：<inputs_needed，口语化；定时无需用户给则写"到点我会提醒你">
2. **我会**：<把各 sub-skill 的 role 串成一句处理过程>
3. **开始方式**：<context.schedule_or_trigger，口语化，如"每周日晚上自动开始">
4. **完成后**：<context.output_or_delivery，口语化，如"周报发到你的微信">
5. **如果缺东西**：<context.failure_handling>
6. **我不会做的事**：<context.confirmation_boundary 及任何被跳过的能力>
```

If `skipped[]` is non-empty, reflect the dropped part honestly in line 6.

### 6. Report and Hand Off

- List each written skill path.
- If `skipped[]` is non-empty, tell the user in plain language which workflow was dropped and why — the `message` is already user-facing. Make clear what they still get without it. (E.g. "桌面 widget 这次没法做，但报告照样会在微信发给你。")
- Mention the `orchestration_path`.
- Proceed to `genSkill:execute`.

> **Structured UI (optional).** If trusted inbound metadata says the client
> supports capability `a2ui:genskill.cards@1` or catalog `genskill.cards@1`,
> report this file-write progress as a `SkillCreate` card **instead of** a prose
> list — read `../genSkill/references/a2ui-emit-guide.md` for the fields (one
> `steps[]` entry per written file, `status:"done"`, `label:"写入 <最后两段路径>"`;
> failures as `status:"error"` with the raw output in `detail`; dropped sub-skills
> as `annotations[]`). If the client does not advertise that capability, keep the
> plain-text report above. Same content — only the rendering changes.

## Authoring Quality Bar

These are the judgment calls that make a generated skill actually usable. The script can't make them — you must. The full craft (with a worked example) is in `../genSkill/references/authoring-good-skills.md`; the essentials:

- **Description = when to use, not what it does.** Start with "Use when…" and describe only triggering conditions. Never summarize the workflow in the description: testing shows an Agent will follow the description and skip the body, so a workflow summary there causes the body to be ignored. (This is the one rule worth borrowing from superpowers/writing-skills.)
- **Name by action, verb-first.** `import-data`, not `data-import`. The slug is already set; make the `# heading` and prose match that voice.
- **Workflow steps are concrete.** "提取关键字段" is too vague — say which fields, from what, into what shape. The plan stayed high-level on purpose; you fill the detail.
- **Keyword coverage.** Pull the everyday words a user would say (from the capability `user_words` and the `role`) into the description, so the skill is found when needed.
- **Self-contained.** Each SKILL.md must run on its own. A consumer skill names its upstream product as an input; it does not assume the producer's SKILL.md is open.
- **Boundaries are real.** Confirmation boundary, memory scope, and failure handling come from the plan — confirm they read correctly for this specific sub-skill, not as generic boilerplate.

## Hard Boundary (Never Cross)

The script hard-blocks always-forbidden capabilities, but you must never author around them either: no payment/transfer, no sending messages to other people, no deleting or publishing external content, no external account changes. If a sub-skill seems to need one, stop and return to `writing-plans` — do not write a skill that does it "manually."

## Fallback (No Script Execution)

If you cannot run the script:
- Show the approved plan JSON and the command the user can run.
- Explain: "我没法直接执行脚本，你可以手动运行这个命令，然后我来把每个做法写好。"
- You can still author the SKILL.md bodies; you just won't have the script's gating/ordering, so check capabilities yourself: use `../genSkill/references/capabilities-summary.md` to confirm supported/disabled, then read the specific `../genSkill/references/capabilities/<file>.md` only for the exact 边界 wording of the capabilities this plan uses.

## Exit Condition

Every surviving sub-skill in the work-list has a complete SKILL.md written to its `output_path`, the orchestration manifest exists, and `flow.md` (the plain-language 做法流程) is written to the plan-slug directory.

**Next step**: Invoke `genSkill:execute` to offer an optional first-run.
