---
name: genSkill:execute
description: Use after a skill is successfully generated. Offers the user an optional first-run to verify the skill works as expected.
---

# Execute Skill (Optional First-Run)

Let the user see the just-created skill working once, immediately after generation.

Reference files are shared from the entry skill directory. If this phase ever
needs a reference file, read it from `../genSkill/references/...`; do not expect
a local `references/` directory under `genSkill-execute`.

## Precondition

This skill may be invoked in **either** case:

- **Just generated** — `genSkill:writing-skills` has finished writing every
  surviving skill's SKILL.md, and the file(s) plus `orchestration.json` (for a
  multi-skill workflow) exist at the reported path(s).
- **Reuse of an existing workflow** — Phase 1's reuse check (Step 0) found a saved
  做法 and the user chose 就用这个. The workflow's directory (SKILL.md files +
  `orchestration.json` for multi-skill) is already on disk; no generation runs.
  Locate it by the slug returned from the `--list` reuse check, then proceed with
  the same flow below.

## Single Skill vs. Multi-Skill

The generator returns one of two shapes:

- **Single skill** — one `output_path`. Run the first-run flow below directly.
- **Multi-skill** — a `skills[]` list plus an `orchestration_path`. Read `orchestration.json` first and set up wiring before offering a first-run. Some sub-skills may have been `skipped` (unsupported capability) — tell the user plainly what they lose, then proceed with what survived.

### Reading orchestration.json

The manifest has up to two keys:

- `scheduled[]` — `{ skill, when, after[] }`. Register a real timed trigger (`trigger.scheduled_reminder`) for each entry by adding an openclaw cron job — do not just remember the time. `when` is the schedule, `after` lists upstream products that must exist first.

  ```
  openclaw cron add \
    --cron "<cron expr from `when`>" \
    --announce \
    --message "调用<skill>,<它的 user_goal>;完成后通过<已确认的 channel>通知我。"
  ```

  - Translate `when` to a standard 5-field cron expression (分 时 日 月 周). E.g. 每周日 09:00 → `0 9 * * 0`; `weekly:SAT 08:00` → `0 8 * * 6`.
  - The `--message` must reference only skills in this workflow and the channel the user confirmed — it is the instruction the Agent receives when cron fires, so it must stay inside the approved scope.
  - `--announce` reports the result to chat; add `--expect-final` when the user must get the finished output (not just a summary).
  - Honor `after`: if the scheduled skill consumes a product, the cron message must run the producing skill first (or the producer must already have run). Do not fire a scheduled skill before its `after` deps exist.
  - Confirm the cron expression with the user in plain language before registering ("每周日早上 9 点" not the raw `0 9 * * 0").
- `immediate[]` — `{ skill, action, bindings[]? }`. Invoke the skill now to perform `action` (e.g. render-and-update-widget). For each binding:
  - `{ trigger, ability }` — bind a UI/event trigger to an Ability (e.g. the widget scan button → scan-code).
  - `{ trigger, target_skill }` — route a trigger's result to another sub-skill (e.g. photo-result → process-data).

Only wire skills that were actually generated. Entries referencing skipped skills are already filtered out of the manifest, but if you spot one, ignore it.

## Process

### 1. Offer the First-Run

Ask the user:

> 已保存这个做法。要不要现在试一次，看看效果？

Wait for the user's response.

**If the user declines**: End with "好的，以后需要时直接告诉我就行。" and stop.

**If the user agrees**: Continue to step 2.

### 2. Collect Minimum Inputs

Based on the `inputs_needed` from the plan:
- Ask the user to provide the minimum inputs for one run
- On the **reuse path** the plan JSON isn't in context — read `inputs_needed` from
  the entry-point skill's `## Inputs` section in its SKILL.md instead.
- One question at a time, same rules as brainstorming (concrete, plain language, max 3 choices if applicable)
- If an input is optional for a first run, skip it and note: "这次先跳过 [x]，下次用的时候再提供也行。"
- For a multi-skill workflow, collect inputs only for the entry-point sub-skill (the one that `consumes` nothing). Downstream sub-skills receive their inputs as upstream products.

### 3. Execute the Workflow

**Single skill**: run through the workflow in its authored SKILL.md, in order. (A legacy single-action plan may carry a flat `steps[]` — follow that.)

**Multi-skill**: run sub-skills in dependency order (a skill runs only after every product it `consumes` has been produced). For each:
- Follow that sub-skill's authored SKILL.md workflow (the plan JSON keeps sub-skills lightweight with no `steps`; the runnable steps live in the SKILL.md written by writing-skills)
- Pass its `produces` output to whatever consumes it
- Honor `immediate` bindings as you reach the skill that owns them

In both cases:
- Respect the `confirmation_boundary` — if the skill says "输出前让用户检查", show the draft before finalizing
- Stay within each skill's `capabilities_used` — do not use capabilities not in the plan

### 4. Present the Result

Show the output to the user and ask:

> 这是按你确认的做法生成的结果，看看是否符合预期？

**If the user is satisfied**: End with "好的，以后需要时直接告诉我按这个做法来就行。"

**If the user wants adjustments**:
- If it's a minor tweak to the output: adjust and re-present
- If it's a fundamental change to the workflow: explain "这个改动需要修改保存的做法本身，要重新编辑吗？" and offer to restart from `genSkill:writing-plans`

> **Structured UI (optional).** If trusted inbound metadata says the client
> supports capability `a2ui:genskill.cards@1` or catalog `genskill.cards@1`,
> present this closing summary as an `ExecuteSummary` card **instead of** prose —
> read `../genSkill/references/a2ui-emit-guide.md` for the fields (`skillPaths` =
> the created SKILL.md path(s); `usageSteps` = how the user invokes it from now on;
> `cta` = the closing call-to-action; optional `intro`). If the client does not
> advertise that capability, keep the plain-text form above. Same content — only
> the rendering changes.

## Exit Condition

One of:
- The user has seen a successful first-run result and is satisfied
- The user explicitly declined to try ("不用了" / "以后再说" / "skip")

## Anti-Patterns

| Wrong | Right |
|-------|-------|
| Running the skill without asking first | Always offer, never force |
| Asking for ALL inputs at once | One at a time, same as brainstorming |
| Using capabilities not in the plan | Strict boundary — only what was approved |
| Silently changing the workflow based on first-run feedback | Explain that changes require re-editing the saved skill |
