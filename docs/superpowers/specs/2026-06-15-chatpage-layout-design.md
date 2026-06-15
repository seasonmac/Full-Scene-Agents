# ChatPage Layout Refresh Design

Date: 2026-06-15

## Scope

Refresh the HarmonyOS `intentXApp` ChatPage using the selected incremental approach B:

- Keep the existing app shell and bottom tab structure in `PostOnboardingTabs.ets`.
- Restructure `ChatPage.ets` into stable internal layers: chat header, optional workflow progress, scrollable message area, and composer.
- Preserve the current composer input visual style because its translucent blue input槽 works better than a heavier v0-style replacement.
- Add a lucide Bot-based Agent avatar resource to the HarmonyOS app and use it consistently for Agent messages.

This change does not redesign Connect, Settings, gateway state handling, or the flow-card business logic.

## Current Problems

The current ChatPage mixes session controls, message scroll content, thinking controls, and composer controls in one page column. On small screens this makes the last Agent response compete with the composer and the parent bottom tab bar.

The session key card also consumes vertical space even though the same context can fit in a compact header subtitle or session drawer. The page already has partial v0 tokens and flow cards, so the next step is layout stabilization rather than a full visual rewrite.

## Target Layout

`PostOnboardingTabs.ets` keeps ownership of the app status bar and bottom tab bar.

`ChatPage.ets` owns four layers:

1. `ChatHeader`
   - Shows active session title, session-list entry, and a compact session key subtitle.
   - Replaces the always-visible `SessionLinkCard`.
   - Keeps copy behavior available from the header/session affordance, not as a full-width card.

2. `WorkflowProgress`
   - Shows `FlowProgressBar` only when structured flow is active.
   - Stays above messages and does not scroll with the conversation.

3. `MessagesList`
   - Owns all chat history scrolling.
   - Uses `layoutWeight(1)` and `minHeight`/bottom padding so long Agent content never overlaps the composer or parent tab bar.
   - Keeps existing structured flow rendering.

4. `Composer`
   - Remains fixed below the message scroll area.
   - Preserves the current translucent input槽: `C_INPUT_BG`, rounded container, slash launcher button, transparent `TextInput`, and current send/abort button behavior.
   - Keeps thinking and refresh controls, but their row must remain inside the fixed composer area.

## Agent Avatar

Agent messages should use the lucide Bot icon, not the existing IntentX app logo.

Implementation target:

- Add an app-local media resource derived from lucide's `Bot` icon, such as `entry/src/main/resources/base/media/lucide_bot.svg` or a PNG export if Harmony resource handling requires raster media.
- Use this resource in the Agent avatar builder for normal Agent messages and streaming messages.
- Keep user messages without a visible avatar unless the current layout needs one for alignment.
- Structured flow cards may keep their current full-card rendering; when they are preceded by an Agent identity row, use the same Bot avatar.

The resource must be committed with the app source so the page does not depend on Web or npm runtime packages.

## Visual Rules

- Composer input style stays close to the current ArkTS implementation.
- Header and flow progress may follow the v0 spacing and hierarchy.
- Remove the permanent white session link card from the vertical flow.
- Keep message bubble widths close to current values unless needed to accommodate the avatar row.
- Avoid changing bottom TabBar styling in this pass.

## Interaction Rules

- Session switching still uses the existing `showSessionList` state and `SessionList` builder.
- Copying the session key must remain possible.
- Flow card selection and pending interaction behavior stays unchanged.
- Sending, aborting, thinking-level changes, skill launcher selection, and refresh must continue to call the same view-model methods.

## Files Expected To Change

- `intentXApp/entry/src/main/ets/pages/ChatPage.ets`
- `intentXApp/entry/src/main/resources/base/media/<lucide-bot-resource>`

Potentially changed only if required by resource lookup:

- `intentXApp/entry/src/main/resources/base/element/string.json`
- `intentXApp/entry/src/main/resources/zh_CN/element/string.json`
- `intentXApp/entry/src/main/resources/en_US/element/string.json`

## Acceptance Criteria

1. On a screen matching the provided screenshot, the last Agent message remains readable and is not covered by thinking controls, composer, or bottom tabs.
2. The composer remains visually close to the current translucent input design.
3. Agent text and streaming messages show a lucide Bot avatar from app resources.
4. The permanent session key card no longer consumes a full message-row height.
5. Workflow progress and structured cards still render when a flow is active.
6. Normal chat still has no workflow progress bar.
7. Existing send, abort, refresh, thinking-level, session-list, and skill-launcher behaviors continue to work.
8. The HarmonyOS project builds after the resource and ArkTS changes.

## Verification Plan

- Run a focused source review to ensure `ChatPage.ets` keeps all existing view-model calls.
- Build `intentXApp` with the local HarmonyOS toolchain if available.
- If a device is available, capture the Chat tab and verify the screenshot case visually.
- If no device is available, complete static verification and report the build/device limitation explicitly.
