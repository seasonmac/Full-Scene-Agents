# ChatPage Workbench UX Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the immersive ChatPage workbench UX for A2UI workflow blocks without changing A2UI parsing or backend logic.

**Architecture:** `ChatPage.ets` owns workbench state and renders a Sheet-style overlay from the current `UIBlock`. New flow components provide phase status and skeleton reveal. Existing flow cards remain the source of detailed content.

**Tech Stack:** HarmonyOS ArkTS/ArkUI, existing A2UI `UIBlock` model, existing `PhotoCaptureService`, Node-based static guard scripts.

---

### Task 1: Add UX Guard

**Files:**
- Create: `scripts/check-chatpage-workbench-ui.mjs`

- [x] **Step 1: Write the failing guard**

Create a static script that asserts `ChatPage.ets` imports and renders workbench components, defines workbench state, renders a fixed footer, protects running stages, and wires CTA to photo selection.

- [x] **Step 2: Run guard and verify RED**

Run: `node scripts/check-chatpage-workbench-ui.mjs`

Expected: FAIL because the workbench files and integration are missing.

### Task 2: Add Phase Workbench Components

**Files:**
- Create: `entry/src/main/ets/components/flow/StatusStrip.ets`
- Create: `entry/src/main/ets/components/flow/PhaseSkeleton.ets`
- Create: `entry/src/main/ets/components/flow/PhaseStage.ets`

- [x] **Step 1: Implement components**

Implement a small, deterministic ArkTS version of the v0 PhaseStage pattern: status strip, skeleton footprint, and revealed content slot.

- [x] **Step 2: Run guard**

Run: `node scripts/check-chatpage-workbench-ui.mjs`

Expected: still FAIL until ChatPage integration is added.

### Task 3: Integrate Workbench Into ChatPage

**Files:**
- Modify: `entry/src/main/ets/pages/ChatPage.ets`

- [x] **Step 1: Add state and helpers**

Add workbench visibility, ready state, active-block selection, phase labels, dismissibility, and footer label helpers.

- [x] **Step 2: Add overlay and sheet builders**

Render a dimmed overlay plus bottom sheet after `Composer()`. Reuse `FlowProgressBar` and existing flow card builders inside `PhaseStage`.

- [x] **Step 3: Move primary workbench actions into footer**

Interactive content inside the workbench uses the same send handlers as inline cards. Footer shows confirm/next/complete actions and stays fixed.

### Task 4: Wire Final CTA To Photo Picker

**Files:**
- Modify: `entry/src/main/ets/pages/ChatPage.ets`

- [x] **Step 1: Import `PhotoCaptureService`**

Use the existing photo picking service rather than adding new media logic.

- [x] **Step 2: Implement CTA handler**

Launch photo selection from the workbench/summary CTA. If no URI is returned, leave chat unchanged.

### Task 5: Verify

**Files:**
- Existing scripts and project commands

- [x] **Step 1: Run static guards**

Run: `node scripts/check-chatpage-ui.mjs && node scripts/check-chatpage-workbench-ui.mjs`

Expected: both pass.

- [x] **Step 2: Run available build/check command**

Run the available Harmony/Node project check command and report exact result.

Result: `node hvigorw.js --help` attempted, but this workspace has no `package.json` for the wrapper's fallback `npm install`, and `hvigor`/`ohpm` are not available on PATH. Static Node guards passed; full ArkTS build must be run in DevEco or an environment with Hvigor installed.
