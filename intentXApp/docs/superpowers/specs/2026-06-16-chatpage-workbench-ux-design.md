# ChatPage Workbench UX Design

## Goal

Close the seven UX gaps between the current ChatPage and the v0 chat-ux direction while keeping A2UI as the only structured UI source. This work does not change parsing, protocol, or backend logic.

## Scope

The implementation adds a focused bottom workbench for active A2UI workflow blocks. The existing inline timeline remains as the historical transcript and fallback. The workbench mirrors the latest live block and provides the immersive interaction layer: Sheet-style container, phase header, progress, waiting/status strip, skeleton-to-content reveal, fixed footer actions, execute-stage protection, and useful final CTA.

## Architecture

`ChatPage.ets` owns workbench visibility and phase state because it already has `flowBlocksState`, `pendingRunCountState`, and the handlers that send A2UI option choices. New focused components live under `entry/src/main/ets/components/flow/`: `PhaseStage`, `StatusStrip`, and `PhaseSkeleton`. Existing flow cards are reused inside the workbench to avoid duplicating UI logic.

The first active structured block is selected from the latest A2UI block with kind `clarify-question`, `approach-proposal`, `plan-confirm`, `skill-create`, or `exec-summary`. The workbench opens when such a block appears, stays open while the workflow phase advances, and can be dismissed except while execute/create work is running.

## UX Behavior

- Normal chat shows no workbench.
- Active workflow opens one persistent bottom workbench over a dimmed chat background.
- The workbench header shows phase title/subtitle and a close affordance only when dismissible.
- A compact `FlowProgressBar` stays inside the workbench header.
- `PhaseStage` shows a status strip and skeleton first, then reveals the real card content.
- The footer is fixed at the bottom and exposes the primary action for interactive blocks.
- Existing inline cards are shown as historical/locked transcript context.
- Execute summary CTA launches image selection via the existing `PhotoCaptureService`.

## Verification

Static UX guard scripts check for the workbench component integration and important contracts. Existing `check-chatpage-ui.mjs` remains valid. Build verification uses the available ArkTS/Harmony project commands where possible.
