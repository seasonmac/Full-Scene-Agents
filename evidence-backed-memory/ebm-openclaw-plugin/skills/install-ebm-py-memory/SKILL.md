# install-ebm-context-memory

## Overview
Install the EBM Context Memory Engine plugin for OpenClaw. This plugin provides Evidence-Backed Memory with a three-plane architecture (Workspace, Skill Graph, Temporal Ledger) powered by a Python sidecar.

## Prerequisites
- OpenClaw >= 2026.3.7
- Python 3.12+ with `ebm_context_engine` package available
- Node.js 22+

## Quick Install

```bash
# From the noteLM workspace root
cd ebm-openclaw-plugin
pnpm install
node ./setup-helper/install.js --mode local
```

## Configuration

The helper writes `~/.openclaw/openclaw.json` automatically. The resulting entry looks like:

```json
{
  "plugins": {
    "entries": {
      "ebm-context-engine": {
        "enabled": true,
        "config": {
          "mode": "local",
          "pythonCommand": "python3",
          "port": 18790,
          "ebmPath": "/path/to/noteLM",
          "configJsonPath": "/path/to/noteLM/ebm/config.json",
          "dbPath": "~/.openclaw/memory/ebm_context.sqlite"
        }
      }
    },
    "slots": {
      "contextEngine": "ebm-context-engine"
    }
  }
}
```

## Modes

### Local Mode (default)
The plugin spawns a Python sidecar process automatically. Set `mode: "local"` and configure `pythonCommand` and `ebmPath`.

### Remote Mode
Connect to an already-running ebm_context_engine server. Set `mode: "remote"` and configure `baseUrl` to point to the server.

```json
{
  "mode": "remote",
  "baseUrl": "http://127.0.0.1:18790"
}
```

## Verification

```bash
node ./setup-helper/install.js --mode remote --base-url http://127.0.0.1:18790 --workdir ~/.openclaw-second
openclaw config get plugins.entries.ebm-context-engine
openclaw config get plugins.slots.contextEngine
```

Expected: the plugin entry exists and `plugins.slots.contextEngine` points to `ebm-context-engine`.
