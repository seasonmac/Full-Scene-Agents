#!/usr/bin/env node
import { cp, mkdir, readFile, writeFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import process from "node:process";

const __dirname = dirname(fileURLToPath(import.meta.url));
const pluginRoot = resolve(__dirname, "..");
const repoRoot = resolve(pluginRoot, "..");
const PLUGIN_ID = "ebm-context-engine";
const VENDORED_PATHS = ["ebm_context_engine", "ebm", "cram"];

function parseArgs(argv) {
  const args = {
    workdir: process.env.OPENCLAW_STATE_DIR || join(process.env.HOME || process.env.USERPROFILE || "", ".openclaw"),
    mode: "local",
    baseUrl: "http://127.0.0.1:18790",
    pythonCommand: "python3",
    port: 18790,
    yes: false,
  };
  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (arg === "--workdir") {
      args.workdir = argv[i + 1] || args.workdir;
      i += 1;
    } else if (arg === "--mode") {
      args.mode = argv[i + 1] || args.mode;
      i += 1;
    } else if (arg === "--base-url") {
      args.baseUrl = argv[i + 1] || args.baseUrl;
      i += 1;
    } else if (arg === "--python-command") {
      args.pythonCommand = argv[i + 1] || args.pythonCommand;
      i += 1;
    } else if (arg === "--port") {
      const parsed = Number(argv[i + 1]);
      if (Number.isFinite(parsed) && parsed > 0) args.port = parsed;
      i += 1;
    } else if (arg === "-y" || arg === "--yes") {
      args.yes = true;
    }
  }
  return args;
}

async function loadJson(path, fallback) {
  if (!existsSync(path)) return fallback;
  try {
    return JSON.parse(await readFile(path, "utf-8"));
  } catch {
    return fallback;
  }
}

function buildPluginConfig(args, vendorRoot) {
  if (args.mode === "remote") {
    return {
      mode: "remote",
      baseUrl: args.baseUrl,
      timeoutMs: 120000,
    };
  }
  return {
    mode: "local",
    pythonCommand: args.pythonCommand,
    port: args.port,
    ebmPyPath: vendorRoot,
    configJsonPath: resolve(vendorRoot, "ebm", "config.json"),
    dbPath: "~/.openclaw/memory/ebm_context_engine.sqlite",
    timeoutMs: 120000,
    healthTimeoutMs: 30000,
    slowPathEnabled: true,
  };
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const workdir = resolve(args.workdir);
  const extensionsDir = join(workdir, "extensions");
  const pluginDest = join(extensionsDir, PLUGIN_ID);
  const vendorRoot = join(workdir, "vendor", "noteLM");
  const configPath = join(workdir, "openclaw.json");

  await mkdir(extensionsDir, { recursive: true });
  await mkdir(vendorRoot, { recursive: true });
  await cp(pluginRoot, pluginDest, {
    recursive: true,
    force: true,
    filter(src) {
      const normalized = src.split(sep);
      return !normalized.includes("node_modules") && !normalized.includes("coverage");
    },
  });
  for (const relativePath of VENDORED_PATHS) {
    await cp(join(repoRoot, relativePath), join(vendorRoot, relativePath), {
      recursive: true,
      force: true,
      filter(src) {
        const normalized = src.split(sep);
        return !normalized.includes(".git")
          && !normalized.includes("node_modules")
          && !normalized.includes("coverage")
          && !normalized.includes("__pycache__");
      },
    });
  }

  const config = await loadJson(configPath, {});
  if (!config.plugins || typeof config.plugins !== "object") {
    config.plugins = {};
  }
  if (!config.plugins.entries || typeof config.plugins.entries !== "object") {
    config.plugins.entries = {};
  }
  if (!config.plugins.slots || typeof config.plugins.slots !== "object") {
    config.plugins.slots = {};
  }

  config.plugins.entries[PLUGIN_ID] = {
    enabled: true,
    config: buildPluginConfig(args, vendorRoot),
  };
  config.plugins.slots.contextEngine = PLUGIN_ID;

  await writeFile(configPath, `${JSON.stringify(config, null, 2)}\n`, "utf-8");
  process.stdout.write(
    `Installed ${PLUGIN_ID} to ${pluginDest}\nUpdated config: ${configPath}\nMode: ${args.mode}\n`,
  );
}

main().catch((err) => {
  process.stderr.write(`${String(err)}\n`);
  process.exitCode = 1;
});
