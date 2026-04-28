import { describe, expect, it } from "vitest";
import { mkdtempSync, readFileSync, existsSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import { execFileSync } from "node:child_process";

describe("setup-helper/install.js", () => {
  it("writes plugin config and slot for local mode", () => {
    const workdir = mkdtempSync(join(tmpdir(), "ebm-py-install-"));
    execFileSync("node", ["./setup-helper/install.js", "--workdir", workdir, "--mode", "local"], {
      cwd: resolve(process.cwd()),
      stdio: "pipe",
    });
    const raw = readFileSync(join(workdir, "openclaw.json"), "utf-8");
    const config = JSON.parse(raw);
    expect(config.plugins.entries["ebm-context-engine"].enabled).toBe(true);
    expect(config.plugins.entries["ebm-context-engine"].config.mode).toBe("local");
    expect(config.plugins.entries["ebm-context-engine"].config.ebmPyPath).toBe(
      join(workdir, "vendor", "noteLM"),
    );
    expect(existsSync(join(workdir, "vendor", "noteLM", "ebm_context_engine"))).toBe(true);
    expect(existsSync(join(workdir, "vendor", "noteLM", "ebm"))).toBe(true);
    expect(existsSync(join(workdir, "vendor", "noteLM", "cram"))).toBe(true);
    expect(config.plugins.slots.contextEngine).toBe("ebm-context-engine");
  });

  it("writes remote config template", () => {
    const workdir = mkdtempSync(join(tmpdir(), "ebm-py-install-"));
    execFileSync(
      "node",
      ["./setup-helper/install.js", "--workdir", workdir, "--mode", "remote", "--base-url", "http://remote:18790"],
      {
        cwd: resolve(process.cwd()),
        stdio: "pipe",
      },
    );
    const raw = readFileSync(join(workdir, "openclaw.json"), "utf-8");
    const config = JSON.parse(raw);
    expect(config.plugins.entries["ebm-context-engine"].config.mode).toBe("remote");
    expect(config.plugins.entries["ebm-context-engine"].config.baseUrl).toBe("http://remote:18790");
  });

  it("does not copy node_modules or coverage", () => {
    const workdir = mkdtempSync(join(tmpdir(), "ebm-py-install-"));
    const fixtureNodeModules = join(process.cwd(), "node_modules", ".ebm-install-helper-test");
    const fixtureCoverage = join(process.cwd(), "coverage", ".ebm-install-helper-test");
    mkdirSync(join(process.cwd(), "node_modules"), { recursive: true });
    mkdirSync(join(process.cwd(), "coverage"), { recursive: true });
    writeFileSync(fixtureNodeModules, "x", "utf-8");
    writeFileSync(fixtureCoverage, "x", "utf-8");

    execFileSync("node", ["./setup-helper/install.js", "--workdir", workdir, "--mode", "local"], {
      cwd: resolve(process.cwd()),
      stdio: "pipe",
    });

    expect(existsSync(join(workdir, "extensions", "ebm-context-engine", "node_modules"))).toBe(false);
    expect(existsSync(join(workdir, "extensions", "ebm-context-engine", "coverage"))).toBe(false);
  });
});
