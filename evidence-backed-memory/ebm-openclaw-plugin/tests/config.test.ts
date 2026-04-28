import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { ebmPyConfigSchema } from "../src/config.js";

const manifest = JSON.parse(
  readFileSync(resolve(process.cwd(), "openclaw.plugin.json"), "utf-8"),
) as {
  configSchema: { additionalProperties?: boolean };
  uiHints: Record<string, unknown>;
};

describe("ebmPyConfigSchema.parse", () => {
  it("returns all defaults for empty input", () => {
    const cfg = ebmPyConfigSchema.parse({});
    expect(cfg.mode).toBe("local");
    expect(cfg.pythonCommand).toBe("python3");
    expect(cfg.port).toBe(18790);
    expect(cfg.baseUrl).toBe("http://127.0.0.1:18790");
    expect(cfg.slowPathEnabled).toBe(true);
    expect(cfg.timeoutMs).toBe(120_000);
    expect(cfg.healthTimeoutMs).toBe(30_000);
    expect(cfg.portScanRange).toBe(10);
    expect(cfg.dbPath).toMatch(/ebm_context_engine\.sqlite$/);
  });

  it("returns defaults for null/undefined input", () => {
    const cfg1 = ebmPyConfigSchema.parse(null);
    expect(cfg1.mode).toBe("local");
    const cfg2 = ebmPyConfigSchema.parse(undefined);
    expect(cfg2.mode).toBe("local");
  });

  it("respects mode=remote", () => {
    const cfg = ebmPyConfigSchema.parse({ mode: "remote", baseUrl: "http://my-server:9000" });
    expect(cfg.mode).toBe("remote");
    expect(cfg.baseUrl).toBe("http://my-server:9000");
  });

  it("strips trailing slashes from baseUrl", () => {
    const cfg = ebmPyConfigSchema.parse({ mode: "remote", baseUrl: "http://host:9000///" });
    expect(cfg.baseUrl).toBe("http://host:9000");
  });

  it("local mode derives baseUrl from port", () => {
    const cfg = ebmPyConfigSchema.parse({ mode: "local", port: 19000 });
    expect(cfg.baseUrl).toBe("http://127.0.0.1:19000");
  });

  it("clamps port to valid range", () => {
    const low = ebmPyConfigSchema.parse({ port: -5 });
    expect(low.port).toBe(1);
    const high = ebmPyConfigSchema.parse({ port: 99999 });
    expect(high.port).toBe(65535);
  });

  it("clamps timeoutMs minimum to 1000", () => {
    const cfg = ebmPyConfigSchema.parse({ timeoutMs: 100 });
    expect(cfg.timeoutMs).toBe(1000);
  });

  it("clamps healthTimeoutMs minimum to 1000", () => {
    const cfg = ebmPyConfigSchema.parse({ healthTimeoutMs: 500 });
    expect(cfg.healthTimeoutMs).toBe(1000);
  });

  it("clamps portScanRange to 0-100", () => {
    const low = ebmPyConfigSchema.parse({ portScanRange: -1 });
    expect(low.portScanRange).toBe(0);
    const high = ebmPyConfigSchema.parse({ portScanRange: 999 });
    expect(high.portScanRange).toBe(100);
  });

  it("accepts string port values", () => {
    const cfg = ebmPyConfigSchema.parse({ port: "19500" });
    expect(cfg.port).toBe(19500);
  });

  it("uses default python command when empty string provided", () => {
    const cfg = ebmPyConfigSchema.parse({ pythonCommand: "  " });
    expect(cfg.pythonCommand).toBe("python3");
  });

  it("trims pythonCommand", () => {
    const cfg = ebmPyConfigSchema.parse({ pythonCommand: "  /usr/bin/python3  " });
    expect(cfg.pythonCommand).toBe("/usr/bin/python3");
  });

  it("expands ~ in ebmPyPath", () => {
    const cfg = ebmPyConfigSchema.parse({ ebmPyPath: "~/projects/noteLM" });
    expect(cfg.ebmPyPath).not.toContain("~");
    expect(cfg.ebmPyPath).toMatch(/projects\/noteLM$/);
  });

  it("expands ~ in dbPath", () => {
    const cfg = ebmPyConfigSchema.parse({ dbPath: "~/my.db" });
    expect(cfg.dbPath).not.toContain("~");
    expect(cfg.dbPath).toMatch(/my\.db$/);
  });

  it("slowPathEnabled defaults to true", () => {
    expect(ebmPyConfigSchema.parse({}).slowPathEnabled).toBe(true);
  });

  it("slowPathEnabled can be set to false", () => {
    expect(ebmPyConfigSchema.parse({ slowPathEnabled: false }).slowPathEnabled).toBe(false);
  });

  it("throws on unknown keys", () => {
    expect(() => ebmPyConfigSchema.parse({ unknownField: true })).toThrow("unknown keys");
  });

  it("manifest schema disables additionalProperties", () => {
    expect(manifest.configSchema.additionalProperties).toBe(false);
  });

  it("manifest includes uiHints for core fields", () => {
    expect(manifest.uiHints.mode).toBeTruthy();
    expect(manifest.uiHints.baseUrl).toBeTruthy();
    expect(manifest.uiHints.slowPathEnabled).toBeTruthy();
  });

  it("derives configJsonPath from ebmPyPath", () => {
    const cfg = ebmPyConfigSchema.parse({ ebmPyPath: "/workspace/noteLM" });
    expect(cfg.configJsonPath).toMatch(/\/workspace\/noteLM\/ebm\/config\.json$/);
  });

  it("explicit configJsonPath overrides derived", () => {
    const cfg = ebmPyConfigSchema.parse({
      ebmPyPath: "/workspace/noteLM",
      configJsonPath: "/custom/config.json",
    });
    expect(cfg.configJsonPath).toMatch(/\/custom\/config\.json$/);
  });

  it("remote mode reads EBM_PY_BASE_URL env fallback", () => {
    const orig = process.env.EBM_PY_BASE_URL;
    try {
      process.env.EBM_PY_BASE_URL = "http://env-server:8000";
      const cfg = ebmPyConfigSchema.parse({ mode: "remote" });
      expect(cfg.baseUrl).toBe("http://env-server:8000");
    } finally {
      if (orig !== undefined) {
        process.env.EBM_PY_BASE_URL = orig;
      } else {
        delete process.env.EBM_PY_BASE_URL;
      }
    }
  });

  // ── Auto-recall config fields ──────────────────────────────

  it("autoRecall defaults to true", () => {
    expect(ebmPyConfigSchema.parse({}).autoRecall).toBe(true);
  });

  it("autoRecall can be disabled", () => {
    expect(ebmPyConfigSchema.parse({ autoRecall: false }).autoRecall).toBe(false);
  });

  it("recallLimit defaults to 6 and clamps minimum to 1", () => {
    expect(ebmPyConfigSchema.parse({}).recallLimit).toBe(6);
    expect(ebmPyConfigSchema.parse({ recallLimit: 0 }).recallLimit).toBe(1);
    expect(ebmPyConfigSchema.parse({ recallLimit: -5 }).recallLimit).toBe(1);
    expect(ebmPyConfigSchema.parse({ recallLimit: 20 }).recallLimit).toBe(20);
  });

  it("recallScoreThreshold defaults to 0.15 and clamps to 0-1", () => {
    expect(ebmPyConfigSchema.parse({}).recallScoreThreshold).toBe(0.15);
    expect(ebmPyConfigSchema.parse({ recallScoreThreshold: -0.5 }).recallScoreThreshold).toBe(0);
    expect(ebmPyConfigSchema.parse({ recallScoreThreshold: 2 }).recallScoreThreshold).toBe(1);
    expect(ebmPyConfigSchema.parse({ recallScoreThreshold: 0.5 }).recallScoreThreshold).toBe(0.5);
  });

  it("recallMaxContentChars defaults to 500 and clamps to 50-10000", () => {
    expect(ebmPyConfigSchema.parse({}).recallMaxContentChars).toBe(500);
    expect(ebmPyConfigSchema.parse({ recallMaxContentChars: 10 }).recallMaxContentChars).toBe(50);
    expect(ebmPyConfigSchema.parse({ recallMaxContentChars: 99999 }).recallMaxContentChars).toBe(10000);
  });

  it("recallTokenBudget defaults to 2000 and clamps to 100-50000", () => {
    expect(ebmPyConfigSchema.parse({}).recallTokenBudget).toBe(2000);
    expect(ebmPyConfigSchema.parse({ recallTokenBudget: 10 }).recallTokenBudget).toBe(100);
    expect(ebmPyConfigSchema.parse({ recallTokenBudget: 100000 }).recallTokenBudget).toBe(50000);
  });

  it("recallTimeoutMs defaults to 5000 and clamps to 500-30000", () => {
    expect(ebmPyConfigSchema.parse({}).recallTimeoutMs).toBe(5000);
    expect(ebmPyConfigSchema.parse({ recallTimeoutMs: 100 }).recallTimeoutMs).toBe(500);
    expect(ebmPyConfigSchema.parse({ recallTimeoutMs: 99999 }).recallTimeoutMs).toBe(30000);
  });

  // ── Ingest-reply-assist config fields ──────────────────────

  it("ingestReplyAssist defaults to true", () => {
    expect(ebmPyConfigSchema.parse({}).ingestReplyAssist).toBe(true);
  });

  it("ingestReplyAssist can be disabled", () => {
    expect(ebmPyConfigSchema.parse({ ingestReplyAssist: false }).ingestReplyAssist).toBe(false);
  });

  it("ingestReplyAssistMinSpeakerTurns defaults to 2 and clamps to 1-12", () => {
    expect(ebmPyConfigSchema.parse({}).ingestReplyAssistMinSpeakerTurns).toBe(2);
    expect(ebmPyConfigSchema.parse({ ingestReplyAssistMinSpeakerTurns: 0 }).ingestReplyAssistMinSpeakerTurns).toBe(1);
    expect(ebmPyConfigSchema.parse({ ingestReplyAssistMinSpeakerTurns: 50 }).ingestReplyAssistMinSpeakerTurns).toBe(12);
  });

  it("ingestReplyAssistMinChars defaults to 120 and clamps to 32-10000", () => {
    expect(ebmPyConfigSchema.parse({}).ingestReplyAssistMinChars).toBe(120);
    expect(ebmPyConfigSchema.parse({ ingestReplyAssistMinChars: 5 }).ingestReplyAssistMinChars).toBe(32);
    expect(ebmPyConfigSchema.parse({ ingestReplyAssistMinChars: 99999 }).ingestReplyAssistMinChars).toBe(10000);
  });

  it("ingestReplyAssistIgnoreSessionPatterns defaults to empty array", () => {
    expect(ebmPyConfigSchema.parse({}).ingestReplyAssistIgnoreSessionPatterns).toEqual([]);
  });

  it("parses ingestReplyAssistIgnoreSessionPatterns from array", () => {
    const cfg = ebmPyConfigSchema.parse({
      ingestReplyAssistIgnoreSessionPatterns: ["test:*", "ci:**"],
    });
    expect(cfg.ingestReplyAssistIgnoreSessionPatterns).toEqual(["test:*", "ci:**"]);
  });

  it("parses ingestReplyAssistIgnoreSessionPatterns from comma-separated string", () => {
    const cfg = ebmPyConfigSchema.parse({
      ingestReplyAssistIgnoreSessionPatterns: "test:*, ci:**",
    });
    expect(cfg.ingestReplyAssistIgnoreSessionPatterns).toEqual(["test:*", "ci:**"]);
  });

  it("filters empty entries from ingestReplyAssistIgnoreSessionPatterns", () => {
    const cfg = ebmPyConfigSchema.parse({
      ingestReplyAssistIgnoreSessionPatterns: ["valid", "", "  ", "also-valid"],
    });
    expect(cfg.ingestReplyAssistIgnoreSessionPatterns).toEqual(["valid", "also-valid"]);
  });
});
