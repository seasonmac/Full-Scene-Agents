import { describe, it, expect } from "vitest";
import { estimateTokenCount, buildMemoryLinesWithBudget } from "../src/memory-ranking.js";
import type { MemorySearchHit } from "../src/client.js";

function hit(overrides: Partial<MemorySearchHit> = {}): MemorySearchHit {
  return {
    id: "fact:1",
    title: "Test",
    content: "some content",
    source: "ledger",
    score: 0.9,
    ...overrides,
  };
}

const DEFAULT_OPTS = {
  recallTokenBudget: 2000,
  recallMaxContentChars: 500,
  recallScoreThreshold: 0.15,
};

describe("estimateTokenCount", () => {
  it("returns 0 for empty string", () => {
    expect(estimateTokenCount("")).toBe(0);
  });

  it("returns 0 for falsy input", () => {
    expect(estimateTokenCount(undefined as unknown as string)).toBe(0);
    expect(estimateTokenCount(null as unknown as string)).toBe(0);
  });

  it("estimates ~chars/4 rounded up", () => {
    expect(estimateTokenCount("abcd")).toBe(1);
    expect(estimateTokenCount("abcde")).toBe(2);
    expect(estimateTokenCount("a".repeat(100))).toBe(25);
    expect(estimateTokenCount("a".repeat(101))).toBe(26);
  });
});

describe("buildMemoryLinesWithBudget", () => {
  it("returns empty for empty hits", () => {
    const result = buildMemoryLinesWithBudget([], DEFAULT_OPTS);
    expect(result.lines).toEqual([]);
    expect(result.estimatedTokens).toBe(0);
  });

  it("formats a single hit with source", () => {
    const result = buildMemoryLinesWithBudget([hit()], DEFAULT_OPTS);
    expect(result.lines).toHaveLength(1);
    expect(result.lines[0]).toBe("- Test [ledger]: some content");
    expect(result.estimatedTokens).toBeGreaterThan(0);
  });

  it("uses id when title is empty", () => {
    const result = buildMemoryLinesWithBudget(
      [hit({ title: "", id: "fact:42" })],
      DEFAULT_OPTS,
    );
    expect(result.lines[0]).toContain("fact:42");
  });

  it("omits source bracket when source is empty", () => {
    const result = buildMemoryLinesWithBudget(
      [hit({ source: "" })],
      DEFAULT_OPTS,
    );
    expect(result.lines[0]).toBe("- Test: some content");
    expect(result.lines[0]).not.toContain("[]");
  });

  it("truncates content exceeding recallMaxContentChars", () => {
    const longContent = "x".repeat(600);
    const result = buildMemoryLinesWithBudget(
      [hit({ content: longContent })],
      DEFAULT_OPTS,
    );
    expect(result.lines[0]).toContain("...");
    const contentPart = result.lines[0]!.split(": ")[1]!;
    expect(contentPart.length).toBeLessThanOrEqual(504); // 500 + "..."
  });

  it("collapses whitespace in content", () => {
    const result = buildMemoryLinesWithBudget(
      [hit({ content: "  hello   world\n\nnewline  " })],
      DEFAULT_OPTS,
    );
    expect(result.lines[0]).toContain("hello world newline");
  });

  it("filters hits below score threshold", () => {
    const result = buildMemoryLinesWithBudget(
      [hit({ score: 0.1 }), hit({ score: 0.2, id: "fact:2", title: "Good" })],
      DEFAULT_OPTS,
    );
    expect(result.lines).toHaveLength(1);
    expect(result.lines[0]).toContain("Good");
  });

  it("first item is always included even if it exceeds remaining budget", () => {
    const result = buildMemoryLinesWithBudget(
      [hit({ content: "a".repeat(200) })],
      { ...DEFAULT_OPTS, recallTokenBudget: 1 },
    );
    expect(result.lines).toHaveLength(1);
    expect(result.estimatedTokens).toBeGreaterThan(1);
  });

  it("stops adding when budget is exhausted", () => {
    const hits = Array.from({ length: 20 }, (_, i) =>
      hit({ id: `fact:${i}`, title: `Item ${i}`, content: "a".repeat(200), score: 0.9 }),
    );
    const result = buildMemoryLinesWithBudget(hits, {
      ...DEFAULT_OPTS,
      recallTokenBudget: 100,
    });
    expect(result.lines.length).toBeLessThan(20);
    expect(result.estimatedTokens).toBeLessThanOrEqual(100 + 60);
  });

  it("handles empty content gracefully", () => {
    const result = buildMemoryLinesWithBudget(
      [hit({ content: "" })],
      DEFAULT_OPTS,
    );
    expect(result.lines).toHaveLength(1);
    expect(result.lines[0]).toBe("- Test [ledger]: ");
  });

  it("accumulates tokens correctly across multiple hits", () => {
    const hits = [
      hit({ id: "a", title: "A", content: "short" }),
      hit({ id: "b", title: "B", content: "also short" }),
    ];
    const result = buildMemoryLinesWithBudget(hits, DEFAULT_OPTS);
    expect(result.lines).toHaveLength(2);
    const manualTotal =
      estimateTokenCount(result.lines[0]!) + estimateTokenCount(result.lines[1]!);
    expect(result.estimatedTokens).toBe(manualTotal);
  });
});
