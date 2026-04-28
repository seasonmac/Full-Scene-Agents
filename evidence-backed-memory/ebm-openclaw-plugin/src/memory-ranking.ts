/**
 * Token budget control and memory formatting for auto-recall injection.
 *
 * Ported from EBM's memory-ranking.ts, adapted for EBM MemorySearchHit.
 */
import type { MemorySearchHit } from "./client.js";

export function estimateTokenCount(text: string): number {
  if (!text) return 0;
  return Math.ceil(text.length / 4);
}

export type BuildMemoryLinesOptions = {
  recallTokenBudget: number;
  recallMaxContentChars: number;
  recallScoreThreshold: number;
};

/**
 * Build memory lines with token budget constraint.
 *
 * First memory is always included even if it exceeds remaining budget
 * (EBM spec §6.2): with recallMaxContentChars=500 a single line
 * is at most ~128 tokens, well within the 2000-token default budget.
 */
export function buildMemoryLinesWithBudget(
  hits: MemorySearchHit[],
  options: BuildMemoryLinesOptions,
): { lines: string[]; estimatedTokens: number } {
  let budgetRemaining = options.recallTokenBudget;
  const lines: string[] = [];
  let totalTokens = 0;

  for (const hit of hits) {
    if (budgetRemaining <= 0) break;
    if (hit.score < options.recallScoreThreshold) continue;

    let content = (hit.content || "").replace(/\s+/g, " ").trim();
    if (content.length > options.recallMaxContentChars) {
      content = content.slice(0, options.recallMaxContentChars) + "...";
    }

    const source = hit.source ? ` [${hit.source}]` : "";
    const line = `- ${hit.title || hit.id}${source}: ${content}`;
    const lineTokens = estimateTokenCount(line);

    if (lineTokens > budgetRemaining && lines.length > 0) break;

    lines.push(line);
    totalTokens += lineTokens;
    budgetRemaining -= lineTokens;
  }

  return { lines, estimatedTokens: totalTokens };
}
