import type { TokenInfo } from "../engine/messages";

// small display helpers shared by the templates: no DOM, no state

export type Tokens = Map<number, TokenInfo>;

export const esc = (s: string) => s.replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "\"": "&quot;" })[c]!);
export const fmt = (n: number) => n.toLocaleString("en-US");
export const pct = (p: number) => p >= 0.001 ? `${(p * 100).toFixed(1)}%` : p.toExponential(0);

/** token as you'd want to SEE it: whitespace made visible, special tokens raw */
export function tokenText(info: TokenInfo | undefined) {
  if (!info) return "…";
  if (info.raw === "") return "∅";
  if (info.special) return info.raw;
  return info.text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "⇥") || "∅";
}

export function tokenTitle(id: number, info: TokenInfo | undefined) {
  if (!info) return `id ${id}`;
  if (info.raw === "") return `id ${id}\npadding row of the output layer: has a logit, but no token`;
  return `id ${id}\nvocab entry: ${info.raw}\ndecodes to: ${JSON.stringify(info.text)}`;
}

/** 0..1 brightness for a probability, log scale: 1e-6 -> 0, 1 -> 1 */
export const heat = (p: number) => Math.max(0, Math.min(1, 1 + Math.log10(Math.max(p, 1e-12)) / 6));
