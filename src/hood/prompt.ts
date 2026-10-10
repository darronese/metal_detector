import type { Tokenizer } from "./tokenizer";

// default one i chose
export const DEFAULT_SYSTEM = "You are a sarcastic, playful assistant";

export type PromptOptions = {
  // '' = no system message (Qwen's template then inserts its own default one)
  system?: string,
  // false = raw text: no roles, no special end token cue, so the model just continues it
  template?: boolean,
};

/**
 * The ONE place a user prompt becomes the text that gets encoded into input_ids.
 * Terminal and browser both call this, so they can't drift apart.
 *
 * @param tok - loaded tokenizer, its chat template decides the special tokens
 * @param userPrompt - what the user typed
 * @param options - system message and whether to use the chat template at all
 * @returns the exact text that will be encoded
 */
export function buildPrompt(tok: Tokenizer, userPrompt: string, { system = DEFAULT_SYSTEM, template = true }: PromptOptions = {}): string {
  if (!template) return userPrompt;

  // empty string is falsy: no system turn at all, rather than an empty one
  const messages = system
    ? [{ role: "system", content: system }, { role: "user", content: userPrompt }]
    : [{ role: "user", content: userPrompt }];
  return tok.applyTemplate(messages);
}
