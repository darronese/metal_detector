// the contract between the page (UI thread) and the worker (model thread)
// both sides import these so a typo in a message type is a compile error
// Step/TokenInfo are defined by the engine (src/hood); type-only, so the page never bundles model code
import type { DataType } from "@huggingface/transformers";
import type { PromptOptions } from "../../src/hood/prompt";
import type { Step } from "../../src/hood/session";
import type { TokenInfo } from "../../src/hood/tokenizer";

export type { Step, TokenInfo };
export type Device = "webgpu" | "wasm";

// what the page asks for when loading. empty = the worker's default (GPU if it works, else CPU)
// device set = only that device, no fallback, so a test of one setup is honest
export type LoadPrefs = { device?: Device, dtype?: DataType };

// page -> worker
export type Request =
  | { type: "load", modelName: string, prefs: LoadPrefs }
  | { type: "start", prompt: string, options: PromptOptions }   // build + encode the prompt, no forward pass yet
  | { type: "step" };                                            // exactly one forward pass

// worker -> page
// tokens: info for every id the message mentions that the page hasn't been sent yet
export type Response =
  | { type: "progress", file: string, progress: number }
  | { type: "ready", device: Device, dtype: string, vocabEntries: number }
  | { type: "prompt", input_ids: number[], tokens: Record<number, TokenInfo> }
  | { type: "step", step: Step, tokens: Record<number, TokenInfo> }
  | { type: "error", message: string };
