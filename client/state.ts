import type { PromptOptions } from "../src/hood/prompt";
import type { Device, Response, Step, TokenInfo } from "./engine/messages";

// everything the page knows, and the only functions allowed to change it.
// no DOM in here: ui/ reads this, main.ts calls these

export const MAX_STEPS = 200;

export type State = {
  // id = the Hugging Face repo that is loaded (or loading)
  model: { status: "idle" | "loading" | "ready", id?: string, device?: Device, dtype?: string }
  // a Generate that is waiting for its model to finish loading
  pending: Run | null
  progress: { file: string, progress: number } | null
  status: string
  // every token id the worker has described so far
  tokens: Map<number, TokenInfo>
  promptIds: number[]
  steps: Step[]
  // index into steps being shown, -1 = prompt only
  view: number
  // clicked column, null = follow the last position
  selected: number | null
  // waiting on the worker (encode or a forward pass)
  busy: boolean
  // generating: keep asking for passes until a stop token or Stop
  running: boolean
};

export type Run = { prompt: string, options: PromptOptions };

/** what the ui should do after a message lands. start = a run that was waiting on the load */
export type Effect = { enter: boolean, follow: boolean, next: boolean, start: Run | null };
const NONE: Effect = { enter: false, follow: false, next: false, start: null };

export function createState(): State {
  return {
    model: { status: "idle" }, pending: null, progress: null, status: "",
    tokens: new Map(), promptIds: [], steps: [], view: -1, selected: null,
    busy: false, running: false,
  };
}

// ---------- reads ----------

export const currentStep = (s: State) => s.view >= 0 ? s.steps[s.view] : null;
export const finished = (s: State) => s.steps.length >= MAX_STEPS || !!s.steps.at(-1)?.eos;

/** the column the inspector shows: the clicked one, or the last position */
export function selectedColumn(s: State) {
  const n = (currentStep(s)?.input_ids ?? s.promptIds).length;
  return s.selected ?? Math.max(n - 1, 0);
}

export const canGenerate = (s: State) => s.model.status !== "loading" && !s.busy && !s.running;

/** Generate loads first when nothing is loaded yet or the dropdown picked a different model */
export const needsLoad = (s: State, id: string) => s.model.status !== "ready" || s.model.id !== id;

// ---------- changes the user asks for ----------

/**
 * @param id - the model to load
 * @param run - the Generate to start once it's ready
 */
export function beginLoad(s: State, id: string, run: Run) {
  s.model = { status: "loading", id };
  s.pending = run;
  s.status = "";
}

/** new prompt: forget the old run, encode, then passes follow automatically */
export function beginGenerate(s: State) {
  s.promptIds = [];
  s.steps = [];
  s.view = -1;
  s.selected = null;
  s.running = true;
  s.busy = true;
  s.status = "encoding prompt…";
}

export function beginPass(s: State) {
  s.busy = true;
  s.status = `forward pass ${s.steps.length + 1}…`;
}

/** the pass in flight still lands, nothing new is requested after it */
export function stopGenerating(s: State) {
  s.running = false;
}

export function select(s: State, col: number) {
  s.selected = col;
}

/**
 * jump to a pass in history, clamped. history opens once the run is over
 *
 * @returns false when there is no history to move through (so the caller can leave keys alone)
 */
export function viewStep(s: State, index: number): boolean {
  if (!s.steps.length || s.running) return false;
  s.view = Math.max(0, Math.min(s.steps.length - 1, index));
  s.selected = null;
  return true;
}

/**
 * every way a run ends (stop token, limit, Stop, error) lands here: open the passes that finished at pass 1
 *
 * @param why - first half of the status line
 */
function endRun(s: State, why: string): Effect {
  s.running = s.busy = false;
  if (!s.steps.length) {
    s.status = why;
    return NONE;
  }
  s.view = 0;
  s.selected = null;
  s.status = `${why}. Showing pass 1 of ${s.steps.length}: press Next (or →) to step through.`;
  return { ...NONE, enter: true, follow: true };
}

// ---------- changes the worker reports ----------

export function receive(s: State, msg: Response): Effect {
  if ("tokens" in msg) for (const [id, info] of Object.entries(msg.tokens)) s.tokens.set(Number(id), info);

  switch (msg.type) {
    case "progress":
      s.progress = { file: msg.file, progress: msg.progress };
      return NONE;
    case "ready": {
      s.model = { status: "ready", id: s.model.id, device: msg.device, dtype: msg.dtype };
      s.progress = null;
      s.status = `model ready: ${msg.vocabEntries.toLocaleString("en-US")} tokens in its vocabulary`;
      const start = s.pending;
      s.pending = null;
      return { ...NONE, start };
    }
    case "prompt":
      s.promptIds = msg.input_ids;
      s.busy = false;
      s.status = `prompt encoded: ${msg.input_ids.length} tokens`;
      return { ...NONE, follow: true, next: s.running };
    case "step": {
      s.steps.push(msg.step);
      s.busy = false;
      if (finished(s)) s.running = false;
      // still generating: just count. the board stays hidden until the run is over
      if (s.running) return { ...NONE, next: true };

      // run over (stop token, limit, or Stop)
      const n = s.steps.length;
      return endRun(s, msg.step.eos ? `done: stop token after ${n} passes`
        : n >= MAX_STEPS ? `done: hit the ${MAX_STEPS}-pass limit without a stop token`
        : `stopped after ${n} passes`);
    }
    case "error":
      if (s.model.status === "loading") s.model = { status: "idle" };
      s.pending = null;
      s.progress = null;
      // a pass can fail mid-run (GPU out of memory, lost device): keep the passes that did finish
      return endRun(s, `error: ${msg.message}`);
  }
}
