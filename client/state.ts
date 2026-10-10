import type { Device, Response, Step, TokenInfo } from './engine/messages'

// everything the page knows, and the only functions allowed to change it.
// no DOM in here: ui/ reads this, main.ts calls these

export const MAX_STEPS = 200

export type State = {
  model: { status: 'idle' | 'loading' | 'ready', device?: Device, dtype?: string }
  progress: { file: string, progress: number } | null
  status: string
  tokens: Map<number, TokenInfo>   // every token id the worker has described so far
  prompt_ids: number[]
  steps: Step[]
  view: number                     // index into steps being shown, -1 = prompt only
  selected: number | null          // clicked column, null = follow the last position
  busy: boolean                    // waiting on the worker (encode or a forward pass)
  running: boolean                 // generating: keep asking for passes until a stop token or Stop
}

/** what the ui should do after a message lands */
export type Effect = { enter: boolean, follow: boolean, next: boolean }
const NONE: Effect = { enter: false, follow: false, next: false }

export function createState(): State {
  return {
    model: { status: 'idle' }, progress: null, status: '',
    tokens: new Map(), prompt_ids: [], steps: [], view: -1, selected: null,
    busy: false, running: false,
  }
}

// ---------- reads ----------

export const currentStep = (s: State) => s.view >= 0 ? s.steps[s.view] : null
export const finished = (s: State) => s.steps.length >= MAX_STEPS || !!s.steps.at(-1)?.eos

/** the column the inspector shows: the clicked one, or the last position */
export function selectedColumn(s: State) {
  const n = (currentStep(s)?.input_ids ?? s.prompt_ids).length
  return s.selected ?? Math.max(n - 1, 0)
}

export const canGenerate = (s: State) => s.model.status === 'ready' && !s.busy && !s.running

// ---------- changes the user asks for ----------

export function beginLoad(s: State) {
  s.model = { status: 'loading' }
  s.status = ''
}

/** new prompt: forget the old run, encode, then passes follow automatically */
export function beginGenerate(s: State) {
  s.prompt_ids = []
  s.steps = []
  s.view = -1
  s.selected = null
  s.running = true
  s.busy = true
  s.status = 'encoding prompt…'
}

export function beginPass(s: State) {
  s.busy = true
  s.status = `forward pass ${s.steps.length + 1}…`
}

/** the pass in flight still lands, nothing new is requested after it */
export function stopGenerating(s: State) {
  s.running = false
}

export function select(s: State, col: number) {
  s.selected = col
}

/** jump to a pass in history, clamped. history opens once the run is over */
export function viewStep(s: State, index: number) {
  if (!s.steps.length || s.running) return
  s.view = Math.max(0, Math.min(s.steps.length - 1, index))
  s.selected = null
}

// ---------- changes the worker reports ----------

export function receive(s: State, msg: Response): Effect {
  if ('tokens' in msg) for (const [id, info] of Object.entries(msg.tokens)) s.tokens.set(Number(id), info)

  switch (msg.type) {
    case 'progress':
      s.progress = { file: msg.file, progress: msg.progress }
      return NONE
    case 'ready':
      s.model = { status: 'ready', device: msg.device, dtype: msg.dtype }
      s.progress = null
      s.status = `ready. ${msg.vocab_entries.toLocaleString('en-US')} tokens in the vocabulary`
      return NONE
    case 'prompt':
      s.prompt_ids = msg.input_ids
      s.busy = false
      s.status = `prompt encoded: ${msg.input_ids.length} tokens`
      return { enter: false, follow: true, next: s.running }
    case 'step': {
      s.steps.push(msg.step)
      s.busy = false
      if (finished(s)) s.running = false
      // still generating: just count. the board stays hidden until the run is over
      if (s.running) return { enter: false, follow: false, next: true }

      // run over (stop token, limit, or Stop): start at the very first pass, Next walks forward
      s.view = 0
      s.selected = null
      const n = s.steps.length
      const done = msg.step.eos ? `done: stop token after ${n} passes`
        : n >= MAX_STEPS ? `done: hit the ${MAX_STEPS}-pass limit without a stop token`
        : `stopped after ${n} passes`
      s.status = `${done}. Showing pass 1: press Next (or →) to step through.`
      return { enter: true, follow: true, next: false }
    }
    case 'error':
      s.busy = s.running = false
      if (s.model.status === 'loading') s.model = { status: 'idle' }
      s.progress = null
      s.status = `error: ${msg.message}`
      return NONE
  }
}
