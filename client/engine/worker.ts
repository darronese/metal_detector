import type { DataType } from '@huggingface/transformers'
import { loadModel } from '../../src/hood/model'
import { loadTokenizer, type Tokenizer } from '../../src/hood/tokenizer'
import { build_prompt } from '../../src/hood/prompt'
import { Session } from '../../src/hood/session'
import type { Device, Request, Response, TokenInfo } from './messages'

// the browser's "backend": runs the engine (src/hood) on a background thread so the page never freezes.
// it only does browser-specific things: pick a device, and turn messages into engine calls and back

// q4 = 4-bit weights, fp32 math. not q4f16/fp16: fp16 math on webgpu gives garbage for Qwen2.5 (fine on cpu)
// both keep fp32 logits, which is what we're inspecting
const DTYPE: Record<Device, DataType> = { webgpu: 'q4', wasm: 'q8' }

let tok: Tokenizer
let session: Session
let sent = new Set<number>()   // token ids the page already has info for

const send = (msg: Response) => postMessage(msg)

async function hasWebGPU() {
  // navigator.gpu can exist but hand back no adapter (blocklisted gpu, flag off)
  const gpu = (navigator as any).gpu
  return !!gpu && !!(await gpu.requestAdapter())
}

async function load(model_name: string) {
  const devices: Device[] = (await hasWebGPU()) ? ['webgpu', 'wasm'] : ['wasm']
  let device = devices[0]
  let model
  for (device of devices) {
    try {
      model = await loadModel(model_name, {
        device,
        dtype: DTYPE[device],
        progress_callback: (info) => {
          if (info.status === 'progress') send({ type: 'progress', file: info.file, progress: info.progress })
        },
      })
      break
    } catch (err) {
      // webgpu can fail at load (out of memory, unsupported op) -> try the next device
      if (device === devices.at(-1)) throw err
    }
  }
  tok = await loadTokenizer(model_name)
  session = new Session(model!, tok)
  sent = new Set()
  send({ type: 'ready', device, dtype: DTYPE[device], vocab_entries: tok.vocab_entries })
}

/** info for ids the page hasn't seen yet, so each one crosses the thread boundary once */
function tokens(ids: Iterable<number>) {
  const out: Record<number, TokenInfo> = {}
  for (const id of ids) {
    if (sent.has(id)) continue
    sent.add(id)
    out[id] = tok.token_info(id)
  }
  return out
}

onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data
  try {
    if (msg.type === 'load') await load(msg.model_name)
    else if (msg.type === 'start') {
      const input_ids = session.start(build_prompt(tok, msg.prompt, msg.options))
      send({ type: 'prompt', input_ids, tokens: tokens(input_ids) })
    }
    else if (msg.type === 'step') {
      const step = await session.next()
      const mentioned = [...step.rows, step.winner, ...step.columns.flatMap(c => c.top.map(t => t.id))]
      send({ type: 'step', step, tokens: tokens(mentioned) })
    }
  } catch (err) {
    send({ type: 'error', message: String(err) })
  }
}
