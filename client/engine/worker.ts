import type { DataType } from "@huggingface/transformers";
import { loadModel } from "../../src/hood/model";
import { loadTokenizer, type Tokenizer } from "../../src/hood/tokenizer";
import { buildPrompt } from "../../src/hood/prompt";
import { Session } from "../../src/hood/session";
import type { Device, LoadPrefs, Request, Response, TokenInfo } from "./messages";

// the browser's "backend": runs the engine (src/hood) on a background thread so the page never freezes.
// it only does browser-specific things: pick a device, and turn messages into engine calls and back

// q4 = 4-bit weights, fp32 math. not q4f16/fp16: fp16 math on webgpu gives garbage for Qwen2.5 (fine on cpu)
// both keep fp32 logits, which is what we're inspecting
const DTYPE: Record<Device, DataType> = { webgpu: "q4", wasm: "q8" };

let tok: Tokenizer;
let session: Session;
let sent = new Set<number>();   // token ids the page already has info for

const send = (msg: Response) => postMessage(msg);

async function hasWebGPU() {
  // navigator.gpu can exist but hand back no adapter (blocklisted gpu, flag off)
  const gpu = (navigator as any).gpu;
  return !!gpu && !!(await gpu.requestAdapter());
}

/**
 * @param modelName - Hugging Face repo id
 * @param prefs - device/dtype the page asked for (phone -> cpu, or a ?device= / ?dtype= test link)
 */
async function load(modelName: string, prefs: LoadPrefs) {
  // asked for one device: use only that. otherwise GPU first, CPU if the GPU is missing or fails
  const devices: Device[] = prefs.device ? [prefs.device] : (await hasWebGPU()) ? ["webgpu", "wasm"] : ["wasm"];
  const dtypeFor = (d: Device) => prefs.dtype ?? DTYPE[d];
  let device = devices[0];
  let model;
  for (device of devices) {
    try {
      model = await loadModel(modelName, {
        device,
        dtype: dtypeFor(device),
        progress_callback: (info) => {
          if (info.status === "progress") send({ type: "progress", file: info.file, progress: info.progress });
        },
      });
      break;
    } catch (err) {
      // webgpu can fail at load (out of memory, unsupported op) -> try the next device
      if (device === devices.at(-1)) throw err;
    }
  }
  tok = await loadTokenizer(modelName);
  session = new Session(model!, tok);
  sent = new Set();
  send({ type: "ready", device, dtype: dtypeFor(device), vocabEntries: tok.vocabEntries });
}

/** info for ids the page hasn't seen yet, so each one crosses the thread boundary once */
function tokens(ids: Iterable<number>) {
  const out: Record<number, TokenInfo> = {};
  for (const id of ids) {
    if (sent.has(id)) continue;
    sent.add(id);
    out[id] = tok.tokenInfo(id);
  }
  return out;
}

onmessage = async (e: MessageEvent<Request>) => {
  const msg = e.data;
  try {
    if (msg.type === "load") await load(msg.modelName, msg.prefs);
    else if (msg.type === "start") {
      const input_ids = session.start(buildPrompt(tok, msg.prompt, msg.options));
      send({ type: "prompt", input_ids, tokens: tokens(input_ids) });
    }
    else if (msg.type === "step") {
      const step = await session.next();
      const mentioned = [...step.rows, step.winner, ...step.columns.flatMap(c => c.top.map(t => t.id))];
      send({ type: "step", step, tokens: tokens(mentioned) });
    }
  } catch (err) {
    send({ type: "error", message: String(err) });
  }
};
