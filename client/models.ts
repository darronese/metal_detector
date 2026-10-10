// the models the page offers. mb = first-load download for each device's dtype
// (webgpu loads q4, wasm loads q8: see DTYPE in engine/worker.ts). the browser caches it after that

export type ModelChoice = {
  // Hugging Face repo id, what loadModel/loadTokenizer get
  id: string,
  name: string,
  note: string,
  mb: { webgpu: number, wasm: number },
}

export const MODELS: ModelChoice[] = [
  { id: 'HuggingFaceTB/SmolLM2-135M-Instruct', name: 'SmolLM2 135M', note: 'small and fast, makes mistakes', mb: { webgpu: 182, wasm: 137 } },
  { id: 'onnx-community/Qwen2.5-0.5B-Instruct', name: 'Qwen2.5 0.5B', note: 'smarter, much bigger download', mb: { webgpu: 786, wasm: 512 } },
]
