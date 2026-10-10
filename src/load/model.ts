import { AutoModelForCausalLM, type ProgressCallback } from '@huggingface/transformers'

/**
 * @param model_name - full Hugging Face repo id, e.g. onnx-community/Qwen2.5-0.5B-Instruct
 * @param progress_callback - called as weights download/load (terminal now, browser later)
 */
export async function loadModel(model_name: string) {
  const model = await AutoModelForCausalLM.from_pretrained(model_name)
  return model
}
