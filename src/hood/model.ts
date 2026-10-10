import { AutoModelForCausalLM, type PretrainedModelOptions } from '@huggingface/transformers'

/**
 * @param model_name - full Hugging Face repo id, e.g. onnx-community/Qwen2.5-0.5B-Instruct
 * @param options - device/dtype/progress_callback; omit in the terminal (cpu, fp32), set in the browser (webgpu, q4)
 */
export async function loadModel(model_name: string, options: PretrainedModelOptions = {}) {
  const model = await AutoModelForCausalLM.from_pretrained(model_name, options)
  return model
}
