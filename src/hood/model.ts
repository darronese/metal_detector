import { AutoModelForCausalLM, type PretrainedModelOptions } from "@huggingface/transformers";

/**
 * @param modelName - full Hugging Face repo id, e.g. onnx-community/Qwen2.5-0.5B-Instruct
 * @param options - device/dtype/progress_callback; omit in the terminal (cpu, fp32), set in the browser (webgpu, q4)
 */
export async function loadModel(modelName: string, options: PretrainedModelOptions = {}) {
  const model = await AutoModelForCausalLM.from_pretrained(modelName, options);
  return model;
}
