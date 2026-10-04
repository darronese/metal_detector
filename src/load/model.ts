import { AutoModel } from '@huggingface/transformers'

async function loadModel(model_name: string) {
  const model = AutoModel.from_pretrained(model_name)
}
