import { defineConfig } from 'vite'

export default defineConfig({
  root: 'client',
  worker: { format: 'es' },
  // transformers.js ships its own onnxruntime wasm files; pre-bundling breaks how it finds them
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
})
