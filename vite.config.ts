import { defineConfig } from 'vite'

export default defineConfig(({ command, isPreview }) => ({
  root: 'client',
  // GitHub Pages serves the repo at darronese.github.io/metal_detector/, so built asset links need that prefix.
  // `vite preview` must use it too (it runs as 'serve', not 'build') or it serves dist/ at / and every asset 404s.
  // `npm run dev` stays at / so localhost works as before
  base: command === 'build' || isPreview ? '/metal_detector/' : '/',
  build: {
    // root is client/, so without this the build would land in client/dist
    outDir: '../dist',
    emptyOutDir: true,
  },
  worker: { format: 'es' },
  // transformers.js ships its own onnxruntime wasm files; pre-bundling breaks how it finds them
  optimizeDeps: { exclude: ['@huggingface/transformers'] },
}))
