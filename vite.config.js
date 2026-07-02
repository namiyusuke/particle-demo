import { defineConfig } from 'vite';

export default defineConfig({
  base:"./",
  server: { open: true },
  // WebGPU はモダンブラウザ前提。トップレベル await を使うため target を上げる
  build: { target: 'esnext' },
  esbuild: { target: 'esnext' },
  // three.webgpu は巨大なので pre-bundle 対象から外すと dev 起動が安定する
  optimizeDeps: {
    exclude: ['three'],
  },
});
