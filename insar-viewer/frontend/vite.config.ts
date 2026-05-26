import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': 'http://127.0.0.1:8050',
      '/overlay': 'http://127.0.0.1:8050',
    },
  },
  optimizeDeps: {
    include: ['react-plotly.js', 'plotly.js'],
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
  },
})
