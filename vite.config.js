import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  base: '/MarineAdventure/',
  plugins: [vue()],
  server: {
    port: 9205
  },
  build: {
    chunkSizeWarningLimit: 1600,
    rollupOptions: {
      output: {
        manualChunks: {
          'mediapipe-pose': ['@mediapipe/pose'],
          'tensorflow': ['@tensorflow/tfjs-core', '@tensorflow/tfjs-backend-webgl']
        }
      }
    }
  },
  optimizeDeps: {
    include: ['@mediapipe/pose', '@tensorflow/tfjs-core', '@tensorflow/tfjs-backend-webgl']
  }
})