import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// /api calls are proxied to Flask, so no CORS configuration is needed in development.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:5000', changeOrigin: true },
    },
  },
});
