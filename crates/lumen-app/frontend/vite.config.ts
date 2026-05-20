import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import tailwindcss from "@tailwindcss/vite";

// Tauri 2 expects build output under `frontend/dist`. Dev server runs on
// :5173 (matches tauri.conf.json::build.devUrl).
// `tailwindcss()` must come BEFORE `svelte()` so utility classes inside
// .svelte files (and `:global()` rules) get processed before Svelte compiles.
export default defineConfig({
  plugins: [tailwindcss(), svelte()],
  clearScreen: false,
  server: {
    port: 5173,
    strictPort: true,
    host: "127.0.0.1",
  },
  build: {
    target: "esnext",
    outDir: "dist",
    emptyOutDir: true,
    sourcemap: false,
  },
});
