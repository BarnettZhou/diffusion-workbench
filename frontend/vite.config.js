import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const API_PROXY = {
  "/api": {
    target: "http://127.0.0.1:8188",
    changeOrigin: true,
    ws: true,
  },
};

export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    // 显式列出全部依赖,避免扫描器漏掉后运行时“发现新依赖”触发二次优化,
    // 导致 hash 变化、旧 chunk 走 transform 路径出现 Content-Length 不匹配。
    include: ["react", "react-dom", "react-dom/client", "react/jsx-dev-runtime"],
  },
  server: {
    host: "0.0.0.0",
    port: 5173,
    proxy: API_PROXY,
  },
  preview: {
    host: "0.0.0.0",
    proxy: API_PROXY,
  },
});
