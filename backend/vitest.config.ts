import { defineConfig } from "vitest/config";
import path from "path";

const backendRoot = path.resolve(import.meta.dirname);
const projectRoot = path.resolve(import.meta.dirname, "..");

export default defineConfig({
  root: backendRoot,
  resolve: {
    alias: {
      "@": path.resolve(projectRoot, "frontend", "src"),
      "@shared": path.resolve(backendRoot, "shared"),
      "@assets": path.resolve(projectRoot, "attached_assets"),
    },
  },
  test: {
    environment: "node",
    include: ["server/**/*.test.ts", "server/**/*.spec.ts"],
  },
});
