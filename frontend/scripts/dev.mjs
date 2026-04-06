import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const frontendDir = path.resolve(__dirname, "..");
const projectRoot = path.resolve(frontendDir, "..");

const isWindows = process.platform === "win32";
const pythonPath = isWindows
  ? path.join(projectRoot, ".venv", "Scripts", "python.exe")
  : path.join(projectRoot, ".venv", "bin", "python");
const pythonCommand = existsSync(pythonPath) ? pythonPath : "python";

const childProcesses = [];

function startProcess(command, args, cwd) {
  const child = spawn(command, args, {
    cwd,
    stdio: "inherit",
    shell: isWindows,
  });

  child.on("error", (error) => {
    console.error(`Failed to start: ${command}`);
    console.error(error);
    process.exitCode = 1;
  });

  childProcesses.push(child);
  return child;
}

const backend = startProcess(
  `"${pythonCommand}" -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000`,
  [],
  projectRoot
);

const vite = startProcess("npm run dev:vite", [], frontendDir);

function shutdown() {
  for (const child of childProcesses) {
    if (!child.killed) {
      child.kill();
    }
  }
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);

backend.on("exit", (code) => {
  if (code && code !== 0) {
    process.exitCode = code;
  }
});

vite.on("exit", (code) => {
  shutdown();
  process.exit(code ?? 0);
});
