import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { NextResponse } from "next/server";
import { getAiCodeLabChallenge, type PythonLabResult, type PythonLabRunResponse } from "@/lib/ml/code-lab";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_CODE_LENGTH = 70_000;
const MAX_OUTPUT_LENGTH = 180_000;
const TIMEOUT_MS = 22_000;
const RESULT_PATTERN = /AI_LAB_RESULT_START\s*([\s\S]*?)\s*AI_LAB_RESULT_END/;

interface RunRequest {
  challengeId?: string;
  code?: string;
}

function trimOutput(value: string) {
  if (value.length <= MAX_OUTPUT_LENGTH) return value;
  return `${value.slice(0, MAX_OUTPUT_LENGTH)}\n...[output kısaltıldı]`;
}

function parseResult(stdout: string): PythonLabResult | undefined {
  const match = stdout.match(RESULT_PATTERN);
  if (!match?.[1]) return undefined;
  return JSON.parse(match[1]) as PythonLabResult;
}

function runPython(filePath: string, cwd: string) {
  const started = Date.now();
  return new Promise<PythonLabRunResponse>((resolve) => {
    const child = spawn("python3", ["-u", filePath], {
      cwd,
      env: {
        ...process.env,
        MPLBACKEND: "Agg",
        OMP_NUM_THREADS: "1",
        MKL_NUM_THREADS: "1",
        PYTHONUNBUFFERED: "1",
      },
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let timedOut = false;

    const timeout = windowlessSetTimeout(() => {
      timedOut = true;
      child.kill("SIGKILL");
    }, TIMEOUT_MS);

    child.stdout.on("data", (chunk: Buffer) => {
      stdout = trimOutput(stdout + chunk.toString("utf8"));
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderr = trimOutput(stderr + chunk.toString("utf8"));
    });
    child.on("error", (error) => {
      windowlessClearTimeout(timeout);
      resolve({
        ok: false,
        stdout,
        stderr,
        durationMs: Date.now() - started,
        error: error.message,
      });
    });
    child.on("close", (code) => {
      windowlessClearTimeout(timeout);
      let result: PythonLabResult | undefined;
      let parseError: string | undefined;
      try {
        result = parseResult(stdout);
      } catch (error) {
        parseError = error instanceof Error ? error.message : "Python sonucu JSON olarak okunamadı.";
      }
      resolve({
        ok: code === 0 && !timedOut && !parseError,
        stdout,
        stderr,
        durationMs: Date.now() - started,
        result,
        error: timedOut
          ? `${TIMEOUT_MS / 1000} saniye timeout.`
          : parseError ?? (code === 0 ? undefined : `Python çıkış kodu ${code}.`),
      });
    });
  });
}

const windowlessSetTimeout: typeof setTimeout = setTimeout;
const windowlessClearTimeout: typeof clearTimeout = clearTimeout;

export async function POST(request: Request) {
  let body: RunRequest;
  try {
    body = (await request.json()) as RunRequest;
  } catch {
    return NextResponse.json(
      { ok: false, stdout: "", stderr: "", durationMs: 0, error: "Geçersiz JSON isteği." },
      { status: 400 }
    );
  }

  const code = body.code ?? "";
  if (typeof code !== "string" || code.trim().length === 0) {
    return NextResponse.json(
      { ok: false, stdout: "", stderr: "", durationMs: 0, error: "Çalıştırılacak Python kodu boş." },
      { status: 400 }
    );
  }
  if (code.length > MAX_CODE_LENGTH) {
    return NextResponse.json(
      { ok: false, stdout: "", stderr: "", durationMs: 0, error: "Kod bu lab hücresi için çok uzun." },
      { status: 413 }
    );
  }

  const challenge = body.challengeId ? getAiCodeLabChallenge(body.challengeId) : undefined;
  const tempDir = await mkdtemp(path.join(tmpdir(), "ai-code-lab-"));
  const filePath = path.join(tempDir, "submission.py");

  try {
    await writeFile(filePath, code, "utf8");
    const response = await runPython(filePath, tempDir);
    return NextResponse.json({ ...response, challengeId: challenge?.id ?? body.challengeId });
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}
