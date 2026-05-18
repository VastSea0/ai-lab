import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { NextResponse } from "next/server";
import { getAiCodeLabChallenge, type PythonLabResult, type PythonLabRunResponse } from "@/lib/ml/code-lab";
import { getCurriculumTask } from "@/lib/ml/curriculum";
import { analyzeTorchStructure } from "@/lib/ml/live-code";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_CODE_LENGTH = 70_000;
const MAX_OUTPUT_LENGTH = 180_000;
const TIMEOUT_MS = 60_000;
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
        passed: false,
        feedback: ["Python process could not be started."],
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
        modelMeta: result?.modelMeta,
        passed: false,
        feedback: [],
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
      {
        ok: false,
        stdout: "",
        stderr: "",
        durationMs: 0,
        passed: false,
        feedback: ["Request JSON could not be parsed."],
        error: "Geçersiz JSON isteği.",
      },
      { status: 400 }
    );
  }

  const code = body.code ?? "";
  if (typeof code !== "string" || code.trim().length === 0) {
    return NextResponse.json(
      {
        ok: false,
        stdout: "",
        stderr: "",
        durationMs: 0,
        passed: false,
        feedback: ["Python code is empty."],
        error: "Çalıştırılacak Python kodu boş.",
      },
      { status: 400 }
    );
  }
  if (code.length > MAX_CODE_LENGTH) {
    return NextResponse.json(
      {
        ok: false,
        stdout: "",
        stderr: "",
        durationMs: 0,
        passed: false,
        feedback: ["Python code is longer than this lab cell allows."],
        error: "Kod bu lab hücresi için çok uzun.",
      },
      { status: 413 }
    );
  }

  const challenge = body.challengeId ? getAiCodeLabChallenge(body.challengeId) : undefined;
  const curriculumTask = body.challengeId ? getCurriculumTask(body.challengeId) : undefined;
  const tempDir = await mkdtemp(path.join(tmpdir(), "ai-code-lab-"));
  const filePath = path.join(tempDir, "submission.py");

  try {
    await writeFile(filePath, code, "utf8");
    const response = await runPython(filePath, tempDir);
    const challengeId = curriculumTask?.id ?? challenge?.id ?? body.challengeId;
    const liveAnalysis = curriculumTask ? analyzeTorchStructure(code, curriculumTask.requirements) : null;
    const result: PythonLabResult | undefined =
      response.result
        ? { ...response.result, sourceCode: code }
        : response.ok && liveAnalysis?.modelMeta
          ? {
              title: curriculumTask?.title,
              sourceCode: code,
              modelMeta: liveAnalysis.modelMeta,
              metrics: [{ label: "live checks", value: `${liveAnalysis.checks.filter((check) => check.passed).length}/${liveAnalysis.checks.length}` }],
              notes: [liveAnalysis.outputPreview ?? "Model structure parsed from code."],
            }
          : undefined;
    const responseWithResult = {
      ...response,
      result,
      modelMeta: result?.modelMeta,
    };

    if (!curriculumTask) {
      return NextResponse.json({
        ...responseWithResult,
        challengeId,
      });
    }

    if (!responseWithResult.ok || !responseWithResult.result) {
      return NextResponse.json({
        ...responseWithResult,
        challengeId,
        passed: false,
        feedback: responseWithResult.error
          ? [`Run did not complete cleanly: ${responseWithResult.error}`]
          : ["No JSON result was emitted between the AI_LAB_RESULT markers."],
      });
    }

    const verification = curriculumTask.verify({
      ...responseWithResult,
      challengeId,
      modelMeta: responseWithResult.result.modelMeta,
    });

    return NextResponse.json({
      ...responseWithResult,
      challengeId,
      passed: verification.passed,
      feedback: verification.feedback,
    });
  } finally {
    await rm(tempDir, { recursive: true, force: true });
  }
}
