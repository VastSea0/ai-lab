"use client";

import { AlertTriangle, CheckCircle2, Circle, Code2, Layers3, Play, RotateCcw, Sparkles, Terminal, XCircle } from "lucide-react";
import { useState, type ReactNode } from "react";
import { type PythonLabResult, type PythonLabRunResponse } from "@/lib/ml/code-lab";
import { CodeEditor } from "@/components/lab/code/CodeEditor";
import { MissionRail } from "@/components/lab/curriculum/MissionRail";
import { analyzeTorchStructure } from "@/lib/ml/live-code";
import { formatNumber } from "@/lib/ml/network";
import {
  completeCurriculumTask,
  firstAvailableCurriculumTask,
  getCurriculumTask,
  isCurriculumTaskUnlocked,
  normalizeCurriculumProgress,
} from "@/lib/ml/curriculum";
import type { CurriculumProgress, CurriculumTask, LiveCodeAnalysis } from "@/lib/ml/types";

export function AiCodeLab({
  onApplyResult,
  onLiveResult,
  preview,
  applyLabel = "Import to 3D",
  progress,
  onProgressChange,
}: {
  onApplyResult: (task: CurriculumTask, result: PythonLabResult) => void;
  onLiveResult?: (task: CurriculumTask, result: PythonLabResult, analysis: LiveCodeAnalysis) => void;
  preview: ReactNode;
  applyLabel?: string;
  progress: CurriculumProgress;
  onProgressChange: (progress: CurriculumProgress) => void;
}) {
  const safeProgress = normalizeCurriculumProgress(progress);
  const [selectedId, setSelectedId] = useState(() => firstAvailableCurriculumTask(safeProgress).id);
  const selected = getCurriculumTask(selectedId) ?? firstAvailableCurriculumTask(safeProgress);
  const [code, setCode] = useState(selected.starterCode);
  const [running, setRunning] = useState(false);
  const [response, setResponse] = useState<PythonLabRunResponse | null>(null);
  const [celebration, setCelebration] = useState<string[] | null>(null);
  const liveAnalysis = analyzeTorchStructure(code, selected.requirements);
  const liveMissionSupported = supportsLiveMission(selected);

  const publishLiveResult = (task: CurriculumTask, nextCode: string, analysis: LiveCodeAnalysis) => {
    if (!analysis.modelMeta) return;
    onLiveResult?.(
      task,
      {
        title: task.title,
        sourceCode: nextCode,
        modelMeta: analysis.modelMeta,
        metrics: [
          {
            label: "checks",
            value: `${analysis.checks.filter((check) => check.passed).length}/${analysis.checks.length}`,
          },
        ],
        notes: analysis.outputPreview ? [analysis.outputPreview] : undefined,
      },
      analysis
    );
  };

  const completeIfLivePassed = (task: CurriculumTask, analysis: LiveCodeAnalysis) => {
    if (!supportsLiveMission(task) || !analysis.passed || safeProgress.completedTaskIds.includes(task.id)) return;
    const nextProgress = completeCurriculumTask(safeProgress, task.id);
    onProgressChange(nextProgress);
    setCelebration(["Mission complete! Next lesson unlocked."]);
  };

  const selectTask = (task: CurriculumTask) => {
    if (!isCurriculumTaskUnlocked(task, safeProgress)) return;
    setSelectedId(task.id);
    setCode(task.starterCode);
    setResponse(null);
    setCelebration(null);
    publishLiveResult(task, task.starterCode, analyzeTorchStructure(task.starterCode, task.requirements));
  };

  const updateCode = (nextCode: string) => {
    const analysis = analyzeTorchStructure(nextCode, selected.requirements);
    setCode(nextCode);
    setResponse(null);
    setCelebration(null);
    publishLiveResult(selected, nextCode, analysis);
    completeIfLivePassed(selected, analysis);
  };

  const runCode = async () => {
    setRunning(true);
    setResponse(null);
    setCelebration(null);
    try {
      const result = await fetch("/api/python-lab", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ challengeId: selected.id, code }),
      });
      const payload = (await result.json()) as PythonLabRunResponse;
      setResponse(payload);
      if (payload.ok && payload.result) {
        onApplyResult(selected, payload.result);
      }
      if (payload.ok && payload.passed) {
        const nextProgress = completeCurriculumTask(safeProgress, selected.id);
        onProgressChange(nextProgress);
        setCelebration(payload.feedback.length > 0 ? payload.feedback : ["Task passed. Next step unlocked."]);
      }
    } catch (error) {
      setResponse({
        ok: false,
        stdout: "",
        stderr: "",
        durationMs: 0,
        passed: false,
        feedback: ["Python cell could not be reached."],
        error: error instanceof Error ? error.message : "Python execution failed.",
      });
    } finally {
      setRunning(false);
    }
  };

  const statusText = !liveMissionSupported
    ? "run Python to grade"
    : liveAnalysis.passed
      ? liveAnalysis.outputPreview ?? "model valid"
      : liveAnalysis.outputPreview ?? "waiting for model";
  const statusTone: "neutral" | "good" | "warning" =
    liveMissionSupported && liveAnalysis.passed ? "good" : "neutral";

  return (
    <div className="grid h-full min-h-0 grid-cols-[200px_minmax(360px,1fr)_minmax(360px,42vw)] gap-0">
      {/* Left: Mission Rail */}
      <aside className="min-h-0 overflow-hidden border-r border-[#d7dde8]">
        <MissionRail selectedTaskId={selected.id} progress={safeProgress} onSelectTask={selectTask} />
      </aside>

      {/* Center: Code Editor */}
      <section className="flex min-h-0 flex-col">
        <CodeEditor
          value={code}
          onChange={updateCode}
          onRun={runCode}
          onReset={() => updateCode(selected.starterCode)}
          running={running}
          statusText={statusText}
          statusTone={statusTone}
        />
      </section>

      {/* Right: Live Preview + Output */}
      <aside className="flex min-h-0 flex-col border-l border-[#d7dde8] bg-[#f4f7fb]">
        {/* Three.js preview */}
        <div className="min-h-0 flex-1 overflow-hidden bg-[#eef3f8]">{preview}</div>

        {/* Bottom panel: Mission + Console */}
        <div className="grid h-60 shrink-0 grid-cols-[1fr_1fr] gap-0 border-t border-[#d7dde8]">
          {/* Mission checklist */}
          <div className="min-h-0 overflow-y-auto border-r border-[#d7dde8] bg-white p-3">
            <div className="mb-2 flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.08em] text-[#64748b]">
              <Layers3 className="h-3.5 w-3.5 text-[#2563eb]" />
              Mission Checklist
            </div>
            <MissionChecklist task={selected} analysis={liveAnalysis} liveSupported={liveMissionSupported} />
            {celebration && (
              <div className="mt-2 rounded-md border border-[#bbf7d0] bg-[#f0fdf4] px-2.5 py-2 text-[11px] leading-4 text-[#166534]">
                <div className="mb-0.5 flex items-center gap-1.5 font-semibold text-[#14532d]">
                  <Sparkles className="h-3.5 w-3.5" />
                  Mission passed
                </div>
                {celebration[0]}
              </div>
            )}
          </div>

          {/* Console / Output */}
          <div className="min-h-0 overflow-y-auto bg-white p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-[10px] font-bold uppercase tracking-[0.08em] text-[#64748b]">
                <Terminal className="h-3.5 w-3.5 text-[#2563eb]" />
                Console
              </div>
              {response && (
                <button
                  type="button"
                  className="inline-flex h-6 items-center gap-1 rounded border border-[#cbd5e1] bg-white px-2 text-[10px] font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
                  onClick={() => {
                    if (response.result) onApplyResult(selected, response.result);
                  }}
                >
                  {applyLabel}
                </button>
              )}
            </div>

            {response ? (
              <ConsoleOutput response={response} />
            ) : (
              <div className="space-y-2">
                <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] px-2.5 py-2 text-[11px] leading-4 text-[#526070]">
                  <div className="flex items-center gap-1.5 font-semibold text-[#18202f]">
                    <Code2 className="h-3.5 w-3.5 text-[#64748b]" />
                    Live Preview
                  </div>
                  <div className="mt-1 text-[#94a3b8]">
                    {liveMissionSupported
                      ? liveAnalysis.outputPreview ?? "Type code to see live model structure."
                      : "Run the code to grade metrics and update the 3D preview."}
                  </div>
                </div>
                {liveMissionSupported && liveAnalysis.checks.length > 0 && (
                  <MiniChecklist checks={liveAnalysis.checks} />
                )}
              </div>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}

function supportsLiveMission(task: CurriculumTask) {
  return task.requirements.modelType === "mlp";
}

function requirementSummary(task: CurriculumTask) {
  const requirements = task.requirements;
  return [
    requirements.scoreThreshold !== undefined ? `score >= ${requirements.scoreThreshold}` : null,
    requirements.accuracyThreshold !== undefined ? `accuracy >= ${requirements.accuracyThreshold}` : null,
    requirements.lossThreshold !== undefined ? `loss <= ${requirements.lossThreshold}` : null,
    requirements.minEpochs !== undefined ? `${requirements.minEpochs}+ epochs` : null,
    requirements.clusters !== undefined ? `${requirements.clusters} clusters` : null,
    requirements.rewardThreshold !== undefined ? `reward > ${requirements.rewardThreshold}` : null,
    requirements.featureShape ? `X shape ${requirements.featureShape.join(" x ")}` : null,
    requirements.targetShape ? `y shape ${requirements.targetShape.join(" x ")}` : null,
  ].filter(Boolean) as string[];
}

function MissionChecklist({
  task,
  analysis,
  liveSupported,
}: {
  task: CurriculumTask;
  analysis: LiveCodeAnalysis;
  liveSupported: boolean;
}) {
  if (!liveSupported) {
    const requirements = requirementSummary(task);

    return (
      <div className="space-y-2">
        <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] px-2.5 py-2 text-[10px] leading-4 text-[#526070]">
          <div className="font-semibold text-[#18202f]">{task.title}</div>
          <div className="mt-1">Run the Python cell to grade this lesson and mirror modelMeta in the preview.</div>
        </div>
        {requirements.length > 0 && (
          <div className="space-y-1">
            {requirements.map((item) => (
              <div
                key={item}
                className="flex items-center gap-2 rounded-md border border-[#e2e8f0] bg-[#fbfdff] px-2 py-1.5 text-[10px] leading-[14px] text-[#526070]"
              >
                <Circle className="h-3.5 w-3.5 shrink-0 text-[#cbd5e1]" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  const passedCount = analysis.checks.filter((c) => c.passed).length;
  const total = Math.max(1, analysis.checks.length);

  return (
    <div className="space-y-2">
      <div className="h-1.5 overflow-hidden rounded-full bg-[#e2e8f0]">
        <div
          className={`h-full rounded-full transition-all ${analysis.passed ? "bg-[#059669]" : "bg-[#2563eb]"}`}
          style={{ width: `${Math.round((passedCount / total) * 100)}%` }}
        />
      </div>
      <div className="space-y-1">
        {analysis.checks.map((check) => (
          <div
            key={check.id}
            className={`flex items-start gap-2 rounded-md border px-2 py-1.5 text-[10px] leading-[14px] ${
              check.passed
                ? "border-[#bbf7d0] bg-[#f0fdf4] text-[#166534]"
                : "border-[#e2e8f0] bg-[#fbfdff] text-[#526070]"
            }`}
          >
            {check.passed ? (
              <CheckCircle2 className="mt-0.5 h-3.5 w-3.5 shrink-0" />
            ) : (
              <Circle className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[#cbd5e1]" />
            )}
            <span>
              <span className="block font-semibold text-[#18202f]">{check.label}</span>
              <span className="text-[#64748b]">{check.detail}</span>
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function MiniChecklist({ checks }: { checks: Array<{ id: string; label: string; passed: boolean }> }) {
  const passed = checks.filter((c) => c.passed).length;
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] px-2.5 py-2">
      <div className="text-[10px] font-bold uppercase tracking-[0.08em] text-[#64748b]">Checks</div>
      <div className="mt-1 text-xs font-semibold text-[#18202f]">
        {passed}/{checks.length} passed
      </div>
    </div>
  );
}

function ConsoleOutput({ response }: { response: PythonLabRunResponse }) {
  const result = response.result;
  const hasError = !response.ok || response.error;

  return (
    <div className="space-y-2">
      {/* Status badge */}
      <div
        className={`flex items-center gap-2 rounded-md border px-2.5 py-1.5 text-[11px] ${
          response.ok
            ? response.passed
              ? "border-[#bbf7d0] bg-[#f0fdf4] text-[#166534]"
              : "border-[#dbe3ee] bg-[#fbfdff] text-[#18202f]"
            : "border-[#fecaca] bg-[#fef2f2] text-[#b91c1c]"
        }`}
      >
        {response.ok ? (
          response.passed ? (
            <CheckCircle2 className="h-4 w-4 shrink-0" />
          ) : (
            <Play className="h-4 w-4 shrink-0 text-[#2563eb]" />
          )
        ) : (
          <XCircle className="h-4 w-4 shrink-0" />
        )}
        <div>
          <span className="font-semibold">
            {response.ok ? (response.passed ? "Passed" : "Ran") : "Error"}
          </span>
          <span className="text-[#64748b]"> · {formatNumber(response.durationMs / 1000, 2)}s</span>
        </div>
      </div>

      {/* Metrics */}
      {result && <ResultMetrics result={result} />}

      {/* Feedback */}
      {response.feedback.length > 0 && (
        <div
          className={`rounded-md border p-2 text-[10px] leading-4 ${
            response.passed
              ? "border-[#bbf7d0] bg-[#f0fdf4] text-[#166534]"
              : "border-[#fed7aa] bg-[#fff7ed] text-[#b45309]"
          }`}
        >
          <div className="mb-1 text-[10px] font-bold uppercase tracking-[0.08em] text-[#64748b]">Grading</div>
          {response.feedback.map((item, i) => (
            <div key={i}>{item}</div>
          ))}
        </div>
      )}

      {/* Stdout/Stderr */}
      {(response.stdout || response.stderr) && (
        <div className="relative">
          <pre className="max-h-32 overflow-auto rounded-md bg-[#0f172a] p-2 text-[10px] leading-[14px] text-[#e2e8f0]">
            {response.stdout}
            {response.stderr && (
              <>
                {"\n--- stderr ---\n"}
                <span className="text-[#f87171]">{response.stderr}</span>
              </>
            )}
          </pre>
        </div>
      )}

      {hasError && response.error && (
        <div className="flex items-start gap-2 rounded-md border border-[#fecaca] bg-[#fef2f2] px-2.5 py-2 text-[10px] leading-4 text-[#b91c1c]">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
          {response.error}
        </div>
      )}
    </div>
  );
}

function ResultMetrics({ result }: { result: PythonLabResult }) {
  const metrics = [
    result.epochs !== undefined ? { label: "epoch", value: result.epochs } : null,
    result.accuracy !== undefined ? { label: "accuracy", value: `${formatNumber(result.accuracy * 100, 1)}%` } : null,
    result.loss !== undefined ? { label: "loss", value: formatNumber(result.loss, 6) } : null,
    result.val_loss !== undefined ? { label: "val loss", value: formatNumber(result.val_loss, 6) } : null,
    result.r2_score !== undefined ? { label: "r2", value: formatNumber(result.r2_score, 4) } : null,
    result.inertia !== undefined ? { label: "inertia", value: formatNumber(result.inertia, 4) } : null,
    result.n_clusters !== undefined ? { label: "clusters", value: result.n_clusters } : null,
    result.avg_reward_last10 !== undefined ? { label: "reward", value: formatNumber(result.avg_reward_last10, 3) } : null,
    result.losses?.length ? { label: "last loss", value: formatNumber(result.losses.at(-1) ?? 0, 6) } : null,
    ...(result.metrics ?? []),
  ].filter(Boolean) as Array<{ label: string; value: string | number }>;

  if (metrics.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-1.5">
      {metrics.slice(0, 4).map((metric) => (
        <div key={metric.label} className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] px-2 py-1.5">
          <div className="truncate text-[9px] font-bold uppercase tracking-[0.08em] text-[#64748b]">{metric.label}</div>
          <div className="mt-0.5 truncate text-xs font-semibold text-[#18202f]">{metric.value}</div>
        </div>
      ))}
    </div>
  );
}
