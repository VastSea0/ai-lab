"use client";

import {
  AlertTriangle,
  CheckCircle2,
  Code2,
  Circle,
  FileCode2,
  Layers3,
  Play,
  RotateCcw,
  Sparkles,
  UploadCloud,
} from "lucide-react";
import { useMemo, useState, type ReactNode } from "react";
import {
  type PythonLabResult,
  type PythonLabRunResponse,
} from "@/lib/ml/code-lab";
import { CurriculumPanel } from "@/components/lab/curriculum/CurriculumPanel";
import { analyzeTorchStructure } from "@/lib/ml/live-code";
import { formatNumber } from "@/lib/ml/network";
import { CompactPanel, PanelTabs } from "@/components/lab/ui/Workbench";
import {
  completeCurriculumTask,
  firstAvailableCurriculumTask,
  getCurriculumTask,
  isCurriculumTaskUnlocked,
  normalizeCurriculumProgress,
} from "@/lib/ml/curriculum";
import type { CurriculumProgress, CurriculumTask, LiveCodeAnalysis, ModelMetaLayer } from "@/lib/ml/types";

export function AiCodeLab({
  onApplyResult,
  onLiveResult,
  preview,
  applyLabel = "Ağa Aktar",
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
  const [editorScrollTop, setEditorScrollTop] = useState(0);
  const liveAnalysis = analyzeTorchStructure(code, selected.requirements);
  const lineNumbers = useMemo(
    () => Array.from({ length: Math.max(1, code.split("\n").length) }, (_, index) => index + 1),
    [code]
  );

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
    if (!analysis.passed || safeProgress.completedTaskIds.includes(task.id)) return;
    const nextProgress = completeCurriculumTask(safeProgress, task.id);
    onProgressChange(nextProgress);
    setCelebration(["Mission complete. Next task unlocked."]);
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
        error: error instanceof Error ? error.message : "Python hücresi çalıştırılamadı.",
      });
    } finally {
      setRunning(false);
    }
  };

  const canApply =
    Boolean(response?.result?.modelMeta) ||
    Boolean(response?.result?.layers?.length && response.result.weights?.length && response.result.biases?.length);

  return (
    <div className="grid h-full min-h-0 grid-cols-[270px_minmax(430px,1fr)_minmax(360px,40vw)] gap-px bg-[#d7dde8]">
      <aside className="min-h-0 bg-white p-3">
        <CurriculumPanel selectedTaskId={selected.id} progress={safeProgress} onSelectTask={selectTask} compact />
      </aside>

      <section className="flex min-h-0 flex-col bg-[#0b1020]">
        <div className="flex h-11 shrink-0 items-center justify-between border-b border-[#1e293b] bg-[#111827] px-3">
          <div className="flex min-w-0 items-center gap-2">
            <div className="inline-flex h-7 items-center gap-2 rounded-md bg-[#0f172a] px-3 text-xs font-semibold text-[#e2e8f0]">
              <FileCode2 className="h-4 w-4 text-[#60a5fa]" />
              main.py
            </div>
            <span className="hidden truncate text-[11px] text-[#94a3b8] xl:inline">{selected.title}</span>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              className="inline-flex h-7 items-center gap-1.5 rounded border border-[#334155] bg-[#0f172a] px-2.5 text-[11px] font-semibold text-[#cbd5e1] hover:border-[#60a5fa] hover:text-white"
              onClick={() => updateCode(selected.starterCode)}
            >
              <RotateCcw className="h-3.5 w-3.5" />
              Reset
            </button>
            <button
              type="button"
              className="inline-flex h-7 items-center gap-1.5 rounded bg-[#2563eb] px-3 text-[11px] font-semibold text-white hover:bg-[#1d4ed8] disabled:cursor-wait disabled:opacity-60"
              disabled={running}
              onClick={runCode}
            >
              <Play className="h-3.5 w-3.5" />
              {running ? "Running" : "Run"}
            </button>
          </div>
        </div>

        <div className="border-b border-[#1e293b] bg-[#0f172a] px-4 py-3">
          <div className="flex items-start justify-between gap-3">
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold text-white">{selected.title}</div>
              <div className="mt-1 line-clamp-2 text-xs leading-5 text-[#94a3b8]">{selected.description}</div>
            </div>
            <span className={`shrink-0 rounded border px-2 py-1 text-[10px] font-bold uppercase tracking-[0.08em] ${
              liveAnalysis.passed ? "border-[#34d399] bg-[#064e3b] text-[#a7f3d0]" : "border-[#334155] bg-[#111827] text-[#94a3b8]"
            }`}>
              {liveAnalysis.passed ? "Complete" : "Live Parse"}
            </span>
          </div>
        </div>

        <div className="grid min-h-0 flex-1 grid-cols-[48px_minmax(0,1fr)] bg-[#0b1020]">
          <div className="relative overflow-hidden border-r border-[#1e293b] bg-[#0f172a] py-3 text-right font-mono text-[12px] leading-5 text-[#64748b]">
            <div style={{ transform: `translateY(-${editorScrollTop}px)` }}>
              {lineNumbers.map((line) => (
                <div key={line} className="h-5 pr-3">
                  {line}
                </div>
              ))}
            </div>
          </div>
          <textarea
            className="h-full w-full resize-none overflow-auto border-0 bg-[#0b1020] px-4 py-3 font-mono text-[12px] leading-5 text-[#dbeafe] caret-[#60a5fa] outline-none selection:bg-[#2563eb]/40"
            value={code}
            onChange={(event) => updateCode(event.target.value)}
            onScroll={(event) => setEditorScrollTop(event.currentTarget.scrollTop)}
            spellCheck={false}
            aria-label="Python challenge kodu"
          />
        </div>

        <div className="flex h-8 shrink-0 items-center justify-between border-t border-[#1e293b] bg-[#111827] px-3 text-[11px] text-[#94a3b8]">
          <div className="flex items-center gap-3">
            <span>Python</span>
            <span>PyTorch</span>
            <span>{lineNumbers.length} lines</span>
          </div>
          <div className="truncate">
            {liveAnalysis.outputPreview ?? "waiting for model"}
          </div>
        </div>
      </section>

      <aside className="flex min-h-0 flex-col bg-[#eef3f8]">
        <div className="min-h-0 flex-1 overflow-hidden">{preview}</div>
        <div className="grid h-[300px] shrink-0 grid-cols-2 gap-px border-t border-[#d7dde8] bg-[#d7dde8]">
          <div className="min-h-0 overflow-y-auto bg-white p-3">
            <LiveMissionPanel analysis={liveAnalysis} />
            {celebration && (
              <div className="mt-3 rounded-md border border-[#bbf7d0] bg-[#f0fdf4] px-3 py-2 text-[11px] leading-5 text-[#166534]">
                <div className="mb-1 flex items-center gap-1.5 font-semibold text-[#14532d]">
                  <Sparkles className="h-3.5 w-3.5" />
                  Mission passed
                </div>
                {celebration[0]}
              </div>
            )}
          </div>
          <div className="min-h-0 overflow-y-auto bg-white p-3">
            <div className="mb-2 flex items-center justify-between gap-2">
              <div className="flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                <Code2 className="h-4 w-4 text-[#2563eb]" />
                Output
              </div>
              <button
                type="button"
                className="inline-flex h-7 items-center gap-1 rounded border border-[#cbd5e1] bg-white px-2 text-[11px] font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb] disabled:cursor-not-allowed disabled:opacity-50"
                disabled={!canApply || !response?.result}
                onClick={() => {
                  if (response?.result) onApplyResult(selected, response.result);
                }}
              >
                <UploadCloud className="h-3.5 w-3.5" />
                {applyLabel}
              </button>
            </div>
            {response ? (
              <RunResult response={response} challenge={selected} />
            ) : (
              <pre className="min-h-28 rounded-md bg-[#0f172a] p-3 text-[11px] leading-5 text-[#e2e8f0]">
                {liveAnalysis.outputPreview ?? "Run the code to see stdout."}
              </pre>
            )}
          </div>
        </div>
      </aside>
    </div>
  );
}

function LiveMissionPanel({ analysis }: { analysis: LiveCodeAnalysis }) {
  const passedCount = analysis.checks.filter((check) => check.passed).length;
  const total = Math.max(1, analysis.checks.length);

  return (
    <CompactPanel title="Mission" icon={<Layers3 className="h-4 w-4" />} bodyClassName="space-y-3">
      <div className="h-2 overflow-hidden rounded-full bg-[#e2e8f0]">
        <div
          className={`h-full rounded-full ${analysis.passed ? "bg-[#059669]" : "bg-[#2563eb]"}`}
          style={{ width: `${Math.round((passedCount / total) * 100)}%` }}
        />
      </div>
      <div className="grid gap-1.5">
        {analysis.checks.map((check) => (
          <div
            key={check.id}
            className={`grid grid-cols-[18px_1fr] gap-2 rounded-md border px-2.5 py-2 text-[11px] leading-4 ${
              check.passed
                ? "border-[#bbf7d0] bg-[#f0fdf4] text-[#166534]"
                : "border-[#e2e8f0] bg-[#fbfdff] text-[#526070]"
            }`}
          >
            {check.passed ? <CheckCircle2 className="mt-0.5 h-4 w-4" /> : <Circle className="mt-0.5 h-4 w-4" />}
            <span>
              <span className="block font-semibold text-[#18202f]">{check.label}</span>
              <span>{check.detail}</span>
            </span>
          </div>
        ))}
      </div>
      <pre className="rounded-md bg-[#0f172a] px-3 py-2 text-[11px] leading-5 text-[#e2e8f0]">
        {analysis.outputPreview ?? "output tensor shape: waiting"}
      </pre>
    </CompactPanel>
  );
}

function RunResult({
  response,
  challenge,
}: {
  response: PythonLabRunResponse;
  challenge: CurriculumTask;
}) {
  const result = response.result;
  const [tab, setTab] = useState<"summary" | "preview" | "logs">("summary");
  const hasPreview = Boolean(
    result?.losses?.length ||
      result?.modelMeta?.lossHistory?.length ||
      result?.modelMeta?.episodeRewards?.length ||
      result?.layers?.length ||
      result?.modelMeta?.layers?.length ||
      result?.policy?.length ||
      result?.notes?.length
  );
  const hasLogs = Boolean(response.stdout || response.stderr);

  return (
    <CompactPanel
      title="Python Sonucu"
      icon={response.ok ? <CheckCircle2 className="h-4 w-4 text-[#047857]" /> : <AlertTriangle className="h-4 w-4 text-[#b91c1c]" />}
      bodyClassName="space-y-3"
    >
      <div
        className={`rounded-md border px-3 py-2 text-[11px] leading-5 ${
          response.ok ? "border-[#bbf7d0] bg-[#f0fdf4]" : "border-[#fecaca] bg-[#fef2f2]"
        }`}
      >
        <div className="font-semibold text-[#18202f]">
          {response.ok ? (response.passed ? "Passed" : "Ran") : "Hata"} · {formatNumber(response.durationMs / 1000, 2)} sn
        </div>
        <div className="text-[#526070]">
          {challenge.title}
          {response.error ? ` · ${response.error}` : ""}
        </div>
      </div>

      <PanelTabs
        compact
        items={[
          { id: "summary", label: "Özet" },
          { id: "preview", label: "Preview", badge: hasPreview ? "var" : undefined },
          { id: "logs", label: "Logs", badge: hasLogs ? "var" : undefined },
        ]}
        value={tab}
        onChange={setTab}
      />

      {tab === "summary" && (
        <div className="space-y-3">
          {result && <ResultMetrics result={result} />}
          {response.feedback.length > 0 && (
            <div className={`rounded-md border p-3 ${response.passed ? "border-[#bbf7d0] bg-[#f0fdf4]" : "border-[#fed7aa] bg-[#fff7ed]"}`}>
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                Grading
              </div>
              <div className="space-y-1 text-[11px] leading-5 text-[#526070]">
                {response.feedback.map((item) => (
                  <div key={item}>{item}</div>
                ))}
              </div>
            </div>
          )}
          {result?.notes && result.notes.length > 0 && (
            <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
              <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                Notlar
              </div>
              <div className="space-y-1 text-[11px] leading-5 text-[#526070]">
                {result.notes.map((note) => (
                  <div key={note}>{note}</div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {tab === "preview" && (
        <div className="space-y-3">
          {!hasPreview && <div className="text-[11px] leading-5 text-[#64748b]">Bu sonuç için ek önizleme yok.</div>}
          {(result?.losses?.length || result?.modelMeta?.lossHistory?.length) && (
            <LossSparkline values={result.losses?.length ? result.losses : result.modelMeta?.lossHistory ?? []} />
          )}
          {result?.modelMeta?.episodeRewards && result.modelMeta.episodeRewards.length > 0 && (
            <EpisodeRewardPreview values={result.modelMeta.episodeRewards} />
          )}
          {(result?.layers || result?.modelMeta?.layers) && <LayerPreview layers={result.layers ?? result.modelMeta?.layers} />}
          {result?.policy && <PolicyPreview policy={result.policy} trajectory={result.trajectory ?? []} />}
        </div>
      )}

      {tab === "logs" && (
        <pre className="max-h-44 overflow-auto rounded-md bg-[#0f172a] p-3 text-[10px] leading-4 text-[#e2e8f0]">
          {hasLogs
            ? `${response.stdout}${response.stderr ? `\n--- stderr ---\n${response.stderr}` : ""}`
            : "stdout / stderr boş"}
        </pre>
      )}
    </CompactPanel>
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
    result.losses?.length ? { label: "son loss", value: formatNumber(result.losses.at(-1) ?? 0, 6) } : null,
    ...(result.metrics ?? []),
  ].filter(Boolean) as Array<{ label: string; value: string | number }>;
  if (metrics.length === 0) return null;

  return (
    <div className="grid grid-cols-2 gap-2">
      {metrics.slice(0, 6).map((metric) => (
        <div key={metric.label} className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] px-3 py-2">
          <div className="truncate text-[10px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            {metric.label}
          </div>
          <div className="mt-1 truncate text-sm font-semibold text-[#18202f]">{metric.value}</div>
        </div>
      ))}
    </div>
  );
}

function LossSparkline({ values }: { values: number[] }) {
  const width = 318;
  const height = 84;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const spread = Math.max(1e-9, max - min);
  const points = values
    .map((value, index) => {
      const x = 10 + (index / Math.max(1, values.length - 1)) * (width - 20);
      const y = height - 14 - ((value - min) / spread) * (height - 26);
      return `${x},${y}`;
    })
    .join(" ");

  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        Python Loss
      </div>
      <svg className="h-[84px] w-full" viewBox={`0 0 ${width} ${height}`}>
        <rect width={width} height={height} rx={6} fill="#fbfdff" stroke="#dbe3ee" />
        <polyline points={points} fill="none" stroke="#2563eb" strokeWidth={2.6} strokeLinecap="round" strokeLinejoin="round" />
        <circle cx={width - 10} cy={height - 14 - ((values.at(-1) ?? min) - min) / spread * (height - 26)} r={3.6} fill="#f97316" />
      </svg>
    </div>
  );
}

function EpisodeRewardPreview({ values }: { values: number[] }) {
  const sample = values.slice(-80);
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        Episode Rewards
      </div>
      <div className="flex h-20 items-end gap-0.5">
        {sample.map((value, index) => {
          const min = Math.min(...sample);
          const max = Math.max(...sample);
          const spread = Math.max(1e-9, max - min);
          const height = 12 + ((value - min) / spread) * 64;
          return (
            <span
              key={`${index}-${value}`}
              className={value >= 0 ? "bg-[#10b981]" : "bg-[#ef4444]"}
              style={{ height, width: `${100 / Math.max(1, sample.length)}%` }}
            />
          );
        })}
      </div>
    </div>
  );
}

function LayerPreview({ layers }: { layers?: Array<{ size: number; activation?: string; kind?: ModelMetaLayer["kind"] }> }) {
  if (!layers?.length) return null;
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        <Layers3 className="h-4 w-4 text-[#2563eb]" />
        Kodlanan Nöronlar
      </div>
      <div className="flex items-center gap-3 overflow-x-auto pb-1">
        {layers.map((layer, layerIndex) => {
          const visible = Math.min(14, layer.size);
          return (
            <div key={`${layerIndex}-${layer.size}`} className="shrink-0">
              <div className="mb-1 text-center text-[10px] font-semibold text-[#64748b]">
                {layer.kind ?? `L${layerIndex}`} · {layer.size}
              </div>
              <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${Math.min(4, visible)}, 10px)` }}>
                {Array.from({ length: visible }, (_, index) => (
                  <span
                    key={index}
                    className={`block h-2.5 w-2.5 rounded-full ${
                      layerIndex === 0 ? "bg-[#38bdf8]" : layerIndex === layers.length - 1 ? "bg-[#f87171]" : "bg-[#fbbf24]"
                    }`}
                  />
                ))}
              </div>
              <div className="mt-1 text-center text-[10px] text-[#64748b]">{layer.activation ?? "linear"}</div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function PolicyPreview({
  policy,
  trajectory,
}: {
  policy: string[];
  trajectory: number[][];
}) {
  const size = Math.round(Math.sqrt(policy.length));
  if (size <= 0 || size * size !== policy.length) return null;
  const arrows: Record<string, string> = {
    up: "↑",
    right: "→",
    down: "↓",
    left: "←",
    goal: "G",
    trap: "!",
  };
  const pathCells = new Set(trajectory.map(([x, y]) => `${x}:${y}`));
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        Policy Simülasyonu
      </div>
      <div className="grid gap-1" style={{ gridTemplateColumns: `repeat(${size}, minmax(0, 1fr))` }}>
        {policy.map((action, index) => {
          const x = index % size;
          const y = Math.floor(index / size);
          const isPath = pathCells.has(`${x}:${y}`);
          return (
            <div
              key={`${action}-${index}`}
              className={`flex aspect-square items-center justify-center rounded-md border text-sm font-bold ${
                action === "goal"
                  ? "border-[#10b981] bg-[#ecfdf5] text-[#047857]"
                  : action === "trap"
                    ? "border-[#ef4444] bg-[#fef2f2] text-[#b91c1c]"
                    : isPath
                      ? "border-[#2563eb] bg-[#eef4ff] text-[#1d4ed8]"
                      : "border-[#dbe3ee] bg-[#fbfdff] text-[#64748b]"
              }`}
            >
              {arrows[action] ?? action.slice(0, 1)}
            </div>
          );
        })}
      </div>
    </div>
  );
}
