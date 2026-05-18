"use client";

import {
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  Circle,
  Layers3,
  Terminal,
  XCircle,
} from "lucide-react";
import { useState, type ReactNode } from "react";
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
import { type PythonLabResult, type PythonLabRunResponse } from "@/lib/ml/code-lab";
import type { CurriculumProgress, CurriculumTask, LiveCodeAnalysis, ModelMeta } from "@/lib/ml/types";

type PreviewTab = "network" | "loss" | "console";

export function AiCodeLab({
  onApplyResult,
  onLiveResult,
  onClearPreview,
  preview,
  applyLabel = "Görselleştir",
  progress,
  onProgressChange,
  layerSummary,
  lossSummary,
  selectionSummary,
}: {
  onApplyResult: (task: CurriculumTask, result: PythonLabResult) => void;
  onLiveResult?: (task: CurriculumTask, result: PythonLabResult, analysis: LiveCodeAnalysis) => void;
  onClearPreview?: () => void;
  preview: ReactNode;
  applyLabel?: string;
  progress: CurriculumProgress;
  onProgressChange: (progress: CurriculumProgress) => void;
  layerSummary: string;
  lossSummary: string;
  selectionSummary: string;
}) {
  const safeProgress = normalizeCurriculumProgress(progress);
  const [selectedId, setSelectedId] = useState(() => firstAvailableCurriculumTask(safeProgress).id);
  const selected = getCurriculumTask(selectedId) ?? firstAvailableCurriculumTask(safeProgress);
  const [code, setCode] = useState(selected.starterCode);
  const [running, setRunning] = useState(false);
  const [response, setResponse] = useState<PythonLabRunResponse | null>(null);
  const [previewTab, setPreviewTab] = useState<PreviewTab>("network");
  const liveAnalysis = analyzeTorchStructure(code, selected.requirements);
  const liveMissionSupported = supportsLiveMission(selected);
  const visibleModelMeta = liveAnalysis.modelMeta ?? response?.result?.modelMeta;

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
    onProgressChange(completeCurriculumTask(safeProgress, task.id));
  };

  const selectTask = (task: CurriculumTask) => {
    if (!isCurriculumTaskUnlocked(task, safeProgress)) return;
    const analysis = analyzeTorchStructure(task.starterCode, task.requirements);
    setSelectedId(task.id);
    setCode(task.starterCode);
    setResponse(null);
    setPreviewTab("network");
    if (analysis.modelMeta) publishLiveResult(task, task.starterCode, analysis);
    else onClearPreview?.();
  };

  const updateCode = (nextCode: string) => {
    const analysis = analyzeTorchStructure(nextCode, selected.requirements);
    setCode(nextCode);
    setResponse(null);
    publishLiveResult(selected, nextCode, analysis);
    completeIfLivePassed(selected, analysis);
  };

  const runCode = async () => {
    const preflight = preflightShapeMismatch(code, liveAnalysis);
    if (preflight) {
      setResponse({
        ok: false,
        stdout: "",
        stderr: "",
        durationMs: 0,
        passed: false,
        feedback: [preflight],
        error: "Input shape mismatch.",
      });
      setPreviewTab("console");
      return;
    }

    setRunning(true);
    setResponse(null);
    try {
      const result = await fetch("/api/python-lab", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ challengeId: selected.id, code }),
      });
      const payload = (await result.json()) as PythonLabRunResponse;
      setResponse(payload);
      setPreviewTab(payload.ok ? "network" : "console");
      if (payload.ok && payload.result) {
        onApplyResult(selected, payload.result);
      }
      if (payload.ok && payload.passed) {
        onProgressChange(completeCurriculumTask(safeProgress, selected.id));
      }
    } catch (error) {
      setResponse({
        ok: false,
        stdout: "",
        stderr: "",
        durationMs: 0,
        passed: false,
        feedback: ["Python hücresine ulaşılamadı."],
        error: error instanceof Error ? error.message : "Python çalıştırılamadı.",
      });
      setPreviewTab("console");
    } finally {
      setRunning(false);
    }
  };

  const statusText = !liveMissionSupported
    ? "Çalıştırınca değerlendirilecek"
    : liveAnalysis.passed
      ? liveAnalysis.outputPreview ?? "model hazır"
      : liveAnalysis.outputPreview ?? "model bekleniyor";

  return (
    <div className="grid h-full min-h-0 grid-cols-[220px_minmax(0,1fr)_320px] overflow-hidden bg-[#f5f6fa]">
      <aside className="min-h-0 border-r border-[#e8eaf2]">
        <MissionRail selectedTaskId={selected.id} progress={safeProgress} onSelectTask={selectTask} />
      </aside>

      <main className="min-h-0 border-r border-[#e8eaf2]">
        <CodeEditor
          value={code}
          onChange={updateCode}
          onRun={runCode}
          onReset={() => updateCode(selected.starterCode)}
          running={running}
          statusText={statusText}
          statusTone={liveMissionSupported && liveAnalysis.passed ? "good" : "neutral"}
        />
      </main>

      <aside className="flex min-h-0 flex-col bg-white">
        <div className="flex h-11 shrink-0 items-center gap-1 border-b border-[#e8eaf2] px-3">
          <PreviewTabButton active={previewTab === "network"} onClick={() => setPreviewTab("network")} icon={<Layers3 className="h-3.5 w-3.5" />}>
            Ağ
          </PreviewTabButton>
          <PreviewTabButton active={previewTab === "loss"} onClick={() => setPreviewTab("loss")} icon={<BarChart3 className="h-3.5 w-3.5" />}>
            Loss
          </PreviewTabButton>
          <PreviewTabButton active={previewTab === "console"} onClick={() => setPreviewTab("console")} icon={<Terminal className="h-3.5 w-3.5" />}>
            Konsol
          </PreviewTabButton>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto p-4">
          {previewTab === "network" && (
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-2">
                <MetricCard label="Mimari" value={layerSummary} tone="purple" sub={`${layerSummary.split("→").length} katman`} />
                <MetricCard label="Loss" value={lossSummary} tone="green" sub={response?.passed ? "Görev geçti" : "Son değer"} />
              </div>
              <section className="rounded-lg border border-[#e8eaf2] bg-[#f8f9fe] p-3">
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#a0a4c0]">
                  2D ağ şeması
                </div>
                <NetworkDiagram2D meta={visibleModelMeta} />
              </section>
              <section className="rounded-lg border border-[#e8eaf2] bg-[#f8f9fe] p-3">
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#a0a4c0]">
                  3D ağ simülasyonu
                </div>
                <div className="h-[280px] overflow-hidden rounded-md border border-[#eef0f8] bg-[#fafbff]">{preview}</div>
              </section>
              <section className="rounded-lg border border-[#e8eaf2] bg-[#f8f9fe] p-3">
                <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#a0a4c0]">
                  Seçili durum
                </div>
                <div className="text-sm font-semibold text-[#3d4069]">{selectionSummary}</div>
                <div className="mt-1 text-[11px] text-[#a0a4c0]">{selected.title}</div>
              </section>
              <MissionChecklist task={selected} analysis={liveAnalysis} liveSupported={liveMissionSupported} />
            </div>
          )}

          {previewTab === "loss" && (
            <div className="space-y-3">
              <MetricCard label="Anlık Loss" value={lossSummary} tone="purple" sub={selected.title} />
              {response?.result ? <ResultMetrics result={response.result} /> : <EmptyPanel text="Çalıştırınca eğitim metrikleri burada görünecek." />}
              {response?.feedback?.length ? <FeedbackPanel response={response} /> : null}
            </div>
          )}

          {previewTab === "console" && (
            <div className="space-y-3">
              {response ? (
                <>
                  <ConsoleOutput response={response} />
                  {response.result && (
                    <button
                      type="button"
                      className="h-8 rounded-md bg-[#6366f1] px-3 text-xs font-semibold text-white"
                      onClick={() => response.result && onApplyResult(selected, response.result)}
                    >
                      {applyLabel}
                    </button>
                  )}
                </>
              ) : (
                <EmptyPanel text={liveMissionSupported ? liveAnalysis.outputPreview ?? "Kod yazınca canlı çıktı burada özetlenir." : "Kodu çalıştırınca stdout ve hata kayıtları burada görünecek."} />
              )}
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}

function preflightShapeMismatch(code: string, analysis: LiveCodeAnalysis) {
  const expectedInput = analysis.modelMeta?.layers.find((layer) => layer.kind === "input")?.size;
  const dummyMatch = code.match(/torch\.randn\s*\(\s*1\s*,\s*(\d+)\s*\)/);
  const dummyInput = dummyMatch ? Number(dummyMatch[1]) : undefined;

  if (!expectedInput || !dummyInput || expectedInput === dummyInput) return null;

  return `Model ${expectedInput} input neuron bekliyor ama dummy_input ${dummyInput} feature ile oluşturulmuş. Şunu düzelt: torch.randn(1, ${expectedInput}).`;
}

function PreviewTabButton({
  active,
  onClick,
  icon,
  children,
}: {
  active: boolean;
  onClick: () => void;
  icon: ReactNode;
  children: ReactNode;
}) {
  return (
    <button
      type="button"
      className={`inline-flex h-8 items-center gap-1.5 rounded-md px-3 text-xs font-semibold transition ${
        active ? "bg-[#f0f1fe] text-[#5254c8]" : "text-[#9599b8] hover:bg-[#f8f9fe] hover:text-[#6b6f90]"
      }`}
      onClick={onClick}
    >
      {icon}
      {children}
    </button>
  );
}

function MetricCard({
  label,
  value,
  sub,
  tone,
}: {
  label: string;
  value: string;
  sub?: string;
  tone?: "green" | "purple";
}) {
  return (
    <div className="rounded-lg border border-[#e8eaf2] bg-[#f8f9fe] px-3 py-3">
      <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#a0a4c0]">{label}</div>
      <div className={`mt-1 truncate text-lg font-semibold ${tone === "green" ? "text-[#16a34a]" : tone === "purple" ? "text-[#6366f1]" : "text-[#1a1c2e]"}`}>
        {value}
      </div>
      {sub && <div className="mt-1 truncate text-[10px] text-[#a0a4c0]">{sub}</div>}
    </div>
  );
}

function NetworkDiagram2D({ meta }: { meta?: ModelMeta }) {
  if (!meta?.layers.length) {
    return <EmptyPanel text="Koddan modelMeta üretildiğinde 2D ağ şeması burada görünecek." />;
  }

  const width = 288;
  const height = 178;
  const layers = meta.layers.slice(0, 6);
  const maxNodes = 8;
  const xGap = layers.length <= 1 ? 0 : (width - 40) / (layers.length - 1);

  const points = layers.map((layer, layerIndex) => {
    const visible = Math.min(maxNodes, Math.max(1, layer.size));
    const yGap = visible <= 1 ? 0 : 112 / (visible - 1);
    const startY = height / 2 - ((visible - 1) * yGap) / 2;
    return Array.from({ length: visible }, (_, nodeIndex) => ({
      id: `${layerIndex}-${nodeIndex}`,
      x: 20 + layerIndex * xGap,
      y: startY + nodeIndex * yGap,
      hidden: layer.size > maxNodes && nodeIndex === visible - 1,
    }));
  });

  return (
    <div className="overflow-hidden rounded-md border border-[#eef0f8] bg-[#fafbff]">
      <svg className="h-[178px] w-full" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="2D neural network diagram">
        {points.slice(0, -1).flatMap((layer, layerIndex) =>
          layer.flatMap((from) =>
            points[layerIndex + 1].map((to) => (
              <line
                key={`${from.id}-${to.id}`}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                stroke="#d8dcf0"
                strokeWidth="0.8"
                opacity="0.7"
              />
            ))
          )
        )}
        {points.map((layer, layerIndex) =>
          layer.map((point) => {
            const kind = layers[layerIndex].kind;
            const fill = kind === "input" ? "#93c5fd" : kind === "output" ? "#67e8f9" : "#a5b4fc";
            const stroke = kind === "output" ? "#06b6d4" : "#6366f1";
            return (
              <g key={point.id}>
                <circle cx={point.x} cy={point.y} r="6.2" fill={fill} stroke={stroke} strokeWidth="1.5" />
                {point.hidden && (
                  <text x={point.x} y={point.y + 3} textAnchor="middle" className="fill-[#5254c8] text-[8px] font-bold">
                    +
                  </text>
                )}
              </g>
            );
          })
        )}
        {layers.map((layer, layerIndex) => (
          <g key={`${layer.kind}-${layerIndex}`}>
            <text
              x={20 + layerIndex * xGap}
              y={18}
              textAnchor="middle"
              className="fill-[#a0a4c0] text-[8px] font-bold uppercase"
            >
              {layer.kind}
            </text>
            <text
              x={20 + layerIndex * xGap}
              y={166}
              textAnchor="middle"
              className="fill-[#6366f1] text-[9px] font-bold"
            >
              {layer.size}
            </text>
          </g>
        ))}
      </svg>
    </div>
  );
}

function EmptyPanel({ text }: { text: string }) {
  return (
    <div className="rounded-lg border border-[#e8eaf2] bg-[#f8f9fe] px-3 py-3 text-[11px] leading-5 text-[#9599b8]">
      {text}
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
    requirements.minEpochs !== undefined ? `${requirements.minEpochs}+ epoch` : null,
    requirements.clusters !== undefined ? `${requirements.clusters} cluster` : null,
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
  const checks = liveSupported
    ? analysis.checks
    : requirementSummary(task).map((detail, index) => ({
        id: `${task.id}-${index}`,
        label: detail,
        detail: "Çalıştırınca kontrol edilir.",
        passed: false,
      }));

  return (
    <section className="border-t border-[#e8eaf2] pt-3">
      <div className="mb-2 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#a0a4c0]">
        Görev kontrol listesi
      </div>
      <div className="space-y-1">
        {checks.map((check) => (
          <div key={check.id} className="flex items-center gap-2 border-b border-[#f0f1f8] py-1.5 last:border-b-0">
            <span
              className={`flex h-4 w-4 shrink-0 items-center justify-center rounded-full border ${
                check.passed ? "border-[#6366f1] bg-[#6366f1] text-white" : "border-[#d7daf0] bg-white text-[#c0c4dc]"
              }`}
            >
              {check.passed ? <CheckCircle2 className="h-3 w-3" /> : <Circle className="h-3 w-3" />}
            </span>
            <span className="min-w-0">
              <span className="block truncate text-[11px] font-semibold text-[#3d4069]">{check.label}</span>
              <span className="block truncate text-[10px] text-[#a0a4c0]">{check.detail}</span>
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}

function FeedbackPanel({ response }: { response: PythonLabRunResponse }) {
  return (
    <div
      className={`rounded-lg border p-3 text-[11px] leading-5 ${
        response.passed ? "border-[#bbf7d0] bg-[#f0fdf4] text-[#166534]" : "border-[#fed7aa] bg-[#fff7ed] text-[#b45309]"
      }`}
    >
      <div className="mb-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#a0a4c0]">Grading</div>
      {response.feedback.map((item) => (
        <div key={item}>{item}</div>
      ))}
    </div>
  );
}

function ConsoleOutput({ response }: { response: PythonLabRunResponse }) {
  const hasError = !response.ok || response.error;

  return (
    <div className="space-y-3">
      <div
        className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-[11px] ${
          response.ok
            ? response.passed
              ? "border-[#bbf7d0] bg-[#f0fdf4] text-[#166534]"
              : "border-[#e8eaf2] bg-[#f8f9fe] text-[#3d4069]"
            : "border-[#fecaca] bg-[#fef2f2] text-[#b91c1c]"
        }`}
      >
        {response.ok ? <CheckCircle2 className="h-4 w-4" /> : <XCircle className="h-4 w-4" />}
        <span className="font-semibold">{response.ok ? (response.passed ? "Passed" : "Ran") : "Error"}</span>
        <span className="text-[#a0a4c0]">{formatNumber(response.durationMs / 1000, 2)}s</span>
      </div>

      {response.result && <ResultMetrics result={response.result} />}
      <FeedbackPanel response={response} />

      {(response.stdout || response.stderr) && (
        <pre className="max-h-60 overflow-auto rounded-lg border border-[#e8eaf2] bg-[#fafbff] p-3 font-mono text-[10px] leading-5 text-[#3d4069]">
          {response.stdout}
          {response.stderr ? `\n--- stderr ---\n${response.stderr}` : ""}
        </pre>
      )}

      {hasError && response.error && (
        <div className="flex items-start gap-2 rounded-lg border border-[#fecaca] bg-[#fef2f2] px-3 py-2 text-[11px] leading-5 text-[#b91c1c]">
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

  if (metrics.length === 0) return <EmptyPanel text="Henüz metrik yok." />;

  return (
    <div className="grid grid-cols-2 gap-2">
      {metrics.slice(0, 6).map((metric) => (
        <div key={metric.label} className="rounded-lg border border-[#e8eaf2] bg-[#f8f9fe] px-3 py-2">
          <div className="truncate text-[9px] font-semibold uppercase tracking-[0.08em] text-[#a0a4c0]">{metric.label}</div>
          <div className="mt-1 truncate text-sm font-semibold text-[#1a1c2e]">{metric.value}</div>
        </div>
      ))}
    </div>
  );
}
