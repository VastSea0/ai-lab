"use client";

import { Activity, ArrowLeft, BrainCircuit, Cuboid, Layers3, Play, RotateCcw } from "lucide-react";
import Link from "next/link";
import { useCallback, useMemo, useState } from "react";
import { ModelSurface3D } from "@/components/lab/canvas/ModelSurface3D";
import { AiCodeLab } from "@/components/lab/code/AiCodeLab";
import { IconToolbar, MetricStrip, StatusToast, WorkbenchShell } from "@/components/lab/ui/Workbench";
import {
  CURRICULUM_PROGRESS_KEY,
  curriculumPhaseSummary,
  firstAvailableCurriculumTask,
  normalizeCurriculumProgress,
} from "@/lib/ml/curriculum";
import { analyzeTorchStructure } from "@/lib/ml/live-code";
import type { CurriculumProgress, CurriculumTask, ModelMeta, PythonLabResult } from "@/lib/ml/types";
import type { DataPoint, Selection, TrainingPhase, TrainingTrace } from "@/lib/ml/network";
import { formatNumber, NeuralNetwork } from "@/lib/ml/network";
import type { Task, TaskId } from "@/lib/ml/tasks";
import { getTask } from "@/lib/ml/tasks";
import type { VisualizationMode } from "@/lib/ml/lab-types";

interface ImportedModel {
  challenge: CurriculumTask;
  result: PythonLabResult;
  modelMeta?: ModelMeta;
  task: Task;
  data: DataPoint[];
  network: NeuralNetwork;
  trace: TrainingTrace;
  lossHistory: number[];
}

function finiteNumber(value: unknown, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function firstTrace(network: NeuralNetwork, data: DataPoint[]) {
  const sizes = network.getLayerSizes();
  const first = data[0] ?? {
    id: "empty",
    inputs: Array(sizes[0] ?? 1).fill(0),
    targets: Array(sizes.at(-1) ?? 1).fill(0),
  };
  return network.inspect(first.inputs, first.targets);
}

function readCurriculumProgress(): CurriculumProgress {
  if (typeof window === "undefined") return normalizeCurriculumProgress(null);
  try {
    return normalizeCurriculumProgress(JSON.parse(window.localStorage.getItem(CURRICULUM_PROGRESS_KEY) ?? "null"));
  } catch {
    return normalizeCurriculumProgress(null);
  }
}

function modelTaskId(meta: ModelMeta | undefined, inputSize: number, outputSize: number): TaskId {
  if (meta?.type === "cnn" || inputSize === 25) return "digit";
  if (meta?.type === "sklearn" && outputSize === 1) return "regression";
  if (meta?.type === "lstm") return "sine";
  return "xor";
}

function dataForImportedResult(result: PythonLabResult, inputSize: number, outputSize: number): DataPoint[] {
  const sourceData = result.dataset?.length
    ? result.dataset
    : [
        {
          id: "meta-0",
          inputs: Array(inputSize).fill(0),
          targets: Array(outputSize).fill(0),
          label: "preview",
        },
      ];

  return sourceData.map((point, index) => ({
    id: String(point.id ?? `code-${index + 1}`),
    inputs: Array.from({ length: inputSize }, (_, inputIndex) => finiteNumber(point.inputs?.[inputIndex], 0)),
    targets: Array.from({ length: outputSize }, (_, targetIndex) => finiteNumber(point.targets?.[targetIndex], 0)),
    label: point.label,
  }));
}

function buildImportedModel(
  challenge: CurriculumTask,
  result: PythonLabResult,
  seed: number
): ImportedModel {
  const network = result.modelMeta
    ? NeuralNetwork.fromMeta(result.modelMeta, seed)
    : result.layers?.length && result.weights?.length && result.biases?.length
      ? NeuralNetwork.fromParameters(
          result.layers.map((layer) => ({
            size: Math.max(1, Math.floor(finiteNumber(layer.size, 1))),
            activation: layer.activation,
          })),
          result.weights,
          result.biases,
          seed
        )
      : null;

  if (!network) {
    throw new Error("Modeli görselleştirmek için modelMeta veya layers/weights/biases alanları gerekli.");
  }

  const layerConfigs = network.getLayerConfigs();
  const inputSize = layerConfigs[0]?.size ?? 1;
  const outputSize = layerConfigs.at(-1)?.size ?? 1;
  const baseTask = getTask(modelTaskId(result.modelMeta, inputSize, outputSize));
  const outputType = outputSize > 1 ? "classification" : "regression";
  const data = dataForImportedResult(result, inputSize, outputSize);
  const classNames =
    outputType === "classification"
      ? Array.from(
          { length: outputSize },
          (_, index) => result.classNames?.[index] ?? baseTask.classNames?.[index] ?? `Sınıf ${index}`
        )
      : undefined;
  const task: Task = {
    ...baseTask,
    name: result.title ?? challenge.title,
    description: challenge.description,
    explanation: challenge.description,
    inputSize,
    outputSize,
    outputType,
    classNames,
    defaultLayers: layerConfigs,
    data,
  };
  const lossHistory = (result.losses ?? result.modelMeta?.lossHistory ?? []).filter(Number.isFinite).slice(-160);

  return {
    challenge,
    result,
    modelMeta: result.modelMeta,
    task,
    data,
    network,
    trace: firstTrace(network, data),
    lossHistory: lossHistory.length > 0 ? lossHistory : [network.evaluateLoss(data)],
  };
}

function buildLiveResult(task: CurriculumTask, code: string): PythonLabResult | null {
  const analysis = analyzeTorchStructure(code, task.requirements);
  if (!analysis.modelMeta) return null;
  return {
    title: task.title,
    sourceCode: code,
    modelMeta: analysis.modelMeta,
    metrics: [
      {
        label: "checks",
        value: `${analysis.checks.filter((check) => check.passed).length}/${analysis.checks.length}`,
      },
    ],
    notes: analysis.outputPreview ? [analysis.outputPreview] : undefined,
  };
}

function initialImportedModel(): ImportedModel | null {
  try {
    const progress = readCurriculumProgress();
    const task = firstAvailableCurriculumTask(progress);
    const result = buildLiveResult(task, task.starterCode);
    return result ? buildImportedModel(task, result, 9421) : null;
  } catch {
    return null;
  }
}

function selectionSummary(selection: Selection | null, trace: TrainingTrace | null) {
  if (!selection || !trace) return "Henüz seçim yok";
  if (selection.type === "neuron") {
    const neuron = trace.neurons.find((item) => item.id === selection.id);
    return neuron
      ? `L${neuron.layerIndex} N${neuron.neuronIndex} · a=${formatNumber(neuron.value, 4)} · δ=${formatNumber(neuron.delta, 4)}`
      : selection.id;
  }
  const edge = trace.edges.find((item) => item.id === selection.id);
  return edge
    ? `${edge.id} · w=${formatNumber(edge.weight, 4)} · grad=${formatNumber(edge.gradient, 4)}`
    : selection.id;
}

export function AiCodeLabWorkspace() {
  const [imported, setImported] = useState<ImportedModel | null>(() => initialImportedModel());
  const [importError, setImportError] = useState<string | null>(null);
  const [curriculumProgress, setCurriculumProgress] = useState<CurriculumProgress>(() => readCurriculumProgress());
  const [selected, setSelected] = useState<Selection | null>(null);
  const [hovered, setHovered] = useState<Selection | null>(null);
  const [phase, setPhase] = useState<TrainingPhase>("idle");
  const [visualizationMode, setVisualizationMode] = useState<VisualizationMode>("weights");

  const updateCurriculumProgress = useCallback((nextProgress: CurriculumProgress) => {
    const clean = normalizeCurriculumProgress(nextProgress);
    setCurriculumProgress(clean);
    window.localStorage.setItem(CURRICULUM_PROGRESS_KEY, JSON.stringify(clean));
  }, []);

  const applyResult = useCallback((challenge: CurriculumTask, result: PythonLabResult) => {
    try {
      const next = buildImportedModel(challenge, result, Date.now() % 100_000);
      setImported(next);
      setSelected(null);
      setHovered(null);
      setPhase("forward");
      setImportError(null);
      window.setTimeout(() => setPhase("idle"), 1300);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Model görselleştirilemedi.");
    }
  }, []);

  const applyLiveResult = useCallback((challenge: CurriculumTask, result: PythonLabResult) => {
    try {
      const next = buildImportedModel(challenge, result, 9421);
      setImported(next);
      setImportError(null);
    } catch (error) {
      setImportError(error instanceof Error ? error.message : "Canlı model görselleştirilemedi.");
    }
  }, []);

  const activeSelection = hovered ?? selected;
  const hasEpisodeRewards = Boolean(imported?.modelMeta?.episodeRewards?.length);
  const activeVisualizationMode =
    visualizationMode === "rl-reward" && !hasEpisodeRewards ? "weights" : visualizationMode;
  const activeLoss = imported?.network.evaluateLoss(imported.data) ?? null;
  const layerText = useMemo(
    () => imported?.network.getLayerSizes().join(" → ") ?? "model bekleniyor",
    [imported]
  );

  return (
    <WorkbenchShell>
      <div className="grid h-full grid-rows-[56px_minmax(0,1fr)]">
        <header className="flex h-full items-center justify-between border-b border-[#dbe3ee] bg-white px-4">
          <div className="flex min-w-0 items-center gap-2.5">
            <Link
              href="/"
              className="inline-flex h-8 w-8 items-center justify-center rounded-md border border-[#cbd5e1] bg-white text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
              aria-label="Sandbox sayfasına dön"
            >
              <ArrowLeft className="h-4 w-4" />
            </Link>
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-[#e8f0ff] text-[#2563eb]">
              <BrainCircuit className="h-[18px] w-[18px]" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-sm font-semibold">AI Kod Labı</div>
              <div className="truncate text-[11px] text-[#607089]">
                Editörde kodla, ağı önizle, görevi tamamla
              </div>
            </div>
          </div>

          <div className="flex min-w-0 items-center gap-2">
            <CurriculumProgressHeader progress={curriculumProgress} />
            <MetricStrip
              items={[
                { label: "Katman", value: layerText },
                { label: "Loss", value: activeLoss === null ? "-" : formatNumber(activeLoss, 6) },
              ]}
            />
            <IconToolbar label="Kod labı ayarları" className="shrink-0">
              <select
                className="h-7 rounded border-0 bg-transparent px-1.5 text-[11px] font-semibold text-[#334155] outline-none"
                value={activeVisualizationMode}
                onChange={(event) => setVisualizationMode(event.target.value as VisualizationMode)}
                aria-label="Görselleştirme modu"
              >
                <option value="weights">Ağırlık</option>
                <option value="gradients">Gradient</option>
                <option value="corrections">Düzeltme</option>
                <option value="rl-reward" disabled={!hasEpisodeRewards}>
                  RL Ödül
                </option>
              </select>
            </IconToolbar>
            <button
              type="button"
              className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md border border-[#cbd5e1] bg-white px-2.5 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
              onClick={() => {
                setPhase("forward");
                window.setTimeout(() => setPhase("backward"), 620);
                window.setTimeout(() => setPhase("idle"), 1380);
              }}
            >
              <Play className="h-4 w-4" />
              Akışı Oynat
            </button>
          </div>
        </header>

        <div className="min-h-0 bg-[#d7dde8]">
          <AiCodeLab
            onApplyResult={applyResult}
            onLiveResult={applyLiveResult}
            applyLabel="Görselleştir"
            progress={curriculumProgress}
            onProgressChange={updateCurriculumProgress}
            preview={
              <section className="relative h-full min-h-0 overflow-hidden bg-[#eef3f8]">
                {imported ? (
                  <ModelSurface3D
                    task={imported.task}
                    network={imported.network}
                    data={imported.data}
                    trace={imported.trace}
                    phase={phase}
                    visualizationMode={activeVisualizationMode}
                    selected={selected}
                    hovered={hovered}
                    onSelect={setSelected}
                    onHover={setHovered}
                    onOpenDetail={setSelected}
                    modelMeta={imported.modelMeta}
                  />
                ) : (
                  <div className="flex h-full items-start justify-start p-5">
                    <div className="max-w-[420px] rounded-md border border-[#dbe3ee] bg-white/88 px-4 py-3 shadow-sm backdrop-blur">
                      <div className="flex items-center gap-2 text-xs font-semibold">
                        <Cuboid className="h-5 w-5 text-[#2563eb]" />
                        Model Simülasyonu
                      </div>
                      <div className="mt-1 text-xs leading-5 text-[#526070]">Model bekleniyor.</div>
                    </div>
                  </div>
                )}

                <div className="absolute bottom-4 right-4 z-20 w-[320px]">
                  <StatusToast
                    title={imported?.challenge.title ?? "Python sonucu bekleniyor"}
                    icon={<Activity className="h-3.5 w-3.5" />}
                  >
                    <div>{selectionSummary(activeSelection, imported?.trace ?? null)}</div>
                    {importError && <div className="mt-1 text-[#b91c1c]">{importError}</div>}
                    {imported && (
                      <div className="mt-1 flex items-center gap-2">
                        <Layers3 className="h-3.5 w-3.5 text-[#64748b]" />
                        {imported.network.getLayerSizes().join(" -> ")}
                      </div>
                    )}
                  </StatusToast>
                </div>

                {imported && (
                  <button
                    type="button"
                    className="absolute left-5 top-5 z-20 inline-flex h-9 items-center gap-2 rounded-md border border-[#cbd5e1] bg-white/90 px-3 text-xs font-semibold text-[#334155] shadow-sm backdrop-blur hover:border-[#2563eb] hover:text-[#2563eb]"
                    onClick={() => {
                      setImported(null);
                      setSelected(null);
                      setHovered(null);
                      setImportError(null);
                    }}
                  >
                    <RotateCcw className="h-4 w-4" />
                    Sahneyi temizle
                  </button>
                )}
              </section>
            }
          />
        </div>
      </div>
    </WorkbenchShell>
  );
}

function CurriculumProgressHeader({ progress }: { progress: CurriculumProgress }) {
  const summary = curriculumPhaseSummary(progress);
  return (
    <div className="hidden min-w-[210px] grid-cols-3 gap-2 rounded-md border border-[#dbe3ee] bg-white px-2.5 py-1.5 lg:grid">
      {summary.map(({ phase, completed, total }) => (
        <div key={phase} className="min-w-0">
          <div className="flex items-center justify-between gap-1 text-[9px] font-bold uppercase tracking-[0.08em] text-[#64748b]">
            <span>P{phase}</span>
            <span>
              {completed}/{total}
            </span>
          </div>
          <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-[#e2e8f0]">
            <div
              className="h-full rounded-full bg-[#2563eb]"
              style={{ width: `${Math.round((completed / Math.max(1, total)) * 100)}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}
