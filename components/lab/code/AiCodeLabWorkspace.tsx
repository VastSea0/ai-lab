"use client";

import { useCallback, useMemo, useState } from "react";
import { ModelSurface3D } from "@/components/lab/canvas/ModelSurface3D";
import { AiCodeLab } from "@/components/lab/code/AiCodeLab";
import { WorkbenchShell } from "@/components/lab/ui/Workbench";
import {
  CURRICULUM_PROGRESS_KEY,
  firstAvailableCurriculumTask,
  normalizeCurriculumProgress,
} from "@/lib/ml/curriculum";
import { analyzeTorchStructure } from "@/lib/ml/live-code";
import type { CurriculumProgress, CurriculumTask, ModelMeta, PythonLabResult } from "@/lib/ml/types";
import type { DataPoint, Selection, TrainingPhase, TrainingTrace } from "@/lib/ml/network";
import { formatNumber, NeuralNetwork } from "@/lib/ml/network";
import type { Task, TaskId } from "@/lib/ml/tasks";
import { getTask } from "@/lib/ml/tasks";

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

function buildImportedModel(challenge: CurriculumTask, result: PythonLabResult, seed: number): ImportedModel {
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

  const clearPreview = useCallback(() => {
    setImported(null);
    setSelected(null);
    setHovered(null);
    setImportError(null);
  }, []);

  const activeSelection = hovered ?? selected;
  const activeLoss = imported?.network.evaluateLoss(imported.data) ?? null;
  const layerText = useMemo(
    () => imported?.network.getLayerSizes().join(" → ") ?? "model bekleniyor",
    [imported]
  );
  const lossText = activeLoss === null ? "-" : formatNumber(activeLoss, 5);

  return (
    <WorkbenchShell>
      <AiCodeLab
        onApplyResult={applyResult}
        onLiveResult={applyLiveResult}
        onClearPreview={clearPreview}
        applyLabel="Görselleştir"
        progress={curriculumProgress}
        onProgressChange={updateCurriculumProgress}
        layerSummary={layerText}
        lossSummary={lossText}
        selectionSummary={importError ?? selectionSummary(activeSelection, imported?.trace ?? null)}
        preview={
          <section className="h-full min-h-0 overflow-hidden bg-[#fafbff]">
            {imported ? (
              <ModelSurface3D
                task={imported.task}
                network={imported.network}
                data={imported.data}
                trace={imported.trace}
                phase={phase}
                visualizationMode="weights"
                selected={selected}
                hovered={hovered}
                onSelect={setSelected}
                onHover={setHovered}
                onOpenDetail={setSelected}
                modelMeta={imported.modelMeta}
                chrome="minimal"
              />
            ) : (
              <div className="flex h-full items-center justify-center px-4 text-center text-[11px] leading-5 text-[#a0a4c0]">
                Model önizlemesi koddan üretilecek.
              </div>
            )}
          </section>
        }
      />
    </WorkbenchShell>
  );
}
