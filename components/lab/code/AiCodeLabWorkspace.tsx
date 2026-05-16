"use client";

import { Activity, ArrowLeft, BrainCircuit, Cuboid, Layers3, Play, RotateCcw } from "lucide-react";
import Link from "next/link";
import { useCallback, useMemo, useState } from "react";
import { ModelSurface3D } from "@/components/lab/canvas/ModelSurface3D";
import { AiCodeLab } from "@/components/lab/code/AiCodeLab";
import type { AiCodeLabChallenge, PythonLabResult } from "@/lib/ml/code-lab";
import type { DataPoint, LayerConfig, Selection, TrainingPhase, TrainingTrace } from "@/lib/ml/network";
import { formatNumber, NeuralNetwork } from "@/lib/ml/network";
import type { Task } from "@/lib/ml/tasks";
import { getTask } from "@/lib/ml/tasks";
import type { VisualizationMode } from "@/lib/ml/lab-types";

interface ImportedModel {
  challenge: AiCodeLabChallenge;
  result: PythonLabResult;
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

function buildImportedModel(
  challenge: AiCodeLabChallenge,
  result: PythonLabResult,
  seed: number
): ImportedModel {
  if (!challenge.taskId) {
    throw new Error("Bu challenge sinir ağı üretmiyor; sonucu policy/simülasyon olarak oku.");
  }
  if (!result.layers?.length || !result.weights?.length || !result.biases?.length) {
    throw new Error("Modeli görselleştirmek için layers, weights ve biases alanları gerekli.");
  }

  const baseTask = getTask(challenge.taskId);
  const layers: LayerConfig[] = result.layers.map((layer) => ({
    size: Math.max(1, Math.floor(finiteNumber(layer.size, 1))),
    activation: layer.activation,
  }));
  const inputSize = layers[0]?.size ?? baseTask.inputSize;
  const outputSize = layers.at(-1)?.size ?? baseTask.outputSize;
  const classNames =
    baseTask.outputType === "classification"
      ? Array.from(
          { length: outputSize },
          (_, index) => result.classNames?.[index] ?? baseTask.classNames?.[index] ?? `Sınıf ${index}`
        )
      : baseTask.classNames;
  const sourceData = result.dataset?.length ? result.dataset : baseTask.data;
  const data = sourceData.map((point, index) => ({
    id: String(point.id ?? `code-${index + 1}`),
    inputs: Array.from({ length: inputSize }, (_, inputIndex) => finiteNumber(point.inputs?.[inputIndex], 0)),
    targets: Array.from({ length: outputSize }, (_, targetIndex) => finiteNumber(point.targets?.[targetIndex], 0)),
    label: point.label,
  }));
  const task: Task = {
    ...baseTask,
    name: result.title ?? challenge.title,
    description: challenge.summary,
    explanation: challenge.prompt,
    inputSize,
    outputSize,
    classNames,
    defaultLayers: layers,
    data,
  };
  const network = NeuralNetwork.fromParameters(layers, result.weights, result.biases, seed);
  const lossHistory = (result.losses ?? []).filter(Number.isFinite).slice(-160);

  return {
    challenge,
    result,
    task,
    data,
    network,
    trace: firstTrace(network, data),
    lossHistory: lossHistory.length > 0 ? lossHistory : [network.evaluateLoss(data)],
  };
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
  const [imported, setImported] = useState<ImportedModel | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Selection | null>(null);
  const [hovered, setHovered] = useState<Selection | null>(null);
  const [phase, setPhase] = useState<TrainingPhase>("idle");
  const [visualizationMode, setVisualizationMode] = useState<VisualizationMode>("weights");

  const applyResult = useCallback((challenge: AiCodeLabChallenge, result: PythonLabResult) => {
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

  const activeSelection = hovered ?? selected;
  const activeLoss = imported?.network.evaluateLoss(imported.data) ?? null;
  const layerText = useMemo(
    () => imported?.network.getLayerSizes().join(" → ") ?? "model bekleniyor",
    [imported]
  );

  return (
    <main className="h-screen min-h-[720px] min-w-[1240px] overflow-hidden bg-[#f5f7fb] text-[#18202f]">
      <div className="grid h-full grid-rows-[60px_minmax(0,1fr)]">
        <header className="flex items-center justify-between border-b border-[#dbe3ee] bg-white px-5">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/"
              className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-[#cbd5e1] bg-white text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
              aria-label="Sandbox sayfasına dön"
            >
              <ArrowLeft className="h-4 w-4" />
            </Link>
            <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-md bg-[#e8f0ff] text-[#2563eb]">
              <BrainCircuit className="h-5 w-5" />
            </div>
            <div className="min-w-0">
              <div className="truncate text-base font-semibold">AI Kod Labı</div>
              <div className="truncate text-xs text-[#607089]">
                Python kütüphaneleriyle kodla, sonucu simülasyona dönüştür
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <HeaderMetric label="Katman" value={layerText} />
            <HeaderMetric label="Loss" value={activeLoss === null ? "-" : formatNumber(activeLoss, 6)} />
            <select
              className="h-9 rounded-md border border-[#cbd5e1] bg-white px-2 text-xs font-semibold text-[#334155]"
              value={visualizationMode}
              onChange={(event) => setVisualizationMode(event.target.value as VisualizationMode)}
              aria-label="Görselleştirme modu"
            >
              <option value="weights">Ağırlık</option>
              <option value="gradients">Gradient</option>
              <option value="corrections">Düzeltme</option>
            </select>
            <button
              type="button"
              className="inline-flex h-9 items-center gap-2 rounded-md border border-[#cbd5e1] bg-white px-3 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
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

        <div className="grid min-h-0 grid-cols-[430px_minmax(0,1fr)] gap-px bg-[#d7dde8]">
          <aside className="min-h-0 overflow-y-auto bg-white px-4 py-4">
            <AiCodeLab onApplyResult={applyResult} applyLabel="Görselleştir" />
          </aside>

          <section className="relative min-h-0 overflow-hidden bg-[#eef3f8]">
            {imported ? (
              <ModelSurface3D
                task={imported.task}
                network={imported.network}
                data={imported.data}
                trace={imported.trace}
                phase={phase}
                visualizationMode={visualizationMode}
                selected={selected}
                hovered={hovered}
                onSelect={setSelected}
                onHover={setHovered}
                onOpenDetail={setSelected}
              />
            ) : (
              <div className="flex h-full items-center justify-center">
                <div className="w-[460px] rounded-md border border-[#dbe3ee] bg-white p-5 shadow-sm">
                  <div className="flex items-center gap-2 text-sm font-semibold">
                    <Cuboid className="h-5 w-5 text-[#2563eb]" />
                    Model Simülasyonu
                  </div>
                  <div className="mt-2 text-sm leading-6 text-[#526070]">
                    Soldaki Python challenge&apos;ını çalıştırıp sonucu görselleştirince eğitilen nöronlar burada
                    ayrı bir sahnede açılır.
                  </div>
                </div>
              </div>
            )}

            <div className="pointer-events-none absolute bottom-5 right-5 w-[360px] rounded-md border border-white/70 bg-white/[0.92] px-3 py-2 text-[11px] leading-5 text-[#526070] shadow-sm backdrop-blur">
              <div className="flex items-center gap-2 font-semibold text-[#18202f]">
                <Activity className="h-3.5 w-3.5 text-[#2563eb]" />
                {imported?.challenge.title ?? "Python sonucu bekleniyor"}
              </div>
              <div>{selectionSummary(activeSelection, imported?.trace ?? null)}</div>
              {importError && <div className="mt-1 text-[#b91c1c]">{importError}</div>}
              {imported && (
                <div className="mt-1 flex items-center gap-2">
                  <Layers3 className="h-3.5 w-3.5 text-[#64748b]" />
                  {imported.network.getLayerSizes().join(" -> ")}
                </div>
              )}
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
        </div>
      </div>
    </main>
  );
}

function HeaderMetric({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-24 rounded-md border border-[#dbe3ee] bg-[#fbfdff] px-3 py-1">
      <div className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        {label}
      </div>
      <div className="max-w-44 truncate text-sm font-semibold" title={value}>
        {value}
      </div>
    </div>
  );
}
