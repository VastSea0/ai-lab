"use client";

import { line, scaleLinear } from "d3";
import {
  Activity,
  BarChart3,
  Box,
  BrainCircuit,
  BookOpen,
  Code2,
  Database,
  Eye,
  FlaskConical,
  ImageIcon,
  Maximize2,
  MessageSquareText,
  Minus,
  MousePointerClick,
  Pause,
  Play,
  Plus,
  RotateCcw,
  SlidersHorizontal,
  Sigma,
  StepForward,
  Trash2,
  Waves,
  Zap
} from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ActivationName,
  DataPoint,
  EdgeSnapshot,
  LayerConfig,
  NeuralNetwork,
  NeuronSnapshot,
  Selection,
  TrainingPhase,
  TrainingTrace,
  activationLabel,
  activationValue,
  edgeId,
  formatNumber,
  sanitizePointValue
} from "@/lib/ml/network";
import type { ConceptId } from "@/lib/ml/concepts";
import { ACTIVATION_CONCEPT_IDS } from "@/lib/ml/concepts";
import {
  DEFAULT_NLP_CONFIG,
  NLP_CLASS_NAMES,
  NLP_VOCABULARY,
  buildNlpVectorizer,
  encodeText,
  parseTextDataset,
  tokenizeText,
  vectorizeExamples,
  vectorizeText,
  type NlpVectorizerConfig,
  type NlpVectorizerModel,
  type TextExample,
} from "@/lib/ml/nlp";
import { TASKS, Task, TaskId } from "@/lib/ml/tasks";
import { DatasetImport } from "@/components/lab/DatasetImport";
import { ModelSurface3D } from "@/components/lab/canvas/ModelSurface3D";
import { StepExplorer } from "@/components/lab/StepExplorer";
import { ConceptBrowser, ConceptDrawer } from "@/components/lab/concepts/ConceptDrawer";
import { useConceptDrawer } from "@/components/lab/hooks/useConceptDrawer";
import {
  DrawerPanel,
  IconToolbar,
  MetricStrip,
  PanelTabs,
  StatusToast,
  WorkbenchShell,
} from "@/components/lab/ui/Workbench";
import { buildEpochTraceRecord, type EpochTraceRecord } from "@/lib/ml/trace";
import type { ConceptMode, DatasetMetadata, VisualizationMode } from "@/lib/ml/lab-types";
import { lessonsForTask } from "@/lib/ml/lessons";
import { presetsForTask } from "@/lib/ml/presets";
import { simulateLearningRates } from "@/lib/ml/experiments";

interface ModelState {
  network: NeuralNetwork;
  trace: TrainingTrace;
  lossHistory: number[];
  epoch: number;
  seed: number;
  history: EpochTraceRecord[];
}

function firstTrace(network: NeuralNetwork, data: DataPoint[]) {
  const first = data[0] ?? {
    id: "empty",
    inputs: Array(network.getLayerSizes()[0]).fill(0),
    targets: Array(network.getLayerSizes().at(-1) ?? 1).fill(0)
  };
  return network.inspect(first.inputs, first.targets);
}

function createModelState(seed: number, task: Task, data = task.data): ModelState {
  const network = NeuralNetwork.create(task.defaultLayers, seed);
  const trace = firstTrace(network, data);
  return {
    network,
    trace,
    lossHistory: [network.evaluateLoss(data)],
    epoch: 0,
    seed,
    history: []
  };
}

function targetName(task: Task, target: number[]) {
  if (task.outputType === "regression") return formatNumber(target[0] ?? 0, 3);
  const index = target.indexOf(Math.max(...target));
  return task.classNames?.[index] ?? `Sınıf ${index}`;
}

function predictionName(task: Task, prediction: number[]) {
  if (task.outputType === "regression") return formatNumber(prediction[0] ?? 0, 3);
  const index = prediction.indexOf(Math.max(...prediction));
  return `${task.classNames?.[index] ?? `Sınıf ${index}`} (${formatNumber(
    prediction[index] ?? 0,
    3
  )})`;
}

function cloneData(data: DataPoint[]) {
  return data.map((point) => ({
    ...point,
    inputs: [...point.inputs],
    targets: [...point.targets]
  }));
}

function oneHotTarget(index: number, size: number) {
  return Array.from({ length: size }, (_, itemIndex) => (itemIndex === index ? 1 : 0));
}

function resizeTargets(targets: number[], size: number) {
  return Array.from({ length: size }, (_, index) => targets[index] ?? 0);
}

function resizeDataTargets(data: DataPoint[], size: number) {
  return data.map((point) => ({
    ...point,
    targets: resizeTargets(point.targets, size),
  }));
}

function nlpTaskWithVectorizer(task: Task, vectorizer: NlpVectorizerModel | null): Task {
  if (task.id !== "sentiment" || !vectorizer) return task;
  const inputSize = Math.max(1, vectorizer.vocabulary.length);
  const outputSize = Math.max(1, vectorizer.labels.length);
  return {
    ...task,
    inputSize,
    outputSize,
    classNames: vectorizer.labels,
    defaultLayers: [
      { size: inputSize },
      { size: Math.min(48, Math.max(10, Math.ceil(inputSize * 0.65))), activation: "relu" },
      { size: Math.min(32, Math.max(6, Math.ceil(inputSize * 0.35))), activation: "relu" },
      { size: outputSize, activation: "sigmoid" },
    ],
  };
}

type CanvasView = "network" | "surface3d";

function initialCanvasView(): CanvasView {
  if (typeof window === "undefined") return "network";
  const params = new URLSearchParams(window.location.search);
  return window.location.hash === "#3d" || params.get("view") === "3d" ? "surface3d" : "network";
}

interface SandboxAppProps {
  initialView?: CanvasView;
}

export function SandboxApp({ initialView = "network" }: SandboxAppProps) {
  const [taskId, setTaskId] = useState<TaskId>("regression");
  const task = useMemo(() => TASKS.find((item) => item.id === taskId) ?? TASKS[0], [taskId]);
  const [customClassNames, setCustomClassNames] = useState<string[]>(() => task.classNames ?? []);
  const [nlpVectorizer, setNlpVectorizer] = useState<NlpVectorizerModel | null>(null);
  const labTask = useMemo<Task>(() => {
    if (task.id === "sentiment") return nlpTaskWithVectorizer(task, nlpVectorizer);
    if (task.outputType !== "classification") return task;
    const classNames = customClassNames.length > 0 ? customClassNames : task.classNames ?? [];
    const outputSize = Math.max(1, classNames.length, task.outputSize);
    return {
      ...task,
      outputSize,
      classNames,
      defaultLayers: task.defaultLayers.map((layer, index, layers) =>
        index === layers.length - 1 ? { ...layer, size: outputSize } : layer
      ),
    };
  }, [customClassNames, nlpVectorizer, task]);
  const [data, setData] = useState<DataPoint[]>(() => cloneData(task.data));
  const [model, setModel] = useState<ModelState>(() => createModelState(1327, task));
  const [learningRate, setLearningRate] = useState(task.defaultLearningRate);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<TrainingPhase>("idle");
  const [focusMode, setFocusMode] = useState(false);
  const [canvasView, setCanvasView] = useState<CanvasView>(initialView);
  const [conceptMode, setConceptMode] = useState<ConceptMode>("beginner");
  const [visualizationMode, setVisualizationMode] = useState<VisualizationMode>("weights");
  const [datasetMetadata, setDatasetMetadata] = useState<DatasetMetadata | null>(null);
  const [selected, setSelected] = useState<Selection | null>(null);
  const [hovered, setHovered] = useState<Selection | null>(null);
  const [detailSelection, setDetailSelection] = useState<Selection | null>(null);
  const { openConceptId, openConcept, closeConcept } = useConceptDrawer();
  const timers = useRef<number[]>([]);

  const clearPhaseTimers = useCallback(() => {
    timers.current.forEach((timer) => window.clearTimeout(timer));
    timers.current = [];
  }, []);

  const playPhaseAnimation = useCallback(() => {
    clearPhaseTimers();
    setPhase("forward");
    timers.current = [
      window.setTimeout(() => setPhase("backward"), 560),
      window.setTimeout(() => setPhase("idle"), 1220)
    ];
  }, [clearPhaseTimers]);

  useEffect(() => () => clearPhaseTimers(), [clearPhaseTimers]);

  useEffect(() => {
    if (canvasView === "surface3d") return undefined;
    if (initialCanvasView() !== "surface3d") return undefined;
    const timer = window.setTimeout(() => setCanvasView("surface3d"), 0);
    return () => window.clearTimeout(timer);
  }, [canvasView]);

  const switchTask = useCallback((nextTaskId: TaskId) => {
    const nextTask = TASKS.find((item) => item.id === nextTaskId) ?? TASKS[0];
    const nextData = cloneData(nextTask.data);
    setRunning(false);
    setSelected(null);
    setHovered(null);
    setDetailSelection(null);
    setDatasetMetadata(null);
    setNlpVectorizer(null);
    setTaskId(nextTask.id);
    setCustomClassNames(nextTask.outputType === "classification" ? nextTask.classNames ?? [] : []);
    setData(nextData);
    setLearningRate(nextTask.defaultLearningRate);
    setModel((previous) => createModelState(previous.seed + 131, nextTask, nextData));
  }, []);

  const rebuildNetwork = useCallback(
    (configs: LayerConfig[]) => {
      setRunning(false);
      setSelected(null);
      setHovered(null);
      setDetailSelection(null);
      setModel((previous) => {
        const seed = previous.seed + 97;
        const next = NeuralNetwork.create(configs, seed);
        return {
          network: next,
          trace: firstTrace(next, data),
          lossHistory: [next.evaluateLoss(data)],
          epoch: 0,
          seed,
          history: []
        };
      });
    },
    [data]
  );

  const resetWeights = useCallback(() => {
    rebuildNetwork(model.network.getLayerConfigs());
  }, [model.network, rebuildNetwork]);

  const updateData = useCallback(
    (nextData: DataPoint[]) => {
      const clean = cloneData(nextData);
      setData(clean);
      setModel((previous) => {
        const network = previous.network.clone();
        return {
          ...previous,
          network,
          trace: firstTrace(network, clean),
          lossHistory: [network.evaluateLoss(clean)],
          epoch: 0,
          history: []
        };
      });
    },
    []
  );

  const resetTaskData = useCallback(() => {
    const clean = cloneData(task.data);
    setDatasetMetadata(null);
    setNlpVectorizer(null);
    setCustomClassNames(task.outputType === "classification" ? task.classNames ?? [] : []);
    setData(clean);
    setLearningRate(task.defaultLearningRate);
    setRunning(false);
    setSelected(null);
    setHovered(null);
    setDetailSelection(null);
    setModel((previous) => createModelState(previous.seed + 89, task, clean));
  }, [task]);

  const addImageExample = useCallback(
    (inputs: number[], labelName: string) => {
      const cleanLabel = labelName.trim() || `Sınıf ${customClassNames.length + 1}`;
      const existingIndex = customClassNames.findIndex(
        (name) => name.toLocaleLowerCase("tr") === cleanLabel.toLocaleLowerCase("tr")
      );
      const nextClassNames =
        existingIndex >= 0 ? customClassNames : [...customClassNames, cleanLabel];
      const labelIndex = existingIndex >= 0 ? existingIndex : nextClassNames.length - 1;
      const outputSize = nextClassNames.length;
      const nextPoint: DataPoint = {
        id: `img${Date.now()}`,
        inputs: [...inputs],
        targets: oneHotTarget(labelIndex, outputSize),
        label: cleanLabel,
      };
      const nextData = resizeDataTargets([...data, nextPoint], outputSize);

      setRunning(false);
      setCustomClassNames(nextClassNames);
      setData(nextData);
      setDatasetMetadata({
        name: "Çizim veri seti",
        rowCount: nextData.length,
        inputColumns: Array.from({ length: inputs.length }, (_, index) => `pixel${index}`),
        targetColumns: nextClassNames,
        normalized: true,
        trainRatio: 1,
        rejectedRows: [],
      });
      setModel((previous) => {
        const currentOutputSize = previous.network.getLayerSizes().at(-1) ?? outputSize;
        const seed = currentOutputSize === outputSize ? previous.seed : previous.seed + 211;
        const network =
          currentOutputSize === outputSize
            ? previous.network.clone()
            : NeuralNetwork.create(
                previous.network.getLayerConfigs().map((config, index, configs) =>
                  index === configs.length - 1
                    ? { ...config, size: outputSize, activation: config.activation ?? "sigmoid" }
                    : { ...config }
                ),
                seed
              );
        return {
          network,
          trace: firstTrace(network, nextData),
          lossHistory: [network.evaluateLoss(nextData)],
          epoch: 0,
          seed,
          history: [],
        };
      });
    },
    [customClassNames, data]
  );

  const applyNlpDataset = useCallback(
    (examples: TextExample[], config: NlpVectorizerConfig) => {
      const vectorizer = buildNlpVectorizer(examples, config);
      const nextData = vectorizeExamples(examples, vectorizer);
      const dynamicTask = nlpTaskWithVectorizer(task, vectorizer);
      setRunning(false);
      setNlpVectorizer(vectorizer);
      setCustomClassNames(vectorizer.labels);
      setData(nextData);
      setLearningRate(0.14);
      setSelected(null);
      setHovered(null);
      setDetailSelection(null);
      setDatasetMetadata({
        name: "NLP metin veri seti",
        rowCount: nextData.length,
        inputColumns: vectorizer.vocabulary,
        targetColumns: vectorizer.labels,
        normalized: true,
        trainRatio: 1,
        rejectedRows: [],
      });
      setModel((previous) => createModelState(previous.seed + 307, dynamicTask, nextData));
    },
    [task]
  );

  const runEpoch = useCallback(() => {
    setModel((previous) => {
      const nextNetwork = previous.network.clone();
      const result = nextNetwork.trainEpochDetailed(data, learningRate);
      const entry = buildEpochTraceRecord(previous.epoch + 1, result, labTask, data, learningRate);
      return {
        ...previous,
        network: nextNetwork,
        trace: result.trace,
        epoch: previous.epoch + 1,
        lossHistory: [...previous.lossHistory, result.lossAfter].slice(-160),
        history: [entry, ...previous.history].slice(0, 10)
      };
    });
    playPhaseAnimation();
  }, [data, labTask, learningRate, playPhaseAnimation]);

  useEffect(() => {
    if (!running) return undefined;
    const interval = window.setInterval(runEpoch, 1250);
    return () => window.clearInterval(interval);
  }, [running, runEpoch]);

  const handleCanvasSelect = useCallback((selection: Selection | null) => {
    setSelected(selection);
    if (selection?.type === "neuron") setFocusMode(true);
  }, []);

  const activeSelection = hovered ?? selected;
  const accuracy = labTask.outputType === "classification" ? model.network.evaluateAccuracy(data) : null;

  return (
    <WorkbenchShell>
      <div className="grid h-full grid-cols-[clamp(238px,23vw,300px)_minmax(0,1fr)_clamp(268px,26vw,340px)] grid-rows-[56px_minmax(0,1fr)_176px] gap-px bg-[#d7dde8]">
        <AppHeader
          task={labTask}
          epoch={model.epoch}
          loss={model.lossHistory.at(-1) ?? 0}
          accuracy={accuracy}
          focusMode={focusMode}
          conceptMode={conceptMode}
          visualizationMode={visualizationMode}
          onConceptModeChange={setConceptMode}
          onVisualizationModeChange={setVisualizationMode}
          onToggleFocus={() => setFocusMode((value) => !value)}
        />

        <ArchitecturePanel
          task={labTask}
          taskId={taskId}
          network={model.network}
          onTaskChange={switchTask}
          onRebuild={rebuildNetwork}
          onApplyPreset={(layers, nextLearningRate) => {
            setLearningRate(nextLearningRate);
            rebuildNetwork(
              layers.map((layer, index) =>
                index === 0
                  ? { ...layer, size: labTask.inputSize }
                  : index === layers.length - 1
                    ? { ...layer, size: labTask.outputSize }
                    : layer
              )
            );
          }}
          onResetWeights={resetWeights}
          onOpenConcept={openConcept}
        />

        <section className="relative overflow-hidden bg-[#eef3f8]">
          {canvasView === "network" ? (
            <>
              <NetworkCanvas
                network={model.network}
                trace={model.trace}
                phase={phase}
                visualizationMode={visualizationMode}
                focusMode={focusMode}
                selected={selected}
                hovered={hovered}
                onSelect={handleCanvasSelect}
                onHover={setHovered}
                onOpenDetail={setDetailSelection}
              />
              <CanvasFocusPanel
                selection={selected}
                trace={model.trace}
                onOpenDetail={setDetailSelection}
                onClose={() => {
                  setSelected(null);
                  setFocusMode(false);
                }}
              />
            </>
          ) : (
            <ModelSurface3D
              task={labTask}
              network={model.network}
              data={data}
              trace={model.trace}
              phase={phase}
              visualizationMode={visualizationMode}
              selected={selected}
              hovered={hovered}
              onSelect={handleCanvasSelect}
              onHover={setHovered}
              onOpenDetail={setDetailSelection}
            />
          )}
          <IconToolbar className="absolute right-4 top-4 z-20" label="Canvas görünümü">
            <button
              type="button"
              className={`inline-flex h-7 items-center gap-1.5 rounded px-2.5 text-[11px] font-semibold ${
                canvasView === "network" ? "bg-[#2563eb] text-white" : "text-[#334155] hover:bg-[#eef4ff]"
              }`}
              onPointerDown={() => setCanvasView("network")}
              onClick={() => setCanvasView("network")}
            >
              <BrainCircuit className="h-4 w-4" />
              Ağ
            </button>
            <button
              type="button"
              className={`inline-flex h-7 items-center gap-1.5 rounded px-2.5 text-[11px] font-semibold ${
                canvasView === "surface3d" ? "bg-[#2563eb] text-white" : "text-[#334155] hover:bg-[#eef4ff]"
              }`}
              onPointerDown={() => setCanvasView("surface3d")}
              onClick={() => setCanvasView("surface3d")}
            >
              <Box className="h-4 w-4" />
              3D
            </button>
          </IconToolbar>
        </section>

        <InspectorPanel
          task={labTask}
          selection={activeSelection}
          trace={model.trace}
          network={model.network}
          data={data}
          metadata={datasetMetadata}
          nlpVectorizer={nlpVectorizer}
          conceptMode={conceptMode}
          onDataChange={updateData}
          onMetadataChange={setDatasetMetadata}
          onResetData={resetTaskData}
          onAddImageExample={addImageExample}
          onApplyNlpDataset={applyNlpDataset}
          onOpenDetail={setDetailSelection}
          onOpenConcept={openConcept}
        />

        <TrainingPanel
          task={labTask}
          epoch={model.epoch}
          history={model.history}
          lossHistory={model.lossHistory}
          learningRate={learningRate}
          running={running}
          phase={phase}
          trace={model.trace}
          network={model.network}
          data={data}
          accuracy={accuracy}
          conceptMode={conceptMode}
          onSelectTarget={setSelected}
          onPhaseChange={setPhase}
          onOpenConcept={openConcept}
          onLearningRateChange={setLearningRate}
          onStep={runEpoch}
          onToggleRun={() => {
            if (!running) runEpoch();
            setRunning((value) => !value);
          }}
          onReset={resetWeights}
        />
        <DetailModal
          selection={detailSelection}
          trace={model.trace}
          learningRate={learningRate}
          conceptMode={conceptMode}
          onOpenConcept={openConcept}
          onClose={() => setDetailSelection(null)}
        />
        <ConceptDrawer
          conceptId={openConceptId}
          conceptMode={conceptMode}
          onOpenConcept={openConcept}
          onClose={closeConcept}
        />
      </div>
    </WorkbenchShell>
  );
}

function AppHeader({
  task,
  epoch,
  loss,
  accuracy,
  focusMode,
  conceptMode,
  visualizationMode,
  onConceptModeChange,
  onVisualizationModeChange,
  onToggleFocus,
}: {
  task: Task;
  epoch: number;
  loss: number;
  accuracy: number | null;
  focusMode: boolean;
  conceptMode: ConceptMode;
  visualizationMode: VisualizationMode;
  onConceptModeChange: (mode: ConceptMode) => void;
  onVisualizationModeChange: (mode: VisualizationMode) => void;
  onToggleFocus: () => void;
}) {
  const metrics = [
    { label: "Epoch", value: String(epoch) },
    { label: "Loss", value: formatNumber(loss, 5) },
    ...(accuracy !== null
      ? [{ label: "Doğruluk", value: `${formatNumber(accuracy * 100, 1)}%`, tone: "good" as const }]
      : []),
  ];

  return (
    <header className="col-span-3 flex h-full items-center justify-between bg-white px-4">
      <div className="flex min-w-0 items-center gap-2.5">
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-[#e8f0ff] text-[#2563eb]">
          <BrainCircuit className="h-[18px] w-[18px]" />
        </div>
        <div className="min-w-0">
          <div className="truncate text-sm font-semibold">Görsel Sinir Ağı Sandbox</div>
          <div className="truncate text-[11px] text-[#607089]">{task.name}</div>
        </div>
      </div>
      <div className="flex min-w-0 items-center gap-2">
        <MetricStrip items={metrics} />
        <Link
          href="/code-lab"
          className="inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md border border-[#cbd5e1] bg-white px-2.5 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
        >
          <Code2 className="h-4 w-4" />
          Kod Labı
        </Link>
        <IconToolbar label="Sandbox ayarları" className="shrink-0">
          <select
            className="h-7 rounded border-0 bg-transparent px-1.5 text-[11px] font-semibold text-[#334155] outline-none"
            value={conceptMode}
            onChange={(event) => onConceptModeChange(event.target.value as ConceptMode)}
            aria-label="Açıklama modu"
          >
            <option value="beginner">Başlangıç</option>
            <option value="math">Matematik</option>
            <option value="engineer">Mühendis</option>
          </select>
          <select
            className="h-7 rounded border-0 bg-transparent px-1.5 text-[11px] font-semibold text-[#334155] outline-none"
            value={visualizationMode}
            onChange={(event) => onVisualizationModeChange(event.target.value as VisualizationMode)}
            aria-label="Canvas görselleştirme modu"
          >
            <option value="weights">Ağırlık</option>
            <option value="gradients">Gradient</option>
            <option value="corrections">Düzeltme</option>
          </select>
        </IconToolbar>
        <button
          type="button"
          className={`inline-flex h-8 shrink-0 items-center gap-1.5 rounded-md border px-2.5 text-xs font-semibold ${
            focusMode
              ? "border-[#2563eb] bg-[#2563eb] text-white"
              : "border-[#cbd5e1] bg-white text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
          }`}
          onClick={onToggleFocus}
        >
          <Eye className="h-4 w-4" />
          Odak
        </button>
      </div>
    </header>
  );
}

interface ArchitecturePanelProps {
  task: Task;
  taskId: TaskId;
  network: NeuralNetwork;
  onTaskChange: (taskId: TaskId) => void;
  onRebuild: (configs: LayerConfig[]) => void;
  onApplyPreset: (configs: LayerConfig[], learningRate: number) => void;
  onResetWeights: () => void;
  onOpenConcept: (conceptId: ConceptId) => void;
}

function ArchitecturePanel({
  task,
  taskId,
  network,
  onTaskChange,
  onRebuild,
  onApplyPreset,
  onResetWeights,
  onOpenConcept
}: ArchitecturePanelProps) {
  const [tab, setTab] = useState<"task" | "layers" | "guide" | "concepts">("task");
  const configs = network.getLayerConfigs();
  const hiddenConfigs = configs.slice(1, -1);

  const rebuildHidden = (nextHidden: LayerConfig[]) => {
    onRebuild([configs[0], ...nextHidden, configs.at(-1) ?? { size: task.outputSize }]);
  };

  const updateActivation = (layerIndex: number, activation: ActivationName) => {
    const next = configs.map((config, index) =>
      index === layerIndex ? { ...config, activation } : { ...config }
    );
    onRebuild(next);
  };

  return (
    <aside className="flex min-h-0 flex-col overflow-hidden bg-white">
      <div className="border-b border-[#e2e8f0] px-3 py-3">
        <PanelTitle icon={<Sigma className="h-4 w-4" />} title="Lab Kurulumu" compact />
        <PanelTabs
          className="mt-3"
          items={[
            { id: "task", label: "Görev" },
            { id: "layers", label: "Mimari" },
            { id: "guide", label: "Rehber" },
            { id: "concepts", label: "Konsept" },
          ]}
          value={tab}
          onChange={setTab}
          compact
        />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
        {tab === "task" && (
          <div className="space-y-3">
            <label className="block">
              <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                Öğrenme görevi
              </span>
              <select
                className="mt-2 h-9 w-full rounded-md border border-[#cbd5e1] bg-white px-2.5 text-xs font-semibold"
                value={taskId}
                onChange={(event) => onTaskChange(event.target.value as TaskId)}
              >
                {TASKS.map((item) => (
                  <option key={item.id} value={item.id}>
                    {item.name} - {item.description}
                  </option>
                ))}
              </select>
            </label>

            <DrawerPanel title="Görev açıklaması" summary={task.description}>
              <div className="text-xs leading-5 text-[#334155]">
                <div className="mb-1 font-semibold">{task.name}</div>
              {task.explanation}
              </div>
            </DrawerPanel>

            <div className="grid grid-cols-2 gap-2">
              <Stat label="giriş" value={`${task.inputSize} özellik`} />
              <Stat label="çıkış" value={`${task.outputSize} değer`} />
              <Stat label="tip" value={task.outputType === "regression" ? "Regresyon" : "Sınıflandırma"} />
              <Stat label="veri" value={`${task.data.length} örnek`} />
            </div>

            <DrawerPanel
              title="Model Preset"
              icon={<SlidersHorizontal className="h-4 w-4" />}
              summary={`${presetsForTask(task.id).length} seçenek`}
            >
              <div className="space-y-2">
                {presetsForTask(task.id).map((preset) => (
                  <button
                    key={preset.id}
                    type="button"
                    className="block w-full rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-2 text-left hover:border-[#2563eb]"
                    onClick={() => onApplyPreset(preset.layers, preset.learningRate)}
                  >
                    <span className="block text-xs font-semibold text-[#18202f]">{preset.name}</span>
                    <span className="mt-0.5 block text-[11px] leading-4 text-[#64748b]">
                      {preset.description} · η={formatNumber(preset.learningRate, 2)}
                    </span>
                  </button>
                ))}
              </div>
            </DrawerPanel>
          </div>
        )}

        {tab === "layers" && (
          <div>
            <div className="space-y-3">
              <LayerRow label="Input" detail={`${task.inputSize} özellik`} count={configs[0].size} locked />
              {hiddenConfigs.map((config, index) => {
                const layerIndex = index + 1;
                return (
                  <LayerRow
                    key={`hidden-${index}`}
                    label={`Hidden ${index + 1}`}
                    detail={activationLabel(config.activation ?? "sigmoid")}
                    count={config.size}
                    activation={config.activation ?? "sigmoid"}
                    onOpenConcept={onOpenConcept}
                    onActivationChange={(activation) => updateActivation(layerIndex, activation)}
                    onDecrease={() => {
                      const next = [...hiddenConfigs];
                      next[index] = { ...next[index], size: Math.max(1, next[index].size - 1) };
                      rebuildHidden(next);
                    }}
                    onIncrease={() => {
                      const max = task.inputSize > 10 ? 18 : 10;
                      const next = [...hiddenConfigs];
                      next[index] = { ...next[index], size: Math.min(max, next[index].size + 1) };
                      rebuildHidden(next);
                    }}
                    onRemove={() => {
                      rebuildHidden(hiddenConfigs.filter((_, itemIndex) => itemIndex !== index));
                    }}
                  />
                );
              })}
              <LayerRow
                label="Output"
                detail={`${task.outputSize} çıktı · ${activationLabel(configs.at(-1)?.activation ?? "linear")}`}
                count={configs.at(-1)?.size ?? task.outputSize}
                locked
              />
            </div>

            <div className="mt-4 grid grid-cols-2 gap-2">
              <ActionButton
                icon={<Plus className="h-4 w-4" />}
                label="Katman"
                onClick={() =>
                  rebuildHidden([
                    ...hiddenConfigs,
                    { size: task.inputSize > 10 ? 8 : 4, activation: "sigmoid" },
                  ])
                }
                disabled={hiddenConfigs.length >= 4}
              />
              <ActionButton
                icon={<RotateCcw className="h-4 w-4" />}
                label="Ağırlık"
                onClick={onResetWeights}
              />
            </div>
          </div>
        )}

        {tab === "guide" && <LearningGuide task={task} onOpenConcept={onOpenConcept} />}
        {tab === "concepts" && <ConceptBrowser onOpenConcept={onOpenConcept} />}
      </div>
    </aside>
  );
}

interface LayerRowProps {
  label: string;
  detail: string;
  count: number;
  locked?: boolean;
  activation?: ActivationName;
  onActivationChange?: (activation: ActivationName) => void;
  onOpenConcept?: (conceptId: ConceptId) => void;
  onDecrease?: () => void;
  onIncrease?: () => void;
  onRemove?: () => void;
}

function LayerRow({
  label,
  detail,
  count,
  locked,
  activation,
  onActivationChange,
  onOpenConcept,
  onDecrease,
  onIncrease,
  onRemove
}: LayerRowProps) {
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-sm font-semibold">{label}</div>
          <div className="mt-0.5 text-xs text-[#64748b]">{detail}</div>
        </div>
        {!locked && (
          <IconButton title="Katmanı sil" onClick={onRemove}>
            <Trash2 className="h-4 w-4" />
          </IconButton>
        )}
      </div>
      <div className="flex items-center justify-between">
        <IconButton title="Nöron azalt" disabled={locked || count <= 1} onClick={onDecrease}>
          <Minus className="h-4 w-4" />
        </IconButton>
        <div className="h-9 min-w-16 rounded-md border border-[#cbd5e1] bg-white px-4 py-2 text-center text-sm font-semibold">
          {count}
        </div>
        <IconButton title="Nöron artır" disabled={locked} onClick={onIncrease}>
          <Plus className="h-4 w-4" />
        </IconButton>
      </div>
      {activation && onActivationChange && (
        <div className="mt-3 grid grid-cols-[1fr_auto] gap-2">
          <select
            className="h-9 w-full rounded-md border border-[#cbd5e1] bg-white px-2 text-xs"
            value={activation}
            onChange={(event) => onActivationChange(event.target.value as ActivationName)}
          >
            <option value="sigmoid">Sigmoid</option>
            <option value="tanh">Tanh</option>
            <option value="relu">ReLU</option>
          </select>
          <IconButton
            title="Aktivasyonu öğren"
            onClick={() => onOpenConcept?.(ACTIVATION_CONCEPT_IDS[activation])}
          >
            <BookOpen className="h-4 w-4" />
          </IconButton>
        </div>
      )}
    </div>
  );
}

function LearningGuide({
  task,
  onOpenConcept,
}: {
  task: Task;
  onOpenConcept: (conceptId: ConceptId) => void;
}) {
  const lessons = lessonsForTask(task);
  const imageCopy =
    task.id === "sentiment"
      ? "NLP görevinde metin önce token'lara ayrılır, sonra kelime var/yok vektörüne çevrilir. Bu vektördeki her değer bir input nöronudur."
      : task.id === "digit"
      ? "Bu görevde her piksel bir giriş nöronudur. Beyaz piksel 0, dolu piksel 1 gibi düşünülür; ağ çizgileri ve şekilleri sayı tahminine çevirir."
      : "Görüntü tanımada da aynı matematik çalışır: x değerleri piksel parlaklıkları olur, ağ bunlardan kenar, çizgi ve sınıf sinyalleri üretmeyi öğrenir.";

  return (
    <div className="space-y-3">
      <PanelTitle icon={<BookOpen className="h-5 w-5" />} title="Başlangıç Rehberi" compact />
      <GuideItem
        title="1. Forward pass"
        text="Girdi değerleri çizgilerden akar. Her nöron önce Σ(x×w)+b hesabını yapar, sonra aktivasyon fonksiyonundan geçirir."
        conceptId="forward-pass"
        onOpenConcept={onOpenConcept}
      />
      <GuideItem
        title="2. Loss"
        text="Tahmin hedefe uzaksa loss büyür. Loss, ağın ne kadar yanıldığını tek bir sayıya indirir."
        conceptId="loss"
        onOpenConcept={onOpenConcept}
      />
      <GuideItem
        title="3. Backpropagation"
        text="Hata geriye doğru paylaşılır. Büyük katkı yapan bağlantı daha büyük gradient alır ve ağırlığı daha fazla değişir."
        conceptId="backprop"
        onOpenConcept={onOpenConcept}
      />
      <GuideItem
        title={task.id === "sentiment" ? "4. NLP mantığı" : "4. Resim mantığı"}
        text={imageCopy}
        conceptId={task.id === "sentiment" ? "nlp" : "classification"}
        onOpenConcept={onOpenConcept}
      />
      <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
        <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Ders Akışı
        </div>
        <div className="space-y-2">
          {lessons.map((lesson, index) => (
            <div key={lesson.id} className="grid grid-cols-[22px_1fr] gap-2 text-xs leading-5">
              <span className="flex h-5 w-5 items-center justify-center rounded-full bg-[#e8f0ff] text-[10px] font-bold text-[#2563eb]">
                {index + 1}
              </span>
              <span>
                <strong className="text-[#18202f]">{lesson.title}</strong>
                <span className="block text-[#526070]">{lesson.text}</span>
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function GuideItem({
  title,
  text,
  conceptId,
  onOpenConcept,
}: {
  title: string;
  text: string;
  conceptId?: ConceptId;
  onOpenConcept?: (conceptId: ConceptId) => void;
}) {
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
      <div className="flex items-center justify-between gap-2">
        <div className="text-xs font-semibold text-[#18202f]">{title}</div>
        {conceptId && onOpenConcept && (
          <button
            type="button"
            className="text-[11px] font-semibold text-[#2563eb]"
            onClick={() => onOpenConcept(conceptId)}
          >
            Öğren
          </button>
        )}
      </div>
      <div className="mt-1 text-xs leading-5 text-[#526070]">{text}</div>
    </div>
  );
}

interface NetworkCanvasProps {
  network: NeuralNetwork;
  trace: TrainingTrace;
  phase: TrainingPhase;
  visualizationMode: VisualizationMode;
  focusMode: boolean;
  selected: Selection | null;
  hovered: Selection | null;
  onSelect: (selection: Selection | null) => void;
  onHover: (selection: Selection | null) => void;
  onOpenDetail: (selection: Selection) => void;
}

function NetworkCanvas({
  network,
  trace,
  phase,
  visualizationMode,
  focusMode,
  selected,
  hovered,
  onSelect,
  onHover,
  onOpenDetail
}: NetworkCanvasProps) {
  const width = 980;
  const height = 612;
  const positions = useMemo(() => {
    const map = new Map<string, { x: number; y: number; radius: number }>();
    const left = 92;
    const right = width - 92;
    const layerGap =
      network.layers.length <= 1 ? 0 : (right - left) / (network.layers.length - 1);

    network.layers.forEach((layer, layerIndex) => {
      const neuronCount = layer.neurons.length;
      const usableHeight = height - 170;
      const gap = neuronCount <= 1 ? 0 : Math.min(64, usableHeight / (neuronCount - 1));
      const radius = neuronCount > 16 ? 8 : neuronCount > 10 ? 11 : Math.max(15, Math.min(23, gap * 0.28 || 22));
      const startY = height / 2 - (gap * (neuronCount - 1)) / 2 + 16;
      layer.neurons.forEach((neuron, neuronIndex) => {
        map.set(neuron.id, {
          x: left + layerIndex * layerGap,
          y: startY + neuronIndex * gap,
          radius
        });
      });
    });
    return map;
  }, [network.layers]);

  const neuronTrace = useMemo(
    () => new Map(trace.neurons.map((neuron) => [neuron.id, neuron])),
    [trace.neurons]
  );
  const edgeTrace = useMemo(
    () => new Map(trace.edges.map((edge) => [edge.id, edge])),
    [trace.edges]
  );
  const activeSelection = hovered ?? selected;
  const viewSelection = focusMode ? selected ?? hovered : activeSelection;
  const focusedViewBox = useMemo(() => {
    if (!focusMode || !viewSelection) return `0 0 ${width} ${height}`;
    if (viewSelection.type === "neuron") {
      const neuron = network.layers[viewSelection.layerIndex]?.neurons[viewSelection.neuronIndex];
      const pos = neuron ? positions.get(neuron.id) : undefined;
      if (!pos) return `0 0 ${width} ${height}`;
      const viewWidth = 360;
      const viewHeight = 250;
      return `${Math.max(0, Math.min(width - viewWidth, pos.x - viewWidth / 2))} ${Math.max(
        0,
        Math.min(height - viewHeight, pos.y - viewHeight / 2)
      )} ${viewWidth} ${viewHeight}`;
    }
    const from = network.layers[viewSelection.fromLayerIndex]?.neurons[viewSelection.fromNeuronIndex];
    const to = network.layers[viewSelection.toLayerIndex]?.neurons[viewSelection.toNeuronIndex];
    const fromPos = from ? positions.get(from.id) : undefined;
    const toPos = to ? positions.get(to.id) : undefined;
    if (!fromPos || !toPos) return `0 0 ${width} ${height}`;
    const midX = (fromPos.x + toPos.x) / 2;
    const midY = (fromPos.y + toPos.y) / 2;
    const viewWidth = 430;
    const viewHeight = 285;
    return `${Math.max(0, Math.min(width - viewWidth, midX - viewWidth / 2))} ${Math.max(
      0,
      Math.min(height - viewHeight, midY - viewHeight / 2)
    )} ${viewWidth} ${viewHeight}`;
  }, [focusMode, network.layers, positions, viewSelection]);

  return (
    <svg
      className="h-full w-full"
      viewBox={focusedViewBox}
      role="img"
      aria-label="Sinir ağı görselleştirmesi"
      onMouseLeave={() => onHover(null)}
    >
      <defs>
        <linearGradient id="canvasWash" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stopColor="#f8fbff" />
          <stop offset="52%" stopColor="#eef6f2" />
          <stop offset="100%" stopColor="#fff7ed" />
        </linearGradient>
      </defs>
      <rect width={width} height={height} fill="url(#canvasWash)" />

      {network.layers.map((layer) => {
        const first = positions.get(layer.neurons[0]?.id);
        return first ? (
          <text
            key={layer.id}
            x={first.x}
            y={78}
            textAnchor="middle"
            className="fill-[#526070] text-[12px] font-semibold uppercase"
          >
            {layer.kind} · {layer.neurons.length}
          </text>
        ) : null;
      })}

      <g>
        {network.weights.flatMap((matrix, layerIndex) =>
          matrix.flatMap((row, fromIndex) =>
            row.map((weight, toIndex) => {
              const from = network.layers[layerIndex].neurons[fromIndex];
              const to = network.layers[layerIndex + 1].neurons[toIndex];
              const fromPos = positions.get(from.id);
              const toPos = positions.get(to.id);
              if (!fromPos || !toPos) return null;

              const id = edgeId(layerIndex, fromIndex, layerIndex + 1, toIndex);
              const selection: Selection = {
                type: "edge",
                id,
                fromLayerIndex: layerIndex,
                fromNeuronIndex: fromIndex,
                toLayerIndex: layerIndex + 1,
                toNeuronIndex: toIndex
              };
              const traceEdge = edgeTrace.get(id);
              const activeNeuronId =
                activeSelection?.type === "neuron" ? activeSelection.id : null;
              const connectedToActiveNeuron =
                activeNeuronId !== null && (from.id === activeNeuronId || to.id === activeNeuronId);
              const active = selected?.id === id || hovered?.id === id || connectedToActiveNeuron;
              const metric =
                visualizationMode === "weights"
                  ? weight
                  : visualizationMode === "gradients"
                    ? traceEdge?.gradient ?? 0
                    : visualizationMode === "corrections"
                      ? traceEdge?.correction ?? 0
                      : weight;
              const metricMagnitude = Math.min(
                1,
                Math.abs(metric) / (visualizationMode === "weights" || visualizationMode === "rl-reward" ? 2.4 : 0.35)
              );
              const stroke =
                visualizationMode === "corrections"
                  ? metric >= 0
                    ? "#059669"
                    : "#f97316"
                  : metric >= 0
                    ? "#2563eb"
                    : "#e11d48";
              const widthByWeight = 0.85 + metricMagnitude * 5.8;
              const contributionWidth =
                1 + Math.min(6, Math.abs(traceEdge?.contribution ?? 0) * 5.5);
              const errorWidth =
                1.1 + Math.min(9, Math.abs(traceEdge?.gradient ?? 0) * 22);

              return (
                <g key={id}>
                  <line
                    x1={fromPos.x}
                    y1={fromPos.y}
                    x2={toPos.x}
                    y2={toPos.y}
                    stroke={stroke}
                    strokeWidth={widthByWeight}
                    strokeOpacity={active ? 0.82 : 0.12 + metricMagnitude * 0.34}
                    strokeLinecap="round"
                  />
                  {phase === "forward" && (
                    <line
                      className="forward-pulse"
                      x1={fromPos.x}
                      y1={fromPos.y}
                      x2={toPos.x}
                      y2={toPos.y}
                      stroke="#0891b2"
                      strokeWidth={contributionWidth}
                      strokeOpacity={0.72}
                      strokeLinecap="round"
                      strokeDasharray="9 17"
                    />
                  )}
                  {phase === "backward" && (
                    <line
                      className="backward-pulse"
                      x1={toPos.x}
                      y1={toPos.y}
                      x2={fromPos.x}
                      y2={fromPos.y}
                      stroke="#f97316"
                      strokeWidth={errorWidth}
                      strokeOpacity={0.72}
                      strokeLinecap="round"
                      strokeDasharray="11 15"
                    />
                  )}
                  <line
                    x1={fromPos.x}
                    y1={fromPos.y}
                    x2={toPos.x}
                    y2={toPos.y}
                    stroke="transparent"
                    strokeWidth={14}
                    className="cursor-pointer"
                    aria-label={`w = ${formatNumber(weight)}`}
                    onMouseEnter={() => onHover(selection)}
                    onMouseLeave={() => onHover(null)}
                    onClick={() => onSelect(selection)}
                    onDoubleClick={() => onOpenDetail(selection)}
                  />
                </g>
              );
            })
          )
        )}
      </g>

      <g>
        {network.layers.flatMap((layer) =>
          layer.neurons.map((neuron) => {
            const pos = positions.get(neuron.id);
            const snapshot = neuronTrace.get(neuron.id);
            if (!pos || !snapshot) return null;

            const selection: Selection = {
              type: "neuron",
              id: neuron.id,
              layerIndex: neuron.layerIndex,
              neuronIndex: neuron.neuronIndex
            };
            const active = selected?.id === neuron.id || hovered?.id === neuron.id;
            const fill =
              neuron.layerKind === "input"
                ? "#e0f2fe"
                : neuron.layerKind === "output"
                  ? "#fee2e2"
                  : "#fef3c7";
            const stroke =
              neuron.layerKind === "input"
                ? "#0284c7"
                : neuron.layerKind === "output"
                  ? "#dc2626"
                  : "#d97706";
            const tiny = pos.radius < 12;

            return (
              <g
                key={neuron.id}
                className="cursor-pointer"
                aria-label={`${snapshot.formula}; a=${formatNumber(
                  snapshot.value
                )}; delta=${formatNumber(snapshot.delta)}`}
                onMouseEnter={() => onHover(selection)}
                onMouseLeave={() => onHover(null)}
                onClick={() => onSelect(selection)}
                onDoubleClick={() => onOpenDetail(selection)}
              >
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={pos.radius}
                  fill={fill}
                  stroke={active ? "#111827" : stroke}
                  strokeWidth={active ? 3 : 1.7}
                  className={phase === "forward" && neuron.layerKind !== "input" ? "neuron-pop" : ""}
                />
                {!tiny && (
                  <>
                    <text
                      x={pos.x}
                      y={pos.y + 4}
                      textAnchor="middle"
                      className="pointer-events-none fill-[#18202f] text-[10px] font-bold"
                    >
                      {formatNumber(snapshot.value, 2)}
                    </text>
                    <text
                      x={pos.x}
                      y={pos.y + pos.radius + 14}
                      textAnchor="middle"
                      className="pointer-events-none fill-[#526070] text-[9px] font-medium"
                    >
                      {neuron.layerKind === "input"
                        ? `x${neuron.neuronIndex + 1}`
                        : `Σ=${formatNumber(snapshot.z, 2)}`}
                    </text>
                  </>
                )}
              </g>
            );
          })
        )}
      </g>
    </svg>
  );
}

interface InspectorPanelProps {
  task: Task;
  selection: Selection | null;
  trace: TrainingTrace;
  network: NeuralNetwork;
  data: DataPoint[];
  metadata: DatasetMetadata | null;
  nlpVectorizer: NlpVectorizerModel | null;
  conceptMode: ConceptMode;
  onDataChange: (data: DataPoint[]) => void;
  onMetadataChange: (metadata: DatasetMetadata | null) => void;
  onResetData: () => void;
  onAddImageExample: (inputs: number[], label: string) => void;
  onApplyNlpDataset: (examples: TextExample[], config: NlpVectorizerConfig) => void;
  onOpenDetail: (selection: Selection) => void;
  onOpenConcept: (conceptId: ConceptId) => void;
}

function InspectorPanel({
  task,
  selection,
  trace,
  network,
  data,
  metadata,
  nlpVectorizer,
  conceptMode,
  onDataChange,
  onMetadataChange,
  onResetData,
  onAddImageExample,
  onApplyNlpDataset,
  onOpenDetail,
  onOpenConcept,
}: InspectorPanelProps) {
  const [tab, setTab] = useState<"inspect" | "data" | "learn">("inspect");
  const neuron =
    selection?.type === "neuron"
      ? trace.neurons.find((item) => item.id === selection.id) ?? null
      : null;
  const edge =
    selection?.type === "edge"
      ? trace.edges.find((item) => item.id === selection.id) ?? null
      : null;

  return (
    <aside className="flex min-h-0 flex-col overflow-hidden bg-white">
      <div className="border-b border-[#e2e8f0] px-3 py-3">
        <PanelTitle icon={<Activity className="h-4 w-4" />} title="Denetçi" compact />
        <PanelTabs
          className="mt-3"
          items={[
            { id: "inspect", label: "Inspect" },
            { id: "data", label: "Veri" },
            { id: "learn", label: "Learn" },
          ]}
          value={tab}
          onChange={setTab}
          compact
        />
      </div>

      {tab === "inspect" ? (
        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
          {selection && (
            <button
              type="button"
              className="mb-3 inline-flex h-9 items-center gap-2 rounded-md border border-[#cbd5e1] bg-white px-3 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
              onClick={() => onOpenDetail(selection)}
            >
              <Maximize2 className="h-4 w-4" />
              Yakın incele
            </button>
          )}
          {neuron && <NeuronInspector neuron={neuron} conceptMode={conceptMode} trace={trace} onOpenConcept={onOpenConcept} />}
          {edge && <EdgeInspector edge={edge} conceptMode={conceptMode} onOpenConcept={onOpenConcept} />}
          {!neuron && !edge && <TraceSummary task={task} trace={trace} conceptMode={conceptMode} onOpenConcept={onOpenConcept} />}
        </div>
      ) : tab === "data" ? (
        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
          <DatasetPanel
            task={task}
            network={network}
            data={data}
            metadata={metadata}
            nlpVectorizer={nlpVectorizer}
            onDataChange={onDataChange}
            onMetadataChange={onMetadataChange}
            onResetData={onResetData}
            onAddImageExample={onAddImageExample}
            onApplyNlpDataset={onApplyNlpDataset}
          />
        </div>
      ) : (
        <div className="min-h-0 flex-1 overflow-y-auto px-3 py-3">
          <div className="space-y-3">
            <LearningGuide task={task} onOpenConcept={onOpenConcept} />
            <DrawerPanel title="Konsept Kütüphanesi" defaultOpen>
              <ConceptBrowser onOpenConcept={onOpenConcept} />
            </DrawerPanel>
          </div>
        </div>
      )}
    </aside>
  );
}

function TraceSummary({
  task,
  trace,
  conceptMode,
  onOpenConcept,
}: {
  task: Task;
  trace: TrainingTrace;
  conceptMode: ConceptMode;
  onOpenConcept: (conceptId: ConceptId) => void;
}) {
  const prediction = predictionName(task, trace.prediction);
  const target = targetName(task, trace.target);
  const inputPreview = trace.input
    .slice(0, 6)
    .map((value) => formatNumber(value, 2))
    .join(", ");

  return (
    <div className="space-y-3">
      <div className="rounded-md border border-[#dbe3ee] p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Son örnek
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <Stat label="girdi" value={trace.input.length > 6 ? `${inputPreview}...` : inputPreview} />
          <Stat label="hedef" value={target} />
          <Stat label="tahmin" value={prediction} />
          <Stat label="loss" value={formatNumber(trace.loss, 5)} />
        </div>
      </div>
      <FormulaBox
        title="Loss hesabı"
        lines={
          task.outputType === "regression"
            ? [
                "L = 1/2 × (ŷ - y)^2",
                `L = 1/2 × (${formatNumber(trace.prediction[0] ?? 0)} - ${formatNumber(
                  trace.target[0] ?? 0
                )})^2`
              ]
            : [
                "L = 1/2 × Σ(çıktı - hedef)^2",
                `çıktı = [${trace.prediction.map((value) => formatNumber(value, 3)).join(", ")}]`,
                `hedef = [${trace.target.map((value) => formatNumber(value, 0)).join(", ")}]`
              ]
        }
      />
      <ConceptChips
        items={
          task.id === "sentiment"
            ? ["nlp", "bag-of-words", "deep-learning", "loss"]
            : ["loss", "forward-pass", task.outputType === "regression" ? "regression" : "classification"]
        }
        onOpenConcept={onOpenConcept}
      />
      <ExplainBox
        title="Ne anlama geliyor?"
        text={
          conceptMode === "math"
            ? "Bu loss, 1/2 × kare hata toplamıdır. Türevi basitleştirmek için 1/2 kullanılır; ∂L/∂ŷ = ŷ - y olur."
            : conceptMode === "engineer"
              ? "Trace snapshot'ı son örneğin forward, delta ve update bilgilerini UI için dondurur. Eğitim geçmişindeki adımlar ayrıca saklanır."
              : "Loss küçülüyorsa ağ, veri setindeki örnekleri daha az hatayla açıklamaya başlıyor. Tek bir epoch, veri setindeki tüm örneklerden bir kez geçmek demektir."
        }
      />
    </div>
  );
}

interface OutgoingSignal {
  edge: EdgeSnapshot;
  toNeuron?: NeuronSnapshot;
}

function outgoingSignals(trace: TrainingTrace, neuron: NeuronSnapshot): OutgoingSignal[] {
  const neurons = new Map(trace.neurons.map((item) => [item.id, item]));
  return trace.edges
    .filter((edge) => edge.fromNeuronId === neuron.id)
    .map((edge) => ({ edge, toNeuron: neurons.get(edge.toNeuronId) }))
    .sort((a, b) => Math.abs(b.edge.contribution) - Math.abs(a.edge.contribution));
}

function CanvasFocusPanel({
  selection,
  trace,
  onOpenDetail,
  onClose,
}: {
  selection: Selection | null;
  trace: TrainingTrace;
  onOpenDetail: (selection: Selection) => void;
  onClose: () => void;
}) {
  const neuron =
    selection?.type === "neuron" ? trace.neurons.find((item) => item.id === selection.id) ?? null : null;
  if (!selection || !neuron) return null;

  return (
    <div className="absolute bottom-4 left-4 z-20 w-[320px]">
      <StatusToast
        title={`Odak: L${neuron.layerIndex} N${neuron.neuronIndex}`}
        icon={<MousePointerClick className="h-4 w-4" />}
        onClose={onClose}
        action={
          <button
            type="button"
            className="inline-flex h-8 w-full items-center justify-center gap-2 rounded-md bg-[#2563eb] px-3 text-[11px] font-semibold text-white hover:bg-[#1d4ed8]"
            onClick={() => onOpenDetail(selection)}
          >
            <Maximize2 className="h-3.5 w-3.5" />
            Tam hesap
          </button>
        }
      >
        <div className="grid grid-cols-3 gap-1.5">
          <Stat label="z" value={formatNumber(neuron.z, 3)} />
          <Stat label="a" value={formatNumber(neuron.value, 3)} />
          <Stat label="δ" value={formatNumber(neuron.delta, 3)} />
        </div>
      </StatusToast>
    </div>
  );
}

function NeuronInspector({
  neuron,
  conceptMode,
  trace,
  onOpenConcept,
}: {
  neuron: NeuronSnapshot;
  conceptMode: ConceptMode;
  trace?: TrainingTrace;
  onOpenConcept?: (conceptId: ConceptId) => void;
}) {
  const topIncoming = [...neuron.incoming]
    .sort((a, b) => Math.abs(b.product) - Math.abs(a.product))
    .slice(0, 6);
  const topOutgoing = trace ? outgoingSignals(trace, neuron).slice(0, 6) : [];

  return (
    <div className="space-y-3">
      <div>
        <div className="text-sm font-semibold">
          {neuron.layerKind} · L{neuron.layerIndex} N{neuron.neuronIndex}
        </div>
        <button
          type="button"
          className="text-xs font-semibold text-[#2563eb]"
          onClick={() => onOpenConcept?.(ACTIVATION_CONCEPT_IDS[neuron.activation])}
        >
          {activationLabel(neuron.activation)}
        </button>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Σ(x×w)+b" value={formatNumber(neuron.z)} />
        <Stat label="aktivasyon" value={formatNumber(neuron.value)} />
        <Stat label="bias" value={formatNumber(neuron.bias)} />
        <Stat label="δ hata" value={formatNumber(neuron.delta)} />
        <Stat label="türev" value={formatNumber(neuron.derivative)} />
        <Stat label="∂L/∂b" value={formatNumber(neuron.gradientBias)} />
      </div>
      <FormulaBox title="Nöron hesabı" lines={[neuron.formula, `a = ${neuron.activation}(Σ)`]} />
      {onOpenConcept && (
        <ConceptChips
          items={["z", "weight", "bias", "activation", "derivative", "delta"]}
          onOpenConcept={onOpenConcept}
        />
      )}
      <ActivationMiniChart neuron={neuron} />
      <ExplainBox
        title="Bu nöron ne yapıyor?"
        text={
          conceptMode === "math"
            ? "z ağırlıklı toplamdır; a=f(z) çıktıdır. Backprop sırasında δ=∂L/∂z tutulur ve bias gradient'i doğrudan δ olur."
            : conceptMode === "engineer"
              ? "Bu değerler `NeuronSnapshot` içinden gelir. Snapshot eğitim adımından sonra saklandığı için denetçi ve modal aynı gerçeği okur."
              : "Gelen sinyalleri ağırlıklarıyla çarpar, bias ekler ve aktivasyon fonksiyonu ile sonucu sıkıştırır ya da geçirir. δ değeri, bu nöronun son hataya ne kadar pay verdiğini gösterir."
        }
      />
      {topIncoming.length > 0 && (
        <div className="rounded-md border border-[#dbe3ee]">
          <div className="border-b border-[#e2e8f0] px-3 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            En güçlü gelen katkılar
          </div>
          <div className="divide-y divide-[#edf2f7]">
            {topIncoming.map((item) => (
              <div
                key={`${item.fromNeuronId}-${neuron.id}`}
                className="grid grid-cols-[1fr_auto] gap-2 px-3 py-2 text-xs"
              >
                <span className="font-medium">{item.fromNeuronId}</span>
                <span className="text-[#64748b]">
                  {formatNumber(item.inputValue, 3)} × {formatNumber(item.weight, 3)} ={" "}
                  {formatNumber(item.product, 3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
      {topOutgoing.length > 0 && (
        <div className="rounded-md border border-[#dbe3ee]">
          <div className="border-b border-[#e2e8f0] px-3 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            Sonraki katmana giden sinyal
          </div>
          <div className="divide-y divide-[#edf2f7]">
            {topOutgoing.map(({ edge, toNeuron }) => (
              <div key={edge.id} className="px-3 py-2 text-xs">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-medium">
                    L{edge.toLayerIndex} N{edge.toNeuronIndex}
                  </span>
                  <span className="font-mono text-[#64748b]">
                    {formatNumber(neuron.value, 3)} × {formatNumber(edge.weightBefore, 3)} ={" "}
                    {formatNumber(edge.contribution, 3)}
                  </span>
                </div>
                {toNeuron && (
                  <div className="mt-1 text-[11px] text-[#64748b]">
                    Bu katkı hedef nöronun z toplamına eklenir; hedef z={formatNumber(toNeuron.z, 4)}.
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function ActivationMiniChart({ neuron }: { neuron: NeuronSnapshot }) {
  const width = 300;
  const height = 104;
  const xScale = scaleLinear().domain([-3, 3]).range([18, width - 12]);
  const yScale = scaleLinear()
    .domain(neuron.activation === "tanh" ? [-1.2, 1.2] : [-0.15, 1.15])
    .range([height - 16, 10]);
  const values = Array.from({ length: 90 }, (_, index) => {
    const z = -3 + (index / 89) * 6;
    return { z, value: activationValue(neuron.activation, z) };
  });
  const path =
    line<{ z: number; value: number }>()
      .x((point) => xScale(point.z))
      .y((point) => yScale(point.value))(values) ?? "";
  const markerX = xScale(Math.max(-3, Math.min(3, neuron.z)));
  const markerY = yScale(neuron.value);

  return (
    <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
      <div className="mb-2 flex items-center justify-between text-xs">
        <span className="font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Aktivasyon Eğrisi
        </span>
        <span className="font-mono text-[#334155]">f&apos;={formatNumber(neuron.derivative, 4)}</span>
      </div>
      <svg className="h-[104px] w-full" viewBox={`0 0 ${width} ${height}`}>
        <rect width={width} height={height} rx={6} fill="#ffffff" stroke="#dbe3ee" />
        <line x1={18} x2={width - 12} y1={yScale(0)} y2={yScale(0)} stroke="#e2e8f0" />
        <line x1={xScale(0)} x2={xScale(0)} y1={10} y2={height - 16} stroke="#e2e8f0" />
        <path d={path} fill="none" stroke="#2563eb" strokeWidth={2.4} />
        <circle cx={markerX} cy={markerY} r={4.5} fill="#f97316" stroke="#ffffff" strokeWidth={1.5} />
      </svg>
    </div>
  );
}

function ConceptChips({
  items,
  onOpenConcept,
}: {
  items: ConceptId[];
  onOpenConcept: (conceptId: ConceptId) => void;
}) {
  const labels: Record<ConceptId, string> = {
    linear: "linear",
    sigmoid: "sigmoid",
    tanh: "tanh",
    relu: "ReLU",
    weight: "w",
    bias: "bias",
    z: "z",
    activation: "activation",
    loss: "loss",
    gradient: "gradient",
    derivative: "türev",
    delta: "δ",
    "learning-rate": "η",
    epoch: "epoch",
    "forward-pass": "forward",
    backprop: "backprop",
    regression: "regresyon",
    classification: "sınıflandırma",
    normalization: "normalizasyon",
    "deep-learning": "deep learning",
    nlp: "NLP",
    tokenization: "tokenization",
    "bag-of-words": "bag-of-words",
    embedding: "embedding",
  };
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item) => (
        <button
          key={item}
          type="button"
          className="rounded-md border border-[#cbd5e1] bg-white px-2 py-1 text-[11px] font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
          onClick={() => onOpenConcept(item)}
        >
          {labels[item]}
        </button>
      ))}
    </div>
  );
}

function EdgeInspector({
  edge,
  conceptMode,
  onOpenConcept,
}: {
  edge: EdgeSnapshot;
  conceptMode: ConceptMode;
  onOpenConcept?: (conceptId: ConceptId) => void;
}) {
  return (
    <div className="space-y-3">
      <div>
        <div className="text-sm font-semibold">
          L{edge.fromLayerIndex} N{edge.fromNeuronIndex} → L{edge.toLayerIndex} N
          {edge.toNeuronIndex}
        </div>
        <div className="text-xs text-[#64748b]">{edge.id}</div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="ağırlık" value={formatNumber(edge.weightBefore)} />
        <Stat label="katkı" value={formatNumber(edge.contribution)} />
        <Stat label="∂L/∂w" value={formatNumber(edge.gradient)} />
        <Stat label="düzeltme" value={formatNumber(edge.correction)} />
        <Stat label="yeni w" value={formatNumber(edge.weightAfter)} />
        <Stat label="hata etkisi" value={formatNumber(edge.errorSignal)} />
      </div>
      <FormulaBox
        title="Bağlantı güncellemesi"
        lines={[
          "katkı = a_önceki × w",
          "∂L/∂w = a_önceki × δ_sonraki",
          `w_yeni = ${formatNumber(edge.weightBefore)} + ${formatNumber(edge.correction)}`
        ]}
      />
      {onOpenConcept && (
        <ConceptChips
          items={["weight", "gradient", "learning-rate", "delta"]}
          onOpenConcept={onOpenConcept}
        />
      )}
      <ExplainBox
        title="Neden değişti?"
        text={
          conceptMode === "math"
            ? "Ağırlık update kuralı w := w - η∂L/∂w. Burada ∂L/∂w = a_prev × δ_next olarak zincir kuralından gelir."
            : conceptMode === "engineer"
              ? "Edge snapshot hem eski ağırlığı hem hesaplanan correction'ı taşır; canvas heatmap aynı gradient/correction değerlerinden beslenir."
              : "Gradient pozitifse ağırlığı azaltmak loss'u düşürmeye çalışır; gradient negatifse ağırlık artırılır. Learning rate bu adımın büyüklüğünü belirler."
        }
      />
    </div>
  );
}

interface DatasetPanelProps {
  task: Task;
  network: NeuralNetwork;
  data: DataPoint[];
  metadata?: DatasetMetadata | null;
  nlpVectorizer?: NlpVectorizerModel | null;
  onDataChange: (data: DataPoint[]) => void;
  onMetadataChange?: (metadata: DatasetMetadata | null) => void;
  onResetData: () => void;
  onAddImageExample?: (inputs: number[], label: string) => void;
  onApplyNlpDataset?: (examples: TextExample[], config: NlpVectorizerConfig) => void;
}

function DatasetPanel({
  task,
  network,
  data,
  metadata,
  nlpVectorizer,
  onDataChange,
  onMetadataChange,
  onResetData,
  onAddImageExample,
  onApplyNlpDataset,
}: DatasetPanelProps) {
  return (
    <div>
      <PanelTitle
        icon={task.id === "digit" ? <ImageIcon className="h-5 w-5" /> : <Database className="h-5 w-5" />}
        title={task.id === "digit" ? "Resim Verisi" : "Veri"}
        compact
      />
      <DatasetImport
        task={task}
        onDataLoaded={onDataChange}
        onMetadata={onMetadataChange}
        onReset={onResetData}
      />
      {metadata && (
        <div className="mt-3 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3 text-xs leading-5 text-[#526070]">
          <div className="font-semibold text-[#18202f]">{metadata.name}</div>
          <div>{metadata.rowCount} örnek · train oranı {Math.round(metadata.trainRatio * 100)}%</div>
          <div>Input: {metadata.inputColumns.join(", ")} · Target: {metadata.targetColumns.join(", ")}</div>
          {metadata.rejectedRows.length > 0 && (
            <div className="mt-1 text-[#b91c1c]">{metadata.rejectedRows.length} satır atlandı.</div>
          )}
        </div>
      )}
      {task.id === "digit" ? (
        <DigitDatasetPanel
          task={task}
          network={network}
          data={data}
          onDataChange={onDataChange}
          onResetData={onResetData}
          onAddImageExample={onAddImageExample}
        />
      ) : task.id === "sentiment" ? (
        <NlpDatasetPanel
          task={task}
          network={network}
          data={data}
          nlpVectorizer={nlpVectorizer}
          onDataChange={onDataChange}
          onApplyNlpDataset={onApplyNlpDataset}
          onResetData={onResetData}
        />
      ) : (
        <NumericDatasetPanel
          task={task}
          network={network}
          data={data}
          onDataChange={onDataChange}
          onResetData={onResetData}
        />
      )}
    </div>
  );
}

function NumericDatasetPanel({ task, network, data, onDataChange }: DatasetPanelProps) {
  const [draft, setDraft] = useState({ x: 0.42, y: 0.58, target: 0.72, cls: 1 });

  const addPoint = () => {
    const inputs =
      task.inputSize === 1
        ? [sanitizePointValue(draft.x)]
        : [sanitizePointValue(draft.x), sanitizePointValue(draft.y)];
    const targets =
      task.outputSize === 1
        ? [sanitizePointValue(draft.target)]
        : Array.from({ length: task.outputSize }, (_, index) => (index === draft.cls ? 1 : 0));
    onDataChange([
      ...data,
      {
        id: `p${Date.now()}`,
        inputs,
        targets,
        label: task.outputSize === 1 ? formatNumber(targets[0], 2) : task.classNames?.[draft.cls]
      }
    ]);
  };

  return (
    <>
      <DataPlot task={task} network={network} data={data} />
      <div className="mt-3 grid grid-cols-[1fr_1fr_auto] gap-2">
        <NumberInput
          label={task.inputSize === 1 ? "x" : "x"}
          value={draft.x}
          onChange={(value) => setDraft((previous) => ({ ...previous, x: value }))}
        />
        {task.inputSize === 1 ? (
          <NumberInput
            label="hedef y"
            value={draft.target}
            onChange={(value) => setDraft((previous) => ({ ...previous, target: value }))}
          />
        ) : (
          <NumberInput
            label="y"
            value={draft.y}
            onChange={(value) => setDraft((previous) => ({ ...previous, y: value }))}
          />
        )}
        <IconButton title="Nokta ekle" onClick={addPoint} className="mt-5">
          <Plus className="h-4 w-4" />
        </IconButton>
      </div>
      {task.outputSize > 1 && (
        <label className="mt-2 block">
          <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[#64748b]">
            Sınıf
          </span>
          <select
            className="mt-1 h-9 w-full rounded-md border border-[#cbd5e1] bg-white px-2 text-sm"
            value={draft.cls}
            onChange={(event) => setDraft((previous) => ({ ...previous, cls: Number(event.target.value) }))}
          >
            {Array.from({ length: task.outputSize }, (_, index) => (
              <option key={index} value={index}>
                {task.classNames?.[index] ?? `Sınıf ${index}`}
              </option>
            ))}
          </select>
        </label>
      )}
      <PointChips task={task} data={data} onDataChange={onDataChange} />
    </>
  );
}

function DigitDatasetPanel({ task, network, data, onAddImageExample }: DatasetPanelProps) {
  const [pixels, setPixels] = useState<number[]>(() => data[0]?.inputs ?? Array(25).fill(0));
  const [labelText, setLabelText] = useState(task.classNames?.[0] ?? "0");
  const [brushValue, setBrushValue] = useState<0 | 1>(1);
  const [painting, setPainting] = useState(false);
  const prediction = network.predictPure(pixels);
  const labels = task.classNames ?? [];

  const paintPixel = (index: number, value = brushValue) => {
    setPixels((previous) =>
      previous.map((pixel, itemIndex) => (itemIndex === index ? value : pixel))
    );
  };

  const addImage = () => {
    onAddImageExample?.(pixels, labelText);
  };

  return (
    <div className="mt-3 grid grid-cols-[122px_1fr] gap-3">
      <div>
        <div
          className="grid h-[122px] w-[122px] touch-none grid-cols-5 grid-rows-5 gap-1 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-2"
          onMouseLeave={() => setPainting(false)}
          onMouseUp={() => setPainting(false)}
        >
          {pixels.map((value, index) => (
            <button
              key={index}
              type="button"
              aria-label={`Piksel ${index + 1}`}
              className="rounded-[3px] border border-[#cbd5e1]"
              style={{ backgroundColor: `rgba(24, 32, 47, ${0.08 + value * 0.86})` }}
              onMouseDown={() => {
                setPainting(true);
                paintPixel(index);
              }}
              onMouseEnter={() => {
                if (painting) paintPixel(index);
              }}
              onClick={() => paintPixel(index, value > 0 ? 0 : 1)}
            />
          ))}
        </div>
        <div className="mt-2 grid grid-cols-3 gap-1">
          <button
            type="button"
            className={`h-8 rounded-md border text-[11px] font-semibold ${
              brushValue === 1 ? "border-[#2563eb] bg-[#e8f0ff] text-[#2563eb]" : "border-[#cbd5e1] text-[#334155]"
            }`}
            onClick={() => setBrushValue(1)}
          >
            Çiz
          </button>
          <button
            type="button"
            className={`h-8 rounded-md border text-[11px] font-semibold ${
              brushValue === 0 ? "border-[#2563eb] bg-[#e8f0ff] text-[#2563eb]" : "border-[#cbd5e1] text-[#334155]"
            }`}
            onClick={() => setBrushValue(0)}
          >
            Sil
          </button>
          <button
            type="button"
            className="h-8 rounded-md border border-[#cbd5e1] text-[11px] font-semibold text-[#334155]"
            onClick={() => setPixels(Array(25).fill(0))}
          >
            Temizle
          </button>
        </div>
        <div className="mt-2 grid grid-cols-[1fr_auto] gap-2">
          <input
            className="h-9 min-w-0 rounded-md border border-[#cbd5e1] bg-white px-2 text-sm"
            value={labelText}
            placeholder="örn. gülen yüz"
            onChange={(event) => setLabelText(event.target.value)}
            aria-label="Çizim etiketi"
            list="image-label-options"
          />
          <datalist id="image-label-options">
            {labels.map((name) => (
              <option key={name} value={name} />
            ))}
          </datalist>
          <IconButton title="Çizimi veri setine ekle" onClick={addImage}>
            <Plus className="h-4 w-4" />
          </IconButton>
        </div>
        {labels.length > 0 && (
          <select
            className="mt-2 h-8 w-full rounded-md border border-[#cbd5e1] bg-white px-2 text-xs"
            value={labels.includes(labelText) ? labelText : ""}
            onChange={(event) => setLabelText(event.target.value)}
            aria-label="Mevcut etiketlerden seç"
          >
            <option value="">Yeni etiket</option>
            {labels.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
        )}
      </div>
      <div className="min-w-0">
        <OutputBars task={task} values={prediction} />
        <div className="mt-2 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-2 text-xs leading-5 text-[#526070]">
          Sürükleyerek çiz, etiketi yaz ve artıya bas. Yeni etiket ayrı bir output nöronu olarak modele eklenir.
        </div>
      </div>
      <div className="col-span-2 mt-1 grid grid-cols-4 gap-2">
        {data.slice(0, 8).map((point) => (
          <button
            key={point.id}
            type="button"
            className="rounded-md border border-[#dbe3ee] p-1 hover:border-[#2563eb]"
            title={`${point.label} örneğini yükle`}
            onClick={() => setPixels(point.inputs)}
          >
            <MiniDigit inputs={point.inputs} />
            <div className="mt-1 text-[11px] font-semibold text-[#526070]">{point.label}</div>
          </button>
        ))}
      </div>
    </div>
  );
}

function NlpDatasetPanel({
  task,
  network,
  data,
  nlpVectorizer,
  onDataChange,
  onApplyNlpDataset,
}: DatasetPanelProps) {
  const [text, setText] = useState("bu çok iyi ve güzel");
  const [label, setLabel] = useState("Pozitif");
  const [draftDataset, setDraftDataset] = useState(
    "text,label\nbu ürün harika ve çok kullanışlı,Pozitif\nhizmet güzel başarılı teşekkürler,Pozitif\nuygulamayı seviyorum deneyim iyi,Pozitif\nbu çok kötü ve berbat,Negatif\nsonuçtan nefret ettim çok zor,Negatif\nüzgün ve kızgın kaldım,Negatif"
  );
  const [config, setConfig] = useState<NlpVectorizerConfig>(DEFAULT_NLP_CONFIG);
  const examples = useMemo(() => parseTextDataset(draftDataset), [draftDataset]);
  const liveVectorizer = useMemo(
    () => nlpVectorizer ?? buildNlpVectorizer(examples, config),
    [config, examples, nlpVectorizer]
  );
  const vector = useMemo(
    () => (nlpVectorizer ? vectorizeText(text, nlpVectorizer) : encodeText(text)),
    [text, nlpVectorizer]
  );
  const tokens = useMemo(() => tokenizeText(text), [text]);
  const prediction = network.predictPure(vector);
  const labelOptions = nlpVectorizer?.labels ?? NLP_CLASS_NAMES;
  const visibleVocabulary = nlpVectorizer?.vocabulary ?? [...NLP_VOCABULARY];

  const addTextExample = () => {
    setDraftDataset((previous) => `${previous.trim()}\n"${text.replaceAll("\"", "\"\"")}",${label}`);
  };

  const buildRealNlpDataset = () => {
    if (examples.length === 0 || !onApplyNlpDataset) return;
    onApplyNlpDataset(examples, config);
  };

  return (
    <div className="mt-3 space-y-3">
      <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
        <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          <MessageSquareText className="h-4 w-4 text-[#2563eb]" />
          Metin Deneyi
        </div>
        <textarea
          className="min-h-20 w-full resize-y rounded-md border border-[#cbd5e1] bg-white p-2 text-sm leading-5 outline-none focus:border-[#2563eb]"
          value={text}
          onChange={(event) => setText(event.target.value)}
          aria-label="NLP metni"
        />
        <div className="mt-2 grid grid-cols-[1fr_auto] gap-2">
          <select
            className="h-9 rounded-md border border-[#cbd5e1] bg-white px-2 text-sm"
            value={label}
            onChange={(event) => setLabel(event.target.value)}
          >
            {labelOptions.map((name) => (
              <option key={name} value={name}>
                {name}
              </option>
            ))}
          </select>
          <IconButton title="Metni veri setine ekle" onClick={addTextExample}>
            <Plus className="h-4 w-4" />
          </IconButton>
        </div>
      </div>

      <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
        <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Eğitim Metinleri
        </div>
        <textarea
          className="min-h-40 w-full resize-y rounded-md border border-[#cbd5e1] bg-[#fbfdff] p-2 font-mono text-[11px] leading-5 text-[#334155] outline-none focus:border-[#2563eb]"
          value={draftDataset}
          onChange={(event) => setDraftDataset(event.target.value)}
          aria-label="NLP eğitim metinleri CSV"
          spellCheck={false}
        />
        <div className="mt-2 grid grid-cols-2 gap-2">
          <label className="block">
            <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
              Max feature
            </span>
            <input
              className="mt-1 h-8 w-full rounded-md border border-[#cbd5e1] px-2 text-xs"
              type="number"
              min={8}
              max={256}
              value={config.maxFeatures}
              onChange={(event) =>
                setConfig((previous) => ({ ...previous, maxFeatures: Number(event.target.value) }))
              }
            />
          </label>
          <label className="block">
            <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
              N-gram
            </span>
            <select
              className="mt-1 h-8 w-full rounded-md border border-[#cbd5e1] px-2 text-xs"
              value={config.ngramMax}
              onChange={(event) =>
                setConfig((previous) => ({ ...previous, ngramMax: Number(event.target.value) as 1 | 2 | 3 }))
              }
            >
              <option value={1}>1</option>
              <option value={2}>1-2</option>
              <option value={3}>1-3</option>
            </select>
          </label>
          <label className="block">
            <span className="text-[10px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
              Min frekans
            </span>
            <input
              className="mt-1 h-8 w-full rounded-md border border-[#cbd5e1] px-2 text-xs"
              type="number"
              min={1}
              max={10}
              value={config.minFrequency}
              onChange={(event) =>
                setConfig((previous) => ({ ...previous, minFrequency: Number(event.target.value) }))
              }
            />
          </label>
          <div className="flex items-end gap-3 pb-1">
            <label className="flex items-center gap-1 text-[11px] font-semibold text-[#334155]">
              <input
                type="checkbox"
                checked={config.useTfIdf}
                onChange={(event) =>
                  setConfig((previous) => ({ ...previous, useTfIdf: event.target.checked }))
                }
              />
              TF-IDF
            </label>
            <label className="flex items-center gap-1 text-[11px] font-semibold text-[#334155]">
              <input
                type="checkbox"
                checked={config.removeStopWords}
                onChange={(event) =>
                  setConfig((previous) => ({ ...previous, removeStopWords: event.target.checked }))
                }
              />
              stopword
            </label>
          </div>
        </div>
        <button
          type="button"
          className="mt-3 inline-flex h-9 w-full items-center justify-center rounded-md bg-[#2563eb] px-3 text-xs font-semibold text-white hover:bg-[#1d4ed8]"
          onClick={buildRealNlpDataset}
        >
          Vocabulary oluştur ve modeli kur
        </button>
      </div>

      <div className="rounded-md border border-[#dbe3ee] bg-white p-3">
        <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Token → Vektör
        </div>
        <div className="mb-2 flex flex-wrap gap-1">
          {tokens.length === 0 ? (
            <span className="text-xs text-[#64748b]">Henüz token yok.</span>
          ) : (
            tokens.map((token, index) => (
              <span key={`${token}-${index}`} className="rounded-md bg-[#eef4ff] px-2 py-1 text-[11px] font-semibold text-[#2563eb]">
                {token}
              </span>
            ))
          )}
        </div>
        <div className="mb-2 grid grid-cols-2 gap-1">
          {visibleVocabulary.slice(0, 40).map((word, index) => (
            <div
              key={`${word}-${index}`}
              className={`flex items-center justify-between rounded-md border px-2 py-1 text-[11px] ${
                vector[index] > 0
                  ? "border-[#2563eb] bg-[#e8f0ff] text-[#1d4ed8]"
                  : "border-[#e2e8f0] bg-[#fbfdff] text-[#64748b]"
              }`}
            >
              <span>{word}</span>
              <span className="font-mono font-semibold">{vector[index]}</span>
            </div>
          ))}
        </div>
        <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-2 text-[11px] leading-5 text-[#526070]">
          {examples.length} metin · {liveVectorizer.vocabulary.length} feature · sınıflar:{" "}
          {liveVectorizer.labels.join(", ")}
        </div>
      </div>

      <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
        <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Model Çıktısı
        </div>
        <OutputBars task={task} values={prediction} />
        <div className="mt-2 text-xs leading-5 text-[#526070]">
          Bu artık dinamik vocabulary + n-gram + TF-IDF destekler. Model yeniden kurulduğunda input nöronları senin veri setindeki feature sayısına göre oluşur.
        </div>
      </div>

      <div className="flex max-h-20 flex-wrap gap-1 overflow-y-auto">
        {data.map((point) => (
          <button
            key={point.id}
            type="button"
            className="rounded-md border border-[#dbe3ee] px-2 py-1 text-[11px] text-[#526070] hover:border-[#ef4444] hover:text-[#b91c1c]"
            onClick={() => onDataChange(data.filter((item) => item.id !== point.id))}
            title="Metin örneğini sil"
          >
            {point.label ?? targetName(task, point.targets)}
          </button>
        ))}
      </div>
    </div>
  );
}

function DataPlot({ task, network, data }: { task: Task; network: NeuralNetwork; data: DataPoint[] }) {
  if (task.inputSize === 1) {
    return <RegressionPlot network={network} data={data} />;
  }
  return <BoundaryPlot task={task} network={network} data={data} />;
}

function RegressionPlot({ network, data }: { network: NeuralNetwork; data: DataPoint[] }) {
  const width = 342;
  const height = 180;
  const xScale = scaleLinear().domain([0, 1]).range([30, width - 14]);
  const yScale = scaleLinear().domain([-0.08, 1.08]).range([height - 24, 12]);
  const curve = useMemo(
    () =>
      Array.from({ length: 80 }, (_, index) => {
        const x = index / 79;
        return { x, y: network.predictPure([x])[0] ?? 0 };
      }),
    [network]
  );
  const path =
    line<{ x: number; y: number }>()
      .x((point) => xScale(point.x))
      .y((point) => yScale(point.y))(curve) ?? "";

  return (
    <svg className="mt-3 h-[180px] w-full rounded-md border border-[#dbe3ee]" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="#fbfdff" />
      <line x1={30} y1={height - 24} x2={width - 14} y2={height - 24} stroke="#cbd5e1" />
      <line x1={30} y1={12} x2={30} y2={height - 24} stroke="#cbd5e1" />
      <path d={path} fill="none" stroke="#0f766e" strokeWidth={3} strokeLinecap="round" />
      {data.map((point) => (
        <circle
          key={point.id}
          cx={xScale(point.inputs[0] ?? 0)}
          cy={yScale(point.targets[0] ?? 0)}
          r={4.2}
          fill="#f97316"
          stroke="#ffffff"
          strokeWidth={1.5}
        />
      ))}
      <text x={30} y={height - 7} className="fill-[#64748b] text-[10px]">
        x
      </text>
      <text x={10} y={18} className="fill-[#64748b] text-[10px]">
        y
      </text>
    </svg>
  );
}

function BoundaryPlot({ task, network, data }: { task: Task; network: NeuralNetwork; data: DataPoint[] }) {
  const width = 342;
  const height = 192;
  const xScale = scaleLinear().domain([0, 1]).range([28, width - 12]);
  const yScale = scaleLinear().domain([0, 1]).range([height - 24, 12]);
  const cells = 22;
  const colors = task.classColors ?? ["#3b82f6", "#ef4444", "#10b981", "#f59e0b"];

  return (
    <svg className="mt-3 h-[192px] w-full rounded-md border border-[#dbe3ee]" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="#fbfdff" />
      {Array.from({ length: cells * cells }, (_, index) => {
        const gx = index % cells;
        const gy = Math.floor(index / cells);
        const x = gx / (cells - 1);
        const y = gy / (cells - 1);
        const out = network.predictPure([x, y]);
        const cls = out.indexOf(Math.max(...out));
        const confidence = Math.max(...out);
        const cellW = (width - 40) / cells;
        const cellH = (height - 36) / cells;
        return (
          <rect
            key={index}
            x={28 + gx * cellW}
            y={12 + gy * cellH}
            width={cellW + 0.4}
            height={cellH + 0.4}
            fill={colors[cls] ?? "#94a3b8"}
            opacity={0.13 + Math.min(0.24, Math.abs(confidence) * 0.12)}
          />
        );
      })}
      <line x1={28} y1={height - 24} x2={width - 12} y2={height - 24} stroke="#cbd5e1" />
      <line x1={28} y1={12} x2={28} y2={height - 24} stroke="#cbd5e1" />
      {data.map((point) => {
        const cls = point.targets.indexOf(Math.max(...point.targets));
        return (
          <circle
            key={point.id}
            cx={xScale(point.inputs[0] ?? 0)}
            cy={yScale(point.inputs[1] ?? 0)}
            r={4.3}
            fill={colors[cls] ?? "#f97316"}
            stroke="#ffffff"
            strokeWidth={1.5}
          />
        );
      })}
    </svg>
  );
}

function PointChips({
  task,
  data,
  onDataChange
}: {
  task: Task;
  data: DataPoint[];
  onDataChange: (data: DataPoint[]) => void;
}) {
  return (
    <div className="mt-3 flex max-h-16 flex-wrap gap-1 overflow-y-auto">
      {data.map((point) => (
        <button
          key={point.id}
          type="button"
          className="rounded-md border border-[#dbe3ee] px-2 py-1 text-[11px] text-[#526070] hover:border-[#ef4444] hover:text-[#b91c1c]"
          onClick={() => onDataChange(data.filter((item) => item.id !== point.id))}
          title="Noktayı sil"
        >
          {task.inputSize === 1
            ? `${formatNumber(point.inputs[0] ?? 0, 2)} → ${formatNumber(point.targets[0] ?? 0, 2)}`
            : `${formatNumber(point.inputs[0] ?? 0, 2)}, ${formatNumber(point.inputs[1] ?? 0, 2)} · ${
                point.label ?? targetName(task, point.targets)
              }`}
        </button>
      ))}
    </div>
  );
}

function MiniDigit({ inputs }: { inputs: number[] }) {
  return (
    <div className="grid grid-cols-5 gap-[2px]">
      {inputs.map((value, index) => (
        <span
          key={index}
          className="block aspect-square rounded-[2px]"
          style={{ backgroundColor: `rgba(24, 32, 47, ${0.08 + value * 0.84})` }}
        />
      ))}
    </div>
  );
}

interface TrainingPanelProps {
  task: Task;
  epoch: number;
  history: EpochTraceRecord[];
  lossHistory: number[];
  learningRate: number;
  running: boolean;
  phase: TrainingPhase;
  trace: TrainingTrace;
  network: NeuralNetwork;
  data: DataPoint[];
  accuracy: number | null;
  conceptMode: ConceptMode;
  onSelectTarget: (selection: Selection | null) => void;
  onPhaseChange: (phase: TrainingPhase) => void;
  onOpenConcept: (conceptId: ConceptId) => void;
  onLearningRateChange: (value: number) => void;
  onStep: () => void;
  onToggleRun: () => void;
  onReset: () => void;
}

function TrainingPanel({
  task,
  epoch,
  history,
  lossHistory,
  learningRate,
  running,
  phase,
  trace,
  network,
  data,
  accuracy,
  conceptMode,
  onSelectTarget,
  onPhaseChange,
  onOpenConcept,
  onLearningRateChange,
  onStep,
  onToggleRun,
  onReset
}: TrainingPanelProps) {
  const [tab, setTab] = useState<"loss" | "step" | "lr">("loss");

  return (
    <section className="col-span-3 grid min-h-0 grid-cols-[292px_minmax(0,1fr)] gap-px bg-[#d7dde8]">
      <div className="min-h-0 overflow-y-auto bg-white px-3 py-3">
        <PanelTitle icon={<Zap className="h-5 w-5" />} title="Eğitim" compact />
        <div className="mt-2 flex items-center gap-2">
          <ActionButton icon={<StepForward className="h-4 w-4" />} label="1 Epoch" onClick={onStep} />
          <IconButton title={running ? "Durdur" : "Sürekli eğit"} onClick={onToggleRun}>
            {running ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
          </IconButton>
          <IconButton title="Ağırlıkları sıfırla" onClick={onReset}>
            <RotateCcw className="h-5 w-5" />
          </IconButton>
        </div>
        <div className="mt-2">
          <div className="mb-2 flex items-center justify-between text-xs font-medium text-[#526070]">
            <span>Learning Rate</span>
            <span>{formatNumber(learningRate, 3)}</span>
          </div>
          <input
            aria-label="Learning Rate"
            className="w-full accent-[#2563eb]"
            min={0.005}
            max={0.8}
            step={0.005}
            type="range"
            value={learningRate}
            onChange={(event) => onLearningRateChange(Number(event.target.value))}
          />
        </div>
        <div className="mt-2 grid grid-cols-2 gap-2">
          <Stat label="epoch" value={String(epoch)} />
          <Stat label="faz" value={phase} />
          <Stat label="loss" value={formatNumber(lossHistory.at(-1) ?? 0, 6)} />
          <Stat label={accuracy === null ? "çıktı" : "doğruluk"} value={accuracy === null ? predictionName(task, trace.predictionAfter) : `${formatNumber(accuracy * 100, 1)}%`} />
        </div>
      </div>

      <div className="flex min-h-0 flex-col overflow-hidden bg-white px-3 py-3">
        <div className="mb-2 flex items-center justify-between gap-3">
          <PanelTabs
            className="w-[420px] max-w-full"
            compact
            items={[
              { id: "loss", label: "Loss", icon: <BarChart3 className="h-3.5 w-3.5" /> },
              { id: "step", label: "Adım", icon: <Waves className="h-3.5 w-3.5" /> },
              { id: "lr", label: "LR Deneyi", icon: <FlaskConical className="h-3.5 w-3.5" /> },
            ]}
            value={tab}
            onChange={setTab}
          />
          <div className="text-xs font-semibold text-[#64748b]">
            {formatNumber(lossHistory.at(-1) ?? 0, 6)}
          </div>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto">
          {tab === "loss" && <LossChart values={lossHistory} />}
          {tab === "step" && (
            <StepExplorer
              task={task}
              trace={trace}
              history={history}
              learningRate={learningRate}
              conceptMode={conceptMode}
              onSelectTarget={onSelectTarget}
              onPhaseChange={onPhaseChange}
              onOpenConcept={onOpenConcept}
            />
          )}
          {tab === "lr" && (
            <LearningRateExperiment
              network={network}
              data={data}
              currentRate={learningRate}
              onApply={onLearningRateChange}
            />
          )}
        </div>
      </div>
    </section>
  );
}

function OutputBars({ task, values }: { task: Task; values: number[] }) {
  const max = Math.max(0.001, ...values.map((value) => Math.abs(value)));
  return (
    <div className="space-y-2">
      {values.map((value, index) => (
        <div key={index}>
          <div className="mb-1 flex items-center justify-between text-[11px] text-[#526070]">
            <span>{task.classNames?.[index] ?? `Çıktı ${index + 1}`}</span>
            <span>{formatNumber(value, 3)}</span>
          </div>
          <div className="h-2 rounded-full bg-[#e2e8f0]">
            <div
              className="h-2 rounded-full"
              style={{
                width: `${Math.min(100, (Math.abs(value) / max) * 100)}%`,
                backgroundColor: task.classColors?.[index] ?? "#2563eb"
              }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}

function LossChart({ values }: { values: number[] }) {
  const width = 650;
  const height = 104;
  const maxLoss = Math.max(0.001, ...values);
  const xScale = scaleLinear()
    .domain([0, Math.max(1, values.length - 1)])
    .range([8, width - 10]);
  const yScale = scaleLinear().domain([0, maxLoss]).range([height - 18, 10]);
  const chartValues = values.length > 1 ? values : [values[0] ?? 0, values[0] ?? 0];
  const path =
    line<number>()
      .x((_, index) => xScale(index))
      .y((value) => yScale(value))(chartValues) ?? "";

  return (
    <svg className="h-[116px] w-full" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} rx={6} fill="#fbfdff" stroke="#dbe3ee" />
      <line x1={8} y1={height - 18} x2={width - 10} y2={height - 18} stroke="#dbe3ee" />
      <path d={path} fill="none" stroke="#2563eb" strokeWidth={3} strokeLinecap="round" />
      {chartValues.map((value, index) => (
        <circle
          key={`${index}-${value}`}
          cx={xScale(index)}
          cy={yScale(value)}
          r={index === chartValues.length - 1 ? 3.6 : 1.8}
          fill={index === chartValues.length - 1 ? "#f97316" : "#60a5fa"}
        />
      ))}
    </svg>
  );
}

function LearningRateExperiment({
  network,
  data,
  currentRate,
  onApply,
}: {
  network: NeuralNetwork;
  data: DataPoint[];
  currentRate: number;
  onApply: (rate: number) => void;
}) {
  const rates = useMemo(() => {
    const candidates = [currentRate * 0.25, currentRate * 0.5, currentRate, currentRate * 1.5, currentRate * 2];
    return Array.from(
      new Set(candidates.map((rate) => Math.max(0.005, Math.min(0.8, Number(rate.toFixed(3))))))
    );
  }, [currentRate]);
  const trials = useMemo(() => simulateLearningRates(network, data, rates), [network, data, rates]);
  const best = trials.reduce((winner, trial) => (trial.lossAfter < winner.lossAfter ? trial : winner), trials[0]);
  const maxLoss = Math.max(0.001, ...trials.map((trial) => trial.lossAfter));

  return (
    <div className="min-h-0 overflow-y-auto bg-white">
      <PanelTitle icon={<FlaskConical className="h-5 w-5" />} title="LR Deneyi" compact />
      <div className="mt-2 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-2 text-[11px] leading-5 text-[#526070]">
        Model kopyalanır, 1 epoch simüle edilir; gerçek ağırlıklar değişmez.
      </div>
      <div className="mt-2 space-y-2">
        {trials.map((trial, index) => {
          const good = trial.delta < 0;
          return (
            <button
              key={`${trial.learningRate}-${index}`}
              type="button"
              className={`block w-full rounded-md border p-2 text-left ${
                best?.learningRate === trial.learningRate
                  ? "border-[#2563eb] bg-[#eef4ff]"
                  : "border-[#dbe3ee] bg-white hover:border-[#2563eb]"
              }`}
              onClick={() => onApply(trial.learningRate)}
            >
              <div className="flex items-center justify-between text-[11px] font-semibold text-[#18202f]">
                <span>η={formatNumber(trial.learningRate, 3)}</span>
                <span className={good ? "text-[#047857]" : "text-[#b91c1c]"}>
                  {good ? "düşer" : "artar"}
                </span>
              </div>
              <div className="mt-1 h-1.5 rounded-full bg-[#e2e8f0]">
                <div
                  className="h-1.5 rounded-full bg-[#2563eb]"
                  style={{ width: `${Math.max(3, (trial.lossAfter / maxLoss) * 100)}%` }}
                />
              </div>
              <div className="mt-1 text-[10px] text-[#64748b]">
                {formatNumber(trial.lossBefore, 4)} → {formatNumber(trial.lossAfter, 4)}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

function DetailModal({
  selection,
  trace,
  learningRate,
  conceptMode,
  onOpenConcept,
  onClose,
}: {
  selection: Selection | null;
  trace: TrainingTrace;
  learningRate: number;
  conceptMode: ConceptMode;
  onOpenConcept: (conceptId: ConceptId) => void;
  onClose: () => void;
}) {
  const neuron =
    selection?.type === "neuron" ? trace.neurons.find((item) => item.id === selection.id) ?? null : null;
  const edge = selection?.type === "edge" ? trace.edges.find((item) => item.id === selection.id) ?? null : null;
  if (!selection || (!neuron && !edge)) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0f172a]/45 p-6">
      <div className="max-h-[88vh] w-full max-w-5xl overflow-hidden rounded-lg border border-[#cbd5e1] bg-white shadow-2xl">
        <div className="flex items-center justify-between border-b border-[#e2e8f0] px-5 py-4">
          <div>
            <div className="flex items-center gap-2 text-base font-semibold">
              <MousePointerClick className="h-5 w-5 text-[#2563eb]" />
              {neuron ? "Nöron Yakınlaştırma" : "Bağlantı / Ağırlık Yakınlaştırma"}
            </div>
            <div className="mt-1 text-xs text-[#64748b]">
              Learning rate η={formatNumber(learningRate, 4)} · mod {conceptMode}
            </div>
          </div>
          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-[#cbd5e1] text-[#334155] hover:border-[#ef4444] hover:text-[#b91c1c]"
            onClick={onClose}
            aria-label="Yakın incelemeyi kapat"
          >
            ×
          </button>
        </div>

        <div className="max-h-[calc(88vh-73px)] overflow-y-auto p-5">
          {neuron && (
            <div className="grid grid-cols-[minmax(0,1fr)_320px] gap-4">
              <div className="space-y-4">
                <NeuronInspector
                  neuron={neuron}
                  conceptMode={conceptMode}
                  trace={trace}
                  onOpenConcept={onOpenConcept}
                />
              </div>
              <div className="space-y-3 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-4">
                <div className="text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                  Tam Denklem
                </div>
                <div className="space-y-2 font-mono text-sm leading-6 text-[#18202f]">
                  <div>{neuron.formula}</div>
                  <div>z = {formatNumber(neuron.z, 8)}</div>
                  <div>a = {neuron.activation}(z) = {formatNumber(neuron.value, 8)}</div>
                  <div>δ = {formatNumber(neuron.delta, 8)}</div>
                  <div>∂L/∂b = {formatNumber(neuron.gradientBias, 8)}</div>
                </div>
              </div>
            </div>
          )}

          {edge && (
            <div className="grid grid-cols-[minmax(0,1fr)_320px] gap-4">
              <div className="space-y-4">
                <EdgeInspector edge={edge} conceptMode={conceptMode} onOpenConcept={onOpenConcept} />
              </div>
              <div className="space-y-3 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-4">
                <div className="text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                  Update Mikroskobu
                </div>
                <div className="space-y-2 font-mono text-sm leading-6 text-[#18202f]">
                  <div>∂L/∂w = {formatNumber(edge.gradient, 8)}</div>
                  <div>Δw = -η × gradient</div>
                  <div>
                    Δw = -{formatNumber(learningRate, 6)} × {formatNumber(edge.gradient, 8)} ={" "}
                    {formatNumber(edge.correction, 8)}
                  </div>
                  <div>
                    w_new = {formatNumber(edge.weightBefore, 8)} + {formatNumber(edge.correction, 8)} ={" "}
                    {formatNumber(edge.weightAfter, 8)}
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function PanelTitle({
  icon,
  title,
  compact
}: {
  icon: React.ReactNode;
  title: string;
  compact?: boolean;
}) {
  return (
    <div className={`flex items-center gap-2 ${compact ? "text-sm" : "text-base"} font-semibold`}>
      <span className="text-[#2563eb]">{icon}</span>
      <span>{title}</span>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-md border border-[#dbe3ee] bg-[#fbfdff] px-3 py-2">
      <div className="truncate text-[11px] font-medium uppercase tracking-[0.06em] text-[#64748b]">
        {label}
      </div>
      <div className="mt-1 truncate text-sm font-semibold text-[#18202f]" title={value}>
        {value}
      </div>
    </div>
  );
}

function FormulaBox({ title, lines, defaultOpen = false }: { title: string; lines: string[]; defaultOpen?: boolean }) {
  return (
    <DrawerPanel title={title} defaultOpen={defaultOpen} className="bg-[#fbfdff]">
      <div className="space-y-1 font-mono text-xs text-[#334155]">
        {lines.map((item) => (
          <div key={item} className="break-words">
            {item}
          </div>
        ))}
      </div>
    </DrawerPanel>
  );
}

function ExplainBox({ title, text }: { title: string; text: string }) {
  return (
    <DrawerPanel title={title}>
      <div className="mt-1 text-xs leading-5 text-[#526070]">{text}</div>
    </DrawerPanel>
  );
}

function NumberInput({
  label,
  value,
  onChange
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
}) {
  return (
    <label className="block">
      <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-[#64748b]">
        {label}
      </span>
      <input
        className="mt-1 h-9 w-full rounded-md border border-[#cbd5e1] px-2 text-sm"
        min={0}
        max={1}
        step={0.01}
        type="number"
        value={value}
        onChange={(event) => onChange(sanitizePointValue(Number(event.target.value)))}
      />
    </label>
  );
}

function IconButton({
  title,
  children,
  onClick,
  disabled,
  className = ""
}: {
  title: string;
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  className?: string;
}) {
  return (
    <button
      type="button"
      title={title}
      aria-label={title}
      disabled={disabled}
      onClick={onClick}
      className={`inline-flex h-9 w-9 shrink-0 items-center justify-center rounded-md border border-[#cbd5e1] bg-white text-[#334155] transition hover:border-[#2563eb] hover:text-[#2563eb] disabled:cursor-not-allowed disabled:opacity-40 ${className}`}
    >
      {children}
    </button>
  );
}

function ActionButton({
  icon,
  label,
  onClick,
  disabled
}: {
  icon: React.ReactNode;
  label: string;
  onClick?: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      type="button"
      disabled={disabled}
      onClick={onClick}
      className="inline-flex h-9 items-center justify-center gap-2 rounded-md border border-[#cbd5e1] bg-white px-3 text-sm font-semibold text-[#334155] transition hover:border-[#2563eb] hover:text-[#2563eb] disabled:cursor-not-allowed disabled:opacity-40"
    >
      {icon}
      <span>{label}</span>
    </button>
  );
}
