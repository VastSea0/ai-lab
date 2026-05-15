"use client";

import { line, scaleLinear } from "d3";
import {
  Activity,
  BarChart3,
  BrainCircuit,
  ChevronDown,
  Database,
  Minus,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Sigma,
  StepForward,
  Trash2,
  Waves,
  Zap
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  DataPoint,
  EdgeSnapshot,
  NeuralNetwork,
  NeuronSnapshot,
  Selection,
  TrainingPhase,
  TrainingTrace,
  edgeId,
  formatNumber,
  sanitizePointValue
} from "@/lib/ml/network";

const DEFAULT_DATA: DataPoint[] = [
  { id: "p1", x: 0.08, y: 0.14 },
  { id: "p2", x: 0.2, y: 0.22 },
  { id: "p3", x: 0.36, y: 0.5 },
  { id: "p4", x: 0.54, y: 0.62 },
  { id: "p5", x: 0.74, y: 0.84 },
  { id: "p6", x: 0.92, y: 0.91 }
];

interface ModelState {
  network: NeuralNetwork;
  trace: TrainingTrace;
  lossHistory: number[];
  epoch: number;
  seed: number;
}

function createModelState(seed: number, data: DataPoint[], sizes = [1, 4, 1]): ModelState {
  const network = NeuralNetwork.create(sizes, seed);
  const first = data[0] ?? { id: "empty", x: 0, y: 0 };
  const trace = network.inspect([first.x], [first.y]);
  return {
    network,
    trace,
    lossHistory: [network.evaluateLoss(data)],
    epoch: 0,
    seed
  };
}

export function SandboxApp() {
  const [data, setData] = useState<DataPoint[]>(DEFAULT_DATA);
  const [model, setModel] = useState<ModelState>(() => createModelState(1327, DEFAULT_DATA));
  const [learningRate, setLearningRate] = useState(0.18);
  const [running, setRunning] = useState(false);
  const [phase, setPhase] = useState<TrainingPhase>("idle");
  const [selected, setSelected] = useState<Selection | null>(null);
  const [hovered, setHovered] = useState<Selection | null>(null);
  const timers = useRef<number[]>([]);

  const clearPhaseTimers = useCallback(() => {
    timers.current.forEach((timer) => window.clearTimeout(timer));
    timers.current = [];
  }, []);

  const playPhaseAnimation = useCallback(() => {
    clearPhaseTimers();
    setPhase("forward");
    timers.current = [
      window.setTimeout(() => setPhase("backward"), 520),
      window.setTimeout(() => setPhase("idle"), 1100)
    ];
  }, [clearPhaseTimers]);

  const inspectWithData = useCallback((network: NeuralNetwork, nextData: DataPoint[]) => {
    const first = nextData[0] ?? { id: "empty", x: 0, y: 0 };
    return network.inspect([first.x], [first.y]);
  }, []);

  const runEpoch = useCallback(() => {
    setModel((previous) => {
      const nextNetwork = previous.network.clone();
      const trace = nextNetwork.trainEpoch(data, learningRate);
      return {
        ...previous,
        network: nextNetwork,
        trace,
        epoch: previous.epoch + 1,
        lossHistory: [...previous.lossHistory, trace.epochLoss].slice(-140)
      };
    });
    playPhaseAnimation();
  }, [data, learningRate, playPhaseAnimation]);

  useEffect(() => {
    if (!running) return undefined;
    const interval = window.setInterval(runEpoch, 1180);
    return () => window.clearInterval(interval);
  }, [running, runEpoch]);

  useEffect(() => {
    return () => clearPhaseTimers();
  }, [clearPhaseTimers]);

  const rebuildNetwork = useCallback(
    (sizes: number[]) => {
      setRunning(false);
      setSelected(null);
      setHovered(null);
      setModel((previous) => {
        const seed = previous.seed + 97;
        const next = NeuralNetwork.create(sizes, seed);
        return {
          network: next,
          trace: inspectWithData(next, data),
          lossHistory: [next.evaluateLoss(data)],
          epoch: 0,
          seed
        };
      });
    },
    [data, inspectWithData]
  );

  const resetWeights = useCallback(() => {
    rebuildNetwork(model.network.getLayerSizes());
  }, [model.network, rebuildNetwork]);

  const updateData = useCallback(
    (nextData: DataPoint[]) => {
      setData(nextData);
      setModel((previous) => {
        const network = previous.network.clone();
        return {
          ...previous,
          network,
          trace: inspectWithData(network, nextData),
          lossHistory: [network.evaluateLoss(nextData)],
          epoch: 0
        };
      });
    },
    [inspectWithData]
  );

  const activeSelection = hovered ?? selected;

  return (
    <main className="h-screen min-h-[720px] min-w-[1120px] overflow-hidden bg-[#f5f7fb] text-[#18202f]">
      <div className="grid h-full grid-cols-[300px_minmax(0,1fr)_360px] grid-rows-[minmax(0,1fr)_176px] gap-px bg-[#d7dde8]">
        <ArchitecturePanel
          network={model.network}
          onRebuild={rebuildNetwork}
          onResetWeights={resetWeights}
        />

        <section className="relative overflow-hidden bg-[#eef3f8]">
          <div className="absolute left-5 top-4 z-10 flex items-center gap-3 rounded-md border border-[#c8d3e1] bg-white/92 px-3 py-2 shadow-panel">
            <BrainCircuit className="h-5 w-5 text-[#2563eb]" />
            <div>
              <div className="text-sm font-semibold">Görsel Sinir Ağı Sandbox</div>
              <div className="text-xs text-[#607089]">
                Epoch {model.epoch} · Loss {formatNumber(model.lossHistory.at(-1) ?? 0, 5)}
              </div>
            </div>
          </div>
          <NetworkCanvas
            network={model.network}
            trace={model.trace}
            phase={phase}
            selected={selected}
            hovered={hovered}
            onSelect={setSelected}
            onHover={setHovered}
          />
        </section>

        <InspectorPanel
          selection={activeSelection}
          trace={model.trace}
          network={model.network}
          data={data}
          onDataChange={updateData}
        />

        <TrainingPanel
          epoch={model.epoch}
          lossHistory={model.lossHistory}
          learningRate={learningRate}
          running={running}
          phase={phase}
          trace={model.trace}
          onLearningRateChange={setLearningRate}
          onStep={runEpoch}
          onToggleRun={() => {
            if (!running) runEpoch();
            setRunning((value) => !value);
          }}
          onReset={resetWeights}
        />
      </div>
    </main>
  );
}

interface ArchitecturePanelProps {
  network: NeuralNetwork;
  onRebuild: (sizes: number[]) => void;
  onResetWeights: () => void;
}

function ArchitecturePanel({ network, onRebuild, onResetWeights }: ArchitecturePanelProps) {
  const sizes = network.getLayerSizes();
  const hiddenSizes = sizes.slice(1, -1);

  const rebuildHidden = (nextHidden: number[]) => {
    onRebuild([1, ...nextHidden, 1]);
  };

  return (
    <aside className="row-span-1 overflow-y-auto bg-white px-4 py-4">
      <PanelTitle icon={<Sigma className="h-5 w-5" />} title="Ağ Mimarisi" />

      <div className="mt-5 space-y-3">
        <LayerRow label="Input" detail="x" count={sizes[0]} locked />
        {hiddenSizes.map((count, index) => (
          <LayerRow
            key={`hidden-${index}`}
            label={`Hidden ${index + 1}`}
            detail="sigmoid"
            count={count}
            onDecrease={() => {
              const next = [...hiddenSizes];
              next[index] = Math.max(1, next[index] - 1);
              rebuildHidden(next);
            }}
            onIncrease={() => {
              const next = [...hiddenSizes];
              next[index] = Math.min(6, next[index] + 1);
              rebuildHidden(next);
            }}
            onRemove={() => {
              rebuildHidden(hiddenSizes.filter((_, itemIndex) => itemIndex !== index));
            }}
          />
        ))}
        <LayerRow label="Output" detail="linear" count={sizes.at(-1) ?? 1} locked />
      </div>

      <div className="mt-5 grid grid-cols-2 gap-2">
        <ActionButton
          icon={<Plus className="h-4 w-4" />}
          label="Katman"
          onClick={() => rebuildHidden([...hiddenSizes, 3])}
          disabled={hiddenSizes.length >= 4}
        />
        <ActionButton
          icon={<RotateCcw className="h-4 w-4" />}
          label="Ağırlık"
          onClick={onResetWeights}
        />
      </div>

      <div className="mt-6 border-t border-[#e2e8f0] pt-4">
        <div className="mb-3 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Katmanlar
        </div>
        <div className="space-y-2">
          {network.layers.map((layer) => (
            <div
              key={layer.id}
              className="flex items-center justify-between rounded-md border border-[#e2e8f0] px-3 py-2"
            >
              <span className="text-sm font-medium">
                L{layer.layerIndex} · {layer.kind}
              </span>
              <span className="text-xs text-[#64748b]">
                {layer.neurons.length} nöron · {layer.activation}
              </span>
            </div>
          ))}
        </div>
      </div>
    </aside>
  );
}

interface LayerRowProps {
  label: string;
  detail: string;
  count: number;
  locked?: boolean;
  onDecrease?: () => void;
  onIncrease?: () => void;
  onRemove?: () => void;
}

function LayerRow({
  label,
  detail,
  count,
  locked,
  onDecrease,
  onIncrease,
  onRemove
}: LayerRowProps) {
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
      <div className="mb-3 flex items-center justify-between">
        <div>
          <div className="text-sm font-semibold">{label}</div>
          <div className="text-xs text-[#64748b]">{detail}</div>
        </div>
        <div className="flex items-center gap-1">
          {!locked && (
            <IconButton title="Katmanı sil" onClick={onRemove}>
              <Trash2 className="h-4 w-4" />
            </IconButton>
          )}
          {locked && <ChevronDown className="h-4 w-4 text-[#94a3b8]" />}
        </div>
      </div>
      <div className="flex items-center justify-between">
        <IconButton title="Nöron azalt" disabled={locked || count <= 1} onClick={onDecrease}>
          <Minus className="h-4 w-4" />
        </IconButton>
        <div className="h-9 min-w-16 rounded-md border border-[#cbd5e1] bg-white px-4 py-2 text-center text-sm font-semibold">
          {count}
        </div>
        <IconButton title="Nöron artır" disabled={locked || count >= 6} onClick={onIncrease}>
          <Plus className="h-4 w-4" />
        </IconButton>
      </div>
    </div>
  );
}

interface NetworkCanvasProps {
  network: NeuralNetwork;
  trace: TrainingTrace;
  phase: TrainingPhase;
  selected: Selection | null;
  hovered: Selection | null;
  onSelect: (selection: Selection | null) => void;
  onHover: (selection: Selection | null) => void;
}

function NetworkCanvas({
  network,
  trace,
  phase,
  selected,
  hovered,
  onSelect,
  onHover
}: NetworkCanvasProps) {
  const width = 940;
  const height = 590;
  const positions = useMemo(() => {
    const map = new Map<string, { x: number; y: number; radius: number }>();
    const left = 95;
    const right = width - 95;
    const usableHeight = height - 165;
    const layerGap =
      network.layers.length <= 1 ? 0 : (right - left) / (network.layers.length - 1);

    network.layers.forEach((layer, layerIndex) => {
      const neuronCount = layer.neurons.length;
      const gap = neuronCount <= 1 ? 0 : Math.min(94, usableHeight / (neuronCount - 1));
      const startY = height / 2 - (gap * (neuronCount - 1)) / 2 + 15;
      const radius = Math.max(17, Math.min(24, gap ? gap * 0.27 : 23));
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

  return (
    <svg
      className="h-full w-full"
      viewBox={`0 0 ${width} ${height}`}
      role="img"
      aria-label="Sinir ağı görselleştirmesi"
      onMouseLeave={() => onHover(null)}
    >
      <defs>
        <linearGradient id="canvasWash" x1="0" x2="1" y1="0" y2="1">
          <stop offset="0%" stopColor="#f8fbff" />
          <stop offset="58%" stopColor="#eef6f2" />
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
            y={72}
            textAnchor="middle"
            className="fill-[#526070] text-[12px] font-semibold uppercase"
          >
            {layer.kind}
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
              const active = selected?.id === id || hovered?.id === id;
              const weightMagnitude = Math.min(1, Math.abs(weight) / 2.4);
              const stroke = weight >= 0 ? "#2563eb" : "#e11d48";
              const widthByWeight = 1.35 + weightMagnitude * 6.4;
              const contributionWidth =
                1.2 + Math.min(6, Math.abs(traceEdge?.contribution ?? 0) * 5.5);
              const errorWidth =
                1.4 + Math.min(9, Math.abs(traceEdge?.gradient ?? 0) * 22);

              return (
                <g key={id}>
                  <line
                    x1={fromPos.x}
                    y1={fromPos.y}
                    x2={toPos.x}
                    y2={toPos.y}
                    stroke={stroke}
                    strokeWidth={widthByWeight}
                    strokeOpacity={active ? 0.75 : 0.22 + weightMagnitude * 0.45}
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
                      strokeOpacity={0.68}
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
                      strokeOpacity={0.7}
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
                    strokeWidth={16}
                    className="cursor-pointer"
                    aria-label={`w = ${formatNumber(weight)}`}
                    onMouseEnter={() => onHover(selection)}
                    onMouseLeave={() => onHover(null)}
                    onClick={() => onSelect(selection)}
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
              >
                <circle
                  cx={pos.x}
                  cy={pos.y}
                  r={pos.radius}
                  fill={fill}
                  stroke={active ? "#111827" : stroke}
                  strokeWidth={active ? 3 : 2}
                  className={phase === "forward" && neuron.layerKind !== "input" ? "neuron-pop" : ""}
                />
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
                  y={pos.y + pos.radius + 16}
                  textAnchor="middle"
                  className="pointer-events-none fill-[#526070] text-[10px] font-medium"
                >
                  {neuron.layerKind === "input"
                    ? `x=${formatNumber(snapshot.value, 2)}`
                    : `Σ=${formatNumber(snapshot.z, 2)}`}
                </text>
                {neuron.layerKind !== "input" && (
                  <text
                    x={pos.x}
                    y={pos.y + pos.radius + 29}
                    textAnchor="middle"
                    className="pointer-events-none fill-[#64748b] text-[9px]"
                  >
                    a={formatNumber(snapshot.value, 2)}
                  </text>
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
  selection: Selection | null;
  trace: TrainingTrace;
  network: NeuralNetwork;
  data: DataPoint[];
  onDataChange: (data: DataPoint[]) => void;
}

function InspectorPanel({ selection, trace, network, data, onDataChange }: InspectorPanelProps) {
  const neuron =
    selection?.type === "neuron"
      ? trace.neurons.find((item) => item.id === selection.id) ?? null
      : null;
  const edge =
    selection?.type === "edge"
      ? trace.edges.find((item) => item.id === selection.id) ?? null
      : null;

  return (
    <aside className="row-span-1 flex min-h-0 flex-col bg-white">
      <div className="border-b border-[#e2e8f0] px-4 py-4">
        <PanelTitle icon={<Activity className="h-5 w-5" />} title="Denetçi" />
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {neuron && <NeuronInspector neuron={neuron} />}
        {edge && <EdgeInspector edge={edge} />}
        {!neuron && !edge && <TraceSummary trace={trace} />}
      </div>

      <DatasetPanel network={network} data={data} onDataChange={onDataChange} />
    </aside>
  );
}

function TraceSummary({ trace }: { trace: TrainingTrace }) {
  return (
    <div className="space-y-3">
      <div className="rounded-md border border-[#dbe3ee] p-3">
        <div className="text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Son Örnek
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2">
          <Stat label="x" value={formatNumber(trace.input[0] ?? 0)} />
          <Stat label="hedef" value={formatNumber(trace.target[0] ?? 0)} />
          <Stat label="tahmin" value={formatNumber(trace.prediction[0] ?? 0)} />
          <Stat label="loss" value={formatNumber(trace.loss, 5)} />
        </div>
      </div>
      <FormulaBox
        title="Loss"
        lines={[
          "L = 1/2 * (ŷ - y)^2",
          `L = 1/2 * (${formatNumber(trace.prediction[0] ?? 0)} - ${formatNumber(
            trace.target[0] ?? 0
          )})^2`
        ]}
      />
    </div>
  );
}

function NeuronInspector({ neuron }: { neuron: NeuronSnapshot }) {
  return (
    <div className="space-y-3">
      <div>
        <div className="text-sm font-semibold">
          {neuron.layerKind} · L{neuron.layerIndex} N{neuron.neuronIndex}
        </div>
        <div className="text-xs text-[#64748b]">{neuron.activation}</div>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Σ(x*w)+b" value={formatNumber(neuron.z)} />
        <Stat label="aktivasyon" value={formatNumber(neuron.value)} />
        <Stat label="bias" value={formatNumber(neuron.bias)} />
        <Stat label="δ hata" value={formatNumber(neuron.delta)} />
        <Stat label="türev" value={formatNumber(neuron.derivative)} />
        <Stat label="∂L/∂b" value={formatNumber(neuron.gradientBias)} />
      </div>
      <FormulaBox title="Hesap" lines={[neuron.formula, `a = ${neuron.activation}(Σ)`]} />
      {neuron.incoming.length > 0 && (
        <div className="rounded-md border border-[#dbe3ee]">
          <div className="border-b border-[#e2e8f0] px-3 py-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
            Gelenler
          </div>
          <div className="divide-y divide-[#edf2f7]">
            {neuron.incoming.map((item) => (
              <div
                key={`${item.fromNeuronId}-${neuron.id}`}
                className="grid grid-cols-[1fr_auto] gap-2 px-3 py-2 text-xs"
              >
                <span className="font-medium">{item.fromNeuronId}</span>
                <span className="text-[#64748b]">
                  {formatNumber(item.inputValue, 3)} * {formatNumber(item.weight, 3)} ={" "}
                  {formatNumber(item.product, 3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function EdgeInspector({ edge }: { edge: EdgeSnapshot }) {
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
        title="Bağlantı"
        lines={[
          "katkı = a_önceki * w",
          "∂L/∂w = a_önceki * δ_sonraki",
          `w' = ${formatNumber(edge.weightBefore)} + ${formatNumber(edge.correction)}`
        ]}
      />
    </div>
  );
}

interface DatasetPanelProps {
  network: NeuralNetwork;
  data: DataPoint[];
  onDataChange: (data: DataPoint[]) => void;
}

function DatasetPanel({ network, data, onDataChange }: DatasetPanelProps) {
  const [draft, setDraft] = useState({ x: 0.42, y: 0.58 });

  const addPoint = () => {
    const point: DataPoint = {
      id: `p${Date.now()}`,
      x: sanitizePointValue(draft.x),
      y: sanitizePointValue(draft.y)
    };
    onDataChange([...data, point]);
  };

  return (
    <div className="border-t border-[#e2e8f0] px-4 py-4">
      <PanelTitle icon={<Database className="h-5 w-5" />} title="Veri" compact />
      <DataPlot network={network} data={data} />
      <div className="mt-3 grid grid-cols-[1fr_1fr_auto] gap-2">
        <NumberInput
          label="x"
          value={draft.x}
          onChange={(value) => setDraft((previous) => ({ ...previous, x: value }))}
        />
        <NumberInput
          label="y"
          value={draft.y}
          onChange={(value) => setDraft((previous) => ({ ...previous, y: value }))}
        />
        <IconButton title="Nokta ekle" onClick={addPoint} className="mt-5">
          <Plus className="h-4 w-4" />
        </IconButton>
      </div>
      <div className="mt-3 flex max-h-16 flex-wrap gap-1 overflow-y-auto">
        {data.map((point) => (
          <button
            key={point.id}
            type="button"
            className="rounded-md border border-[#dbe3ee] px-2 py-1 text-[11px] text-[#526070] hover:border-[#ef4444] hover:text-[#b91c1c]"
            onClick={() => onDataChange(data.filter((item) => item.id !== point.id))}
            title="Noktayı sil"
          >
            {formatNumber(point.x, 2)}, {formatNumber(point.y, 2)}
          </button>
        ))}
      </div>
    </div>
  );
}

function DataPlot({ network, data }: { network: NeuralNetwork; data: DataPoint[] }) {
  const width = 328;
  const height = 178;
  const xScale = scaleLinear().domain([0, 1]).range([30, width - 14]);
  const yScale = scaleLinear().domain([-0.08, 1.08]).range([height - 24, 12]);
  const curve = useMemo(
    () =>
      Array.from({ length: 72 }, (_, index) => {
        const x = index / 71;
        return { x, y: network.predict(x) };
      }),
    [network]
  );
  const path =
    line<{ x: number; y: number }>()
      .x((point) => xScale(point.x))
      .y((point) => yScale(point.y))(curve) ?? "";

  return (
    <svg className="mt-3 h-[178px] w-full rounded-md border border-[#dbe3ee]" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} fill="#fbfdff" />
      <line x1={30} y1={height - 24} x2={width - 14} y2={height - 24} stroke="#cbd5e1" />
      <line x1={30} y1={12} x2={30} y2={height - 24} stroke="#cbd5e1" />
      <path d={path} fill="none" stroke="#0f766e" strokeWidth={3} strokeLinecap="round" />
      {data.map((point) => (
        <circle
          key={point.id}
          cx={xScale(point.x)}
          cy={yScale(point.y)}
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

interface TrainingPanelProps {
  epoch: number;
  lossHistory: number[];
  learningRate: number;
  running: boolean;
  phase: TrainingPhase;
  trace: TrainingTrace;
  onLearningRateChange: (value: number) => void;
  onStep: () => void;
  onToggleRun: () => void;
  onReset: () => void;
}

function TrainingPanel({
  epoch,
  lossHistory,
  learningRate,
  running,
  phase,
  trace,
  onLearningRateChange,
  onStep,
  onToggleRun,
  onReset
}: TrainingPanelProps) {
  return (
    <section className="col-span-3 grid grid-cols-[360px_1fr_320px] gap-px bg-[#d7dde8]">
      <div className="bg-white px-4 py-4">
        <PanelTitle icon={<Zap className="h-5 w-5" />} title="Eğitim" compact />
        <div className="mt-4 flex items-center gap-2">
          <ActionButton icon={<StepForward className="h-4 w-4" />} label="1 Epoch" onClick={onStep} />
          <IconButton title={running ? "Durdur" : "Sürekli eğit"} onClick={onToggleRun}>
            {running ? <Pause className="h-5 w-5" /> : <Play className="h-5 w-5" />}
          </IconButton>
          <IconButton title="Ağırlıkları sıfırla" onClick={onReset}>
            <RotateCcw className="h-5 w-5" />
          </IconButton>
        </div>
        <div className="mt-4">
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
      </div>

      <div className="bg-white px-4 py-4">
        <div className="mb-3 flex items-center justify-between">
          <PanelTitle icon={<BarChart3 className="h-5 w-5" />} title="Loss" compact />
          <div className="text-xs font-semibold text-[#64748b]">
            {formatNumber(lossHistory.at(-1) ?? 0, 6)}
          </div>
        </div>
        <LossChart values={lossHistory} />
      </div>

      <div className="bg-white px-4 py-4">
        <PanelTitle icon={<Waves className="h-5 w-5" />} title="Zaman Çizelgesi" compact />
        <div className="mt-4 grid grid-cols-2 gap-2">
          <Stat label="epoch" value={String(epoch)} />
          <Stat label="faz" value={phase} />
          <Stat label="ŷ önce" value={formatNumber(trace.prediction[0] ?? 0)} />
          <Stat label="ŷ sonra" value={formatNumber(trace.predictionAfter[0] ?? 0)} />
        </div>
      </div>
    </section>
  );
}

function LossChart({ values }: { values: number[] }) {
  const width = 620;
  const height = 100;
  const maxLoss = Math.max(0.001, ...values);
  const xScale = scaleLinear()
    .domain([0, Math.max(1, values.length - 1)])
    .range([8, width - 10]);
  const yScale = scaleLinear().domain([0, maxLoss]).range([height - 16, 8]);
  const chartValues = values.length > 1 ? values : [values[0] ?? 0, values[0] ?? 0];
  const path =
    line<number>()
      .x((_, index) => xScale(index))
      .y((value) => yScale(value))(chartValues) ?? "";

  return (
    <svg className="h-[108px] w-full" viewBox={`0 0 ${width} ${height}`}>
      <rect width={width} height={height} rx={6} fill="#fbfdff" stroke="#dbe3ee" />
      <line x1={8} y1={height - 16} x2={width - 10} y2={height - 16} stroke="#dbe3ee" />
      <path d={path} fill="none" stroke="#2563eb" strokeWidth={3} strokeLinecap="round" />
      {chartValues.map((value, index) => (
        <circle
          key={`${index}-${value}`}
          cx={xScale(index)}
          cy={yScale(value)}
          r={index === chartValues.length - 1 ? 3.5 : 1.8}
          fill={index === chartValues.length - 1 ? "#f97316" : "#60a5fa"}
        />
      ))}
    </svg>
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

function FormulaBox({ title, lines }: { title: string; lines: string[] }) {
  return (
    <div className="rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
      <div className="mb-2 text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        {title}
      </div>
      <div className="space-y-1 font-mono text-xs text-[#334155]">
        {lines.map((item) => (
          <div key={item} className="break-words">
            {item}
          </div>
        ))}
      </div>
    </div>
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
      className={`inline-flex h-9 w-9 items-center justify-center rounded-md border border-[#cbd5e1] bg-white text-[#334155] transition hover:border-[#2563eb] hover:text-[#2563eb] disabled:cursor-not-allowed disabled:opacity-40 ${className}`}
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
