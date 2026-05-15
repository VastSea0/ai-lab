import type {
  DataPoint,
  EdgeSnapshot,
  EpochTrainingResult,
  NeuronSnapshot,
  Selection,
  TrainingTrace,
} from "./network";
import { formatNumber } from "./network";
import type { Task } from "./tasks";

export type CalculationPhase = "input" | "forward" | "loss" | "backward" | "update";

export interface CalculationStep {
  id: string;
  phase: CalculationPhase;
  title: string;
  summary: string;
  equations: string[];
  details: string[];
  target?: Selection;
}

export interface SampleTraceRecord {
  sampleId?: string;
  sampleName: string;
  input: number[];
  target: number[];
  prediction: number[];
  predictionAfter: number[];
  loss: number;
  steps: CalculationStep[];
}

export interface EpochTraceRecord {
  epoch: number;
  learningRate: number;
  lossBefore: number;
  lossAfter: number;
  epochLoss: number;
  samples: SampleTraceRecord[];
}

function neuronTitle(neuron: NeuronSnapshot) {
  return `L${neuron.layerIndex} N${neuron.neuronIndex}`;
}

function edgeTitle(edge: EdgeSnapshot) {
  return `L${edge.fromLayerIndex}N${edge.fromNeuronIndex} → L${edge.toLayerIndex}N${edge.toNeuronIndex}`;
}

function neuronSelection(neuron: NeuronSnapshot): Selection {
  return {
    type: "neuron",
    id: neuron.id,
    layerIndex: neuron.layerIndex,
    neuronIndex: neuron.neuronIndex,
  };
}

function edgeSelection(edge: EdgeSnapshot): Selection {
  return {
    type: "edge",
    id: edge.id,
    fromLayerIndex: edge.fromLayerIndex,
    fromNeuronIndex: edge.fromNeuronIndex,
    toLayerIndex: edge.toLayerIndex,
    toNeuronIndex: edge.toNeuronIndex,
  };
}

function targetLabel(task: Task, target: number[]) {
  if (task.outputType === "regression") return formatNumber(target[0] ?? 0, 3);
  const index = target.indexOf(Math.max(...target));
  return task.classNames?.[index] ?? `Sınıf ${index}`;
}

function predictionLabel(task: Task, prediction: number[]) {
  if (task.outputType === "regression") return formatNumber(prediction[0] ?? 0, 3);
  const index = prediction.indexOf(Math.max(...prediction));
  return `${task.classNames?.[index] ?? `Sınıf ${index}`} (${formatNumber(
    prediction[index] ?? 0,
    3
  )})`;
}

export function buildCalculationSteps(
  trace: TrainingTrace,
  task: Task,
  learningRate: number
): CalculationStep[] {
  const steps: CalculationStep[] = [];
  const neurons = new Map(trace.neurons.map((neuron) => [neuron.id, neuron]));

  steps.push({
    id: "input",
    phase: "input",
    title: "Girdiler ağa verildi",
    summary: `${trace.input.length} giriş değeri input nöronlarına yazıldı.`,
    equations: trace.input.map((value, index) => `x${index + 1} = ${formatNumber(value, 4)}`),
    details: [
      "Input nöronları ağırlık hesabı yapmaz; sadece veri değerini bir sonraki katmana taşır.",
    ],
  });

  trace.neurons
    .filter((neuron) => neuron.layerKind !== "input")
    .forEach((neuron) => {
      const termLines = neuron.incoming.map(
        (item) =>
          `${item.fromNeuronId}: ${formatNumber(item.inputValue, 5)} × ${formatNumber(
            item.weight,
            5
          )} = ${formatNumber(item.product, 5)}`
      );
      const productSum = neuron.incoming.reduce((sum, item) => sum + item.product, 0);
      steps.push({
        id: `forward-${neuron.id}`,
        phase: "forward",
        title: `Forward: ${neuronTitle(neuron)}`,
        summary: `Önce ağırlıklı toplam, sonra ${neuron.activation} aktivasyonu hesaplandı.`,
        equations: [
          `Σ(x×w) = ${formatNumber(productSum, 6)}`,
          `z = Σ(x×w) + b = ${formatNumber(productSum, 6)} + ${formatNumber(
            neuron.bias,
            6
          )} = ${formatNumber(neuron.z, 6)}`,
          `a = ${neuron.activation}(z) = ${formatNumber(neuron.value, 6)}`,
        ],
        details: termLines,
        target: neuronSelection(neuron),
      });
    });

  const predictionText = trace.prediction.map((value) => formatNumber(value, 6)).join(", ");
  const targetText = trace.target.map((value) => formatNumber(value, 6)).join(", ");
  steps.push({
    id: "loss",
    phase: "loss",
    title: "Loss hesaplandı",
    summary: `Tahmin ${predictionLabel(task, trace.prediction)}, hedef ${targetLabel(task, trace.target)}.`,
    equations: [
      "L = 1/2 × Σ(ŷ - y)^2",
      `ŷ = [${predictionText}]`,
      `y = [${targetText}]`,
      `L = ${formatNumber(trace.loss, 8)}`,
    ],
    details: trace.prediction.map((output, index) => {
      const target = trace.target[index] ?? 0;
      const error = output - target;
      return `çıktı ${index + 1}: 1/2 × (${formatNumber(output, 6)} - ${formatNumber(
        target,
        6
      )})² = ${formatNumber(0.5 * error * error, 8)}`;
    }),
  });

  trace.neurons
    .filter((neuron) => neuron.layerKind === "output")
    .forEach((neuron) => {
      const target = trace.target[neuron.neuronIndex] ?? 0;
      steps.push({
        id: `backward-output-${neuron.id}`,
        phase: "backward",
        title: `Output hatası: ${neuronTitle(neuron)}`,
        summary: "Çıktı nöronunda hata doğrudan tahmin ile hedef farkından gelir.",
        equations: [
          `∂L/∂a = a - y = ${formatNumber(neuron.value, 6)} - ${formatNumber(
            target,
            6
          )} = ${formatNumber(neuron.value - target, 6)}`,
          `δ = ∂L/∂a × f'(z) = ${formatNumber(neuron.value - target, 6)} × ${formatNumber(
            neuron.derivative,
            6
          )} = ${formatNumber(neuron.delta, 6)}`,
        ],
        details: [
          "δ değeri, bu nöronun loss'u artırma yönünü ve büyüklüğünü taşır.",
        ],
        target: neuronSelection(neuron),
      });
    });

  trace.neurons
    .filter((neuron) => neuron.layerKind === "hidden")
    .sort((a, b) => b.layerIndex - a.layerIndex || a.neuronIndex - b.neuronIndex)
    .forEach((neuron) => {
      const outgoing = trace.edges.filter((edge) => edge.fromNeuronId === neuron.id);
      const terms = outgoing.map((edge) => {
        const next = neurons.get(edge.toNeuronId);
        return `${formatNumber(edge.weightBefore, 6)} × ${formatNumber(
          next?.delta ?? 0,
          6
        )}`;
      });
      const backSignal = outgoing.reduce((sum, edge) => {
        const next = neurons.get(edge.toNeuronId);
        return sum + edge.weightBefore * (next?.delta ?? 0);
      }, 0);
      steps.push({
        id: `backward-hidden-${neuron.id}`,
        phase: "backward",
        title: `Hidden hatası: ${neuronTitle(neuron)}`,
        summary: "Hidden nöronun hatası, kendisinden sonraki nöronlardan geri toplanır.",
        equations: [
          `geri sinyal = Σ(w_sonraki × δ_sonraki) = ${terms.join(" + ") || "0"}`,
          `geri sinyal = ${formatNumber(backSignal, 6)}`,
          `δ = geri sinyal × f'(z) = ${formatNumber(backSignal, 6)} × ${formatNumber(
            neuron.derivative,
            6
          )} = ${formatNumber(neuron.delta, 6)}`,
        ],
        details: outgoing.map((edge) => {
          const next = neurons.get(edge.toNeuronId);
          return `${edgeTitle(edge)}: w ${formatNumber(edge.weightBefore, 5)} × δ ${
            next ? formatNumber(next.delta, 5) : "0"
          }`;
        }),
        target: neuronSelection(neuron),
      });
    });

  trace.edges.forEach((edge) => {
    steps.push({
      id: `update-${edge.id}`,
      phase: "update",
      title: `Ağırlık güncelle: ${edgeTitle(edge)}`,
      summary: "Bu bağlantının ağırlığı gradient yönünün tersine hareket ettirildi.",
      equations: [
        `katkı = a_önceki × w = ${formatNumber(edge.contribution, 6)}`,
        `∂L/∂w = a_önceki × δ_sonraki = ${formatNumber(edge.gradient, 8)}`,
        `Δw = -η × ∂L/∂w = -${formatNumber(learningRate, 6)} × ${formatNumber(
          edge.gradient,
          8
        )} = ${formatNumber(edge.correction, 8)}`,
        `w_yeni = ${formatNumber(edge.weightBefore, 8)} + ${formatNumber(
          edge.correction,
          8
        )} = ${formatNumber(edge.weightAfter, 8)}`,
      ],
      details: [
        edge.gradient > 0
          ? "Gradient pozitif olduğu için ağırlık azaltılır."
          : edge.gradient < 0
            ? "Gradient negatif olduğu için ağırlık artırılır."
            : "Gradient sıfıra yakın; bu bağlantı bu adımda neredeyse değişmedi.",
      ],
      target: edgeSelection(edge),
    });
  });

  trace.neurons
    .filter((neuron) => neuron.layerKind !== "input")
    .forEach((neuron) => {
      const biasBefore = neuron.bias + learningRate * neuron.delta;
      steps.push({
        id: `bias-${neuron.id}`,
        phase: "update",
        title: `Bias güncelle: ${neuronTitle(neuron)}`,
        summary: "Bias da ağırlık gibi gradient yönünün tersine güncellenir.",
        equations: [
          `∂L/∂b = δ = ${formatNumber(neuron.delta, 8)}`,
          `b_yeni = b_eski - η × δ`,
          `b_yeni = ${formatNumber(biasBefore, 8)} - ${formatNumber(
            learningRate,
            6
          )} × ${formatNumber(neuron.delta, 8)} = ${formatNumber(neuron.bias, 8)}`,
        ],
        details: [
          "Bias, nöronun aktivasyon eşiğini bağlantılardan bağımsız olarak sağa/sola kaydırır.",
        ],
        target: neuronSelection(neuron),
      });
    });

  return steps;
}

export function buildEpochTraceRecord(
  epoch: number,
  result: EpochTrainingResult,
  task: Task,
  data: DataPoint[],
  learningRate: number
): EpochTraceRecord {
  const byId = new Map(data.map((point) => [point.id, point]));
  return {
    epoch,
    learningRate,
    lossBefore: result.lossBefore,
    lossAfter: result.lossAfter,
    epochLoss: result.epochLoss,
    samples: result.samples.map((trace, index) => {
      const point = trace.sampleId ? byId.get(trace.sampleId) : undefined;
      return {
        sampleId: trace.sampleId,
        sampleName: point?.label ?? point?.id ?? `Örnek ${index + 1}`,
        input: trace.input,
        target: trace.target,
        prediction: trace.prediction,
        predictionAfter: trace.predictionAfter,
        loss: trace.loss,
        steps: buildCalculationSteps(trace, task, learningRate),
      };
    }),
  };
}
