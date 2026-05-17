import type { ModelMeta, ModelMetaLayer } from "./types";

export type ActivationName = "input" | "linear" | "sigmoid" | "tanh" | "relu";
export type LayerKind = "input" | "hidden" | "output";
export type TrainingPhase = "idle" | "forward" | "backward";

export interface DataPoint {
  id: string;
  inputs: number[];
  targets: number[];
  /** optional label for classification display */
  label?: string;
}

export interface IncomingContribution {
  fromNeuronId: string;
  fromLayerIndex: number;
  fromNeuronIndex: number;
  inputValue: number;
  weight: number;
  product: number;
}

export interface NeuronSnapshot {
  id: string;
  layerIndex: number;
  neuronIndex: number;
  layerKind: LayerKind;
  activation: ActivationName;
  bias: number;
  z: number;
  value: number;
  delta: number;
  derivative: number;
  incoming: IncomingContribution[];
  formula: string;
  gradientBias: number;
}

export interface EdgeSnapshot {
  id: string;
  fromNeuronId: string;
  toNeuronId: string;
  fromLayerIndex: number;
  fromNeuronIndex: number;
  toLayerIndex: number;
  toNeuronIndex: number;
  weight: number;
  weightBefore: number;
  weightAfter: number;
  contribution: number;
  gradient: number;
  correction: number;
  errorSignal: number;
}

export interface TrainingTrace {
  input: number[];
  target: number[];
  prediction: number[];
  predictionAfter: number[];
  loss: number;
  epochLoss: number;
  sampleId?: string;
  neurons: NeuronSnapshot[];
  edges: EdgeSnapshot[];
}

export interface EpochTrainingResult {
  trace: TrainingTrace;
  samples: TrainingTrace[];
  lossBefore: number;
  lossAfter: number;
  epochLoss: number;
}

export type Selection =
  | { type: "neuron"; id: string; layerIndex: number; neuronIndex: number }
  | {
      type: "edge";
      id: string;
      fromLayerIndex: number;
      fromNeuronIndex: number;
      toLayerIndex: number;
      toNeuronIndex: number;
    };

// ─── Math helpers ───────────────────────────────────────────────────────────

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function roundForFormula(value: number) {
  return Number.isFinite(value) ? value.toFixed(3) : "0.000";
}

export function activationValue(name: ActivationName, z: number): number {
  if (name === "sigmoid") return 1 / (1 + Math.exp(-z));
  if (name === "tanh") return Math.tanh(z);
  if (name === "relu") return Math.max(0, z);
  return z; // linear / input
}

export function activationDerivative(name: ActivationName, z: number): number {
  if (name === "sigmoid") {
    const y = activationValue(name, z);
    return y * (1 - y);
  }
  if (name === "tanh") {
    const y = Math.tanh(z);
    return 1 - y * y;
  }
  if (name === "relu") return z > 0 ? 1 : 0;
  return 1;
}

export function activationLabel(name: ActivationName): string {
  const labels: Record<ActivationName, string> = {
    input: "Giriş",
    linear: "Lineer",
    sigmoid: "Sigmoid  σ(z)=1/(1+e⁻ᶻ)",
    tanh: "Tanh  tanh(z)",
    relu: "ReLU  max(0,z)",
  };
  return labels[name];
}

function layerKind(index: number, count: number): LayerKind {
  if (index === 0) return "input";
  if (index === count - 1) return "output";
  return "hidden";
}

function activationFromMeta(layer: ModelMetaLayer, index: number, count: number): ActivationName {
  if (layer.kind === "input" || index === 0) return "input";
  const raw = layer.activation?.toLocaleLowerCase("en-US");
  if (raw === "linear" || raw === "sigmoid" || raw === "tanh" || raw === "relu") return raw;
  if (layer.kind === "output" || index === count - 1) return "linear";
  if (layer.kind === "lstm") return "tanh";
  if (layer.kind === "pool") return "linear";
  return "relu";
}

// ─── Seeded random ──────────────────────────────────────────────────────────

export class SeededRandom {
  private seed: number;
  constructor(seed = 1327) {
    this.seed = seed >>> 0;
  }
  next() {
    this.seed = (1664525 * this.seed + 1013904223) >>> 0;
    return this.seed / 4294967296;
  }
  centered(scale = 1) {
    return (this.next() * 2 - 1) * scale;
  }
}

// ─── Neuron ─────────────────────────────────────────────────────────────────

export class Neuron {
  readonly id: string;
  readonly layerIndex: number;
  readonly neuronIndex: number;
  readonly layerKind: LayerKind;
  activation: ActivationName;
  bias: number;
  z = 0;
  value = 0;
  delta = 0;
  derivative = 1;
  incoming: IncomingContribution[] = [];
  gradientBias = 0;

  constructor(
    layerIndex: number,
    neuronIndex: number,
    layerKindValue: LayerKind,
    activation: ActivationName,
    bias = 0
  ) {
    this.layerIndex = layerIndex;
    this.neuronIndex = neuronIndex;
    this.layerKind = layerKindValue;
    this.activation = activation;
    this.bias = bias;
    this.id = `L${layerIndex}:N${neuronIndex}`;
  }

  clone() {
    const next = new Neuron(
      this.layerIndex,
      this.neuronIndex,
      this.layerKind,
      this.activation,
      this.bias
    );
    next.z = this.z;
    next.value = this.value;
    next.delta = this.delta;
    next.derivative = this.derivative;
    next.gradientBias = this.gradientBias;
    next.incoming = this.incoming.map((item) => ({ ...item }));
    return next;
  }

  snapshot(): NeuronSnapshot {
    const inputFormula =
      this.layerKind === "input"
        ? `x = ${roundForFormula(this.value)}`
        : this.incoming
            .map(
              (item) =>
                `${roundForFormula(item.inputValue)}×${roundForFormula(item.weight)}`
            )
            .join(" + ");
    const sumFormula =
      this.layerKind === "input"
        ? inputFormula
        : `${inputFormula || "0"} + b(${roundForFormula(this.bias)})`;

    return {
      id: this.id,
      layerIndex: this.layerIndex,
      neuronIndex: this.neuronIndex,
      layerKind: this.layerKind,
      activation: this.activation,
      bias: this.bias,
      z: this.z,
      value: this.value,
      delta: this.delta,
      derivative: this.derivative,
      incoming: this.incoming.map((item) => ({ ...item })),
      formula: sumFormula,
      gradientBias: this.gradientBias,
    };
  }
}

// ─── Layer ──────────────────────────────────────────────────────────────────

export class Layer {
  id: string;
  layerIndex: number;
  kind: LayerKind;
  activation: ActivationName;
  neurons: Neuron[];

  constructor(
    layerIndex: number,
    size: number,
    totalLayers: number,
    rng: SeededRandom,
    activation?: ActivationName
  ) {
    this.layerIndex = layerIndex;
    this.kind = layerKind(layerIndex, totalLayers);
    this.activation = activation ?? this.defaultActivation();
    this.id = `L${layerIndex}`;
    this.neurons = Array.from({ length: size }, (_, neuronIndex) => {
      const bias = this.kind === "input" ? 0 : rng.centered(0.25);
      return new Neuron(layerIndex, neuronIndex, this.kind, this.activation, bias);
    });
  }

  private defaultActivation(): ActivationName {
    if (this.kind === "input") return "input";
    if (this.kind === "output") return "linear";
    return "sigmoid";
  }

  clone() {
    const next = Object.create(Layer.prototype) as Layer;
    next.layerIndex = this.layerIndex;
    next.kind = this.kind;
    next.activation = this.activation;
    next.id = this.id;
    next.neurons = this.neurons.map((neuron) => neuron.clone());
    return next;
  }
}

// ─── NeuralNetwork ──────────────────────────────────────────────────────────

export interface LayerConfig {
  size: number;
  activation?: ActivationName;
}

export class NeuralNetwork {
  layers: Layer[];
  weights: number[][][];

  private constructor(layers: Layer[], weights: number[][][]) {
    this.layers = layers;
    this.weights = weights;
  }

  static create(
    layerConfigs: (number | LayerConfig)[],
    seed = 1327
  ): NeuralNetwork {
    const rng = new SeededRandom(seed);
    const configs: LayerConfig[] = layerConfigs.map((c) =>
      typeof c === "number" ? { size: c } : c
    );
    const cleanSizes = configs.map((c) => Math.max(1, Math.floor(c.size)));
    const layers = configs.map(
      (config, index) =>
        new Layer(index, Math.max(1, Math.floor(config.size)), configs.length, rng, config.activation)
    );
    const weights = cleanSizes.slice(0, -1).map((fromSize, layerIndex) => {
      const toSize = cleanSizes[layerIndex + 1];
      const scale = Math.sqrt(2 / (fromSize + toSize));
      return Array.from({ length: fromSize }, () =>
        Array.from({ length: toSize }, () => rng.centered(scale))
      );
    });
    return new NeuralNetwork(layers, weights);
  }

  static fromParameters(
    layerConfigs: LayerConfig[],
    weights: number[][][],
    biases: number[][],
    seed = 1327
  ): NeuralNetwork {
    const network = NeuralNetwork.create(layerConfigs, seed);
    if (weights.length !== network.weights.length) {
      throw new Error("Ağırlık katman sayısı mimari ile eşleşmiyor.");
    }

    network.weights = network.weights.map((matrix, layerIndex) => {
      const importedMatrix = weights[layerIndex];
      if (!importedMatrix || importedMatrix.length !== matrix.length) {
        throw new Error(`Ağırlık matrisi L${layerIndex} giriş boyutu ile eşleşmiyor.`);
      }
      return matrix.map((row, fromIndex) => {
        const importedRow = importedMatrix[fromIndex];
        if (!importedRow || importedRow.length !== row.length) {
          throw new Error(`Ağırlık matrisi L${layerIndex} çıkış boyutu ile eşleşmiyor.`);
        }
        return row.map((_, toIndex) => {
          const value = Number(importedRow[toIndex]);
          return Number.isFinite(value) ? value : 0;
        });
      });
    });

    network.layers.forEach((layer, layerIndex) => {
      layer.neurons.forEach((neuron, neuronIndex) => {
        const value = Number(biases[layerIndex]?.[neuronIndex] ?? 0);
        neuron.bias = layer.kind === "input" ? 0 : Number.isFinite(value) ? value : 0;
      });
    });

    return network;
  }

  static fromMeta(meta: ModelMeta, seed = 1327): NeuralNetwork {
    if (!meta.layers.length) {
      throw new Error("modelMeta.layers boş; görselleştirilecek mimari yok.");
    }

    const layerConfigs = meta.layers.map((layer, index) => ({
      size: Math.max(1, Math.floor(Number.isFinite(layer.size) ? layer.size : 1)),
      activation: activationFromMeta(layer, index, meta.layers.length),
    }));
    const network = NeuralNetwork.create(layerConfigs, seed);
    const importedWeights = meta.weights;

    if (!importedWeights || importedWeights.length !== network.weights.length) {
      return network;
    }

    const shapesMatch = importedWeights.every((matrix, layerIndex) => {
      const target = network.weights[layerIndex];
      return (
        Array.isArray(matrix) &&
        matrix.length === target.length &&
        matrix.every((row, rowIndex) => Array.isArray(row) && row.length === target[rowIndex].length)
      );
    });

    if (!shapesMatch) return network;

    network.weights = network.weights.map((matrix, layerIndex) =>
      matrix.map((row, fromIndex) =>
        row.map((_, toIndex) => {
          const value = Number(importedWeights[layerIndex][fromIndex][toIndex]);
          return Number.isFinite(value) ? value : 0;
        })
      )
    );

    return network;
  }

  clone(): NeuralNetwork {
    return new NeuralNetwork(
      this.layers.map((layer) => layer.clone()),
      this.weights.map((matrix) => matrix.map((row) => [...row]))
    );
  }

  getLayerConfigs(): LayerConfig[] {
    return this.layers.map((layer) => ({
      size: layer.neurons.length,
      activation: layer.activation,
    }));
  }

  getLayerSizes(): number[] {
    return this.layers.map((layer) => layer.neurons.length);
  }

  setLayerActivation(layerIndex: number, activation: ActivationName) {
    const layer = this.layers[layerIndex];
    if (!layer || layer.kind === "input") return;
    layer.activation = activation;
    layer.neurons.forEach((n) => {
      n.activation = activation;
    });
  }

  forward(inputs: number[]): number[] {
    const inputLayer = this.layers[0];
    inputLayer.neurons.forEach((neuron, index) => {
      const value = inputs[index] ?? 0;
      neuron.value = value;
      neuron.z = value;
      neuron.delta = 0;
      neuron.derivative = 1;
      neuron.gradientBias = 0;
      neuron.incoming = [];
    });

    for (let layerIndex = 1; layerIndex < this.layers.length; layerIndex += 1) {
      const previous = this.layers[layerIndex - 1];
      const current = this.layers[layerIndex];
      current.neurons.forEach((neuron, neuronIndex) => {
        let z = neuron.bias;
        neuron.incoming = previous.neurons.map((previousNeuron, previousIndex) => {
          const weight = this.weights[layerIndex - 1][previousIndex][neuronIndex];
          const product = previousNeuron.value * weight;
          z += product;
          return {
            fromNeuronId: previousNeuron.id,
            fromLayerIndex: previousNeuron.layerIndex,
            fromNeuronIndex: previousNeuron.neuronIndex,
            inputValue: previousNeuron.value,
            weight,
            product,
          };
        });
        neuron.z = z;
        neuron.value = activationValue(neuron.activation, z);
        neuron.derivative = activationDerivative(neuron.activation, z);
        neuron.delta = 0;
        neuron.gradientBias = 0;
      });
    }

    return this.layers[this.layers.length - 1].neurons.map((n) => n.value);
  }

  predictPure(inputs: number[]): number[] {
    let values = inputs;
    for (let layerIndex = 1; layerIndex < this.layers.length; layerIndex += 1) {
      const layer = this.layers[layerIndex];
      values = layer.neurons.map((neuron, neuronIndex) => {
        const z = values.reduce(
          (sum, value, previousIndex) =>
            sum + value * this.weights[layerIndex - 1][previousIndex][neuronIndex],
          neuron.bias
        );
        return activationValue(neuron.activation, z);
      });
    }
    return values;
  }

  predict(x: number): number {
    return this.predictPure([x])[0] ?? 0;
  }

  /** Returns the index of the class with the highest output (for classification). */
  predictClass(inputs: number[]): number {
    const out = this.predictPure(inputs);
    return out.indexOf(Math.max(...out));
  }

  inspect(input: number[], target: number[]): TrainingTrace {
    const prediction = this.forward(input);
    const loss = this.lossForPrediction(prediction, target);
    this.calculateDeltas(target);
    return {
      input,
      target,
      prediction,
      predictionAfter: prediction,
      loss,
      epochLoss: loss,
      neurons: this.neuronSnapshots(),
      edges: this.edgeSnapshots([], 0),
      sampleId: undefined,
    };
  }

  trainSample(
    input: number[],
    target: number[],
    learningRate: number,
    sampleId?: string
  ): TrainingTrace {
    const prediction = this.forward(input);
    const loss = this.lossForPrediction(prediction, target);
    const gradientMatrices = this.calculateDeltas(target);
    const edgeTrace = this.edgeSnapshots(gradientMatrices, learningRate);

    for (let layerIndex = 1; layerIndex < this.layers.length; layerIndex += 1) {
      const current = this.layers[layerIndex];
      const previous = this.layers[layerIndex - 1];
      current.neurons.forEach((neuron, neuronIndex) => {
        neuron.bias += -learningRate * neuron.delta;
        previous.neurons.forEach((_, previousIndex) => {
          this.weights[layerIndex - 1][previousIndex][neuronIndex] +=
            -learningRate * gradientMatrices[layerIndex - 1][previousIndex][neuronIndex];
        });
      });
    }

    const neuronTrace = this.neuronSnapshots();
    const predictionAfter = this.forward(input);
    return {
      input,
      target,
      prediction,
      predictionAfter,
      loss,
      epochLoss: loss,
      sampleId,
      neurons: neuronTrace,
      edges: edgeTrace,
    };
  }

  trainEpoch(data: DataPoint[], learningRate: number): TrainingTrace {
    return this.trainEpochDetailed(data, learningRate).trace;
  }

  trainEpochDetailed(data: DataPoint[], learningRate: number): EpochTrainingResult {
    if (data.length === 0) {
      const sizes = this.getLayerSizes();
      const trace = this.inspect(
        Array(sizes[0]).fill(0),
        Array(sizes[sizes.length - 1]).fill(0)
      );
      return {
        trace,
        samples: [trace],
        lossBefore: 0,
        lossAfter: 0,
        epochLoss: 0,
      };
    }

    const lossBefore = this.evaluateLoss(data);
    let totalLoss = 0;
    let trace!: TrainingTrace;
    const samples: TrainingTrace[] = [];
    data.forEach((point) => {
      trace = this.trainSample(point.inputs, point.targets, learningRate, point.id);
      totalLoss += trace.loss;
      samples.push(trace);
    });

    const epochLoss = totalLoss / data.length;
    const lossAfter = this.evaluateLoss(data);
    trace = { ...trace, epochLoss };
    return { trace, samples, lossBefore, lossAfter, epochLoss };
  }

  evaluateLoss(data: DataPoint[]): number {
    if (data.length === 0) return 0;
    const total = data.reduce((sum, point) => {
      const prediction = this.predictPure(point.inputs);
      return sum + this.lossForPrediction(prediction, point.targets);
    }, 0);
    return total / data.length;
  }

  evaluateAccuracy(data: DataPoint[]): number {
    if (data.length === 0) return 0;
    const correct = data.filter((point) => {
      const predicted = this.predictClass(point.inputs);
      const actual = point.targets.indexOf(Math.max(...point.targets));
      return predicted === actual;
    }).length;
    return correct / data.length;
  }

  private lossForPrediction(prediction: number[], target: number[]): number {
    return prediction.reduce((sum, output, index) => {
      const error = output - (target[index] ?? 0);
      return sum + 0.5 * error * error;
    }, 0);
  }

  private calculateDeltas(target: number[]): number[][][] {
    const lastLayer = this.layers[this.layers.length - 1];
    lastLayer.neurons.forEach((neuron, index) => {
      const dLossDActivation = neuron.value - (target[index] ?? 0);
      neuron.delta = dLossDActivation * neuron.derivative;
      neuron.gradientBias = neuron.delta;
    });

    for (let layerIndex = this.layers.length - 2; layerIndex >= 1; layerIndex -= 1) {
      const layer = this.layers[layerIndex];
      const nextLayer = this.layers[layerIndex + 1];
      layer.neurons.forEach((neuron, neuronIndex) => {
        const backpropagatedError = nextLayer.neurons.reduce(
          (sum, nextNeuron, nextIndex) =>
            sum + this.weights[layerIndex][neuronIndex][nextIndex] * nextNeuron.delta,
          0
        );
        neuron.delta = backpropagatedError * neuron.derivative;
        neuron.gradientBias = neuron.delta;
      });
    }

    return this.weights.map((matrix, layerIndex) =>
      matrix.map((row, fromIndex) =>
        row.map((_, toIndex) => {
          const previousValue = this.layers[layerIndex].neurons[fromIndex].value;
          const nextDelta = this.layers[layerIndex + 1].neurons[toIndex].delta;
          return previousValue * nextDelta;
        })
      )
    );
  }

  private neuronSnapshots(): NeuronSnapshot[] {
    return this.layers.flatMap((layer) =>
      layer.neurons.map((neuron) => neuron.snapshot())
    );
  }

  private edgeSnapshots(gradients: number[][][], learningRate: number): EdgeSnapshot[] {
    const snapshots: EdgeSnapshot[] = [];
    this.weights.forEach((matrix, layerIndex) => {
      matrix.forEach((row, fromIndex) => {
        row.forEach((weight, toIndex) => {
          const from = this.layers[layerIndex].neurons[fromIndex];
          const to = this.layers[layerIndex + 1].neurons[toIndex];
          const gradient = gradients[layerIndex]?.[fromIndex]?.[toIndex] ?? 0;
          const correction = -learningRate * gradient;
          snapshots.push({
            id: edgeId(layerIndex, fromIndex, layerIndex + 1, toIndex),
            fromNeuronId: from.id,
            toNeuronId: to.id,
            fromLayerIndex: layerIndex,
            fromNeuronIndex: fromIndex,
            toLayerIndex: layerIndex + 1,
            toNeuronIndex: toIndex,
            weight,
            weightBefore: weight,
            weightAfter: weight + correction,
            contribution: from.value * weight,
            gradient,
            correction,
            errorSignal: Math.abs(weight * to.delta),
          });
        });
      });
    });
    return snapshots;
  }
}

// ─── Utilities ──────────────────────────────────────────────────────────────

export function edgeId(
  fromLayerIndex: number,
  fromNeuronIndex: number,
  toLayerIndex: number,
  toNeuronIndex: number
): string {
  return `E${fromLayerIndex}:${fromNeuronIndex}->${toLayerIndex}:${toNeuronIndex}`;
}

export function formatNumber(value: number, digits = 4): string {
  if (!Number.isFinite(value)) return "0";
  if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
    return value.toExponential(2);
  }
  return value.toFixed(digits);
}

export function sanitizePointValue(value: number): number {
  return clamp(value, 0, 1);
}
