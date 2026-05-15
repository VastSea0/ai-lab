import { DataPoint, LayerConfig } from "./network";

export type TaskId =
  | "regression"
  | "xor"
  | "circle"
  | "spiral"
  | "digit";

export interface Task {
  id: TaskId;
  name: string;
  emoji: string;
  description: string;
  /** What the network is doing, in plain Turkish */
  explanation: string;
  inputSize: number;
  outputSize: number;
  outputType: "regression" | "classification";
  defaultLayers: LayerConfig[];
  defaultLearningRate: number;
  data: DataPoint[];
  /** axis labels for the 2-D scatter/boundary plot */
  axisLabels?: { x: string; y: string };
  /** class colours indexed by class index */
  classColors?: string[];
  /** class names */
  classNames?: string[];
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function pt(id: string, inputs: number[], targets: number[], label?: string): DataPoint {
  return { id, inputs, targets, label };
}

function oneHot(index: number, size: number): number[] {
  return Array.from({ length: size }, (_, i) => (i === index ? 1 : 0));
}

// ─── 1. Basit Regresyon ─────────────────────────────────────────────────────

const regressionData: DataPoint[] = [
  pt("r1", [0.08], [0.14]),
  pt("r2", [0.20], [0.22]),
  pt("r3", [0.36], [0.50]),
  pt("r4", [0.54], [0.62]),
  pt("r5", [0.74], [0.84]),
  pt("r6", [0.92], [0.91]),
];

// ─── 2. XOR ─────────────────────────────────────────────────────────────────

const xorData: DataPoint[] = [
  pt("x1", [0, 0], [0]),
  pt("x2", [0, 1], [1]),
  pt("x3", [1, 0], [1]),
  pt("x4", [1, 1], [0]),
];

// ─── 3. Dairesel Sınır ──────────────────────────────────────────────────────

function makeCircle(): DataPoint[] {
  const points: DataPoint[] = [];
  let index = 0;
  for (let angle = 0; angle < Math.PI * 2; angle += Math.PI / 6) {
    // inner – class 1
    const r1 = 0.22;
    points.push(
      pt(`c${index++}`, [0.5 + r1 * Math.cos(angle), 0.5 + r1 * Math.sin(angle)], oneHot(0, 2), "İç")
    );
    // outer – class 2
    const r2 = 0.38 + (index % 3) * 0.02;
    points.push(
      pt(`c${index++}`, [0.5 + r2 * Math.cos(angle), 0.5 + r2 * Math.sin(angle)], oneHot(1, 2), "Dış")
    );
  }
  return points;
}

const circleData = makeCircle();

// ─── 4. Sarmal (Spiral) ─────────────────────────────────────────────────────

function makeSpiral(): DataPoint[] {
  const points: DataPoint[] = [];
  const n = 10;
  for (let i = 0; i < n; i++) {
    const t = (i / n) * 2.5;
    // class A – blue
    const ax = 0.5 + (0.08 + t * 0.16) * Math.cos(t * 2.8);
    const ay = 0.5 + (0.08 + t * 0.16) * Math.sin(t * 2.8);
    points.push(pt(`s${i}a`, [ax, ay], oneHot(0, 2), "A"));
    // class B – red
    const bx = 0.5 + (0.08 + t * 0.16) * Math.cos(t * 2.8 + Math.PI);
    const by = 0.5 + (0.08 + t * 0.16) * Math.sin(t * 2.8 + Math.PI);
    points.push(pt(`s${i}b`, [bx, by], oneHot(1, 2), "B"));
  }
  return points;
}

const spiralData = makeSpiral();

// ─── 5. 5×5 Rakam (0 vs 1) ──────────────────────────────────────────────────
// Each digit is a 5×5 grid (25 pixels, 0 or 1)

const DIGIT_0 = [
  0,1,1,1,0,
  1,0,0,0,1,
  1,0,0,0,1,
  1,0,0,0,1,
  0,1,1,1,0,
];

const DIGIT_1 = [
  0,0,1,0,0,
  0,1,1,0,0,
  0,0,1,0,0,
  0,0,1,0,0,
  0,1,1,1,0,
];

const DIGIT_2 = [
  0,1,1,1,0,
  0,0,0,1,0,
  0,1,1,1,0,
  1,0,0,0,0,
  1,1,1,1,0,
];

const DIGIT_3 = [
  0,1,1,1,0,
  0,0,0,1,0,
  0,1,1,1,0,
  0,0,0,1,0,
  0,1,1,1,0,
];

const digitData: DataPoint[] = [
  pt("d0", DIGIT_0.map(v => v), oneHot(0, 4), "0"),
  pt("d1", DIGIT_1.map(v => v), oneHot(1, 4), "1"),
  pt("d2", DIGIT_2.map(v => v), oneHot(2, 4), "2"),
  pt("d3", DIGIT_3.map(v => v), oneHot(3, 4), "3"),
];

// ─── Task Registry ──────────────────────────────────────────────────────────

export const TASKS: Task[] = [
  {
    id: "regression",
    name: "Regresyon",
    emoji: "📈",
    description: "Sayılar arası ilişki öğren",
    explanation:
      "Ağ, verilen x değerlerine karşılık gelen y değerlerini tahmin etmeye çalışır. Tıpkı bir 'en iyi çizgi' bulmak gibi — ama düz değil, eğri olabilir!",
    inputSize: 1,
    outputSize: 1,
    outputType: "regression",
    defaultLayers: [{ size: 1 }, { size: 5, activation: "sigmoid" }, { size: 1, activation: "linear" }],
    defaultLearningRate: 0.18,
    data: regressionData,
    axisLabels: { x: "x", y: "y" },
  },
  {
    id: "xor",
    name: "XOR Problemi",
    emoji: "⚡",
    description: "Doğrusal olmayan mantık öğren",
    explanation:
      "XOR: iki giriş aynıysa 0, farklıysa 1 üretir. Tek katmanlı ağlar bunu çözemez — gizli katman şart! Bu yüzden derin öğrenmenin temel örneğidir.",
    inputSize: 2,
    outputSize: 1,
    outputType: "regression",
    defaultLayers: [{ size: 2 }, { size: 4, activation: "sigmoid" }, { size: 1, activation: "sigmoid" }],
    defaultLearningRate: 0.5,
    data: xorData,
    axisLabels: { x: "Giriş A", y: "Giriş B" },
    classColors: ["#3b82f6", "#ef4444"],
    classNames: ["XOR=0", "XOR=1"],
  },
  {
    id: "circle",
    name: "Dairesel Sınır",
    emoji: "⭕",
    description: "2 sınıf: iç/dış daire",
    explanation:
      "Noktalar iki gruba ayrılmış: dairenin içi ve dışı. Ağ, aralarındaki yuvarlak sınırı öğrenmeye çalışır. İki giriş var: x ve y koordinatları.",
    inputSize: 2,
    outputSize: 2,
    outputType: "classification",
    defaultLayers: [
      { size: 2 },
      { size: 6, activation: "sigmoid" },
      { size: 4, activation: "sigmoid" },
      { size: 2, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.4,
    data: circleData,
    axisLabels: { x: "x", y: "y" },
    classColors: ["#3b82f6", "#ef4444"],
    classNames: ["İç", "Dış"],
  },
  {
    id: "spiral",
    name: "Sarmal",
    emoji: "🌀",
    description: "Zor 2-sınıf problem",
    explanation:
      "Sarmal şeklinde iç içe geçmiş iki sınıf. Bu problem, düz çizgilerle çözülemez. Ağın 'eğri' sınırlar öğrenmesi gerekir. Derin ağlar için klasik zorluktur!",
    inputSize: 2,
    outputSize: 2,
    outputType: "classification",
    defaultLayers: [
      { size: 2 },
      { size: 8, activation: "tanh" },
      { size: 6, activation: "tanh" },
      { size: 2, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.3,
    data: spiralData,
    axisLabels: { x: "x", y: "y" },
    classColors: ["#3b82f6", "#ef4444"],
    classNames: ["A", "B"],
  },
  {
    id: "digit",
    name: "Rakam Tanıma",
    emoji: "🔢",
    description: "5×5 piksel → 0,1,2,3",
    explanation:
      "Her rakam 25 piksellik bir ızgara. Ağ pikselleri alıp hangi rakam olduğunu tahmin eder. Gerçek görüntü tanımanın (MNIST gibi) küçük versiyonu!",
    inputSize: 25,
    outputSize: 4,
    outputType: "classification",
    defaultLayers: [
      { size: 25 },
      { size: 12, activation: "relu" },
      { size: 8, activation: "relu" },
      { size: 4, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.25,
    data: digitData,
    classColors: ["#6366f1", "#f59e0b", "#10b981", "#ef4444"],
    classNames: ["0", "1", "2", "3"],
  },
];

export function getTask(id: TaskId): Task {
  return TASKS.find((t) => t.id === id) ?? TASKS[0];
}
