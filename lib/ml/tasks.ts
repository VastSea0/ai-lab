import { DataPoint, LayerConfig } from "./network";
import { NLP_CLASS_NAMES, NLP_VOCABULARY, textPoint } from "./nlp";

export type TaskId =
  | "regression"
  | "sine"
  | "and"
  | "xor"
  | "linear"
  | "checker"
  | "circle"
  | "moons"
  | "blobs"
  | "spiral"
  | "digit"
  | "sentiment";

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

function clamp01(value: number) {
  return Math.max(0, Math.min(1, value));
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

// ─── 2. Sinüs Regresyon ─────────────────────────────────────────────────────

const sineData: DataPoint[] = Array.from({ length: 13 }, (_, index) => {
  const x = index / 12;
  const y = 0.5 + Math.sin(x * Math.PI * 2) * 0.34;
  return pt(`sin${index}`, [x], [clamp01(y)]);
});

// ─── 3. AND Mantık Kapısı ───────────────────────────────────────────────────

const andData: DataPoint[] = [
  pt("a1", [0, 0], oneHot(0, 2), "0"),
  pt("a2", [0, 1], oneHot(0, 2), "0"),
  pt("a3", [1, 0], oneHot(0, 2), "0"),
  pt("a4", [1, 1], oneHot(1, 2), "1"),
];

// ─── 4. XOR ─────────────────────────────────────────────────────────────────

const xorData: DataPoint[] = [
  pt("x1", [0, 0], oneHot(0, 2), "0"),
  pt("x2", [0, 1], oneHot(1, 2), "1"),
  pt("x3", [1, 0], oneHot(1, 2), "1"),
  pt("x4", [1, 1], oneHot(0, 2), "0"),
];

// ─── 5. Lineer Sınıflandırma ────────────────────────────────────────────────

const linearData: DataPoint[] = [
  pt("l1", [0.12, 0.22], oneHot(0, 2), "Alt"),
  pt("l2", [0.20, 0.42], oneHot(0, 2), "Alt"),
  pt("l3", [0.35, 0.28], oneHot(0, 2), "Alt"),
  pt("l4", [0.42, 0.46], oneHot(0, 2), "Alt"),
  pt("l5", [0.62, 0.20], oneHot(0, 2), "Alt"),
  pt("l6", [0.78, 0.18], oneHot(0, 2), "Alt"),
  pt("l7", [0.26, 0.76], oneHot(1, 2), "Üst"),
  pt("l8", [0.38, 0.68], oneHot(1, 2), "Üst"),
  pt("l9", [0.58, 0.62], oneHot(1, 2), "Üst"),
  pt("l10", [0.70, 0.78], oneHot(1, 2), "Üst"),
  pt("l11", [0.82, 0.54], oneHot(1, 2), "Üst"),
  pt("l12", [0.90, 0.72], oneHot(1, 2), "Üst"),
];

// ─── 6. Dama Tahtası / XOR Alanı ────────────────────────────────────────────

const checkerData: DataPoint[] = [
  pt("ch1", [0.12, 0.12], oneHot(0, 2), "A"),
  pt("ch2", [0.28, 0.28], oneHot(0, 2), "A"),
  pt("ch3", [0.72, 0.16], oneHot(1, 2), "B"),
  pt("ch4", [0.86, 0.34], oneHot(1, 2), "B"),
  pt("ch5", [0.18, 0.74], oneHot(1, 2), "B"),
  pt("ch6", [0.34, 0.88], oneHot(1, 2), "B"),
  pt("ch7", [0.68, 0.70], oneHot(0, 2), "A"),
  pt("ch8", [0.86, 0.88], oneHot(0, 2), "A"),
];

// ─── 7. Dairesel Sınır ──────────────────────────────────────────────────────

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

// ─── 8. Ay Hilalleri ────────────────────────────────────────────────────────

function makeMoons(): DataPoint[] {
  const points: DataPoint[] = [];
  for (let i = 0; i < 12; i += 1) {
    const angle = (i / 11) * Math.PI;
    const topX = 0.26 + 0.38 * Math.cos(angle);
    const topY = 0.55 + 0.25 * Math.sin(angle);
    const bottomX = 0.72 - 0.38 * Math.cos(angle);
    const bottomY = 0.42 - 0.25 * Math.sin(angle);
    points.push(pt(`m${i}a`, [clamp01(topX), clamp01(topY)], oneHot(0, 2), "Ay A"));
    points.push(pt(`m${i}b`, [clamp01(bottomX), clamp01(bottomY)], oneHot(1, 2), "Ay B"));
  }
  return points;
}

const moonsData = makeMoons();

// ─── 9. Çok Sınıflı Kümeler ─────────────────────────────────────────────────

function makeBlobs(): DataPoint[] {
  const centers = [
    { x: 0.24, y: 0.28, label: "Mavi" },
    { x: 0.74, y: 0.30, label: "Turuncu" },
    { x: 0.50, y: 0.78, label: "Yeşil" },
  ];
  const offsets = [
    [-0.08, -0.03],
    [-0.04, 0.06],
    [0.02, -0.07],
    [0.06, 0.02],
    [0.09, 0.08],
    [0.00, 0.00],
  ];
  return centers.flatMap((center, classIndex) =>
    offsets.map(([dx, dy], itemIndex) =>
      pt(
        `b${classIndex}-${itemIndex}`,
        [clamp01(center.x + dx), clamp01(center.y + dy)],
        oneHot(classIndex, centers.length),
        center.label
      )
    )
  );
}

const blobsData = makeBlobs();

// ─── 10. Sarmal (Spiral) ────────────────────────────────────────────────────

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

// ─── 6. Mini NLP: Duygu Analizi ─────────────────────────────────────────────

const sentimentData: DataPoint[] = [
  textPoint("nlp1", "bu çok iyi ve güzel", "Pozitif"),
  textPoint("nlp2", "harika başarılı seviyorum", "Pozitif"),
  textPoint("nlp3", "mutlu ve güzel bir deney", "Pozitif"),
  textPoint("nlp4", "bu kötü ve berbat", "Negatif"),
  textPoint("nlp5", "üzgün kızgın ve zor", "Negatif"),
  textPoint("nlp6", "nefret ediyorum berbat", "Negatif"),
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
    id: "sine",
    name: "Sinüs Regresyon",
    emoji: "〰️",
    description: "Dalgalı ilişki öğren",
    explanation:
      "x arttıkça y düz çizgi gibi gitmez; yukarı ve aşağı dalgalanır. Bu örnek hidden katmanların eğri fonksiyonları nasıl yaklaştırdığını görmek için iyi bir başlangıçtır.",
    inputSize: 1,
    outputSize: 1,
    outputType: "regression",
    defaultLayers: [
      { size: 1 },
      { size: 8, activation: "tanh" },
      { size: 6, activation: "tanh" },
      { size: 1, activation: "linear" },
    ],
    defaultLearningRate: 0.08,
    data: sineData,
    axisLabels: { x: "x", y: "sinüs y" },
  },
  {
    id: "and",
    name: "AND Kapısı",
    emoji: "∧",
    description: "En basit mantık sınıflandırması",
    explanation:
      "AND kapısı sadece iki giriş de 1 olduğunda 1 üretir. Bu görev, ağırlık ve bias'ın düz bir karar sınırı oluşturmasını görmek için temiz bir örnektir.",
    inputSize: 2,
    outputSize: 2,
    outputType: "classification",
    defaultLayers: [
      { size: 2 },
      { size: 2, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.35,
    data: andData,
    axisLabels: { x: "Giriş A", y: "Giriş B" },
    classColors: ["#64748b", "#2563eb"],
    classNames: ["0", "1"],
  },
  {
    id: "xor",
    name: "XOR Problemi",
    emoji: "⚡",
    description: "Doğrusal olmayan mantık öğren",
    explanation:
      "XOR: iki giriş aynıysa 0, farklıysa 1 üretir. Tek katmanlı ağlar bunu çözemez — gizli katman şart! Bu yüzden derin öğrenmenin temel örneğidir.",
    inputSize: 2,
    outputSize: 2,
    outputType: "classification",
    defaultLayers: [{ size: 2 }, { size: 4, activation: "sigmoid" }, { size: 2, activation: "sigmoid" }],
    defaultLearningRate: 0.5,
    data: xorData,
    axisLabels: { x: "Giriş A", y: "Giriş B" },
    classColors: ["#3b82f6", "#ef4444"],
    classNames: ["XOR=0", "XOR=1"],
  },
  {
    id: "linear",
    name: "Çizgi Sınırı",
    emoji: "📏",
    description: "Düz karar sınırı öğren",
    explanation:
      "Noktalar iki gruba ayrılır ve aradaki sınır yaklaşık düz bir çizgidir. Bu görev, lineer ayrılabilir veriyi ve bias'ın çizgiyi nasıl kaydırdığını anlamak için idealdir.",
    inputSize: 2,
    outputSize: 2,
    outputType: "classification",
    defaultLayers: [
      { size: 2 },
      { size: 3, activation: "sigmoid" },
      { size: 2, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.22,
    data: linearData,
    axisLabels: { x: "x", y: "y" },
    classColors: ["#14b8a6", "#f97316"],
    classNames: ["Alt", "Üst"],
  },
  {
    id: "checker",
    name: "Dama Tahtası",
    emoji: "▦",
    description: "XOR'un alan versiyonu",
    explanation:
      "Aynı sınıflar çapraz köşelerde durur. Düz çizgi yetmez; ağın birden fazla nöronla alanı parçalara bölmesi gerekir.",
    inputSize: 2,
    outputSize: 2,
    outputType: "classification",
    defaultLayers: [
      { size: 2 },
      { size: 6, activation: "relu" },
      { size: 4, activation: "relu" },
      { size: 2, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.28,
    data: checkerData,
    axisLabels: { x: "x", y: "y" },
    classColors: ["#6366f1", "#f43f5e"],
    classNames: ["A", "B"],
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
    id: "moons",
    name: "Ay Hilalleri",
    emoji: "🌙",
    description: "İki iç içe eğri sınıf",
    explanation:
      "İki sınıf ay hilali gibi birbirine dolanır. Modelin tek bir çizgi değil, bükülen bir karar sınırı öğrenmesi gerekir.",
    inputSize: 2,
    outputSize: 2,
    outputType: "classification",
    defaultLayers: [
      { size: 2 },
      { size: 10, activation: "tanh" },
      { size: 6, activation: "tanh" },
      { size: 2, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.18,
    data: moonsData,
    axisLabels: { x: "x", y: "y" },
    classColors: ["#0ea5e9", "#f97316"],
    classNames: ["Ay A", "Ay B"],
  },
  {
    id: "blobs",
    name: "Çok Sınıflı Kümeler",
    emoji: "🎯",
    description: "3 sınıfı ayır",
    explanation:
      "Bu görevde output katmanı üç nörondan oluşur. Her nöron bir sınıf skorudur; eğitim ilerledikçe doğru sınıf nöronunun yükselmesini izleyebilirsin.",
    inputSize: 2,
    outputSize: 3,
    outputType: "classification",
    defaultLayers: [
      { size: 2 },
      { size: 7, activation: "relu" },
      { size: 3, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.2,
    data: blobsData,
    axisLabels: { x: "x", y: "y" },
    classColors: ["#2563eb", "#f97316", "#10b981"],
    classNames: ["Mavi", "Turuncu", "Yeşil"],
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
  {
    id: "sentiment",
    name: "NLP Duygu Analizi",
    emoji: "💬",
    description: "Metin → pozitif/negatif",
    explanation:
      "Metin önce küçük bir sözlükle sayısal vektöre çevrilir. Her kelime bir input nöronu olur; ağ bu kelime sinyallerinden pozitif/negatif sınıfını öğrenmeye çalışır.",
    inputSize: NLP_VOCABULARY.length,
    outputSize: NLP_CLASS_NAMES.length,
    outputType: "classification",
    defaultLayers: [
      { size: NLP_VOCABULARY.length },
      { size: 8, activation: "relu" },
      { size: 6, activation: "relu" },
      { size: NLP_CLASS_NAMES.length, activation: "sigmoid" },
    ],
    defaultLearningRate: 0.22,
    data: sentimentData,
    classColors: ["#10b981", "#ef4444"],
    classNames: NLP_CLASS_NAMES,
  },
];

export function getTask(id: TaskId): Task {
  return TASKS.find((t) => t.id === id) ?? TASKS[0];
}
