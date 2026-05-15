import type { LayerConfig } from "./network";
import type { TaskId } from "./tasks";

export interface ModelPreset {
  id: string;
  taskId: TaskId;
  name: string;
  description: string;
  learningRate: number;
  layers: LayerConfig[];
}

export const MODEL_PRESETS: ModelPreset[] = [
  {
    id: "regression-smooth",
    taskId: "regression",
    name: "Yumuşak Eğri",
    description: "Tek gizli katmanla basit ama gözle okunabilir regresyon.",
    learningRate: 0.18,
    layers: [{ size: 1 }, { size: 5, activation: "sigmoid" }, { size: 1, activation: "linear" }],
  },
  {
    id: "regression-deeper",
    taskId: "regression",
    name: "Derin Regresyon",
    description: "İki gizli katmanla eğrinin nasıl büküldüğünü göstermek için.",
    learningRate: 0.12,
    layers: [
      { size: 1 },
      { size: 6, activation: "tanh" },
      { size: 4, activation: "tanh" },
      { size: 1, activation: "linear" },
    ],
  },
  {
    id: "xor-classic",
    taskId: "xor",
    name: "XOR Klasik",
    description: "Doğrusal olmayan karar için küçük sigmoid ağ.",
    learningRate: 0.5,
    layers: [{ size: 2 }, { size: 4, activation: "sigmoid" }, { size: 1, activation: "sigmoid" }],
  },
  {
    id: "circle-boundary",
    taskId: "circle",
    name: "Yuvarlak Sınır",
    description: "Dairesel sınıflandırmada sınırı adım adım izlemek için.",
    learningRate: 0.4,
    layers: [
      { size: 2 },
      { size: 6, activation: "sigmoid" },
      { size: 4, activation: "sigmoid" },
      { size: 2, activation: "sigmoid" },
    ],
  },
  {
    id: "spiral-tanh",
    taskId: "spiral",
    name: "Sarmal Tanh",
    description: "Tanh katmanlarıyla zor karar sınırlarını denemek için.",
    learningRate: 0.3,
    layers: [
      { size: 2 },
      { size: 8, activation: "tanh" },
      { size: 6, activation: "tanh" },
      { size: 2, activation: "sigmoid" },
    ],
  },
  {
    id: "digit-relu",
    taskId: "digit",
    name: "Mini Görüntü Ağı",
    description: "5x5 piksel girdilerinde ReLU etkisini görmek için.",
    learningRate: 0.25,
    layers: [
      { size: 25 },
      { size: 12, activation: "relu" },
      { size: 8, activation: "relu" },
      { size: 4, activation: "sigmoid" },
    ],
  },
];

export function presetsForTask(taskId: TaskId) {
  return MODEL_PRESETS.filter((preset) => preset.taskId === taskId);
}
