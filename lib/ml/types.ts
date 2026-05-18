import type { DataPoint, LayerConfig } from "./network";
import type { TaskId } from "./tasks";

export type CodeLabTrack = "core-ml" | "deep-learning" | "reinforcement";
export type CodeLabDifficulty = "kolay" | "orta" | "zor";

export interface AiCodeLabChallenge {
  id: string;
  title: string;
  track: CodeLabTrack;
  difficulty: CodeLabDifficulty;
  taskId?: TaskId;
  libraries: string[];
  concepts: string[];
  summary: string;
  prompt: string;
  starterCode: string;
  applyToNetwork: boolean;
}

export type CurriculumPhase = 1 | 2 | 3;

export interface CurriculumProgress {
  completedTaskIds: string[];
  unlockedPhases: CurriculumPhase[];
}

export interface CurriculumRequirements {
  modelType?: ModelMetaType;
  inputSize?: number;
  hiddenLayers?: number;
  hiddenSize?: number;
  outputSize?: number;
  featureShape?: number[];
  targetShape?: number[];
  scoreThreshold?: number;
  accuracyThreshold?: number;
  lossThreshold?: number;
  minEpochs?: number;
  layerCount?: number;
  activations?: string[];
  clusters?: number;
  rewardThreshold?: number;
  episodes?: number;
  outputShape?: number[];
}

export interface VerifyResult {
  passed: boolean;
  feedback: string[];
}

export interface CurriculumTask {
  id: string;
  phase: CurriculumPhase;
  title: string;
  description: string;
  requirements: CurriculumRequirements;
  starterCode: string;
  verify: (response: PythonLabRunResponse) => VerifyResult;
}

export interface LiveCodeCheck {
  id: string;
  label: string;
  passed: boolean;
  detail: string;
}

export interface LiveCodeAnalysis {
  modelMeta?: ModelMeta;
  checks: LiveCodeCheck[];
  passed: boolean;
  feedback: string[];
  outputPreview?: string;
}

export type ModelMetaType = "mlp" | "cnn" | "lstm" | "sklearn" | "dqn";

export type ModelMetaLayerKind =
  | "input"
  | "hidden"
  | "output"
  | "conv"
  | "pool"
  | "lstm";

export interface ModelMetaLayer {
  kind: ModelMetaLayerKind;
  size: number;
  activation?: string;
}

export interface ModelMeta {
  type: ModelMetaType;
  layers: ModelMetaLayer[];
  weights?: number[][][];
  episodeRewards?: number[];
  lossHistory?: number[];
}

export interface PythonLabMetric {
  label: string;
  value: string | number;
}

export interface PythonLabResult {
  [key: string]: unknown;
  title?: string;
  classNames?: string[];
  layers?: LayerConfig[];
  weights?: number[][][];
  biases?: number[][];
  dataset?: DataPoint[];
  losses?: number[];
  loss?: number;
  val_loss?: number;
  r2_score?: number;
  score?: number;
  accuracy?: number;
  epochs?: number;
  notes?: string[];
  metrics?: PythonLabMetric[];
  modelMeta?: ModelMeta;
  inertia?: number;
  n_clusters?: number;
  qTable?: number[][];
  q_table?: number[][];
  policy?: string[];
  trajectory?: number[][];
  avg_reward_last10?: number;
  epsilon_final?: number;
}

export interface PythonLabRunResponse {
  ok: boolean;
  challengeId?: string;
  stdout: string;
  stderr: string;
  durationMs: number;
  result?: PythonLabResult;
  modelMeta?: ModelMeta;
  passed: boolean;
  feedback: string[];
  error?: string;
}
