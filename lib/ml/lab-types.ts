import type { Selection } from "./network";

export type ConceptMode = "beginner" | "math" | "engineer";

export type VisualizationMode = "weights" | "gradients" | "corrections";

export interface PlaybackState {
  epochIndex: number;
  sampleIndex: number;
  stepIndex: number;
  playing: boolean;
  speedMs: number;
}

export interface DatasetMetadata {
  name: string;
  rowCount: number;
  inputColumns: string[];
  targetColumns: string[];
  normalized: boolean;
  trainRatio: number;
  rejectedRows: string[];
}

export interface ColumnMapping {
  inputColumns: string[];
  targetColumns: string[];
  labelColumn?: string;
  normalize: boolean;
  trainRatio: number;
}

export interface LessonStep {
  id: string;
  title: string;
  text: string;
  target?: Selection;
}

export interface LessonProgress {
  completedStepIds: string[];
  activeStepId: string;
}
