"use client";

import { useEffect, useMemo, useState } from "react";
import type { Selection, TrainingTrace } from "@/lib/ml/network";
import { formatNumber } from "@/lib/ml/network";
import type { Task } from "@/lib/ml/tasks";
import {
  buildCalculationSteps,
  type CalculationPhase,
  type EpochTraceRecord,
  type SampleTraceRecord,
} from "@/lib/ml/trace";

const phaseLabels: Record<CalculationPhase, string> = {
  input: "Girdi",
  forward: "Forward",
  loss: "Loss",
  backward: "Backprop",
  update: "Güncelleme",
};

function predictionName(task: Task, prediction: number[]) {
  if (task.outputType === "regression") return formatNumber(prediction[0] ?? 0, 3);
  const index = prediction.indexOf(Math.max(...prediction));
  return `${task.classNames?.[index] ?? `Sınıf ${index}`} (${formatNumber(
    prediction[index] ?? 0,
    3
  )})`;
}

function targetName(task: Task, target: number[]) {
  if (task.outputType === "regression") return formatNumber(target[0] ?? 0, 3);
  const index = target.indexOf(Math.max(...target));
  return task.classNames?.[index] ?? `Sınıf ${index}`;
}

function fallbackSample(
  trace: TrainingTrace,
  task: Task,
  learningRate: number
): SampleTraceRecord {
  return {
    sampleId: trace.sampleId,
    sampleName: trace.sampleId ?? "Seçili örnek",
    input: trace.input,
    target: trace.target,
    prediction: trace.prediction,
    predictionAfter: trace.predictionAfter,
    loss: trace.loss,
    steps: buildCalculationSteps(trace, task, learningRate),
  };
}

export function StepExplorer({
  task,
  trace,
  history,
  learningRate,
  onSelectTarget,
}: {
  task: Task;
  trace: TrainingTrace;
  history: EpochTraceRecord[];
  learningRate: number;
  onSelectTarget: (selection: Selection | null) => void;
}) {
  const [epochIndex, setEpochIndex] = useState(0);
  const [sampleIndex, setSampleIndex] = useState(0);
  const [stepIndex, setStepIndex] = useState(0);

  const boundedEpochIndex = Math.min(epochIndex, Math.max(0, history.length - 1));
  const selectedEpoch = history[boundedEpochIndex];
  const samples = useMemo(
    () => selectedEpoch?.samples ?? [fallbackSample(trace, task, learningRate)],
    [selectedEpoch, trace, task, learningRate]
  );
  const boundedSampleIndex = Math.min(sampleIndex, Math.max(0, samples.length - 1));
  const selectedSample = samples[boundedSampleIndex];
  const boundedStepIndex = Math.min(stepIndex, Math.max(0, selectedSample.steps.length - 1));
  const selectedStep = selectedSample.steps[boundedStepIndex];

  useEffect(() => {
    if (selectedStep?.target) onSelectTarget(selectedStep.target);
  }, [selectedStep, onSelectTarget]);

  return (
    <div className="mt-3 grid h-[164px] grid-cols-[132px_142px_1fr] gap-2">
      <div className="min-h-0 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-2">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Epoch
        </div>
        <select
          className="h-8 w-full rounded-md border border-[#cbd5e1] bg-white px-2 text-xs"
          value={boundedEpochIndex}
          onChange={(event) => {
            setEpochIndex(Number(event.target.value));
            setSampleIndex(0);
            setStepIndex(0);
          }}
        >
          {history.length === 0 ? (
            <option value={0}>Henüz eğitim yok</option>
          ) : (
            history.map((item, index) => (
              <option key={item.epoch} value={index}>
                Epoch {item.epoch}
              </option>
            ))
          )}
        </select>
        <div className="mt-2 space-y-1 text-[11px] text-[#526070]">
          <div>Loss önce: {formatNumber(selectedEpoch?.lossBefore ?? trace.epochLoss, 5)}</div>
          <div>Loss sonra: {formatNumber(selectedEpoch?.lossAfter ?? trace.epochLoss, 5)}</div>
          <div>η: {formatNumber(selectedEpoch?.learningRate ?? learningRate, 3)}</div>
        </div>
      </div>

      <div className="min-h-0 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-2">
        <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
          Örnek
        </div>
        <select
          className="h-8 w-full rounded-md border border-[#cbd5e1] bg-white px-2 text-xs"
          value={boundedSampleIndex}
          onChange={(event) => {
            setSampleIndex(Number(event.target.value));
            setStepIndex(0);
          }}
        >
          {samples.map((sample, index) => (
            <option key={`${sample.sampleId ?? "sample"}-${index}`} value={index}>
              {index + 1}. {sample.sampleName}
            </option>
          ))}
        </select>
        <div className="mt-2 space-y-1 text-[11px] text-[#526070]">
          <div>Hedef: {targetName(task, selectedSample.target)}</div>
          <div>Önce: {predictionName(task, selectedSample.prediction)}</div>
          <div>Sonra: {predictionName(task, selectedSample.predictionAfter)}</div>
        </div>
      </div>

      <div className="grid min-h-0 grid-cols-[168px_1fr] gap-2">
        <div className="min-h-0 overflow-y-auto rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-1">
          {selectedSample.steps.map((step, index) => (
            <button
              key={step.id}
              type="button"
              className={`mb-1 block w-full rounded-md px-2 py-1.5 text-left text-[11px] ${
                index === boundedStepIndex
                  ? "bg-[#2563eb] text-white"
                  : "bg-white text-[#334155] hover:bg-[#eef3f8]"
              }`}
              onClick={() => setStepIndex(index)}
            >
              <span className="block font-semibold">{index + 1}. {phaseLabels[step.phase]}</span>
              <span className="block truncate">{step.title}</span>
            </button>
          ))}
        </div>

        <div className="min-h-0 overflow-y-auto rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-2">
          <div className="text-xs font-semibold text-[#18202f]">{selectedStep.title}</div>
          <div className="mt-1 text-[11px] leading-5 text-[#526070]">{selectedStep.summary}</div>
          <div className="mt-2 space-y-1 font-mono text-[11px] text-[#334155]">
            {selectedStep.equations.map((equation) => (
              <div key={equation} className="break-words">
                {equation}
              </div>
            ))}
          </div>
          {selectedStep.details.length > 0 && (
            <div className="mt-2 border-t border-[#e2e8f0] pt-2">
              <div className="mb-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                Terimler
              </div>
              <div className="space-y-1 font-mono text-[10px] text-[#526070]">
                {selectedStep.details.map((detail) => (
                  <div key={detail} className="break-words">
                    {detail}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
