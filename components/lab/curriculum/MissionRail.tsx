"use client";

import { BrainCircuit, Check, LockKeyhole, Play } from "lucide-react";
import {
  CURRICULUM,
  curriculumPhaseSummary,
  isCurriculumTaskUnlocked,
  tasksForPhase,
} from "@/lib/ml/curriculum";
import type { CurriculumPhase, CurriculumProgress, CurriculumTask } from "@/lib/ml/types";

const phaseTitles: Record<CurriculumPhase, string> = {
  1: "Temel Ağlar",
  2: "Eğitim",
  3: "Uygulamalı",
};

function completedTotal(progress: CurriculumProgress) {
  return {
    completed: progress.completedTaskIds.length,
    total: CURRICULUM.length,
  };
}

interface MissionRailProps {
  selectedTaskId: string;
  progress: CurriculumProgress;
  onSelectTask: (task: CurriculumTask) => void;
}

export function MissionRail({ selectedTaskId, progress, onSelectTask }: MissionRailProps) {
  const summary = curriculumPhaseSummary(progress);
  const totalProgress = completedTotal(progress);
  const progressPercent = Math.round((totalProgress.completed / Math.max(1, totalProgress.total)) * 100);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden bg-white">
      <div className="border-b border-[#e8eaf2] px-4 py-4">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-[#6366f1] text-white">
            <BrainCircuit className="h-5 w-5" />
          </div>
          <div className="min-w-0">
            <div className="truncate text-sm font-semibold text-[#1a1c2e]">AI Kod Labı</div>
            <div className="truncate text-[11px] text-[#9599b8]">Machine Learning Studio</div>
          </div>
        </div>
        <div className="mt-4 flex items-center justify-between text-[10px]">
          <span className="font-medium text-[#9599b8]">Genel ilerleme</span>
          <span className="font-semibold text-[#6366f1]">
            {totalProgress.completed} / {totalProgress.total}
          </span>
        </div>
        <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[#eef0f8]">
          <div className="h-full rounded-full bg-[#6366f1]" style={{ width: `${progressPercent}%` }} />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto py-3">
        {summary.map(({ phase, completed, total }) => {
          const phaseTasks = tasksForPhase(phase);
          const phaseUnlocked = progress.unlockedPhases.includes(phase);

          return (
            <section key={phase} className="pb-4">
              <div className="px-4 pb-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#c0c4dc]">
                Bölüm {phase} · {phaseTitles[phase]}
              </div>
              <div>
                {phaseTasks.map((task) => {
                  const unlocked = phaseUnlocked && isCurriculumTaskUnlocked(task, progress);
                  const completedTask = progress.completedTaskIds.includes(task.id);
                  const current = selectedTaskId === task.id;

                  return (
                    <button
                      key={task.id}
                      type="button"
                      disabled={!unlocked}
                      onClick={() => onSelectTask(task)}
                      className={`grid w-full grid-cols-[24px_1fr] items-start gap-3 border-r-2 px-4 py-2.5 text-left transition ${
                        current
                          ? "border-[#6366f1] bg-[#f0f1fe]"
                          : "border-transparent hover:bg-[#f8f9fe]"
                      } ${unlocked ? "" : "cursor-not-allowed opacity-55"}`}
                    >
                      <span
                        className={`mt-0.5 flex h-5 w-5 items-center justify-center rounded-full border ${
                          completedTask
                            ? "border-[#6366f1] bg-[#6366f1] text-white"
                            : current
                              ? "border-[#6366f1] bg-[#eef0fe] text-[#6366f1]"
                              : unlocked
                                ? "border-[#d7daf0] bg-white text-[#9599b8]"
                                : "border-[#e2e4ed] bg-[#f3f4f8] text-[#c0c4dc]"
                        }`}
                      >
                        {completedTask ? (
                          <Check className="h-3 w-3" />
                        ) : current ? (
                          <Play className="h-2.5 w-2.5" />
                        ) : unlocked ? (
                          <span className="h-2 w-2 rounded-full bg-current" />
                        ) : (
                          <LockKeyhole className="h-3 w-3" />
                        )}
                      </span>
                      <span className="min-w-0">
                        <span
                          className={`block truncate text-[13px] font-semibold leading-5 ${
                            current ? "text-[#5254c8]" : "text-[#3d4069]"
                          }`}
                        >
                          {task.title}
                        </span>
                        <span className="block truncate text-[11px] leading-4 text-[#a0a4c0]">
                          {completedTask ? "Görev tamamlandı" : unlocked ? `${completed}/${total} bölüm ilerlemesi` : "Kilitli"}
                        </span>
                      </span>
                    </button>
                  );
                })}
              </div>
            </section>
          );
        })}
      </div>
    </div>
  );
}

export function isCurriculumTaskId(value: string) {
  return CURRICULUM.some((task) => task.id === value);
}
