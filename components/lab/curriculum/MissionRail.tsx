"use client";

import { CheckCircle2, Circle, LockKeyhole, PlayCircle } from "lucide-react";
import {
  CURRICULUM,
  curriculumPhaseSummary,
  isCurriculumTaskUnlocked,
  tasksForPhase,
} from "@/lib/ml/curriculum";
import type { CurriculumPhase, CurriculumProgress, CurriculumTask } from "@/lib/ml/types";

const phaseTitles: Record<CurriculumPhase, string> = {
  1: "Neural Network Basics",
  2: "Training Basics",
  3: "Applied Models",
};

const phaseColors: Record<CurriculumPhase, string> = {
  1: "#2563eb",
  2: "#059669",
  3: "#d97706",
};

interface MissionRailProps {
  selectedTaskId: string;
  progress: CurriculumProgress;
  onSelectTask: (task: CurriculumTask) => void;
}

export function MissionRail({ selectedTaskId, progress, onSelectTask }: MissionRailProps) {
  const summary = curriculumPhaseSummary(progress);

  return (
    <div className="flex h-full flex-col overflow-y-auto bg-[#f4f7fb] px-2 py-3">
      <div className="mb-3 px-1">
        <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-[#94a3b8]">Course Path</div>
      </div>

      {summary.map(({ phase, completed, total }) => {
        const phaseUnlocked = progress.unlockedPhases.includes(phase);
        const phaseTasks = tasksForPhase(phase);
        const color = phaseColors[phase];

        return (
          <div key={phase} className="mb-4">
            {/* Phase header */}
            <div className="mb-2 flex items-center gap-2 px-1">
              <div
                className="flex h-5 w-5 items-center justify-center rounded text-[10px] font-bold text-white"
                style={{ backgroundColor: color }}
              >
                {phase}
              </div>
              <div className="min-w-0">
                <div className="truncate text-[11px] font-bold text-[#18202f]">{phaseTitles[phase]}</div>
                <div className="text-[10px] font-semibold text-[#64748b]">
                  {completed}/{total}
                </div>
              </div>
            </div>

            {/* Mission path */}
            <div className="relative pl-1">
              {phaseTasks.map((task, index) => {
                const unlocked = phaseUnlocked && isCurriculumTaskUnlocked(task, progress);
                const completedTask = progress.completedTaskIds.includes(task.id);
                const current = task.id === selectedTaskId;
                const isLast = index === phaseTasks.length - 1;

                return (
                  <div key={task.id} className="relative">
                    {/* Connector line */}
                    {!isLast && (
                      <div
                        className="absolute left-[14px] top-[26px] w-[2px]"
                        style={{
                          height: "calc(100% - 8px)",
                          backgroundColor: completedTask ? color : "#dbe3ee",
                        }}
                      />
                    )}

                    <button
                      type="button"
                      disabled={!unlocked}
                      onClick={() => onSelectTask(task)}
                      className={`group relative mb-1 flex w-full items-start gap-2.5 rounded-lg border px-2 py-2 text-left transition ${
                        current
                          ? "border-[#2563eb]/40 bg-white shadow-sm ring-1 ring-[#2563eb]/30"
                          : unlocked
                            ? "border-[#dbe3ee] bg-white hover:border-[#2563eb]/50"
                            : "border-transparent opacity-60"
                      }`}
                    >
                      {/* Mission node circle */}
                      <div className="relative mt-0.5 shrink-0">
                        {completedTask ? (
                          <div
                            className="flex h-5 w-5 items-center justify-center rounded-full text-white"
                            style={{ backgroundColor: color }}
                          >
                            <CheckCircle2 className="h-3.5 w-3.5" />
                          </div>
                        ) : current ? (
                          <div
                            className="flex h-5 w-5 items-center justify-center rounded-full border-2 bg-white"
                            style={{ borderColor: color }}
                          >
                            <div className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                          </div>
                        ) : unlocked ? (
                          <div className="flex h-5 w-5 items-center justify-center rounded-full border-2 border-[#cbd5e1] bg-white group-hover:border-[#2563eb]/50">
                            <PlayCircle className="h-3 w-3 text-[#64748b]" />
                          </div>
                        ) : (
                          <div className="flex h-5 w-5 items-center justify-center rounded-full border-2 border-[#e2e8f0] bg-[#f8fafc]">
                            <LockKeyhole className="h-3 w-3 text-[#cbd5e1]" />
                          </div>
                        )}
                      </div>

                      {/* Mission text */}
                      <div className="min-w-0">
                        <div
                          className={`truncate text-[11px] font-semibold leading-4 ${
                            current ? "text-[#2563eb]" : completedTask ? "text-[#18202f]" : "text-[#334155]"
                          }`}
                        >
                          {task.title}
                        </div>
                        <div className="mt-0.5 line-clamp-2 text-[10px] leading-[14px] text-[#64748b]">
                          {task.description}
                        </div>
                      </div>
                    </button>
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

export function isCurriculumTaskId(value: string) {
  return CURRICULUM.some((task) => task.id === value);
}
