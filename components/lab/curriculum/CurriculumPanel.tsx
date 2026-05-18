"use client";

import { CheckCircle2, ChevronDown, LockKeyhole, PlayCircle } from "lucide-react";
import {
  CURRICULUM,
  curriculumPhaseSummary,
  isCurriculumTaskUnlocked,
  tasksForPhase,
} from "@/lib/ml/curriculum";
import type { CurriculumPhase, CurriculumProgress, CurriculumTask } from "@/lib/ml/types";
import { CompactPanel } from "@/components/lab/ui/Workbench";

const phaseLabels: Record<CurriculumPhase, string> = {
  1: "Phase 1 · Traditional ML",
  2: "Phase 2 · Deep Learning",
  3: "Phase 3 · Reinforcement",
};

export function CurriculumPanel({
  selectedTaskId,
  progress,
  onSelectTask,
  compact = false,
}: {
  selectedTaskId: string;
  progress: CurriculumProgress;
  onSelectTask: (task: CurriculumTask) => void;
  compact?: boolean;
}) {
  const summary = curriculumPhaseSummary(progress);

  return (
    <CompactPanel
      title="Curriculum"
      icon={<PlayCircle className="h-4 w-4" />}
      className="h-full overflow-hidden"
      bodyClassName="h-[calc(100%-41px)] space-y-2 overflow-y-auto"
    >
      {summary.map(({ phase, completed, total }) => {
        const phaseUnlocked = progress.unlockedPhases.includes(phase);
        const phaseTasks = tasksForPhase(phase);
        return (
          <details key={phase} className="group rounded-md border border-[#dbe3ee] bg-[#fbfdff]" open={phaseUnlocked}>
            <summary className="flex cursor-pointer list-none items-center justify-between gap-3 px-3 py-2">
              <span className="min-w-0">
                <span className="block truncate text-xs font-semibold text-[#18202f]">{phaseLabels[phase]}</span>
                <span className="mt-0.5 block text-[10px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                  {completed}/{total} complete
                </span>
              </span>
              <span className="flex shrink-0 items-center gap-2">
                {!phaseUnlocked && <LockKeyhole className="h-3.5 w-3.5 text-[#94a3b8]" />}
                <ChevronDown className="h-4 w-4 text-[#64748b] transition group-open:rotate-180" />
              </span>
            </summary>
            <div className="space-y-1.5 border-t border-[#edf2f7] p-2">
              {phaseTasks.map((task) => {
                const unlocked = isCurriculumTaskUnlocked(task, progress);
                const completedTask = progress.completedTaskIds.includes(task.id);
                const current = task.id === selectedTaskId;
                return (
                  <button
                    key={task.id}
                    type="button"
                    className={`grid w-full grid-cols-[18px_1fr] gap-2 rounded-md border p-2.5 text-left transition ${
                      current
                        ? "border-[#2563eb] bg-[#eef4ff]"
                        : unlocked
                          ? "border-[#dbe3ee] bg-white hover:border-[#2563eb]"
                          : "border-[#e2e8f0] bg-[#f8fafc] opacity-70"
                    }`}
                    disabled={!unlocked}
                    onClick={() => onSelectTask(task)}
                  >
                    <span className="mt-0.5">
                      {completedTask ? (
                        <CheckCircle2 className="h-4 w-4 text-[#059669]" />
                      ) : unlocked ? (
                        <PlayCircle className="h-4 w-4 text-[#2563eb]" />
                      ) : (
                        <LockKeyhole className="h-4 w-4 text-[#94a3b8]" />
                      )}
                    </span>
                    <span className="min-w-0">
                      <span className="block truncate text-xs font-semibold text-[#18202f]">{task.title}</span>
                      {!compact && (
                        <span className="mt-1 line-clamp-2 block text-[11px] leading-4 text-[#526070]">
                          {task.description}
                        </span>
                      )}
                    </span>
                  </button>
                );
              })}
            </div>
          </details>
        );
      })}

      <div className="grid grid-cols-3 gap-1.5 pt-1">
        {summary.map(({ phase, completed, total }) => (
          <div key={`bar-${phase}`} className="min-w-0">
            <div className="mb-1 text-[10px] font-bold text-[#64748b]">P{phase}</div>
            <div className="h-1.5 overflow-hidden rounded-full bg-[#e2e8f0]">
              <div
                className="h-full rounded-full bg-[#2563eb]"
                style={{ width: `${Math.round((completed / Math.max(1, total)) * 100)}%` }}
              />
            </div>
          </div>
        ))}
      </div>
    </CompactPanel>
  );
}

export function isCurriculumTaskId(value: string) {
  return CURRICULUM.some((task) => task.id === value);
}
