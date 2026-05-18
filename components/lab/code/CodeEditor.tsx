"use client";

import { FileCode2, Play, RotateCcw } from "lucide-react";
import { useMemo, useRef, useState } from "react";

function buildLineNumbers(count: number) {
  return Array.from({ length: Math.max(1, count) }, (_, index) => index + 1);
}

export interface CodeEditorProps {
  value: string;
  onChange: (value: string) => void;
  onRun: () => void;
  onReset: () => void;
  running: boolean;
  statusText?: string;
  statusTone?: "neutral" | "good" | "warning";
}

export function CodeEditor({
  value,
  onChange,
  onRun,
  onReset,
  running,
  statusText,
  statusTone = "neutral",
}: CodeEditorProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const lineNumbers = useMemo(() => buildLineNumbers(value.split("\n").length), [value]);
  const statusColor =
    statusTone === "good"
      ? "text-[#16a34a]"
      : statusTone === "warning"
        ? "text-[#d97706]"
        : "text-[#6366f1]";

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Tab") {
      event.preventDefault();
      const target = event.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const nextValue = `${value.slice(0, start)}    ${value.slice(end)}`;
      onChange(nextValue);
      requestAnimationFrame(() => {
        target.selectionStart = start + 4;
        target.selectionEnd = start + 4;
      });
    }

    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "enter") {
      event.preventDefault();
      onRun();
    }
  };

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#fafbff]">
      <div className="flex h-11 shrink-0 items-center justify-between border-b border-[#e8eaf2] bg-white px-4">
        <div className="flex h-full items-end gap-1">
          <div className="inline-flex h-10 items-center gap-2 border-b-2 border-[#6366f1] bg-[#f0f1fe] px-4 text-sm font-semibold text-[#5254c8]">
            <FileCode2 className="h-4 w-4" />
            main.py
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="inline-flex h-8 items-center gap-1.5 rounded-md border border-[#e2e4ed] bg-white px-3 text-xs font-semibold text-[#6b6f90] transition hover:border-[#c0c4dc] hover:text-[#3d4069]"
            onClick={onReset}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Sıfırla
          </button>
          <button
            type="button"
            className="inline-flex h-8 items-center gap-1.5 rounded-md bg-[#6366f1] px-4 text-xs font-semibold text-white shadow-sm transition hover:bg-[#5558e8] disabled:cursor-wait disabled:opacity-60"
            disabled={running}
            onClick={onRun}
          >
            <Play className="h-3.5 w-3.5" />
            {running ? "Çalışıyor" : "Çalıştır"}
          </button>
        </div>
      </div>

      <div className="grid min-h-0 flex-1 grid-cols-[46px_minmax(0,1fr)] overflow-hidden">
        <div className="overflow-hidden border-r border-[#eef0f8] bg-[#fafbff] py-4 text-right font-mono text-[12px] leading-[22px] text-[#cdd0e8]">
          <div style={{ transform: `translateY(-${scrollTop}px)` }}>
            {lineNumbers.map((line) => (
              <div key={line} className="h-[22px] pr-3">
                {line}
              </div>
            ))}
          </div>
        </div>
        <textarea
          ref={textareaRef}
          className="h-full w-full resize-none overflow-auto border-0 bg-[#fafbff] px-5 py-4 font-mono text-[13px] leading-[22px] text-[#3d4069] caret-[#6366f1] outline-none selection:bg-[#dedeff]"
          style={{ tabSize: 4 }}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          onScroll={(event) => setScrollTop(event.currentTarget.scrollTop)}
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
          autoCorrect="off"
          aria-label="Python code editor"
        />
      </div>

      <div className="flex h-7 shrink-0 items-center justify-between border-t border-[#e2e4ed] bg-[#f0f1fe] px-4 text-[10px]">
        <div className="flex items-center gap-3 text-[#9599b8]">
          <span className="font-semibold text-[#16a34a]">Çalışıyor</span>
          <span>Python 3.11</span>
          <span>PyTorch</span>
          <span>{lineNumbers.length} satır</span>
        </div>
        <div className={`truncate font-semibold ${statusColor}`}>{statusText ?? "model bekleniyor"}</div>
      </div>
    </div>
  );
}
