"use client";

import { FileCode2, Play, RotateCcw } from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Simple regex-based Python syntax highlighter
const PYTHON_KEYWORDS =
  /\b(import|from|as|def|class|return|if|elif|else|for|while|try|except|finally|with|yield|lambda|pass|break|continue|raise|assert|del|global|nonlocal|in|is|not|and|or|None|True|False|self|super|print|range|len|enumerate|zip|map|filter|sum|min|max|abs|round|int|float|str|list|dict|tuple|set|open)\b/g;
const PYTHON_STRINGS = /("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')/g;
const PYTHON_COMMENTS = /(#.*$)/gm;
const PYTHON_NUMBERS = /\b\d+\.?\d*\b/g;
const PYTHON_DECORATORS = /(@\w+)/g;
const PYTHON_TYPES = /\b(nn\.(?:Module|Linear|ReLU|Sigmoid|Tanh|Softmax|Conv2d|LSTM|Sequential|MSELoss|BCELoss|CrossEntropyLoss|Adam|SGD)|torch\.(?:Tensor|tensor|randn|zeros|ones|manual_seed|no_grad|relu|sigmoid|tanh|softmax|mm|cat|stack|unsqueeze|squeeze|flatten|reshape|max|argmax|mean|sum|abs|sqrt|exp|log|clamp|round|float|int|long))\b/g;

function escapeHtml(str: string) {
  return str
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function highlightPython(code: string): string {
  let html = escapeHtml(code);

  // Comments first (so they override other coloring inside them)
  html = html.replace(PYTHON_COMMENTS, '<span class="editor-comment">$1</span>');

  // We need to protect already-colored spans. A simpler approach:
  // split by comment spans, highlight inside non-comment parts, then rejoin.
  const parts = html.split(/(<span class="editor-comment">.*?<\/span>)/g);
  const highlighted = parts.map((part, index) => {
    if (index % 2 === 1) return part; // keep comment spans as-is
    return part
      .replace(PYTHON_STRINGS, '<span class="editor-string">$1</span>')
      .replace(PYTHON_TYPES, '<span class="editor-type">$1</span>')
      .replace(PYTHON_KEYWORDS, '<span class="editor-keyword">$1</span>')
      .replace(PYTHON_NUMBERS, '<span class="editor-number">$1</span>')
      .replace(PYTHON_DECORATORS, '<span class="editor-decorator">$1</span>');
  });

  return highlighted.join("");
}

function buildLineNumbers(count: number) {
  return Array.from({ length: Math.max(1, count) }, (_, i) => i + 1);
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
  const preRef = useRef<HTMLPreElement>(null);
  const lineNumbersRef = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const lineCount = useMemo(() => value.split("\n").length, [value]);
  const lineNumbers = useMemo(() => buildLineNumbers(lineCount), [lineCount]);

  const syncScroll = useCallback(() => {
    const top = textareaRef.current?.scrollTop ?? 0;
    setScrollTop(top);
    if (preRef.current) preRef.current.scrollTop = top;
    if (lineNumbersRef.current) lineNumbersRef.current.scrollTop = top;
  }, []);

  useEffect(() => {
    syncScroll();
  }, [value, syncScroll]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Tab") {
      e.preventDefault();
      const target = e.currentTarget;
      const start = target.selectionStart;
      const end = target.selectionEnd;
      const newValue = value.substring(0, start) + "    " + value.substring(end);
      onChange(newValue);
      requestAnimationFrame(() => {
        target.selectionStart = target.selectionEnd = start + 4;
      });
    }
    if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "enter") {
      e.preventDefault();
      onRun();
    }
  };

  const highlighted = useMemo(() => highlightPython(value), [value]);

  const statusColor =
    statusTone === "good"
      ? "text-emerald-400"
      : statusTone === "warning"
        ? "text-amber-400"
        : "text-slate-400";

  return (
    <div className="flex h-full min-h-0 flex-col bg-[#0b1020]">
      {/* Tab bar */}
      <div className="flex h-9 shrink-0 items-center justify-between border-b border-[#1e293b] bg-[#111827] px-3">
        <div className="flex items-center gap-2">
          <div className="inline-flex h-7 items-center gap-2 rounded-md bg-[#0f172a] px-3 text-xs font-semibold text-[#e2e8f0]">
            <FileCode2 className="h-3.5 w-3.5 text-[#60a5fa]" />
            main.py
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            className="inline-flex h-7 items-center gap-1.5 rounded border border-[#334155] bg-[#0f172a] px-2.5 text-[11px] font-semibold text-[#cbd5e1] hover:border-[#60a5fa] hover:text-white transition"
            onClick={onReset}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Reset
          </button>
          <button
            type="button"
            className="inline-flex h-7 items-center gap-1.5 rounded bg-[#2563eb] px-3 text-[11px] font-semibold text-white hover:bg-[#1d4ed8] disabled:cursor-wait disabled:opacity-60 transition"
            disabled={running}
            onClick={onRun}
          >
            <Play className="h-3.5 w-3.5" />
            {running ? "Running…" : "Run"}
          </button>
        </div>
      </div>

      {/* Editor body */}
      <div className="grid min-h-0 flex-1 grid-cols-[44px_1fr]">
        {/* Line numbers */}
        <div
          ref={lineNumbersRef}
          className="overflow-hidden border-r border-[#1e293b] bg-[#0f172a] py-3 text-right font-mono text-[12px] leading-5 text-[#64748b] select-none"
        >
          <div style={{ transform: `translateY(-${scrollTop}px)` }}>
            {lineNumbers.map((n) => (
              <div key={n} className="h-5 pr-3">
                {n}
              </div>
            ))}
          </div>
        </div>

        {/* Code area with syntax highlight overlay */}
        <div className="relative min-h-0">
          {/* Highlighted code layer */}
          <pre
            ref={preRef}
            aria-hidden="true"
            className="absolute inset-0 m-0 overflow-hidden whitespace-pre-wrap break-words border-0 bg-transparent py-3 pl-4 pr-4 font-mono text-[12px] leading-5"
            style={{ pointerEvents: "none", userSelect: "none" }}
            dangerouslySetInnerHTML={{ __html: `${highlighted}\n` }}
          />
          {/* Input layer */}
          <textarea
            ref={textareaRef}
            className="absolute inset-0 h-full w-full resize-none overflow-auto whitespace-pre-wrap break-words border-0 bg-transparent py-3 pl-4 pr-4 font-mono text-[12px] leading-5 text-transparent caret-[#60a5fa] outline-none selection:bg-[#2563eb]/40"
            style={{ tabSize: 4 }}
            value={value}
            onChange={(e) => onChange(e.target.value)}
            onScroll={syncScroll}
            onKeyDown={handleKeyDown}
            spellCheck={false}
            autoCapitalize="off"
            autoComplete="off"
            autoCorrect="off"
            aria-label="Python code editor"
          />
        </div>
      </div>

      {/* Status bar */}
      <div className="flex h-7 shrink-0 items-center justify-between border-t border-[#1e293b] bg-[#111827] px-3 text-[11px]">
        <div className="flex items-center gap-3 text-[#94a3b8]">
          <span className="font-semibold text-[#cbd5e1]">Python</span>
          <span className="text-[#64748b]">·</span>
          <span>PyTorch</span>
          <span className="text-[#64748b]">·</span>
          <span>{lineNumbers.length} lines</span>
        </div>
        <div className={`truncate font-medium ${statusColor}`}>
          {statusText ?? "waiting for model"}
        </div>
      </div>
    </div>
  );
}
