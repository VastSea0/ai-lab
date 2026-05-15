"use client";

import { FileUp, RotateCcw } from "lucide-react";
import { useRef, useState } from "react";
import type { DataPoint } from "@/lib/ml/network";
import { datasetTemplate, parseDatasetText } from "@/lib/ml/dataset";
import type { Task } from "@/lib/ml/tasks";

export function DatasetImport({
  task,
  onDataLoaded,
  onReset,
}: {
  task: Task;
  onDataLoaded: (data: DataPoint[]) => void;
  onReset: () => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [message, setMessage] = useState<string>("CSV veya JSON yükleyebilirsin.");
  const [error, setError] = useState<string | null>(null);

  const handleFile = async (file: File) => {
    try {
      const text = await file.text();
      const parsed = parseDatasetText(text, task);
      if (parsed.length === 0) {
        throw new Error("Dosyada okunabilir veri bulunamadı.");
      }
      onDataLoaded(parsed);
      setError(null);
      setMessage(`${parsed.length} örnek yüklendi: ${file.name}`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Veri okunamadı.");
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  return (
    <div className="mt-3 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
      <div className="flex items-center justify-between gap-2">
        <button
          type="button"
          className="inline-flex h-9 items-center gap-2 rounded-md border border-[#cbd5e1] bg-white px-3 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
          onClick={() => inputRef.current?.click()}
        >
          <FileUp className="h-4 w-4" />
          Veri Yükle
        </button>
        <button
          type="button"
          className="inline-flex h-9 items-center gap-2 rounded-md border border-[#cbd5e1] bg-white px-3 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
          onClick={onReset}
        >
          <RotateCcw className="h-4 w-4" />
          Örnek Veri
        </button>
      </div>
      <input
        ref={inputRef}
        className="hidden"
        type="file"
        accept=".csv,.json,.txt,text/csv,application/json"
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) void handleFile(file);
        }}
      />
      <div className={`mt-2 text-[11px] leading-5 ${error ? "text-[#b91c1c]" : "text-[#526070]"}`}>
        {error ?? message}
      </div>
      <details className="mt-2">
        <summary className="cursor-pointer text-[11px] font-semibold text-[#2563eb]">
          Beklenen format
        </summary>
        <pre className="mt-2 max-h-24 overflow-auto rounded-md bg-white p-2 text-[10px] text-[#334155]">
          {datasetTemplate(task)}
        </pre>
      </details>
    </div>
  );
}
