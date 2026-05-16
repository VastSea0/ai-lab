"use client";

import { Check, ClipboardEdit, FileUp, RotateCcw, Table2, X } from "lucide-react";
import { useMemo, useRef, useState } from "react";
import type { DataPoint } from "@/lib/ml/network";
import { datasetTemplate, parseDatasetWithMapping, previewDatasetText } from "@/lib/ml/dataset";
import type { ColumnMapping, DatasetMetadata } from "@/lib/ml/lab-types";
import type { Task } from "@/lib/ml/tasks";

export function DatasetImport({
  task,
  onDataLoaded,
  onReset,
  onMetadata,
}: {
  task: Task;
  onDataLoaded: (data: DataPoint[]) => void;
  onReset: () => void;
  onMetadata?: (metadata: DatasetMetadata | null) => void;
}) {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [message, setMessage] = useState<string>("CSV veya JSON yükleyebilirsin.");
  const [error, setError] = useState<string | null>(null);
  const [manualOpen, setManualOpen] = useState(false);
  const [manualDraft, setManualDraft] = useState(() => ({
    taskId: task.id,
    text: datasetTemplate(task),
  }));
  const manualText = manualDraft.taskId === task.id ? manualDraft.text : datasetTemplate(task);
  const [studio, setStudio] = useState<{
    fileName: string;
    text: string;
    mapping: ColumnMapping;
  } | null>(null);

  const preview = useMemo(() => {
    if (!studio) return null;
    try {
      return previewDatasetText(studio.text);
    } catch {
      return null;
    }
  }, [studio]);

  const openStudioFromText = (text: string, sourceName: string) => {
    const previewResult = previewDatasetText(text);
    if (previewResult.headers.length === 0 || previewResult.rows.length === 0) {
      throw new Error("Önizleme için başlık ve en az bir satır gerekli.");
    }
    const defaultInputs = previewResult.headers.slice(0, task.inputSize);
    const defaultTargets = previewResult.headers.slice(task.inputSize, task.inputSize + Math.max(1, task.outputSize));
    setStudio({
      fileName: sourceName,
      text,
      mapping: {
        inputColumns: defaultInputs,
        targetColumns: defaultTargets.length > 0 ? defaultTargets : previewResult.headers.slice(-1),
        normalize: true,
        trainRatio: 0.8,
      },
    });
  };

  const openFile = async (file: File) => {
    try {
      const text = await file.text();
      openStudioFromText(text, file.name);
      setError(null);
      setMessage(`${file.name} için Dataset Studio açıldı.`);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Veri okunamadı.");
    } finally {
      if (inputRef.current) inputRef.current.value = "";
    }
  };

  const openManualStudio = () => {
    try {
      openStudioFromText(manualText, "manuel-veri");
      setError(null);
      setMessage("Manuel veri için Dataset Studio açıldı.");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Manuel veri okunamadı.");
    }
  };

  const applyStudio = () => {
    if (!studio) return;
    try {
      const result = parseDatasetWithMapping(studio.text, task, studio.mapping, studio.fileName.replace(/\W+/g, "-"));
      if (result.data.length === 0) throw new Error("Eşleşen okunabilir veri bulunamadı.");
      onDataLoaded(result.data);
      onMetadata?.(result.metadata);
      setStudio(null);
      setError(null);
      setMessage(
        `${result.data.length} örnek yüklendi. ${result.metadata.normalized ? "Normalize edildi." : "Ham değerler kullanıldı."}`
      );
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Veri uygulanamadı.");
    }
  };

  const updateColumn = (kind: "inputColumns" | "targetColumns", index: number, value: string) => {
    setStudio((previous) => {
      if (!previous) return previous;
      const nextColumns = [...previous.mapping[kind]];
      nextColumns[index] = value;
      return {
        ...previous,
        mapping: { ...previous.mapping, [kind]: nextColumns },
      };
    });
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
          className={`inline-flex h-9 items-center gap-2 rounded-md border px-3 text-xs font-semibold ${
            manualOpen
              ? "border-[#2563eb] bg-[#e8f0ff] text-[#2563eb]"
              : "border-[#cbd5e1] bg-white text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
          }`}
          onClick={() => setManualOpen((value) => !value)}
        >
          <ClipboardEdit className="h-4 w-4" />
          CSV Yaz
        </button>
        <button
          type="button"
                className="inline-flex h-9 items-center gap-2 rounded-md border border-[#cbd5e1] bg-white px-3 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
          onClick={() => {
            onReset();
            onMetadata?.(null);
            setMessage("Görevin örnek verisi tekrar yüklendi.");
            setError(null);
          }}
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
          if (file) void openFile(file);
        }}
      />
      <div className={`mt-2 text-[11px] leading-5 ${error ? "text-[#b91c1c]" : "text-[#526070]"}`}>
        {error ?? message}
      </div>
      {manualOpen && (
        <div className="mt-3 rounded-md border border-[#dbe3ee] bg-white p-3">
          <div className="mb-2 flex items-center justify-between gap-2">
            <div>
              <div className="text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                Manuel CSV / JSON
              </div>
              <div className="mt-1 text-[11px] leading-5 text-[#526070]">
                Buraya veri yaz veya yapıştır; sonra kolonları Dataset Studio’da eşleştir.
              </div>
            </div>
            <button
              type="button"
              className="inline-flex h-8 items-center rounded-md border border-[#cbd5e1] bg-white px-2 text-[11px] font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
              onClick={() => setManualDraft({ taskId: task.id, text: datasetTemplate(task) })}
            >
              Şablon
            </button>
          </div>
          <textarea
            className="min-h-32 w-full resize-y rounded-md border border-[#cbd5e1] bg-[#fbfdff] p-2 font-mono text-xs leading-5 text-[#334155] outline-none focus:border-[#2563eb]"
            value={manualText}
            onChange={(event) => setManualDraft({ taskId: task.id, text: event.target.value })}
            spellCheck={false}
            aria-label="Manuel CSV veya JSON verisi"
          />
          <div className="mt-2 flex items-center justify-between gap-2">
            <div className="text-[11px] text-[#64748b]">
              CSV başlıklı olabilir: <span className="font-mono">x,target</span>
            </div>
            <button
              type="button"
              className="inline-flex h-9 items-center gap-2 rounded-md bg-[#2563eb] px-3 text-xs font-semibold text-white hover:bg-[#1d4ed8]"
              onClick={openManualStudio}
            >
              <Table2 className="h-4 w-4" />
              Önizle ve Eşle
            </button>
          </div>
        </div>
      )}
      <details className="mt-2">
        <summary className="cursor-pointer text-[11px] font-semibold text-[#2563eb]">
          Beklenen format
        </summary>
        <pre className="mt-2 max-h-24 overflow-auto rounded-md bg-white p-2 text-[10px] text-[#334155]">
          {datasetTemplate(task)}
        </pre>
      </details>

      {studio && preview && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-[#0f172a]/40 p-6">
          <div className="flex max-h-[86vh] w-full max-w-5xl flex-col overflow-hidden rounded-lg border border-[#cbd5e1] bg-white shadow-2xl">
            <div className="flex items-center justify-between border-b border-[#e2e8f0] px-5 py-4">
              <div className="min-w-0">
                <div className="flex items-center gap-2 text-base font-semibold">
                  <Table2 className="h-5 w-5 text-[#2563eb]" />
                  Dataset Studio
                </div>
                <div className="mt-1 truncate text-xs text-[#64748b]">
                  {studio.fileName} · {preview.headers.length} kolon · {preview.rows.length} satır önizleme
                </div>
              </div>
              <button
                type="button"
                className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-[#cbd5e1] text-[#334155] hover:border-[#ef4444] hover:text-[#b91c1c]"
                onClick={() => setStudio(null)}
                aria-label="Dataset Studio kapat"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="grid min-h-0 flex-1 grid-cols-[300px_minmax(0,1fr)] gap-px bg-[#e2e8f0]">
              <div className="overflow-y-auto bg-[#fbfdff] p-4">
                <div className="text-xs font-semibold uppercase tracking-[0.08em] text-[#64748b]">
                  Kolon Eşleme
                </div>
                <div className="mt-3 space-y-3">
                  {Array.from({ length: task.inputSize }, (_, index) => (
                    <ColumnSelect
                      key={`input-${index}`}
                      label={`Input x${index + 1}`}
                      columns={preview.headers}
                      value={studio.mapping.inputColumns[index] ?? ""}
                      onChange={(value) => updateColumn("inputColumns", index, value)}
                    />
                  ))}
                  {Array.from({ length: task.outputSize === 1 ? 1 : task.outputSize }, (_, index) => (
                    <ColumnSelect
                      key={`target-${index}`}
                      label={task.outputSize === 1 ? "Hedef y" : `Target ${index + 1}`}
                      columns={preview.headers}
                      value={studio.mapping.targetColumns[index] ?? ""}
                      onChange={(value) => updateColumn("targetColumns", index, value)}
                    />
                  ))}
                </div>

                <label className="mt-4 flex items-center gap-2 text-xs font-medium text-[#334155]">
                  <input
                    type="checkbox"
                    checked={studio.mapping.normalize}
                    onChange={(event) =>
                      setStudio((previous) =>
                        previous
                          ? {
                              ...previous,
                              mapping: { ...previous.mapping, normalize: event.target.checked },
                            }
                          : previous
                      )
                    }
                  />
                  Sayısal kolonları 0-1 aralığına normalize et
                </label>
                <label className="mt-4 block">
                  <span className="text-xs font-medium text-[#334155]">
                    Train oranı: {Math.round(studio.mapping.trainRatio * 100)}%
                  </span>
                  <input
                    className="mt-2 w-full accent-[#2563eb]"
                    type="range"
                    min={0.5}
                    max={1}
                    step={0.05}
                    value={studio.mapping.trainRatio}
                    onChange={(event) =>
                      setStudio((previous) =>
                        previous
                          ? {
                              ...previous,
                              mapping: { ...previous.mapping, trainRatio: Number(event.target.value) },
                            }
                          : previous
                      )
                    }
                  />
                </label>
                <div className="mt-4 rounded-md border border-[#dbe3ee] bg-white p-3 text-xs leading-5 text-[#526070]">
                  Bu sürümde split bilgisi metadata olarak tutulur; eğitim tüm yüklenen örneklerle devam eder.
                </div>
              </div>

              <div className="min-w-0 overflow-auto bg-white p-4">
                <table className="min-w-full border-separate border-spacing-0 text-xs">
                  <thead>
                    <tr>
                      {preview.headers.map((header) => (
                        <th key={header} className="sticky top-0 border-b border-[#dbe3ee] bg-white px-3 py-2 text-left font-semibold text-[#334155]">
                          {header}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {preview.rows.map((row, rowIndex) => (
                      <tr key={rowIndex}>
                        {preview.headers.map((header, cellIndex) => (
                          <td key={`${rowIndex}-${header}`} className="border-b border-[#edf2f7] px-3 py-2 text-[#526070]">
                            {row[cellIndex] ?? ""}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="flex items-center justify-between border-t border-[#e2e8f0] px-5 py-4">
              <div className="text-xs text-[#64748b]">
                Input: {studio.mapping.inputColumns.join(", ")} · Target: {studio.mapping.targetColumns.join(", ")}
              </div>
              <button
                type="button"
                className="inline-flex h-9 items-center gap-2 rounded-md bg-[#2563eb] px-4 text-sm font-semibold text-white hover:bg-[#1d4ed8]"
                onClick={applyStudio}
              >
                <Check className="h-4 w-4" />
                Veriyi Uygula
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function ColumnSelect({
  label,
  columns,
  value,
  onChange,
}: {
  label: string;
  columns: string[];
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="block">
      <span className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
        {label}
      </span>
      <select
        className="mt-1 h-9 w-full rounded-md border border-[#cbd5e1] bg-white px-2 text-xs"
        value={value}
        onChange={(event) => onChange(event.target.value)}
      >
        {columns.map((column) => (
          <option key={column} value={column}>
            {column}
          </option>
        ))}
      </select>
    </label>
  );
}
