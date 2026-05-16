"use client";

import { BookOpen, X } from "lucide-react";
import type { ConceptId } from "@/lib/ml/concepts";
import { CONCEPTS, getConcept } from "@/lib/ml/concepts";
import type { ConceptMode } from "@/lib/ml/lab-types";
import { ActivationPlayground } from "./ActivationPlayground";

export function ConceptDrawer({
  conceptId,
  conceptMode,
  onOpenConcept,
  onClose,
}: {
  conceptId: ConceptId | null;
  conceptMode: ConceptMode;
  onOpenConcept: (conceptId: ConceptId) => void;
  onClose: () => void;
}) {
  if (!conceptId) return null;
  const concept = getConcept(conceptId);

  return (
    <div className="fixed inset-0 z-[60] bg-[#0f172a]/35">
      <aside className="ml-auto flex h-full w-[420px] max-w-[calc(100vw-32px)] flex-col border-l border-[#cbd5e1] bg-white shadow-2xl">
        <div className="flex items-start justify-between gap-3 border-b border-[#e2e8f0] px-5 py-4">
          <div>
            <div className="flex items-center gap-2 text-base font-semibold text-[#18202f]">
              <BookOpen className="h-5 w-5 text-[#2563eb]" />
              {concept.title}
            </div>
            <div className="mt-1 text-xs capitalize text-[#64748b]">{concept.category}</div>
          </div>
          <button
            type="button"
            className="inline-flex h-9 w-9 items-center justify-center rounded-md border border-[#cbd5e1] text-[#334155] hover:border-[#ef4444] hover:text-[#b91c1c]"
            onClick={onClose}
            aria-label="Konsept panelini kapat"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-5 py-4">
          <p className="text-sm leading-6 text-[#334155]">{concept.summary}</p>
          <div className="mt-4 rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
              {conceptMode === "beginner" ? "Başlangıç" : conceptMode === "math" ? "Matematik" : "Mühendis"}
            </div>
            <div className="mt-2 text-sm leading-6 text-[#334155]">{concept.copy[conceptMode]}</div>
          </div>
          {concept.formula && (
            <div className="mt-3 rounded-md border border-[#dbe3ee] bg-white p-3">
              <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">Formül</div>
              <div className="mt-2 font-mono text-sm text-[#18202f]">{concept.formula}</div>
            </div>
          )}
          <div className="mt-3 rounded-md border border-[#dbe3ee] bg-white p-3">
            <div className="text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">Örnek</div>
            <div className="mt-2 text-sm leading-6 text-[#334155]">{concept.example}</div>
          </div>

          {concept.category === "activation" && (
            <div className="mt-4">
              <ActivationPlayground
                initialActivation={
                  concept.id === "linear" || concept.id === "sigmoid" || concept.id === "tanh" || concept.id === "relu"
                    ? concept.id
                    : "sigmoid"
                }
              />
            </div>
          )}

          <div className="mt-4">
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
              İlgili kavramlar
            </div>
            <div className="flex flex-wrap gap-2">
              {concept.related.map((relatedId) => {
                const related = getConcept(relatedId);
                return (
                  <button
                    key={related.id}
                    type="button"
                    className="rounded-md border border-[#cbd5e1] bg-white px-2 py-1 text-xs font-semibold text-[#334155] hover:border-[#2563eb] hover:text-[#2563eb]"
                    onClick={() => onOpenConcept(related.id)}
                  >
                    {related.title}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      </aside>
    </div>
  );
}

export function ConceptBrowser({
  onOpenConcept,
}: {
  onOpenConcept: (conceptId: ConceptId) => void;
}) {
  const categories = [
    { id: "activation", label: "Aktivasyonlar" },
    { id: "neuron", label: "Nöron Hesabı" },
    { id: "training", label: "Eğitim" },
    { id: "backprop", label: "Backprop" },
    { id: "data", label: "Veri" },
    { id: "task", label: "Görevler" },
  ] as const;

  return (
    <div className="space-y-4">
      {categories.map((category) => {
        const items = CONCEPTS.filter((concept) => concept.category === category.id);
        if (items.length === 0) return null;
        return (
          <div key={category.id}>
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-[#64748b]">
              {category.label}
            </div>
            <div className="space-y-2">
              {items.map((concept) => (
                <button
                  key={concept.id}
                  type="button"
                  className="block w-full rounded-md border border-[#dbe3ee] bg-[#fbfdff] p-3 text-left hover:border-[#2563eb]"
                  onClick={() => onOpenConcept(concept.id)}
                >
                  <span className="block text-xs font-semibold text-[#18202f]">{concept.title}</span>
                  <span className="mt-1 block text-[11px] leading-5 text-[#526070]">{concept.summary}</span>
                </button>
              ))}
            </div>
          </div>
        );
      })}
    </div>
  );
}
