"use client";

import { useCallback, useState } from "react";
import type { ConceptId } from "@/lib/ml/concepts";

export function useConceptDrawer(initialConcept: ConceptId = "activation") {
  const [openConceptId, setOpenConceptId] = useState<ConceptId | null>(null);

  const openConcept = useCallback((conceptId: ConceptId = initialConcept) => {
    setOpenConceptId(conceptId);
  }, [initialConcept]);

  const closeConcept = useCallback(() => {
    setOpenConceptId(null);
  }, []);

  return {
    openConceptId,
    openConcept,
    closeConcept,
  };
}
