import { describe, expect, it } from "vitest";
import { ACTIVATION_CONCEPT_IDS, CONCEPTS, getConcept } from "./concepts";

describe("concept registry", () => {
  it("contains complete learning copy for every concept", () => {
    expect(CONCEPTS.length).toBeGreaterThan(10);
    CONCEPTS.forEach((concept) => {
      expect(concept.title).toBeTruthy();
      expect(concept.summary).toBeTruthy();
      expect(concept.example).toBeTruthy();
      expect(concept.copy.beginner).toBeTruthy();
      expect(concept.copy.math).toBeTruthy();
      expect(concept.copy.engineer).toBeTruthy();
    });
  });

  it("maps activation names to existing concepts", () => {
    Object.values(ACTIVATION_CONCEPT_IDS).forEach((conceptId) => {
      expect(getConcept(conceptId).id).toBe(conceptId);
    });
  });
});
