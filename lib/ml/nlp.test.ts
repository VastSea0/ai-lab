import { describe, expect, it } from "vitest";
import {
  DEFAULT_NLP_CONFIG,
  buildNlpVectorizer,
  parseTextDataset,
  targetForLabels,
  vectorizeExamples,
  vectorizeText,
} from "./nlp";

describe("real NLP vectorizer", () => {
  it("builds a dynamic vocabulary from labeled text", () => {
    const examples = parseTextDataset(
      "text,label\nharika ürün çok iyi,Pozitif\nberbat kötü deneyim,Negatif\nharika destek başarılı,Pozitif"
    );
    const model = buildNlpVectorizer(examples, {
      ...DEFAULT_NLP_CONFIG,
      maxFeatures: 10,
      ngramMax: 2,
    });

    expect(examples).toHaveLength(3);
    expect(model.labels).toEqual(["Pozitif", "Negatif"]);
    expect(model.vocabulary.length).toBeGreaterThan(0);
    expect(model.vocabulary.length).toBeLessThanOrEqual(10);
  });

  it("vectorizes text and examples with matching dimensions", () => {
    const examples = parseTextDataset("text,label\niyi güzel,Pozitif\nkötü berbat,Negatif");
    const model = buildNlpVectorizer(examples, DEFAULT_NLP_CONFIG);
    const vector = vectorizeText("iyi güzel", model);
    const data = vectorizeExamples(examples, model);

    expect(vector).toHaveLength(model.vocabulary.length);
    expect(data[0].inputs).toHaveLength(model.vocabulary.length);
    expect(data[0].targets).toHaveLength(model.labels.length);
    expect(targetForLabels("Negatif", model.labels)).toEqual([0, 1]);
  });
});
