import type { DataPoint } from "./network";

export const NLP_VOCABULARY = [
  "iyi",
  "harika",
  "güzel",
  "mutlu",
  "seviyorum",
  "başarılı",
  "kötü",
  "berbat",
  "üzgün",
  "nefret",
  "zor",
  "kızgın",
] as const;

export type NlpLabel = "Pozitif" | "Negatif";

export const NLP_CLASS_NAMES: NlpLabel[] = ["Pozitif", "Negatif"];

export function tokenizeText(text: string): string[] {
  return text
    .toLocaleLowerCase("tr")
    .replace(/[^\p{L}\p{N}\s]/gu, " ")
    .split(/\s+/)
    .filter(Boolean);
}

const STOP_WORDS = new Set([
  "ve",
  "ile",
  "bir",
  "bu",
  "şu",
  "çok",
  "daha",
  "da",
  "de",
  "için",
  "ama",
  "fakat",
  "the",
  "a",
  "an",
  "and",
  "or",
  "to",
  "of",
  "is",
  "it",
]);

export interface TextExample {
  id?: string;
  text: string;
  label: string;
}

export interface NlpVectorizerConfig {
  maxFeatures: number;
  minFrequency: number;
  ngramMax: 1 | 2 | 3;
  useTfIdf: boolean;
  removeStopWords: boolean;
}

export interface NlpVectorizerModel {
  vocabulary: string[];
  labels: string[];
  idf: number[];
  config: NlpVectorizerConfig;
  documentCount: number;
}

export const DEFAULT_NLP_CONFIG: NlpVectorizerConfig = {
  maxFeatures: 64,
  minFrequency: 1,
  ngramMax: 2,
  useTfIdf: true,
  removeStopWords: true,
};

function splitCsvLine(line: string) {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    const next = line[i + 1];
    if (char === "\"" && next === "\"") {
      current += "\"";
      i += 1;
    } else if (char === "\"") {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      cells.push(current.trim());
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current.trim());
  return cells;
}

function normalizeTokens(text: string, removeStopWords: boolean) {
  const tokens = tokenizeText(text);
  return removeStopWords ? tokens.filter((token) => !STOP_WORDS.has(token)) : tokens;
}

export function makeNgrams(tokens: string[], ngramMax: 1 | 2 | 3): string[] {
  const terms = [...tokens];
  for (let n = 2; n <= ngramMax; n += 1) {
    for (let index = 0; index <= tokens.length - n; index += 1) {
      terms.push(tokens.slice(index, index + n).join("_"));
    }
  }
  return terms;
}

export function parseTextDataset(text: string): TextExample[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed) as unknown;
    const rows = Array.isArray(parsed)
      ? parsed
      : Array.isArray((parsed as { data?: unknown[] }).data)
        ? (parsed as { data: unknown[] }).data
        : [];
    return rows.flatMap((row, index) => {
      const record = row as Record<string, unknown>;
      const textValue = record.text ?? record.sentence ?? record.input ?? record.review;
      const labelValue = record.label ?? record.class ?? record.target;
      if (!textValue || !labelValue) return [];
      return {
        id: String(record.id ?? `text-${index + 1}`),
        text: String(textValue),
        label: String(labelValue),
      };
    });
  }

  const lines = trimmed.split(/\r?\n/).filter(Boolean);
  const first = splitCsvLine(lines[0] ?? "");
  const hasHeader = first.some((cell) => /[a-zA-ZğüşöçıİĞÜŞÖÇ_]/.test(cell));
  const headers = hasHeader ? first.map((cell) => cell.toLocaleLowerCase("tr")) : ["text", "label"];
  const rows = hasHeader ? lines.slice(1) : lines;
  const textIndex = Math.max(0, headers.findIndex((header) => ["text", "sentence", "input", "review"].includes(header)));
  const labelIndex = Math.max(1, headers.findIndex((header) => ["label", "class", "target"].includes(header)));

  return rows.flatMap((line, index) => {
    const cells = splitCsvLine(line);
    const textValue = cells[textIndex];
    const labelValue = cells[labelIndex];
    if (!textValue || !labelValue) return [];
    return {
      id: `text-${index + 1}`,
      text: textValue,
      label: labelValue,
    };
  });
}

export function buildNlpVectorizer(
  examples: TextExample[],
  config: NlpVectorizerConfig = DEFAULT_NLP_CONFIG
): NlpVectorizerModel {
  const labels = Array.from(new Set(examples.map((example) => example.label.trim()).filter(Boolean)));
  const documentFrequency = new Map<string, number>();
  const totalFrequency = new Map<string, number>();

  examples.forEach((example) => {
    const terms = makeNgrams(normalizeTokens(example.text, config.removeStopWords), config.ngramMax);
    const seen = new Set<string>();
    terms.forEach((term) => {
      totalFrequency.set(term, (totalFrequency.get(term) ?? 0) + 1);
      if (!seen.has(term)) {
        documentFrequency.set(term, (documentFrequency.get(term) ?? 0) + 1);
        seen.add(term);
      }
    });
  });

  const vocabulary = Array.from(documentFrequency.entries())
    .filter(([, frequency]) => frequency >= config.minFrequency)
    .sort((a, b) => {
      const scoreB = b[1] * 10 + (totalFrequency.get(b[0]) ?? 0);
      const scoreA = a[1] * 10 + (totalFrequency.get(a[0]) ?? 0);
      return scoreB - scoreA || a[0].localeCompare(b[0], "tr");
    })
    .slice(0, config.maxFeatures)
    .map(([term]) => term);

  const documentCount = Math.max(1, examples.length);
  const idf = vocabulary.map((term) => {
    const df = documentFrequency.get(term) ?? 0;
    return Math.log((1 + documentCount) / (1 + df)) + 1;
  });

  return {
    vocabulary,
    labels: labels.length > 0 ? labels : [...NLP_CLASS_NAMES],
    idf,
    config,
    documentCount,
  };
}

export function vectorizeText(text: string, model: NlpVectorizerModel): number[] {
  const terms = makeNgrams(normalizeTokens(text, model.config.removeStopWords), model.config.ngramMax);
  const counts = new Map<string, number>();
  terms.forEach((term) => counts.set(term, (counts.get(term) ?? 0) + 1));
  const maxCount = Math.max(1, ...Array.from(counts.values()));
  return model.vocabulary.map((term, index) => {
    const tf = (counts.get(term) ?? 0) / maxCount;
    const value = model.config.useTfIdf ? tf * model.idf[index] : tf;
    return Math.min(1, value);
  });
}

export function targetForLabels(label: string, labels: string[]): number[] {
  const normalized = label.trim().toLocaleLowerCase("tr");
  const index = Math.max(
    0,
    labels.findIndex((item) => item.trim().toLocaleLowerCase("tr") === normalized)
  );
  return labels.map((_, itemIndex) => (itemIndex === index ? 1 : 0));
}

export function vectorizeExamples(
  examples: TextExample[],
  model: NlpVectorizerModel
): DataPoint[] {
  return examples.map((example, index) => ({
    id: example.id ?? `text-${index + 1}`,
    inputs: vectorizeText(example.text, model),
    targets: targetForLabels(example.label, model.labels),
    label: example.label,
  }));
}

export function encodeText(text: string): number[] {
  const tokens = new Set(tokenizeText(text));
  return NLP_VOCABULARY.map((word) => (tokens.has(word) ? 1 : 0));
}

export function nlpTarget(label: string): number[] {
  const normalized = label.toLocaleLowerCase("tr").trim();
  const index =
    normalized.includes("neg") ||
    normalized.includes("köt") ||
    normalized.includes("kotu") ||
    normalized.includes("üz") ||
    normalized.includes("uz") ||
    normalized.includes("0")
      ? 1
      : 0;
  return NLP_CLASS_NAMES.map((_, itemIndex) => (itemIndex === index ? 1 : 0));
}

export function textPoint(id: string, text: string, label: NlpLabel): DataPoint {
  return {
    id,
    inputs: encodeText(text),
    targets: nlpTarget(label),
    label,
  };
}
