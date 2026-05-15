import type { DataPoint } from "./network";
import { sanitizePointValue } from "./network";
import type { Task } from "./tasks";

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

function parseNumber(value: unknown, fallback = 0) {
  const number = typeof value === "number" ? value : Number(String(value ?? "").replace(",", "."));
  return Number.isFinite(number) ? number : fallback;
}

function oneHot(index: number, size: number) {
  return Array.from({ length: size }, (_, itemIndex) => (itemIndex === index ? 1 : 0));
}

function labelIndex(value: unknown, task: Task) {
  const raw = String(value ?? "").trim();
  const numeric = Number(raw);
  if (Number.isFinite(numeric)) return Math.max(0, Math.min(task.outputSize - 1, numeric));
  const match = task.classNames?.findIndex((name) => name.toLowerCase() === raw.toLowerCase());
  return Math.max(0, match ?? 0);
}

function normalizeInputs(values: unknown[], task: Task) {
  return Array.from({ length: task.inputSize }, (_, index) =>
    sanitizePointValue(parseNumber(values[index], 0))
  );
}

function normalizeTargets(values: unknown[], task: Task) {
  if (task.outputSize === 1) {
    return [sanitizePointValue(parseNumber(values[0], 0))];
  }
  if (values.length >= task.outputSize) {
    return Array.from({ length: task.outputSize }, (_, index) =>
      sanitizePointValue(parseNumber(values[index], 0))
    );
  }
  return oneHot(labelIndex(values[0], task), task.outputSize);
}

function pointFromArray(row: unknown[], task: Task, index: number): DataPoint {
  const inputs = normalizeInputs(row.slice(0, task.inputSize), task);
  const rest = row.slice(task.inputSize);
  const targets = normalizeTargets(rest, task);
  return {
    id: `import-${index + 1}`,
    inputs,
    targets,
    label: task.outputSize === 1 ? undefined : task.classNames?.[targets.indexOf(Math.max(...targets))],
  };
}

function firstDefined(record: Record<string, unknown>, keys: string[]) {
  for (const key of keys) {
    if (record[key] !== undefined && record[key] !== "") return record[key];
  }
  return undefined;
}

function pointFromObject(record: Record<string, unknown>, task: Task, index: number): DataPoint {
  if (Array.isArray(record.inputs) && Array.isArray(record.targets)) {
    return {
      id: String(record.id ?? `import-${index + 1}`),
      inputs: normalizeInputs(record.inputs, task),
      targets: normalizeTargets(record.targets, task),
      label: record.label ? String(record.label) : undefined,
    };
  }

  const inputValues = Array.from({ length: task.inputSize }, (_, itemIndex) => {
    const oneBased = itemIndex + 1;
    return firstDefined(record, [
      `x${oneBased}`,
      `input${oneBased}`,
      `feature${oneBased}`,
      `p${itemIndex}`,
      `pixel${itemIndex}`,
      itemIndex === 0 ? "x" : "",
      itemIndex === 1 ? "y" : "",
    ]);
  });

  const targetValues =
    task.outputSize === 1
      ? [firstDefined(record, ["target", "y", "label", "class", "output"])]
      : Array.from({ length: task.outputSize }, (_, itemIndex) =>
          firstDefined(record, [`target${itemIndex}`, `target${itemIndex + 1}`, `y${itemIndex + 1}`])
        );

  const hasExplicitTargets = targetValues.some((value) => value !== undefined);
  const targets = hasExplicitTargets
    ? normalizeTargets(targetValues, task)
    : normalizeTargets([firstDefined(record, ["label", "class", "target", "y"])], task);

  return {
    id: String(record.id ?? `import-${index + 1}`),
    inputs: normalizeInputs(inputValues, task),
    targets,
    label: record.label ? String(record.label) : record.class ? String(record.class) : undefined,
  };
}

function parseCsv(text: string, task: Task) {
  const lines = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  if (lines.length === 0) return [];

  const first = splitCsvLine(lines[0]);
  const hasHeader = first.some((cell) => /[a-zA-Z_]/.test(cell));
  const rows = hasHeader ? lines.slice(1) : lines;
  const headers = hasHeader ? first : [];

  return rows.map((line, index) => {
    const cells = splitCsvLine(line);
    if (!hasHeader) return pointFromArray(cells, task, index);
    const record = Object.fromEntries(headers.map((header, cellIndex) => [header, cells[cellIndex]]));
    return pointFromObject(record, task, index);
  });
}

export function parseDatasetText(text: string, task: Task): DataPoint[] {
  const trimmed = text.trim();
  if (!trimmed) return [];

  if (trimmed.startsWith("[") || trimmed.startsWith("{")) {
    const parsed = JSON.parse(trimmed) as unknown;
    const rows = Array.isArray(parsed)
      ? parsed
      : Array.isArray((parsed as { data?: unknown[] }).data)
        ? (parsed as { data: unknown[] }).data
        : [];
    return rows.map((row, index) =>
      Array.isArray(row)
        ? pointFromArray(row, task, index)
        : pointFromObject(row as Record<string, unknown>, task, index)
    );
  }

  return parseCsv(trimmed, task);
}

export function datasetTemplate(task: Task) {
  if (task.id === "digit") {
    const headers = [
      ...Array.from({ length: task.inputSize }, (_, index) => `p${index}`),
      "label",
    ];
    return `${headers.join(",")}\n${Array.from({ length: task.inputSize }, (_, index) =>
      index % 6 === 0 ? 1 : 0
    ).join(",")},0`;
  }

  if (task.inputSize === 1) {
    return "x,target\n0.10,0.18\n0.35,0.46\n0.80,0.88";
  }

  if (task.outputSize === 1) {
    return "x,y,target\n0,0,0\n0,1,1\n1,0,1\n1,1,0";
  }

  return `x,y,label\n0.20,0.35,${task.classNames?.[0] ?? 0}\n0.72,0.68,${
    task.classNames?.[1] ?? 1
  }`;
}
