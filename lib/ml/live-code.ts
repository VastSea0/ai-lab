import type {
  CurriculumRequirements,
  LiveCodeAnalysis,
  LiveCodeCheck,
  ModelMeta,
  ModelMetaLayer,
} from "./types";

type LinearLayer = {
  from: number;
  to: number;
};

function stripComments(code: string) {
  return code
    .split("\n")
    .map((line) => line.replace(/#.*$/, ""))
    .join("\n");
}

function parseInteger(value: string | undefined) {
  if (!value) return undefined;
  const number = Number(value.trim());
  return Number.isInteger(number) && number > 0 ? number : undefined;
}

function parseAssignments(code: string) {
  const values = new Map<string, number>();
  for (const match of code.matchAll(/\b([A-Za-z_]\w*)\s*=\s*(-?\d+)\b/g)) {
    const value = parseInteger(match[2]);
    if (value !== undefined) values.set(match[1], value);
  }

  for (const match of code.matchAll(/\b\w+\s*=\s*[A-Za-z_]\w*\(([\s\S]*?)\)/g)) {
    const args = match[1];
    for (const pair of args.matchAll(/\b([A-Za-z_]\w*)\s*=\s*(-?\d+)\b/g)) {
      const value = parseInteger(pair[2]);
      if (value !== undefined) values.set(pair[1], value);
    }
  }

  return values;
}

function resolveSize(token: string, values: Map<string, number>) {
  const clean = token.trim();
  return parseInteger(clean) ?? values.get(clean);
}

function parseLinearLayers(code: string): LinearLayer[] {
  const clean = stripComments(code);
  const values = parseAssignments(clean);
  const layers: LinearLayer[] = [];

  for (const match of clean.matchAll(/nn\.Linear\s*\(\s*([^,\n)]+)\s*,\s*([^,\n)]+)\s*\)/g)) {
    const from = resolveSize(match[1], values);
    const to = resolveSize(match[2], values);
    if (from && to) layers.push({ from, to });
  }

  return layers;
}

function detectActivation(code: string) {
  const lower = code.toLocaleLowerCase("en-US");
  if (lower.includes("nn.relu") || lower.includes(".relu(") || lower.includes("torch.relu")) return "relu";
  if (lower.includes("nn.tanh") || lower.includes(".tanh(") || lower.includes("torch.tanh")) return "tanh";
  if (lower.includes("nn.sigmoid") || lower.includes(".sigmoid(") || lower.includes("torch.sigmoid")) return "sigmoid";
  return "linear";
}

export function parseTorchModelMeta(code: string): ModelMeta | undefined {
  const linearLayers = parseLinearLayers(code);
  if (linearLayers.length === 0) return undefined;

  const activation = detectActivation(code);
  const layers: ModelMetaLayer[] = [
    { kind: "input", size: linearLayers[0].from },
    ...linearLayers.slice(0, -1).map((layer) => ({
      kind: "hidden" as const,
      size: layer.to,
      activation,
    })),
    {
      kind: "output",
      size: linearLayers.at(-1)?.to ?? 1,
      activation: "linear",
    },
  ];

  return {
    type: "mlp",
    layers,
  };
}

function passCheck(id: string, label: string, detail: string): LiveCodeCheck {
  return { id, label, detail, passed: true };
}

function failCheck(id: string, label: string, detail: string): LiveCodeCheck {
  return { id, label, detail, passed: false };
}

export function analyzeTorchStructure(
  code: string,
  requirements: CurriculumRequirements
): LiveCodeAnalysis {
  const meta = parseTorchModelMeta(code);
  const layers = meta?.layers ?? [];
  const input = layers[0];
  const hidden = layers.filter((layer) => layer.kind === "hidden");
  const output = layers.at(-1);
  const checks: LiveCodeCheck[] = [];

  checks.push(
    meta
      ? passCheck("linear", "Use nn.Linear layers", `Found ${layers.length - 1} Linear layer(s).`)
      : failCheck("linear", "Use nn.Linear layers", "Add at least one nn.Linear(...) layer.")
  );

  if (requirements.inputSize !== undefined) {
    checks.push(
      input?.size === requirements.inputSize
        ? passCheck("input", `${requirements.inputSize} input neurons`, `Input layer has ${input.size} neurons.`)
        : failCheck(
            "input",
            `${requirements.inputSize} input neurons`,
            `Input layer is ${input?.size ?? "missing"}, need ${requirements.inputSize}.`
          )
    );
  }

  if (requirements.hiddenLayers !== undefined) {
    checks.push(
      hidden.length === requirements.hiddenLayers
        ? passCheck("hidden-count", `${requirements.hiddenLayers} hidden layers`, `Found ${hidden.length} hidden layers.`)
        : failCheck(
            "hidden-count",
            `${requirements.hiddenLayers} hidden layers`,
            `Found ${hidden.length}, need ${requirements.hiddenLayers}.`
          )
    );
  }

  if (requirements.hiddenSize !== undefined) {
    const badLayer = hidden.find((layer) => layer.size !== requirements.hiddenSize);
    checks.push(
      hidden.length > 0 && !badLayer
        ? passCheck("hidden-size", `${requirements.hiddenSize} neurons per hidden layer`, "Every hidden layer has the target size.")
        : failCheck(
            "hidden-size",
            `${requirements.hiddenSize} neurons per hidden layer`,
            badLayer
              ? `One hidden layer has ${badLayer.size}, need ${requirements.hiddenSize}.`
              : "Add hidden layers before this can pass."
          )
    );
  }

  if (requirements.outputSize !== undefined) {
    checks.push(
      output?.kind === "output" && output.size === requirements.outputSize
        ? passCheck("output", `${requirements.outputSize} output neuron`, `Output layer has ${output.size} neuron(s).`)
        : failCheck(
            "output",
            `${requirements.outputSize} output neuron`,
            `Output layer is ${output?.size ?? "missing"}, need ${requirements.outputSize}.`
          )
    );
  }

  if (requirements.activations?.length) {
    const activations = new Set(layers.map((layer) => layer.activation?.toLocaleLowerCase("en-US")));
    requirements.activations.forEach((activation) => {
      checks.push(
        activations.has(activation)
          ? passCheck(`activation-${activation}`, `Use ${activation}`, `${activation} is present in the model.`)
          : failCheck(`activation-${activation}`, `Use ${activation}`, `Add ${activation} between hidden layers.`)
      );
    });
  }

  const passed = checks.length > 0 && checks.every((check) => check.passed);
  const feedback = checks.map((check) => check.detail);
  const outputPreview =
    meta && output
      ? `output tensor shape: torch.Size([1, ${output.size}])`
      : "output tensor shape: waiting for a complete model";

  return {
    modelMeta: meta,
    checks,
    passed,
    feedback,
    outputPreview,
  };
}
