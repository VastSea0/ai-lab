import { analyzeTorchStructure } from "./live-code";
import type {
  CurriculumPhase,
  CurriculumProgress,
  CurriculumTask,
  PythonLabResult,
  PythonLabRunResponse,
  VerifyResult,
} from "./types";

export const CURRICULUM_PROGRESS_KEY = "curriculum_progress";

type Check = {
  passed: boolean;
  pass: string;
  fail: string;
};

function grade(checks: Check[]): VerifyResult {
  const failures = checks.filter((check) => !check.passed).map((check) => check.fail);
  const feedback = failures.length > 0 ? failures : checks.map((check) => check.pass);
  return {
    passed: failures.length === 0,
    feedback: feedback.length > 0 ? feedback : ["No checks were defined for this lesson."],
  };
}

function resultOf(response: PythonLabRunResponse): PythonLabResult {
  return response.result ?? {};
}

function sourceOf(response: PythonLabRunResponse) {
  const source = response.result?.sourceCode;
  return typeof source === "string" ? source : "";
}

function asNumber(value: unknown, fallback = Number.NaN) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function structuralGrade(response: PythonLabRunResponse, requirements: CurriculumTask["requirements"]) {
  const analysis = analyzeTorchStructure(sourceOf(response), requirements);
  return {
    passed: analysis.passed,
    feedback: analysis.feedback.length > 0 ? analysis.feedback : ["Write the requested PyTorch structure."],
  };
}

const STRUCTURE_STARTER = String.raw`import torch
import torch.nn as nn

class MyNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x

model = MyNeuralNetwork(input_size=3, hidden_size=16, output_size=1)
dummy_input = torch.randn(1, 3)
prediction = model(dummy_input)

print("output tensor shape:", prediction.shape)
`;

const SINGLE_LINEAR_STARTER = String.raw`import torch
import torch.nn as nn

# Mission:
# Create a single layer that accepts 3 input features and returns 1 output.
# Change only the layer shape and dummy input shape.

model = nn.Linear(1, 1)
dummy_input = torch.randn(1, 1)
prediction = model(dummy_input)

print("output tensor shape:", prediction.shape)
`;

const ONE_HIDDEN_STARTER = String.raw`import torch
import torch.nn as nn

# Mission:
# Build input 3 -> hidden 8 -> output 1.
# Use ReLU between hidden and output.

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(3, 4)      # TODO: hidden layer should have 8 neurons
        self.output = nn.Linear(4, 1)      # TODO: match this to the hidden size
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x

model = TinyNet()
dummy_input = torch.randn(1, 3)
prediction = model(dummy_input)

print("output tensor shape:", prediction.shape)
`;

const TWO_HIDDEN_STARTER = String.raw`import torch
import torch.nn as nn

# Mission:
# Upgrade TinyNet into input 3 -> hidden 8 -> hidden 8 -> output 1.
# Add a second hidden layer and use ReLU after each hidden layer.

class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = nn.Linear(3, 8)
        # TODO: add self.hidden2 = nn.Linear(8, 8)
        self.output = nn.Linear(8, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.relu(x)
        # TODO: pass x through hidden2 and ReLU
        x = self.output(x)
        return x

model = TinyNet()
dummy_input = torch.randn(1, 3)
prediction = model(dummy_input)

print("output tensor shape:", prediction.shape)
`;

const LOSS_STARTER = String.raw`import json
import torch
import torch.nn as nn

torch.manual_seed(7)

X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = 2 * X + 1

model = nn.Linear(1, 1)
criterion = nn.MSELoss()

# TODO: replace this fake prediction with the real model output
prediction = torch.zeros_like(y)
loss = criterion(prediction, y)

result = {
    "title": "First Loss",
    "loss": float(loss.detach()),
    "modelMeta": {
        "type": "mlp",
        "layers": [
            {"kind": "input", "size": 1},
            {"kind": "output", "size": 1, "activation": "linear"}
        ],
        "lossHistory": [float(loss.detach())]
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const OPTIMIZER_STEP_STARTER = String.raw`import json
import torch
import torch.nn as nn

torch.manual_seed(11)

X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = 2 * X + 1

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

before_loss = criterion(model(X), y)

# TODO:
# 1. clear gradients
# 2. send the loss backward through the model
# 3. update the parameters once

after_loss = criterion(model(X), y)

result = {
    "title": "One Optimizer Step",
    "before_loss": float(before_loss.detach()),
    "after_loss": float(after_loss.detach()),
    "loss": float(after_loss.detach()),
    "source_checks": {
        "has_backward": ".backward()" in open(__file__).read(),
        "has_step": ".step()" in open(__file__).read()
    },
    "modelMeta": {
        "type": "mlp",
        "layers": [
            {"kind": "input", "size": 1},
            {"kind": "output", "size": 1, "activation": "linear"}
        ],
        "lossHistory": [float(before_loss.detach()), float(after_loss.detach())]
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const TRAINING_LOOP_STARTER = String.raw`import json
import torch
import torch.nn as nn

torch.manual_seed(17)

X = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
y = 2 * X + 1

model = nn.Linear(1, 1)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

epochs = 5  # TODO: train for at least 50 epochs
loss_history = []

for epoch in range(epochs):
    prediction = model(X)
    loss = criterion(prediction, y)
    # TODO: add the three parameter-update lines here
    loss_history.append(float(loss.detach()))

final_loss = loss_history[-1]

result = {
    "title": "Tiny Training Loop",
    "epochs": epochs,
    "loss": final_loss,
    "losses": loss_history,
    "source_checks": {
        "has_backward": ".backward()" in open(__file__).read(),
        "has_step": ".step()" in open(__file__).read()
    },
    "modelMeta": {
        "type": "mlp",
        "layers": [
            {"kind": "input", "size": 1},
            {"kind": "output", "size": 1, "activation": "linear"}
        ],
        "lossHistory": loss_history
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const XOR_STARTER = String.raw`import json
import torch
import torch.nn as nn

torch.manual_seed(23)

X = torch.tensor([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])

# Mission:
# Replace this weak model with a small MLP:
# input 2 -> hidden 8 -> output 1, with ReLU and Sigmoid.
model = nn.Sequential(
    nn.Linear(2, 1),
    nn.Sigmoid()
)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
epochs = 100
loss_history = []

for epoch in range(epochs):
    prediction = model(X)
    loss = criterion(prediction, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == epochs - 1:
        loss_history.append(float(loss.detach()))

with torch.no_grad():
    prediction = model(X)
    final_loss = float(criterion(prediction, y).detach())
    accuracy = float(((prediction >= 0.5).float() == y).float().mean())

linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
weights = [layer.weight.detach().T.tolist() for layer in linear_layers]

result = {
    "title": "XOR MLP",
    "loss": final_loss,
    "accuracy": accuracy,
    "epochs": epochs,
    "losses": loss_history,
    "modelMeta": {
        "type": "mlp",
        "layers": [
            {"kind": "input", "size": 2},
            # TODO: add hidden layer metadata after adding the layer
            {"kind": "output", "size": 1, "activation": "sigmoid"}
        ],
        "weights": weights,
        "lossHistory": loss_history
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

export const CURRICULUM: CurriculumTask[] = [
  {
    id: "phase1-create-neural-network",
    phase: 1,
    title: "Create A Neural Network",
    description: "Build a PyTorch nn.Module with 3 input neurons, two hidden layers, ReLU, and 1 output neuron.",
    requirements: {
      modelType: "mlp",
      inputSize: 3,
      hiddenLayers: 2,
      hiddenSize: 16,
      outputSize: 1,
      activations: ["relu"],
    },
    starterCode: STRUCTURE_STARTER,
    verify(response) {
      return structuralGrade(response, {
        modelType: "mlp",
        inputSize: 3,
        hiddenLayers: 2,
        hiddenSize: 16,
        outputSize: 1,
        activations: ["relu"],
      });
    },
  },
  {
    id: "phase1-single-linear",
    phase: 1,
    title: "Single Linear Layer",
    description: "Change a tiny layer into 3 inputs -> 1 output and make the dummy tensor match it.",
    requirements: {
      modelType: "mlp",
      inputSize: 3,
      hiddenLayers: 0,
      outputSize: 1,
    },
    starterCode: SINGLE_LINEAR_STARTER,
    verify(response) {
      return structuralGrade(response, {
        modelType: "mlp",
        inputSize: 3,
        hiddenLayers: 0,
        outputSize: 1,
      });
    },
  },
  {
    id: "phase1-one-hidden-layer",
    phase: 1,
    title: "One Hidden Layer",
    description: "Build input 3 -> hidden 8 -> output 1 and use ReLU in the forward pass.",
    requirements: {
      modelType: "mlp",
      inputSize: 3,
      hiddenLayers: 1,
      hiddenSize: 8,
      outputSize: 1,
      activations: ["relu"],
    },
    starterCode: ONE_HIDDEN_STARTER,
    verify(response) {
      return structuralGrade(response, {
        modelType: "mlp",
        inputSize: 3,
        hiddenLayers: 1,
        hiddenSize: 8,
        outputSize: 1,
        activations: ["relu"],
      });
    },
  },
  {
    id: "phase1-two-hidden-layers",
    phase: 1,
    title: "Two Hidden Layers",
    description: "Add a second hidden layer so the network becomes 3 -> 8 -> 8 -> 1.",
    requirements: {
      modelType: "mlp",
      inputSize: 3,
      hiddenLayers: 2,
      hiddenSize: 8,
      outputSize: 1,
      activations: ["relu"],
    },
    starterCode: TWO_HIDDEN_STARTER,
    verify(response) {
      return structuralGrade(response, {
        modelType: "mlp",
        inputSize: 3,
        hiddenLayers: 2,
        hiddenSize: 8,
        outputSize: 1,
        activations: ["relu"],
      });
    },
  },
  {
    id: "phase2-first-loss",
    phase: 2,
    title: "First Loss",
    description: "Use the model output, not a fake tensor, to calculate MSE loss.",
    requirements: {
      modelType: "mlp",
      lossThreshold: 8,
    },
    starterCode: LOSS_STARTER,
    verify(response) {
      const result = resultOf(response);
      const loss = asNumber(result.loss);
      const source = sourceOf(response);
      return grade([
        {
          passed: /prediction\s*=\s*model\s*\(\s*X\s*\)/.test(source),
          pass: "Loss is calculated from model(X).",
          fail: "Replace the fake prediction with prediction = model(X).",
        },
        {
          passed: Number.isFinite(loss),
          pass: `Loss was reported as ${loss.toFixed(4)}.`,
          fail: "result.loss is missing or not a number.",
        },
      ]);
    },
  },
  {
    id: "phase2-one-optimizer-step",
    phase: 2,
    title: "One Optimizer Step",
    description: "Write the three core training lines: zero_grad, backward, step.",
    requirements: {
      modelType: "mlp",
    },
    starterCode: OPTIMIZER_STEP_STARTER,
    verify(response) {
      const result = resultOf(response);
      const before = asNumber(result.before_loss);
      const after = asNumber(result.after_loss);
      const source = sourceOf(response);
      return grade([
        {
          passed: /\w+\.backward\(\)/.test(source),
          pass: "backward() is present.",
          fail: "Call before_loss.backward() before stepping the optimizer.",
        },
        {
          passed: /optimizer\.step\(\)/.test(source),
          pass: "optimizer.step() is present.",
          fail: "Call optimizer.step() after backward().",
        },
        {
          passed: after < before,
          pass: `Loss decreased from ${before.toFixed(4)} to ${after.toFixed(4)}.`,
          fail: `Loss did not decrease. Before=${Number.isFinite(before) ? before.toFixed(4) : "missing"}, after=${Number.isFinite(after) ? after.toFixed(4) : "missing"}.`,
        },
      ]);
    },
  },
  {
    id: "phase2-training-loop",
    phase: 2,
    title: "Training Loop",
    description: "Turn one optimizer step into a real loop and train for at least 50 epochs.",
    requirements: {
      modelType: "mlp",
      minEpochs: 50,
      lossThreshold: 0.1,
    },
    starterCode: TRAINING_LOOP_STARTER,
    verify(response) {
      const result = resultOf(response);
      const epochs = asNumber(result.epochs);
      const loss = asNumber(result.loss);
      const source = sourceOf(response);
      return grade([
        {
          passed: /\w+\.backward\(\)/.test(source) && /optimizer\.step\(\)/.test(source),
          pass: "The loop includes backward() and optimizer.step().",
          fail: "The loop must clear gradients, call loss.backward(), then optimizer.step().",
        },
        {
          passed: epochs >= 50,
          pass: `Training ran for ${epochs} epochs.`,
          fail: `epochs is ${Number.isFinite(epochs) ? epochs : "missing"}, need at least 50.`,
        },
        {
          passed: loss <= 0.1,
          pass: `Final loss ${loss.toFixed(4)} is <= 0.1.`,
          fail: `Final loss was ${Number.isFinite(loss) ? loss.toFixed(4) : "missing"}, need <= 0.1.`,
        },
      ]);
    },
  },
  {
    id: "phase3-xor-mlp",
    phase: 3,
    title: "XOR MLP",
    description: "Build a hidden-layer MLP that learns XOR, then export modelMeta for the visualizer.",
    requirements: {
      modelType: "mlp",
      inputSize: 2,
      hiddenLayers: 1,
      hiddenSize: 8,
      outputSize: 1,
      activations: ["relu"],
      lossThreshold: 0.2,
      minEpochs: 100,
    },
    starterCode: XOR_STARTER,
    verify(response) {
      const result = resultOf(response);
      const loss = asNumber(result.loss);
      const accuracy = asNumber(result.accuracy);
      const structural = analyzeTorchStructure(sourceOf(response), this.requirements);
      return grade([
        {
          passed: structural.passed,
          pass: "The code contains the requested 2 -> 8 -> 1 ReLU MLP structure.",
          fail: structural.feedback.find((item) => item.includes("need") || item.includes("missing")) ?? "Build a 2 -> 8 -> 1 ReLU MLP.",
        },
        {
          passed: loss <= 0.2,
          pass: `XOR loss ${loss.toFixed(4)} is <= 0.2.`,
          fail: `XOR loss was ${Number.isFinite(loss) ? loss.toFixed(4) : "missing"}, need <= 0.2.`,
        },
        {
          passed: accuracy >= 1,
          pass: "XOR accuracy reached 100%.",
          fail: `XOR accuracy was ${Number.isFinite(accuracy) ? `${Math.round(accuracy * 100)}%` : "missing"}, need 100%.`,
        },
      ]);
    },
  },
];

export function getCurriculumTask(id: string): CurriculumTask | undefined {
  return CURRICULUM.find((task) => task.id === id);
}

export function tasksForPhase(phase: CurriculumPhase) {
  return CURRICULUM.filter((task) => task.phase === phase);
}

export function normalizeCurriculumProgress(value: unknown): CurriculumProgress {
  const record = typeof value === "object" && value !== null ? (value as Partial<CurriculumProgress>) : {};
  const knownTaskIds = new Set(CURRICULUM.map((task) => task.id));
  const completedTaskIds = Array.isArray(record.completedTaskIds)
    ? record.completedTaskIds.filter((id): id is string => typeof id === "string" && knownTaskIds.has(id))
    : [];
  const unlockedPhases = new Set<CurriculumPhase>([1]);

  if (tasksForPhase(1).every((task) => completedTaskIds.includes(task.id))) unlockedPhases.add(2);
  if (tasksForPhase(2).every((task) => completedTaskIds.includes(task.id))) unlockedPhases.add(3);

  return {
    completedTaskIds: Array.from(new Set(completedTaskIds)),
    unlockedPhases: Array.from(unlockedPhases).sort(),
  };
}

export function isCurriculumTaskUnlocked(task: CurriculumTask, progress: CurriculumProgress) {
  if (!progress.unlockedPhases.includes(task.phase)) return false;
  const phaseTasks = tasksForPhase(task.phase);
  const index = phaseTasks.findIndex((item) => item.id === task.id);
  if (index <= 0) return index === 0;
  return phaseTasks.slice(0, index).every((item) => progress.completedTaskIds.includes(item.id));
}

export function nextCurriculumTask(taskId: string): CurriculumTask | undefined {
  const index = CURRICULUM.findIndex((task) => task.id === taskId);
  return index >= 0 ? CURRICULUM[index + 1] : undefined;
}

export function firstAvailableCurriculumTask(progress: CurriculumProgress): CurriculumTask {
  return (
    CURRICULUM.find(
      (task) => isCurriculumTaskUnlocked(task, progress) && !progress.completedTaskIds.includes(task.id)
    ) ?? CURRICULUM[0]
  );
}

export function completeCurriculumTask(progress: CurriculumProgress, taskId: string): CurriculumProgress {
  const completedTaskIds = Array.from(new Set([...progress.completedTaskIds, taskId]));
  const task = getCurriculumTask(taskId);
  const nextTask = nextCurriculumTask(taskId);
  const unlockedPhases = new Set<CurriculumPhase>(progress.unlockedPhases);

  if (task && nextTask && task.phase !== nextTask.phase) {
    const phaseComplete = tasksForPhase(task.phase).every((item) => completedTaskIds.includes(item.id));
    if (phaseComplete) unlockedPhases.add(nextTask.phase);
  }

  return normalizeCurriculumProgress({
    completedTaskIds,
    unlockedPhases: Array.from(unlockedPhases),
  });
}

export function curriculumPhaseSummary(progress: CurriculumProgress) {
  return ([1, 2, 3] as CurriculumPhase[]).map((phase) => {
    const tasks = tasksForPhase(phase);
    const completed = tasks.filter((task) => progress.completedTaskIds.includes(task.id)).length;
    return { phase, completed, total: tasks.length };
  });
}
