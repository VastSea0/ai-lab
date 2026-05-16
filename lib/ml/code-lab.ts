import type { ActivationName, DataPoint, LayerConfig } from "./network";
import type { TaskId } from "./tasks";

export type CodeLabTrack = "core-ml" | "deep-learning" | "reinforcement";
export type CodeLabDifficulty = "kolay" | "orta" | "zor";

export interface AiCodeLabChallenge {
  id: string;
  title: string;
  track: CodeLabTrack;
  difficulty: CodeLabDifficulty;
  taskId?: TaskId;
  libraries: string[];
  concepts: string[];
  summary: string;
  prompt: string;
  starterCode: string;
  applyToNetwork: boolean;
}

export interface PythonLabMetric {
  label: string;
  value: string | number;
}

export interface PythonLabResult {
  title?: string;
  classNames?: string[];
  layers?: LayerConfig[];
  weights?: number[][][];
  biases?: number[][];
  dataset?: DataPoint[];
  losses?: number[];
  accuracy?: number;
  epochs?: number;
  notes?: string[];
  metrics?: PythonLabMetric[];
  qTable?: number[][];
  policy?: string[];
  trajectory?: number[][];
}

export interface PythonLabRunResponse {
  ok: boolean;
  challengeId?: string;
  stdout: string;
  stderr: string;
  durationMs: number;
  result?: PythonLabResult;
  error?: string;
}

const COMMON_EMIT_HELPER = String.raw`
def emit_ai_lab_result(result):
    print("AI_LAB_RESULT_START")
    print(json.dumps(result, ensure_ascii=False))
    print("AI_LAB_RESULT_END")
`;

export const AI_CODE_LAB_CHALLENGES: AiCodeLabChallenge[] = [
  {
    id: "numpy-and-perceptron",
    title: "AND Perceptron",
    track: "core-ml",
    difficulty: "kolay",
    taskId: "and",
    libraries: ["numpy"],
    concepts: ["perceptron", "one-hot", "gradient descent"],
    summary: "Lineer ayrılabilen AND kapısını tek katmanlı NumPy modeliyle çöz.",
    prompt:
      "Amaç: iki inputtan iki sınıf skoru üret. Kod çalışınca ağırlıklar bu lab'ın nöron canvas'ına aktarılabilir.",
    applyToNetwork: true,
    starterCode: String.raw`import json
import numpy as np

${COMMON_EMIT_HELPER}

np.random.seed(3)

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([
    [1.0, 0.0],
    [1.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0],
])

W = np.random.randn(2, 2) * 0.45
b = np.zeros(2)
lr = 0.42
losses = []

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

for epoch in range(650):
    out = sigmoid(X @ W + b)
    error = out - y
    loss = 0.5 * np.mean(np.sum(error ** 2, axis=1))
    delta = error * out * (1.0 - out)
    W -= lr * (X.T @ delta) / len(X)
    b -= lr * delta.mean(axis=0)
    if epoch % 50 == 0 or epoch == 649:
        losses.append(float(loss))

pred = np.argmax(sigmoid(X @ W + b), axis=1)
truth = np.argmax(y, axis=1)

emit_ai_lab_result({
    "title": "NumPy AND Perceptron",
    "classNames": ["0", "1"],
    "layers": [
        {"size": 2, "activation": "input"},
        {"size": 2, "activation": "sigmoid"},
    ],
    "weights": [W.tolist()],
    "biases": [[], b.tolist()],
    "dataset": [
        {"id": "a1", "inputs": [0, 0], "targets": [1, 0], "label": "0"},
        {"id": "a2", "inputs": [0, 1], "targets": [1, 0], "label": "0"},
        {"id": "a3", "inputs": [1, 0], "targets": [1, 0], "label": "0"},
        {"id": "a4", "inputs": [1, 1], "targets": [0, 1], "label": "1"},
    ],
    "losses": losses,
    "accuracy": float(np.mean(pred == truth)),
    "epochs": 650,
    "notes": [
        "AND lineer ayrılabilir olduğu için hidden layer gerekmiyor.",
        "W matrisi from-neuron x to-neuron formatında canvas'a aktarılır."
    ],
})
`,
  },
  {
    id: "numpy-xor-hidden-layer",
    title: "XOR Hidden Layer",
    track: "core-ml",
    difficulty: "orta",
    taskId: "xor",
    libraries: ["numpy"],
    concepts: ["non-linear boundary", "hidden layer", "backprop"],
    summary: "XOR'u çözen küçük bir MLP yaz ve hidden nöronların neden gerektiğini gör.",
    prompt:
      "Amaç: tek katmanın çözemediği XOR için 2-4-2 ağ kur. Eğitim sonucu doğrudan görsel ağın ağırlıklarına yüklenebilir.",
    applyToNetwork: true,
    starterCode: String.raw`import json
import numpy as np

${COMMON_EMIT_HELPER}

np.random.seed(11)

X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 1.0],
    [1.0, 0.0],
])

W1 = np.random.randn(2, 4) * 0.85
b1 = np.zeros(4)
W2 = np.random.randn(4, 2) * 0.85
b2 = np.zeros(2)
lr = 0.55
losses = []

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

for epoch in range(2200):
    z1 = X @ W1 + b1
    h = np.tanh(z1)
    out = sigmoid(h @ W2 + b2)

    error = out - y
    loss = 0.5 * np.mean(np.sum(error ** 2, axis=1))

    d_out = error * out * (1.0 - out)
    dW2 = h.T @ d_out / len(X)
    db2 = d_out.mean(axis=0)

    d_h = d_out @ W2.T
    d_z1 = d_h * (1.0 - h ** 2)
    dW1 = X.T @ d_z1 / len(X)
    db1 = d_z1.mean(axis=0)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    if epoch % 100 == 0 or epoch == 2199:
        losses.append(float(loss))

pred = np.argmax(sigmoid(np.tanh(X @ W1 + b1) @ W2 + b2), axis=1)
truth = np.argmax(y, axis=1)

emit_ai_lab_result({
    "title": "NumPy XOR MLP",
    "classNames": ["XOR=0", "XOR=1"],
    "layers": [
        {"size": 2, "activation": "input"},
        {"size": 4, "activation": "tanh"},
        {"size": 2, "activation": "sigmoid"},
    ],
    "weights": [W1.tolist(), W2.tolist()],
    "biases": [[], b1.tolist(), b2.tolist()],
    "dataset": [
        {"id": "x1", "inputs": [0, 0], "targets": [1, 0], "label": "XOR=0"},
        {"id": "x2", "inputs": [0, 1], "targets": [0, 1], "label": "XOR=1"},
        {"id": "x3", "inputs": [1, 0], "targets": [0, 1], "label": "XOR=1"},
        {"id": "x4", "inputs": [1, 1], "targets": [1, 0], "label": "XOR=0"},
    ],
    "losses": losses,
    "accuracy": float(np.mean(pred == truth)),
    "epochs": 2200,
    "notes": [
        "Hidden layer iki farklı ara özellik öğrenerek XOR kararını mümkün kılar.",
        "Tanh hidden katmanı negatif/pozitif ara sinyalleri görünür yapar."
    ],
})
`,
  },
  {
    id: "torch-mini-digit",
    title: "Mini Digit PyTorch",
    track: "deep-learning",
    difficulty: "orta",
    taskId: "digit",
    libraries: ["torch"],
    concepts: ["tensor", "deep network", "pixel features"],
    summary: "5x5 piksel rakamlarını PyTorch ile eğit, sonra ağı bu lab'ın nöronlarına aktar.",
    prompt:
      "Amaç: küçük bir görüntü sınıflandırıcıyı PyTorch ile gerçekten eğitmek ve eğitilmiş ağı görsel canvas üzerinde incelemek.",
    applyToNetwork: true,
    starterCode: String.raw`import json
import torch
import torch.nn as nn

${COMMON_EMIT_HELPER}

torch.manual_seed(7)

digits = [
    [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],
    [0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],
    [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0],
    [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0],
]
labels = ["0", "1", "2", "3"]
X = torch.tensor(digits, dtype=torch.float32)
y = torch.eye(4)

model = nn.Sequential(
    nn.Linear(25, 12),
    nn.ReLU(),
    nn.Linear(12, 8),
    nn.ReLU(),
    nn.Linear(8, 4),
    nn.Sigmoid(),
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.045)
loss_fn = nn.MSELoss()
losses = []

for epoch in range(700):
    optimizer.zero_grad()
    out = model(X)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0 or epoch == 699:
        losses.append(float(loss.detach()))

with torch.no_grad():
    out = model(X)
    pred = torch.argmax(out, dim=1)
    truth = torch.arange(4)
    accuracy = float((pred == truth).float().mean())

linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
weights = [layer.weight.detach().T.tolist() for layer in linear_layers]
biases = [[]] + [layer.bias.detach().tolist() for layer in linear_layers]

emit_ai_lab_result({
    "title": "PyTorch Mini Digit Classifier",
    "classNames": labels,
    "layers": [
        {"size": 25, "activation": "input"},
        {"size": 12, "activation": "relu"},
        {"size": 8, "activation": "relu"},
        {"size": 4, "activation": "sigmoid"},
    ],
    "weights": weights,
    "biases": biases,
    "dataset": [
        {"id": f"d{i}", "inputs": digits[i], "targets": y[i].tolist(), "label": labels[i]}
        for i in range(4)
    ],
    "losses": losses,
    "accuracy": accuracy,
    "epochs": 700,
    "notes": [
        "PyTorch Linear ağırlıkları out x in tuttuğu için canvas'a aktarırken transpoze edilir.",
        "Bu küçük veri seti ezberlenir; amaç gerçek görüntü görevindeki katman akışını görünür yapmak."
    ],
})
`,
  },
  {
    id: "numpy-q-learning-grid",
    title: "Q-Learning Grid",
    track: "reinforcement",
    difficulty: "zor",
    libraries: ["numpy"],
    concepts: ["reinforcement learning", "q-table", "policy"],
    summary: "Mini grid dünyasında Q-learning çalıştır ve policy simülasyonunu üret.",
    prompt:
      "Amaç: bu bölümün ileride oyun oynayan ajanlara genişleyebilmesi için state/action tabanlı bir sonuç formatı üretmek.",
    applyToNetwork: false,
    starterCode: String.raw`import json
import numpy as np

${COMMON_EMIT_HELPER}

np.random.seed(5)

size = 4
actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
action_names = ["up", "right", "down", "left"]
goal = (3, 3)
trap = (1, 2)
q = np.zeros((size * size, len(actions)))

def state_id(pos):
    return pos[1] * size + pos[0]

def step(pos, action_index):
    dx, dy = actions[action_index]
    nx = int(np.clip(pos[0] + dx, 0, size - 1))
    ny = int(np.clip(pos[1] + dy, 0, size - 1))
    next_pos = (nx, ny)
    reward = -0.04
    done = False
    if next_pos == trap:
        reward = -1.0
        done = True
    if next_pos == goal:
        reward = 1.0
        done = True
    return next_pos, reward, done

alpha = 0.25
gamma = 0.92
epsilon = 0.22
episode_rewards = []

for episode in range(600):
    pos = (0, 0)
    total_reward = 0.0
    for _ in range(40):
        s = state_id(pos)
        if np.random.rand() < epsilon:
            action = np.random.randint(len(actions))
        else:
            action = int(np.argmax(q[s]))
        next_pos, reward, done = step(pos, action)
        ns = state_id(next_pos)
        q[s, action] += alpha * (reward + gamma * np.max(q[ns]) - q[s, action])
        total_reward += reward
        pos = next_pos
        if done:
            break
    episode_rewards.append(total_reward)

policy = []
for y in range(size):
    for x in range(size):
        pos = (x, y)
        if pos == goal:
            policy.append("goal")
        elif pos == trap:
            policy.append("trap")
        else:
            policy.append(action_names[int(np.argmax(q[state_id(pos)]))])

trajectory = []
pos = (0, 0)
for _ in range(12):
    trajectory.append([pos[0], pos[1]])
    if pos == goal or pos == trap:
        break
    action = int(np.argmax(q[state_id(pos)]))
    pos, _, done = step(pos, action)
    if done:
        trajectory.append([pos[0], pos[1]])
        break

emit_ai_lab_result({
    "title": "NumPy Q-Learning GridWorld",
    "qTable": q.round(4).tolist(),
    "policy": policy,
    "trajectory": trajectory,
    "metrics": [
        {"label": "son 50 ödül", "value": round(float(np.mean(episode_rewards[-50:])), 3)},
        {"label": "episode", "value": 600}
    ],
    "notes": [
        "Bu sonuç sinir ağı değil; policy ve trajectory formatı ileride oyun simülasyonuna bağlanacak.",
        "Aynı Code Lab registry, supervised model ve reinforcement simülasyonunu birlikte taşıyabilir."
    ],
})
`,
  },
];

export function getAiCodeLabChallenge(id: string): AiCodeLabChallenge {
  return AI_CODE_LAB_CHALLENGES.find((challenge) => challenge.id === id) ?? AI_CODE_LAB_CHALLENGES[0];
}

export function isActivationName(value: unknown): value is ActivationName {
  return value === "input" || value === "linear" || value === "sigmoid" || value === "tanh" || value === "relu";
}
