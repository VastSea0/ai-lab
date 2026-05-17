import type {
  CurriculumPhase,
  CurriculumProgress,
  CurriculumTask,
  ModelMeta,
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
    feedback: feedback.length > 0 ? feedback : ["No curriculum checks were defined for this task."],
  };
}

function resultOf(response: PythonLabRunResponse): PythonLabResult {
  return response.result ?? {};
}

function asNumber(value: unknown, fallback = Number.NaN) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function asNumberArray(value: unknown): number[] {
  return asArray(value)
    .map((item) => Number(item))
    .filter((item) => Number.isFinite(item));
}

function sameShape(value: unknown, expected: number[] | undefined) {
  if (!expected) return true;
  const actual = asNumberArray(value);
  return actual.length === expected.length && actual.every((item, index) => item === expected[index]);
}

function metaOf(result: PythonLabResult): ModelMeta | undefined {
  return result.modelMeta;
}

function layerKinds(meta: ModelMeta | undefined) {
  return meta?.layers.map((layer) => layer.kind) ?? [];
}

function activations(meta: ModelMeta | undefined) {
  return (meta?.layers ?? [])
    .map((layer) => layer.activation?.toLocaleLowerCase("en-US"))
    .filter(Boolean) as string[];
}

function twoDimensionalArray(value: unknown) {
  return Array.isArray(value) && value.length > 0 && value.every((row) => Array.isArray(row));
}

const LINEAR_REGRESSION_STARTER = String.raw`import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

np.random.seed(4)

x = np.linspace(-1.0, 1.0, 80)
noise = np.random.normal(0.0, 0.12, size=x.shape)
y = 3.25 * x + 1.4 + noise
frame = pd.DataFrame({"study_hours": x, "score": y})

X = frame[["study_hours"]].values
target = frame["score"].values

model = LinearRegression()
model.fit(X, target)
prediction = model.predict(X)
r2 = float(r2_score(target, prediction))
mse = float(np.mean((prediction - target) ** 2))

coef = float(model.coef_[0])
intercept = float(model.intercept_)

result = {
    "title": "Scikit-Learn Linear Regression",
    "r2_score": r2,
    "feature_shape": list(X.shape),
    "target_shape": list(target.shape),
    "metrics": [
        {"label": "r2_score", "value": round(r2, 4)},
        {"label": "mse", "value": round(mse, 4)}
    ],
    "modelMeta": {
        "type": "sklearn",
        "layers": [
            {"kind": "input", "size": 1},
            {"kind": "output", "size": 1, "activation": "linear"}
        ],
        "weights": [[[coef]]],
        "lossHistory": [mse]
    },
    "notes": [
        "LinearRegression learned one coefficient and one intercept.",
        "The 3D mirror shows the fitted one-input, one-output structure."
    ]
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const DECISION_TREE_STARTER = String.raw`import json
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

np.random.seed(8)

n = 120
x1 = np.random.rand(n)
x2 = np.random.rand(n)
label = ((x1 > 0.54) | ((x2 > 0.62) & (x1 > 0.28))).astype(int)
frame = pd.DataFrame({"signal_a": x1, "signal_b": x2, "label": label})

X = frame[["signal_a", "signal_b"]].values
y = frame["label"].values

model = DecisionTreeClassifier(max_depth=3, random_state=8)
model.fit(X, y)
prediction = model.predict(X)
accuracy = float(accuracy_score(y, prediction))

result = {
    "title": "Scikit-Learn Decision Tree",
    "accuracy": accuracy,
    "feature_shape": list(X.shape),
    "target_shape": list(y.shape),
    "metrics": [
        {"label": "depth", "value": int(model.get_depth())},
        {"label": "leaves", "value": int(model.get_n_leaves())}
    ],
    "modelMeta": {
        "type": "sklearn",
        "layers": [
            {"kind": "input", "size": 2},
            {"kind": "hidden", "size": int(model.get_n_leaves()), "activation": "relu"},
            {"kind": "output", "size": 2, "activation": "sigmoid"}
        ]
    },
    "notes": [
        "The hidden layer approximates terminal tree leaves for visualization.",
        "The grader checks both accuracy and the input-layer metadata."
    ]
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const KMEANS_STARTER = String.raw`import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

np.random.seed(13)

centers = np.array([[0.18, 0.24], [0.78, 0.30], [0.50, 0.82]])
points = []
for center in centers:
    points.append(center + np.random.normal(0.0, 0.055, size=(40, 2)))
X = np.clip(np.vstack(points), 0.0, 1.0)
frame = pd.DataFrame(X, columns=["x", "y"])

model = KMeans(n_clusters=3, random_state=13, n_init=10)
labels = model.fit_predict(frame.values)

result = {
    "title": "Scikit-Learn KMeans",
    "inertia": float(model.inertia_),
    "n_clusters": int(model.n_clusters),
    "feature_shape": list(frame.values.shape),
    "classNames": ["cluster 0", "cluster 1", "cluster 2"],
    "metrics": [
        {"label": "inertia", "value": round(float(model.inertia_), 4)},
        {"label": "clusters", "value": int(model.n_clusters)}
    ],
    "dataset": [
        {
            "id": "k" + str(i),
            "inputs": [float(row[0]), float(row[1])],
            "targets": [1.0 if labels[i] == c else 0.0 for c in range(3)],
            "label": "cluster " + str(int(labels[i]))
        }
        for i, row in enumerate(frame.values[:36])
    ],
    "modelMeta": {
        "type": "sklearn",
        "layers": [
            {"kind": "input", "size": 2},
            {"kind": "hidden", "size": 3, "activation": "relu"},
            {"kind": "output", "size": 3, "activation": "sigmoid"}
        ]
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const MLP_STARTER = String.raw`import json
import numpy as np
import torch
import torch.nn as nn

np.random.seed(21)
torch.manual_seed(21)

half = 60
angles = np.linspace(0.0, np.pi, half)
top = np.stack([0.28 + 0.32 * np.cos(angles), 0.58 + 0.22 * np.sin(angles)], axis=1)
bottom = np.stack([0.72 - 0.32 * np.cos(angles), 0.40 - 0.22 * np.sin(angles)], axis=1)
X_np = np.vstack([top, bottom]).astype("float32")
X_np += np.random.normal(0.0, 0.025, size=X_np.shape).astype("float32")
y_np = np.concatenate([np.zeros(half), np.ones(half)]).reshape(-1, 1).astype("float32")

X = torch.tensor(X_np)
y = torch.tensor(y_np)

# --- YOUR CODE: define your model below ---
model = nn.Sequential(
    nn.Linear(2, 16),
    nn.ReLU(),
    nn.Linear(16, 8),
    nn.ReLU(),
    nn.Linear(8, 1),
    nn.Sigmoid()
)
# --- END YOUR CODE ---

optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
criterion = nn.BCELoss()
loss_history = []
epochs = 260

for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == epochs - 1:
        loss_history.append(float(loss.detach()))

with torch.no_grad():
    output = model(X)
    final_loss = float(criterion(output, y).detach())
    predicted = (output >= 0.5).float()
    accuracy = float((predicted == y).float().mean())

linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
weights = [layer.weight.detach().T.tolist() for layer in linear_layers]

result = {
    "title": "PyTorch MLP Moons",
    "loss": final_loss,
    "accuracy": accuracy,
    "epochs": epochs,
    "losses": loss_history,
    "classNames": ["moon A", "moon B"],
    "dataset": [
        {
            "id": "m" + str(i),
            "inputs": [float(X_np[i, 0]), float(X_np[i, 1])],
            "targets": [1.0 - float(y_np[i, 0]), float(y_np[i, 0])],
            "label": "moon B" if y_np[i, 0] > 0.5 else "moon A"
        }
        for i in range(0, len(X_np), 6)
    ],
    "modelMeta": {
        "type": "mlp",
        "layers": [
            {"kind": "input", "size": 2},
            {"kind": "hidden", "size": 16, "activation": "relu"},
            {"kind": "hidden", "size": 8, "activation": "relu"},
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

const CNN_STARTER = String.raw`import json
import numpy as np
import torch
import torch.nn as nn

np.random.seed(31)
torch.manual_seed(31)

digits = np.array([
    [0,1,1,1,0, 1,0,0,0,1, 1,0,0,0,1, 1,0,0,0,1, 0,1,1,1,0],
    [0,0,1,0,0, 0,1,1,0,0, 0,0,1,0,0, 0,0,1,0,0, 0,1,1,1,0],
    [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 1,0,0,0,0, 1,1,1,1,0],
    [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
], dtype="float32")
labels = np.array([0, 1, 2, 3], dtype="int64")

X_np = np.repeat(digits, 8, axis=0)
y_np = np.repeat(labels, 8, axis=0)
noise = np.random.normal(0.0, 0.04, size=X_np.shape).astype("float32")
X_np = np.clip(X_np + noise, 0.0, 1.0)

X = torch.tensor(X_np.reshape(-1, 1, 5, 5), dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.long)

class TinyDigitCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 3 * 3, 4)
        )

    def forward(self, x):
        return self.net(x)

model = TinyDigitCNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
criterion = nn.CrossEntropyLoss()
loss_history = []
epochs = 220

for epoch in range(epochs):
    optimizer.zero_grad()
    logits = model(X)
    loss = criterion(logits, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0 or epoch == epochs - 1:
        loss_history.append(float(loss.detach()))

with torch.no_grad():
    logits = model(X)
    predicted = torch.argmax(logits, dim=1)
    accuracy = float((predicted == y).float().mean())
    final_loss = float(criterion(logits, y).detach())

result = {
    "title": "PyTorch 5x5 Digit CNN",
    "loss": final_loss,
    "accuracy": accuracy,
    "epochs": epochs,
    "losses": loss_history,
    "classNames": ["0", "1", "2", "3"],
    "dataset": [
        {
            "id": "d" + str(i),
            "inputs": digits[i].tolist(),
            "targets": [1.0 if i == c else 0.0 for c in range(4)],
            "label": str(i)
        }
        for i in range(4)
    ],
    "modelMeta": {
        "type": "cnn",
        "layers": [
            {"kind": "input", "size": 25},
            {"kind": "conv", "size": 36, "activation": "relu"},
            {"kind": "output", "size": 4, "activation": "linear"}
        ],
        "lossHistory": loss_history
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const LSTM_STARTER = String.raw`import json
import numpy as np
import torch
import torch.nn as nn

np.random.seed(41)
torch.manual_seed(41)

values = np.linspace(0.0, 1.0, 96).astype("float32")
series = 0.5 + 0.35 * np.sin(values * np.pi * 2.0) + 0.10 * values
series = series.astype("float32")

seq_len = 5
X_np = []
y_np = []
for i in range(len(series) - seq_len):
    X_np.append(series[i:i + seq_len])
    y_np.append(series[i + seq_len])
X_np = np.array(X_np, dtype="float32")
y_np = np.array(y_np, dtype="float32").reshape(-1, 1)

split = 68
X_train = torch.tensor(X_np[:split, :, None])
y_train = torch.tensor(y_np[:split])
X_val = torch.tensor(X_np[split:, :, None])
y_val = torch.tensor(y_np[split:])

class SequencePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=8, batch_first=True)
        self.head = nn.Linear(8, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :])

model = SequencePredictor()
optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
criterion = nn.MSELoss()
loss_history = []
epochs = 300

for epoch in range(epochs):
    optimizer.zero_grad()
    prediction = model(X_train)
    loss = criterion(prediction, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 12 == 0 or epoch == epochs - 1:
        loss_history.append(float(loss.detach()))

with torch.no_grad():
    train_loss = float(criterion(model(X_train), y_train).detach())
    val_loss = float(criterion(model(X_val), y_val).detach())

result = {
    "title": "PyTorch LSTM Sequence Predictor",
    "loss": train_loss,
    "val_loss": val_loss,
    "epochs": epochs,
    "losses": loss_history,
    "metrics": [
        {"label": "val_loss", "value": round(val_loss, 5)}
    ],
    "modelMeta": {
        "type": "lstm",
        "layers": [
            {"kind": "input", "size": 1},
            {"kind": "lstm", "size": 8, "activation": "tanh"},
            {"kind": "output", "size": 1, "activation": "linear"}
        ],
        "lossHistory": loss_history
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const Q_TABLE_STARTER = String.raw`import json
import numpy as np

np.random.seed(51)

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
    reward = -0.02
    done = False
    if next_pos == trap:
        reward = -1.0
        done = True
    if next_pos == goal:
        reward = 1.0
        done = True
    return next_pos, reward, done

alpha = 0.30
gamma = 0.94
epsilon = 0.55
episode_rewards = []

for episode in range(520):
    pos = (0, 0)
    total_reward = 0.0
    for _ in range(44):
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
    epsilon = max(0.04, epsilon * 0.992)
    episode_rewards.append(float(total_reward))

eval_rewards = []
for _ in range(10):
    pos = (0, 0)
    total_reward = 0.0
    for _ in range(20):
        action = int(np.argmax(q[state_id(pos)]))
        pos, reward, done = step(pos, action)
        total_reward += reward
        if done:
            break
    eval_rewards.append(float(total_reward))

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
for _ in range(16):
    trajectory.append([pos[0], pos[1]])
    if pos == goal or pos == trap:
        break
    action = int(np.argmax(q[state_id(pos)]))
    pos, _, done = step(pos, action)
    if done:
        trajectory.append([pos[0], pos[1]])
        break

avg_reward_last10 = float(np.mean(eval_rewards))

result = {
    "title": "Q-Table GridWorld",
    "q_table": q.round(4).tolist(),
    "qTable": q.round(4).tolist(),
    "avg_reward_last10": avg_reward_last10,
    "episodes": 520,
    "policy": policy,
    "trajectory": trajectory,
    "reward_structure": {"goal": 1.0, "trap": -1.0, "step": -0.02},
    "metrics": [
        {"label": "avg reward last10", "value": round(avg_reward_last10, 3)},
        {"label": "episodes", "value": 520}
    ],
    "modelMeta": {
        "type": "dqn",
        "layers": [
            {"kind": "input", "size": 16},
            {"kind": "hidden", "size": 4, "activation": "linear"},
            {"kind": "output", "size": 4, "activation": "linear"}
        ],
        "episodeRewards": episode_rewards
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

const DQN_STARTER = String.raw`import json
import numpy as np
import torch
import torch.nn as nn

np.random.seed(61)
torch.manual_seed(61)

size = 4
num_states = size * size
num_actions = 4
actions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
goal = (3, 3)
trap = (1, 2)

def state_id(pos):
    return pos[1] * size + pos[0]

def one_hot_state(state):
    vector = np.zeros(num_states, dtype="float32")
    vector[state] = 1.0
    return vector

def step(pos, action_index):
    dx, dy = actions[action_index]
    nx = int(np.clip(pos[0] + dx, 0, size - 1))
    ny = int(np.clip(pos[1] + dy, 0, size - 1))
    next_pos = (nx, ny)
    reward = -0.02
    done = False
    if next_pos == trap:
        reward = -1.0
        done = True
    if next_pos == goal:
        reward = 1.0
        done = True
    return next_pos, reward, done

model = nn.Sequential(
    nn.Linear(num_states, 32),
    nn.ReLU(),
    nn.Linear(32, num_actions)
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
criterion = nn.MSELoss()
gamma = 0.94
epsilon = 0.65
episode_rewards = []
loss_history = []
episodes = 700

for episode in range(episodes):
    pos = (0, 0)
    total_reward = 0.0
    for _ in range(44):
        state = state_id(pos)
        state_tensor = torch.tensor(one_hot_state(state)).unsqueeze(0)
        q_values = model(state_tensor)
        if np.random.rand() < epsilon:
            action = int(np.random.randint(num_actions))
        else:
            action = int(torch.argmax(q_values, dim=1).item())

        next_pos, reward, done = step(pos, action)
        next_state = state_id(next_pos)
        next_tensor = torch.tensor(one_hot_state(next_state)).unsqueeze(0)

        with torch.no_grad():
            next_q = model(next_tensor)
            target_value = reward if done else reward + gamma * float(torch.max(next_q))

        target = q_values.detach().clone()
        target[0, action] = target_value

        loss = criterion(q_values, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_reward += reward
        pos = next_pos
        if done:
            break
    epsilon = max(0.05, epsilon * 0.992)
    episode_rewards.append(float(total_reward))
    if episode % 20 == 0 or episode == episodes - 1:
        loss_history.append(float(loss.detach()))

eval_rewards = []
for _ in range(10):
    pos = (0, 0)
    total_reward = 0.0
    for _ in range(20):
        state_tensor = torch.tensor(one_hot_state(state_id(pos))).unsqueeze(0)
        with torch.no_grad():
            action = int(torch.argmax(model(state_tensor), dim=1).item())
        pos, reward, done = step(pos, action)
        total_reward += reward
        if done:
            break
    eval_rewards.append(float(total_reward))

avg_reward_last10 = float(np.mean(eval_rewards))
epsilon_final = float(epsilon)

with torch.no_grad():
    q_values_start = model(torch.tensor(one_hot_state(0)).unsqueeze(0)).squeeze(0).tolist()

linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
weights = [layer.weight.detach().T.tolist() for layer in linear_layers]

result = {
    "title": "DQN GridWorld",
    "avg_reward_last10": avg_reward_last10,
    "epsilon_final": epsilon_final,
    "episodes": episodes,
    "q_values_start": q_values_start,
    "q_value_shape": [num_states, num_actions],
    "losses": loss_history,
    "metrics": [
        {"label": "avg reward last10", "value": round(avg_reward_last10, 3)},
        {"label": "epsilon final", "value": round(epsilon_final, 3)}
    ],
    "modelMeta": {
        "type": "dqn",
        "layers": [
            {"kind": "input", "size": num_states},
            {"kind": "hidden", "size": 32, "activation": "relu"},
            {"kind": "output", "size": num_actions, "activation": "linear"}
        ],
        "weights": weights,
        "episodeRewards": episode_rewards,
        "lossHistory": loss_history
    }
}

print("AI_LAB_RESULT_START")
print(json.dumps(result))
print("AI_LAB_RESULT_END")
`;

export const CURRICULUM: CurriculumTask[] = [
  {
    id: "phase1-linear-regression",
    phase: 1,
    title: "Linear Regression",
    description: "Fit a Scikit-Learn LinearRegression model and report R2 plus feature and target shapes.",
    requirements: {
      modelType: "sklearn",
      featureShape: [80, 1],
      targetShape: [80],
      scoreThreshold: 0.85,
    },
    starterCode: LINEAR_REGRESSION_STARTER,
    verify(response) {
      const result = resultOf(response);
      const meta = metaOf(result);
      const r2 = asNumber(result.r2_score);
      return grade([
        {
          passed: meta?.type === "sklearn",
          pass: "Model metadata identifies a Scikit-Learn model.",
          fail: "modelMeta.type must be 'sklearn' for this task.",
        },
        {
          passed: sameShape(result.feature_shape, [80, 1]),
          pass: "Feature shape is [80, 1].",
          fail: "Feature shape must be [80, 1]. Keep X as a 2D matrix with one feature.",
        },
        {
          passed: sameShape(result.target_shape, [80]),
          pass: "Target shape is [80].",
          fail: "Target shape must be [80]. Keep y as a one-dimensional target vector.",
        },
        {
          passed: r2 >= 0.85,
          pass: `R2 score ${r2.toFixed(3)} meets the 0.85 target.`,
          fail: `R2 score was ${Number.isFinite(r2) ? r2.toFixed(3) : "missing"}, need >= 0.85. Check X/y alignment and model.fit().`,
        },
      ]);
    },
  },
  {
    id: "phase1-decision-tree",
    phase: 1,
    title: "Decision Tree Classifier",
    description: "Train a DecisionTreeClassifier and expose a simple tree-shaped modelMeta for the mirror.",
    requirements: {
      modelType: "sklearn",
      featureShape: [120, 2],
      targetShape: [120],
      scoreThreshold: 0.8,
    },
    starterCode: DECISION_TREE_STARTER,
    verify(response) {
      const result = resultOf(response);
      const meta = metaOf(result);
      const accuracy = asNumber(result.accuracy);
      return grade([
        {
          passed: sameShape(result.feature_shape, [120, 2]),
          pass: "Feature shape is [120, 2].",
          fail: "Feature shape must be [120, 2]. Preserve the two-feature training matrix.",
        },
        {
          passed: sameShape(result.target_shape, [120]),
          pass: "Target shape is [120].",
          fail: "Target shape must be [120]. Keep labels as one value per row.",
        },
        {
          passed: accuracy >= 0.8,
          pass: `Accuracy ${accuracy.toFixed(3)} meets the 0.80 target.`,
          fail: `Accuracy was ${Number.isFinite(accuracy) ? accuracy.toFixed(3) : "missing"}, need >= 0.80. Try increasing max_depth or checking labels.`,
        },
        {
          passed: layerKinds(meta).includes("input"),
          pass: "modelMeta.layers includes an input layer.",
          fail: "modelMeta.layers must include at least one layer with kind 'input'.",
        },
      ]);
    },
  },
  {
    id: "phase1-kmeans",
    phase: 1,
    title: "KMeans Clustering",
    description: "Cluster a small 2D dataset into three Scikit-Learn KMeans clusters.",
    requirements: {
      modelType: "sklearn",
      featureShape: [120, 2],
      clusters: 3,
    },
    starterCode: KMEANS_STARTER,
    verify(response) {
      const result = resultOf(response);
      const inertia = asNumber(result.inertia);
      const clusters = asNumber(result.n_clusters);
      return grade([
        {
          passed: sameShape(result.feature_shape, [120, 2]),
          pass: "Feature shape is [120, 2].",
          fail: "Feature shape must be [120, 2]. Keep the full 2D clustering matrix.",
        },
        {
          passed: Number.isFinite(inertia),
          pass: `Inertia was reported as ${inertia.toFixed(3)}.`,
          fail: "result.inertia is missing. Use float(model.inertia_) in the emitted JSON.",
        },
        {
          passed: clusters === 3,
          pass: "KMeans reported exactly 3 clusters.",
          fail: `n_clusters was ${Number.isFinite(clusters) ? clusters : "missing"}, need exactly 3.`,
        },
      ]);
    },
  },
  {
    id: "phase2-mlp-moons",
    phase: 2,
    title: "MLP Moons",
    description: "Build a PyTorch MLP with ReLU activations for a non-linear two-moons classification task.",
    requirements: {
      modelType: "mlp",
      lossThreshold: 0.15,
      minEpochs: 50,
      layerCount: 4,
      activations: ["relu"],
    },
    starterCode: MLP_STARTER,
    verify(response) {
      const result = resultOf(response);
      const meta = metaOf(result);
      const loss = asNumber(result.loss);
      const epochs = asNumber(result.epochs);
      const activationList = activations(meta);
      return grade([
        {
          passed: loss <= 0.15,
          pass: `Loss ${loss.toFixed(4)} is <= 0.15.`,
          fail: `Loss was ${Number.isFinite(loss) ? loss.toFixed(4) : "missing"}, need <= 0.15. Try more epochs or a lower learning rate.`,
        },
        {
          passed: epochs >= 50,
          pass: `Training ran for ${epochs} epochs.`,
          fail: `Epoch count was ${Number.isFinite(epochs) ? epochs : "missing"}, need >= 50.`,
        },
        {
          passed: (meta?.layers.length ?? 0) === 4,
          pass: "modelMeta has the required 4 layers.",
          fail: `Layer count was ${meta?.layers.length ?? 0}, need exactly 4 including input and output.`,
        },
        {
          passed: activationList.includes("relu"),
          pass: "ReLU activation is present in modelMeta.",
          fail: "ReLU activation is missing. Include at least one layer with activation 'relu'.",
        },
      ]);
    },
  },
  {
    id: "phase2-cnn-digits",
    phase: 2,
    title: "CNN 5x5 Digits",
    description: "Train a tiny torch.nn Conv2d classifier for 5x5 digit images.",
    requirements: {
      modelType: "cnn",
      accuracyThreshold: 0.75,
      activations: ["relu"],
    },
    starterCode: CNN_STARTER,
    verify(response) {
      const result = resultOf(response);
      const meta = metaOf(result);
      const accuracy = asNumber(result.accuracy);
      return grade([
        {
          passed: layerKinds(meta).includes("conv"),
          pass: "modelMeta.layers includes a conv layer.",
          fail: "modelMeta.layers must include at least one layer with kind 'conv'.",
        },
        {
          passed: accuracy >= 0.75,
          pass: `Accuracy ${accuracy.toFixed(3)} meets the 0.75 target.`,
          fail: `Accuracy was ${Number.isFinite(accuracy) ? accuracy.toFixed(3) : "missing"}, need >= 0.75. Check the Conv2d path and training loop.`,
        },
      ]);
    },
  },
  {
    id: "phase2-lstm-sequence",
    phase: 2,
    title: "LSTM Sequence",
    description: "Use torch.nn.LSTM to predict the next value in a short synthetic sequence.",
    requirements: {
      modelType: "lstm",
      lossThreshold: 0.03,
      minEpochs: 50,
    },
    starterCode: LSTM_STARTER,
    verify(response) {
      const result = resultOf(response);
      const meta = metaOf(result);
      const valLoss = asNumber(result.val_loss);
      return grade([
        {
          passed: layerKinds(meta).includes("lstm"),
          pass: "modelMeta.layers includes an LSTM layer.",
          fail: "modelMeta.layers must include at least one layer with kind 'lstm'.",
        },
        {
          passed: Number.isFinite(valLoss),
          pass: `Validation loss was reported as ${valLoss.toFixed(5)}.`,
          fail: "result.val_loss is missing. Emit validation loss after training.",
        },
        {
          passed: valLoss <= 0.03,
          pass: `Validation loss ${valLoss.toFixed(5)} is <= 0.03.`,
          fail: `Validation loss was ${Number.isFinite(valLoss) ? valLoss.toFixed(5) : "missing"}, need <= 0.03.`,
        },
      ]);
    },
  },
  {
    id: "phase3-q-table-grid",
    phase: 3,
    title: "Q-Table GridWorld",
    description: "Learn a Q-table policy in a custom grid environment and emit rewards for the 3D chart.",
    requirements: {
      rewardThreshold: 0,
      episodes: 500,
      outputShape: [16, 4],
    },
    starterCode: Q_TABLE_STARTER,
    verify(response) {
      const result = resultOf(response);
      const qTable = result.q_table ?? result.qTable;
      const reward = asNumber(result.avg_reward_last10);
      const rewardStructure = result.reward_structure as Record<string, unknown> | undefined;
      return grade([
        {
          passed: twoDimensionalArray(qTable),
          pass: "q_table is a 2D array.",
          fail: "q_table must be present as a 2D array of states by actions.",
        },
        {
          passed:
            rewardStructure?.goal === 1 ||
            rewardStructure?.goal === 1.0 ||
            (Number(rewardStructure?.goal) > 0 && Number(rewardStructure?.trap) < 0),
          pass: "Reward structure includes positive goal and negative trap rewards.",
          fail: "Reward structure must include a positive goal reward and a negative trap reward.",
        },
        {
          passed: reward > 0,
          pass: `Average reward over the final evaluation was ${reward.toFixed(3)}.`,
          fail: `avg_reward_last10 was ${Number.isFinite(reward) ? reward.toFixed(3) : "missing"}, need > 0.`,
        },
      ]);
    },
  },
  {
    id: "phase3-dqn-grid",
    phase: 3,
    title: "DQN GridWorld",
    description: "Train a small PyTorch DQN for the same grid and emit Q-value and reward metadata.",
    requirements: {
      modelType: "dqn",
      rewardThreshold: 0.4,
      episodes: 500,
      outputShape: [16, 4],
    },
    starterCode: DQN_STARTER,
    verify(response) {
      const result = resultOf(response);
      const meta = metaOf(result);
      const epsilon = asNumber(result.epsilon_final);
      const reward = asNumber(result.avg_reward_last10);
      return grade([
        {
          passed: meta?.type === "dqn",
          pass: "modelMeta.type is dqn.",
          fail: "modelMeta.type must be 'dqn'.",
        },
        {
          passed: epsilon < 0.1,
          pass: `epsilon_final ${epsilon.toFixed(3)} is < 0.1.`,
          fail: `epsilon_final was ${Number.isFinite(epsilon) ? epsilon.toFixed(3) : "missing"}, need < 0.1.`,
        },
        {
          passed: reward > 0.4,
          pass: `Average reward ${reward.toFixed(3)} clears the 0.4 threshold.`,
          fail: `avg_reward_last10 was ${Number.isFinite(reward) ? reward.toFixed(3) : "missing"}, need > 0.4.`,
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
  const completedTaskIds = Array.isArray(record.completedTaskIds)
    ? record.completedTaskIds.filter((id): id is string => typeof id === "string")
    : [];
  const unlockedPhases: CurriculumPhase[] = Array.isArray(record.unlockedPhases)
    ? record.unlockedPhases.filter((phase): phase is CurriculumPhase => phase === 1 || phase === 2 || phase === 3)
    : [1];

  return {
    completedTaskIds: Array.from(new Set(completedTaskIds)),
    unlockedPhases: Array.from(new Set<CurriculumPhase>([1, ...unlockedPhases])).sort(),
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
