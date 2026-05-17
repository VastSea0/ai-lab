"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Maximize2, Minimize2, RotateCcw, ZoomIn, ZoomOut } from "lucide-react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import type {
  DataPoint,
  EdgeSnapshot,
  NeuralNetwork,
  Selection,
  TrainingPhase,
  TrainingTrace,
} from "@/lib/ml/network";
import { edgeId, formatNumber } from "@/lib/ml/network";
import type { VisualizationMode } from "@/lib/ml/lab-types";
import type { Task } from "@/lib/ml/tasks";
import type { ModelMeta } from "@/lib/ml/types";

interface ModelSurface3DProps {
  task: Task;
  network: NeuralNetwork;
  data: DataPoint[];
  trace: TrainingTrace;
  phase: TrainingPhase;
  visualizationMode: VisualizationMode;
  selected: Selection | null;
  hovered: Selection | null;
  onSelect: (selection: Selection | null) => void;
  onHover: (selection: Selection | null) => void;
  onOpenDetail: (selection: Selection) => void;
  modelMeta?: ModelMeta;
}

type SceneMode = "architecture" | "surface";
type RenderMode = SceneMode | "reward";

interface InteractiveObject extends THREE.Object3D {
  userData: {
    selection?: Selection;
  };
}

interface SignalDefinition {
  id: string;
  forwardFrom: THREE.Vector3;
  forwardTo: THREE.Vector3;
  backwardFrom: THREE.Vector3;
  backwardTo: THREE.Vector3;
  forwardDelay: number;
  backwardDelay: number;
  duration: number;
  radius: number;
  weightMagnitude: number;
  forwardColor: string;
  backwardColor: string;
}

interface PulseBurst {
  id: number;
  phase: Exclude<TrainingPhase, "idle">;
  startedAt: number;
}

interface SignalParticle {
  key: string;
  phase: Exclude<TrainingPhase, "idle">;
  from: THREE.Vector3;
  to: THREE.Vector3;
  duration: number;
  startTime: number;
  mesh: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
  trail: THREE.Line<THREE.BufferGeometry, THREE.LineBasicMaterial>;
  trailPositions: Float32Array;
}

interface RewardSceneAnimation {
  marker: THREE.Mesh<THREE.SphereGeometry, THREE.MeshStandardMaterial>;
  positions: THREE.Vector3[];
  values: number[];
}

const DEFAULT_ARCHITECTURE_CAMERA = new THREE.Vector3(5.8, 3.6, 6.5);
const DEFAULT_SURFACE_CAMERA = new THREE.Vector3(5.2, 4.4, 6.1);
const SCENE_TARGET = new THREE.Vector3(0, 0, 0);
const PULSE_MAX_AGE_SECONDS = 7.5;
const SIGNAL_PARTICLE_LIMIT = 520;

function classColor(task: Task, index: number) {
  return task.classColors?.[index] ?? ["#2563eb", "#ef4444", "#10b981", "#f59e0b"][index % 4];
}

function bestClass(values: number[]) {
  return values.indexOf(Math.max(...values));
}

function toSceneX(value: number) {
  return (value - 0.5) * 4.8;
}

function toSceneY(value: number) {
  return (value - 0.5) * 4.8;
}

function outputHeight(value: number) {
  return (value - 0.5) * 2.25;
}

function metricForEdge(edge: EdgeSnapshot | undefined, fallbackWeight: number, mode: VisualizationMode) {
  if (mode === "weights") return fallbackWeight;
  if (mode === "gradients") return edge?.gradient ?? 0;
  if (mode === "rl-reward") return fallbackWeight;
  return edge?.correction ?? 0;
}

function colorForMetric(value: number, mode: VisualizationMode) {
  if (mode === "corrections") return value >= 0 ? "#059669" : "#f97316";
  if (mode === "rl-reward") return value >= 0 ? "#2563eb" : "#e11d48";
  return value >= 0 ? "#2563eb" : "#e11d48";
}

function easeInOut(value: number) {
  return value < 0.5 ? 2 * value * value : 1 - Math.pow(-2 * value + 2, 2) / 2;
}

function smoothStep(value: number) {
  const clamped = Math.max(0, Math.min(1, value));
  return clamped * clamped * (3 - 2 * clamped);
}

function layerTitle(kind: string, index: number) {
  if (kind === "input") return "INPUT";
  if (kind === "output") return "OUTPUT";
  return `HIDDEN ${index}`;
}

function signalLane(
  from: THREE.Vector3,
  to: THREE.Vector3,
  side: 1 | -1
) {
  const direction = new THREE.Vector3().subVectors(to, from);
  const length = direction.length();
  if (length < 1e-6) {
    return { from: from.clone(), to: to.clone() };
  }
  direction.normalize();
  const lane = new THREE.Vector3().crossVectors(direction, new THREE.Vector3(0, 1, 0));
  if (lane.lengthSq() < 1e-5) lane.set(0, 0, 1);
  lane.normalize().multiplyScalar(0.115 * side);
  lane.y += 0.032 * side;
  const endpointPadding = Math.min(0.18, length * 0.18);

  return {
    from: from.clone().addScaledVector(direction, endpointPadding).add(lane),
    to: to.clone().addScaledVector(direction, -endpointPadding).add(lane),
  };
}

function disposeParticle(particle: SignalParticle, scene: THREE.Scene) {
  scene.remove(particle.mesh);
  scene.remove(particle.trail);
  particle.mesh.geometry.dispose();
  particle.mesh.material.dispose();
  particle.trail.geometry.dispose();
  particle.trail.material.dispose();
}

function clearScene(scene: THREE.Scene) {
  scene.traverse((object) => {
    const mesh = object as THREE.Mesh;
    if (mesh.geometry) mesh.geometry.dispose();
    const material = mesh.material;
    if (Array.isArray(material)) {
      material.forEach((item) => item.dispose());
    } else if (material) {
      material.dispose();
    }
  });
}

function addLabelSprite(text: string, color = "#18202f", width = 360, height = 108) {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.clearRect(0, 0, width, height);
  context.fillStyle = "rgba(255, 255, 255, 0.84)";
  context.strokeStyle = "rgba(203, 213, 225, 0.92)";
  context.lineWidth = 3;
  const radius = 14;
  context.beginPath();
  context.roundRect(8, 8, width - 16, height - 16, radius);
  context.fill();
  context.stroke();
  context.font = "700 28px Arial";
  context.fillStyle = color;
  context.textAlign = "center";
  const lines = text.split("\n");
  lines.forEach((line, index) => {
    context.fillText(line, width / 2, 42 + index * 32);
  });
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(width / 190, height / 190, 1);
  return sprite;
}

function addCylinderBetween(
  from: THREE.Vector3,
  to: THREE.Vector3,
  radius: number,
  material: THREE.Material
) {
  const direction = new THREE.Vector3().subVectors(to, from);
  const length = direction.length();
  const geometry = new THREE.CylinderGeometry(radius, radius, length, 10, 1);
  const cylinder = new THREE.Mesh(geometry, material);
  cylinder.position.copy(from).add(to).multiplyScalar(0.5);
  cylinder.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction.normalize());
  return cylinder;
}

function buildSurfaceScene(scene: THREE.Scene, task: Task, network: NeuralNetwork, data: DataPoint[]) {
  const grid = new THREE.GridHelper(6, 12, "#cbd5e1", "#e2e8f0");
  grid.position.y = -1.18;
  scene.add(grid);

  const xLabel = addLabelSprite(task.axisLabels?.x ?? "x", "#526070", 220, 82);
  if (xLabel) {
    xLabel.position.set(2.85, -1.05, -2.75);
    scene.add(xLabel);
  }
  const yLabel = addLabelSprite(task.axisLabels?.y ?? (task.inputSize === 1 ? "model y" : "y"), "#526070", 260, 82);
  if (yLabel) {
    yLabel.position.set(-2.9, -1.05, 2.75);
    scene.add(yLabel);
  }

  if (task.inputSize === 1) {
    const curvePoints = Array.from({ length: 90 }, (_, index) => {
      const input = index / 89;
      const output = network.predictPure([input])[0] ?? 0;
      return new THREE.Vector3(toSceneX(input), outputHeight(output), 0);
    });
    const curve = new THREE.CatmullRomCurve3(curvePoints);
    const tube = new THREE.TubeGeometry(curve, 96, 0.035, 8, false);
    scene.add(
      new THREE.Mesh(
        tube,
        new THREE.MeshStandardMaterial({
          color: "#0f766e",
          roughness: 0.42,
          metalness: 0.08,
          emissive: "#0f766e",
          emissiveIntensity: 0.08,
        })
      )
    );

    data.forEach((point) => {
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(0.08, 18, 18),
        new THREE.MeshStandardMaterial({ color: "#f97316", roughness: 0.36 })
      );
      sphere.position.set(toSceneX(point.inputs[0] ?? 0), outputHeight(point.targets[0] ?? 0), 0.22);
      scene.add(sphere);
    });
    return;
  }

  const segments = 38;
  const vertices: number[] = [];
  const colors: number[] = [];
  const indices: number[] = [];
  for (let yIndex = 0; yIndex <= segments; yIndex += 1) {
    for (let xIndex = 0; xIndex <= segments; xIndex += 1) {
      const x = xIndex / segments;
      const y = yIndex / segments;
      const output = network.predictPure([x, y]);
      const cls = task.outputType === "classification" ? bestClass(output) : 0;
      const confidence =
        task.outputType === "classification"
          ? Math.max(...output)
          : Math.max(0, Math.min(1, output[0] ?? 0));
      vertices.push(toSceneX(x), -0.75 + confidence * 1.35, toSceneY(y));
      const color = new THREE.Color(classColor(task, cls));
      colors.push(color.r, color.g, color.b);
    }
  }
  for (let yIndex = 0; yIndex < segments; yIndex += 1) {
    for (let xIndex = 0; xIndex < segments; xIndex += 1) {
      const a = yIndex * (segments + 1) + xIndex;
      const b = a + 1;
      const c = a + segments + 1;
      const d = c + 1;
      indices.push(a, c, b, b, c, d);
    }
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
  geometry.setIndex(indices);
  geometry.computeVertexNormals();
  scene.add(
    new THREE.Mesh(
      geometry,
      new THREE.MeshStandardMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        roughness: 0.6,
        metalness: 0.02,
        transparent: true,
        opacity: 0.78,
      })
    )
  );

  data.forEach((point) => {
    const targetIndex = point.targets.length > 1 ? bestClass(point.targets) : Math.round(point.targets[0] ?? 0);
    const predicted = network.predictPure([point.inputs[0] ?? 0, point.inputs[1] ?? 0]);
    const marker = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 18, 18),
      new THREE.MeshStandardMaterial({ color: classColor(task, targetIndex), roughness: 0.32 })
    );
    marker.position.set(
      toSceneX(point.inputs[0] ?? 0),
      -0.66 + Math.max(...predicted) * 1.35 + 0.12,
      toSceneY(point.inputs[1] ?? 0)
    );
    scene.add(marker);
  });
}

function sampledRewards(values: number[], limit = 96) {
  const clean = values.filter(Number.isFinite);
  if (clean.length <= limit) return clean;
  const step = clean.length / limit;
  return Array.from({ length: limit }, (_, index) => clean[Math.min(clean.length - 1, Math.floor(index * step))]);
}

function buildRewardScene(scene: THREE.Scene, rewards: number[]): RewardSceneAnimation | null {
  const values = sampledRewards(rewards);
  if (values.length === 0) return null;

  const grid = new THREE.GridHelper(7, 14, "#cbd5e1", "#e2e8f0");
  grid.position.y = -1.18;
  scene.add(grid);

  const min = Math.min(...values);
  const max = Math.max(...values);
  const magnitude = Math.max(1e-6, Math.max(Math.abs(min), Math.abs(max)));
  const spread = Math.max(1e-6, max - min);
  const width = 6.2;
  const gap = width / values.length;
  const positions: THREE.Vector3[] = [];
  const red = new THREE.Color("#ef4444");
  const yellow = new THREE.Color("#f59e0b");
  const green = new THREE.Color("#10b981");

  values.forEach((reward, index) => {
    const normalized = (reward - min) / spread;
    const color = reward < 0
      ? new THREE.Color().lerpColors(red, yellow, Math.max(0, Math.min(1, normalized)))
      : new THREE.Color().lerpColors(yellow, green, Math.max(0, Math.min(1, reward / magnitude)));
    const height = 0.08 + Math.abs(reward) / magnitude * 2.6;
    const geometry = new THREE.BoxGeometry(Math.max(0.025, gap * 0.62), height, 0.24);
    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: 0.36,
      metalness: 0.04,
      emissive: color,
      emissiveIntensity: 0.04,
    });
    const bar = new THREE.Mesh(geometry, material);
    const x = -width / 2 + index * gap + gap / 2;
    bar.position.set(x, -1.12 + height / 2, 0);
    scene.add(bar);
    positions.push(new THREE.Vector3(x, -1.02 + height + 0.18, 0));
  });

  const zeroLine = addCylinderBetween(
    new THREE.Vector3(-width / 2, -1.1, -0.22),
    new THREE.Vector3(width / 2, -1.1, -0.22),
    0.012,
    new THREE.MeshBasicMaterial({ color: "#475569", transparent: true, opacity: 0.55 })
  );
  scene.add(zeroLine);

  const title = addLabelSprite(`Episode reward\n${values.length} samples`, "#334155", 360, 108);
  if (title) {
    title.position.set(0, 2.05, 0);
    title.scale.set(1.9, 0.62, 1);
    scene.add(title);
  }

  const minLabel = addLabelSprite(`min ${formatNumber(min, 2)}`, "#b91c1c", 220, 74);
  if (minLabel) {
    minLabel.position.set(-3.25, -1.0, 0.65);
    scene.add(minLabel);
  }
  const maxLabel = addLabelSprite(`max ${formatNumber(max, 2)}`, "#047857", 220, 74);
  if (maxLabel) {
    maxLabel.position.set(3.25, -1.0, 0.65);
    scene.add(maxLabel);
  }

  const marker = new THREE.Mesh(
    new THREE.SphereGeometry(0.115, 24, 24),
    new THREE.MeshStandardMaterial({
      color: "#111827",
      emissive: "#38bdf8",
      emissiveIntensity: 0.8,
      roughness: 0.18,
    })
  );
  marker.position.copy(positions[0]);
  scene.add(marker);

  const halo = new THREE.Mesh(
    new THREE.TorusGeometry(0.2, 0.014, 10, 48),
    new THREE.MeshBasicMaterial({ color: "#111827", transparent: true, opacity: 0.7 })
  );
  marker.add(halo);

  return { marker, positions, values };
}

function buildArchitectureScene({
  scene,
  task,
  network,
  trace,
  visualizationMode,
  selected,
  pickables,
  signalDefinitions,
}: {
  scene: THREE.Scene;
  task: Task;
  network: NeuralNetwork;
  trace: TrainingTrace;
  visualizationMode: VisualizationMode;
  selected: Selection | null;
  pickables: THREE.Object3D[];
  signalDefinitions: SignalDefinition[];
}) {
  const layerCount = network.layers.length;
  const layerGap = layerCount <= 1 ? 0 : 5.25 / (layerCount - 1);
  const maxVisibleWeights = 520;
  const neuronPositions = new Map<string, THREE.Vector3>();
  const neuronTrace = new Map(trace.neurons.map((neuron) => [neuron.id, neuron]));
  const edgeTrace = new Map(trace.edges.map((edge) => [edge.id, edge]));
  const totalNeurons = network.layers.reduce((sum, layer) => sum + layer.neurons.length, 0);
  const showDenseLabels = totalNeurons <= 14;
  const activeId = selected?.id ?? null;
  let visibleWeightCount = 0;
  let signalCount = 0;

  network.layers.forEach((layer, layerIndex) => {
    const x = -2.625 + layerIndex * layerGap;
    const count = layer.neurons.length;
    const verticalGap = count <= 1 ? 0 : Math.min(0.42, 3.45 / (count - 1));
    const startY = -((count - 1) * verticalGap) / 2;

    const layerLabel = addLabelSprite(`${layerTitle(layer.kind, layerIndex)} · ${count}`, "#475569", 290, 74);
    if (layerLabel) {
      layerLabel.position.set(x, 2.12, 0);
      layerLabel.scale.set(1.08, 0.34, 1);
      scene.add(layerLabel);
    }

    layer.neurons.forEach((neuron, neuronIndex) => {
      const spiral = count > 18 ? (neuronIndex % 5 - 2) * 0.1 : 0;
      const position = new THREE.Vector3(x, startY + neuronIndex * verticalGap, spiral);
      neuronPositions.set(neuron.id, position);

      const snapshot = neuronTrace.get(neuron.id);
      const isActive = activeId === neuron.id;
      const color =
        layer.kind === "input" ? "#0284c7" : layer.kind === "output" ? "#dc2626" : "#d97706";
      const radius = isActive ? 0.13 : count > 20 ? 0.055 : count > 12 ? 0.075 : 0.095;
      const material = new THREE.MeshStandardMaterial({
        color,
        emissive: isActive ? color : "#000000",
        emissiveIntensity: isActive ? 0.55 : 0,
        roughness: 0.34,
      });
      const sphere = new THREE.Mesh(new THREE.SphereGeometry(radius, 18, 18), material) as InteractiveObject;
      sphere.position.copy(position);
      const selection: Selection = {
        type: "neuron",
        id: neuron.id,
        layerIndex: neuron.layerIndex,
        neuronIndex: neuron.neuronIndex,
      };
      sphere.userData.selection = selection;
      pickables.push(sphere);
      scene.add(sphere);

      if (isActive) {
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(radius * 1.95, 0.012, 8, 44),
          new THREE.MeshBasicMaterial({ color: "#111827" })
        );
        ring.position.copy(position);
        scene.add(ring);
      }

      if (snapshot && (showDenseLabels || isActive || (layer.kind === "output" && count <= 6))) {
        const label = addLabelSprite(
          `${formatNumber(snapshot.value, 2)}\n${
            layer.kind === "input" ? `x${neuronIndex + 1}` : `Σ=${formatNumber(snapshot.z, 2)}`
          }`,
          isActive ? "#111827" : "#334155",
          190,
          94
        );
        if (label) {
          label.position.copy(position).add(new THREE.Vector3(0, radius + 0.22, 0));
          label.scale.multiplyScalar(isActive ? 1.25 : 0.78);
          scene.add(label);
        }
      }
    });
  });

  network.weights.forEach((matrix, layerIndex) => {
    let layerVisibleCount = 0;
    matrix.forEach((row, fromIndex) => {
      row.forEach((weight, toIndex) => {
        const from = network.layers[layerIndex].neurons[fromIndex];
        const to = network.layers[layerIndex + 1].neurons[toIndex];
        const fromPosition = neuronPositions.get(from.id);
        const toPosition = neuronPositions.get(to.id);
        if (!fromPosition || !toPosition) return;

        const id = edgeId(layerIndex, fromIndex, layerIndex + 1, toIndex);
        const snapshot = edgeTrace.get(id);
        const metric = metricForEdge(snapshot, weight, visualizationMode);
        const magnitude = Math.min(1, Math.abs(metric) / (visualizationMode === "weights" ? 2.4 : 0.35));
        const isConnectedToActive =
          activeId !== null && (from.id === activeId || to.id === activeId || id === activeId);
        const shouldShow =
          isConnectedToActive ||
          (visibleWeightCount < maxVisibleWeights && (layerVisibleCount < 120 || Math.abs(weight) > 0.05));
        if (!shouldShow) return;
        visibleWeightCount += 1;
        layerVisibleCount += 1;

        const color = colorForMetric(metric, visualizationMode);
        const lineGeometry = new THREE.BufferGeometry().setFromPoints([fromPosition, toPosition]);
        const line = new THREE.Line(
          lineGeometry,
          new THREE.LineBasicMaterial({
            color,
            transparent: true,
            opacity: isConnectedToActive ? 0.9 : 0.18 + magnitude * 0.36,
          })
        );
        scene.add(line);

        const selection: Selection = {
          type: "edge",
          id,
          fromLayerIndex: layerIndex,
          fromNeuronIndex: fromIndex,
          toLayerIndex: layerIndex + 1,
          toNeuronIndex: toIndex,
        };
        const pickRadius = isConnectedToActive ? 0.035 : 0.018;
        const pickTube = addCylinderBetween(
          fromPosition,
          toPosition,
          pickRadius,
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0 })
        ) as InteractiveObject;
        pickTube.userData.selection = selection;
        pickables.push(pickTube);
        scene.add(pickTube);

        const pulsePriority =
          isConnectedToActive ||
          layerVisibleCount < 30 ||
          Math.abs(snapshot?.gradient ?? 0) > 0.035 ||
          Math.abs(snapshot?.contribution ?? 0) > 0.1;
        if (pulsePriority && signalCount < 112) {
          const pulseStrength = Math.max(
            Math.abs(snapshot?.contribution ?? 0),
            Math.abs(snapshot?.gradient ?? 0) * 1.8,
            Math.abs(weight) * 0.18
          );
          const forwardLane = signalLane(fromPosition, toPosition, 1);
          const backwardLane = signalLane(toPosition, fromPosition, 1);
          signalDefinitions.push({
            id,
            forwardFrom: forwardLane.from,
            forwardTo: forwardLane.to,
            backwardFrom: backwardLane.from,
            backwardTo: backwardLane.to,
            forwardDelay: layerIndex * 0.5 + (signalCount % 16) * 0.028,
            backwardDelay:
              (network.weights.length - 1 - layerIndex) * 0.5 + (signalCount % 16) * 0.028,
            duration: 1.75 + layerCount * 0.1,
            radius: 0.038 + Math.min(0.045, pulseStrength * 0.13),
            weightMagnitude: Math.abs(weight),
            forwardColor: colorForMetric(weight, "weights"),
            backwardColor: colorForMetric(metric, visualizationMode),
          });
          signalCount += 1;
        }
      });
    });
  });

  const label = addLabelSprite(`${task.inputSize} input → ${task.outputSize} output`, "#334155", 420, 90);
  if (label) {
    label.position.set(0, -2.05, 0);
    label.scale.set(2.2, 0.48, 1);
    scene.add(label);
  }
}

function selectionTitle(selection: Selection | null) {
  if (!selection) return "Bir nöron veya bağlantıya tıkla";
  if (selection.type === "neuron") return `L${selection.layerIndex} N${selection.neuronIndex}`;
  return `L${selection.fromLayerIndex}N${selection.fromNeuronIndex} → L${selection.toLayerIndex}N${selection.toNeuronIndex}`;
}

export function ModelSurface3D({
  task,
  network,
  data,
  trace,
  phase,
  visualizationMode,
  selected,
  hovered,
  onSelect,
  onHover,
  onOpenDetail,
  modelMeta,
}: ModelSurface3DProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const cameraPoseRef = useRef<{ position: THREE.Vector3; target: THREE.Vector3 } | null>(null);
  const pulseBurstsRef = useRef<PulseBurst[]>([]);
  const pulseBurstIdRef = useRef(0);
  const [mode, setMode] = useState<SceneMode>("architecture");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const canShowSurface = task.inputSize <= 2;
  const episodeRewards = useMemo(
    () => (modelMeta?.episodeRewards ?? []).filter(Number.isFinite),
    [modelMeta]
  );
  const showRewardScene = visualizationMode === "rl-reward" && episodeRewards.length > 0;
  const activeMode: RenderMode = showRewardScene ? "reward" : canShowSurface ? mode : "architecture";
  const activeSelection = hovered ?? selected;
  const selectedNeuron = activeSelection?.type === "neuron"
    ? trace.neurons.find((neuron) => neuron.id === activeSelection.id)
    : null;
  const selectedEdge = activeSelection?.type === "edge"
    ? trace.edges.find((edge) => edge.id === activeSelection.id)
    : null;
  const sceneLabel =
    activeMode === "reward" ? "RL ödül grafiği" : activeMode === "architecture" ? "Ağ mimarisi" : "Karar yüzeyi";

  const resetCamera = () => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    camera.position.copy(activeMode === "surface" ? DEFAULT_SURFACE_CAMERA : DEFAULT_ARCHITECTURE_CAMERA);
    controls.target.copy(SCENE_TARGET);
    controls.update();
    cameraPoseRef.current = {
      position: camera.position.clone(),
      target: controls.target.clone(),
    };
  };

  const zoomCamera = (factor: number) => {
    const camera = cameraRef.current;
    const controls = controlsRef.current;
    if (!camera || !controls) return;
    const direction = new THREE.Vector3().subVectors(camera.position, controls.target);
    const nextDistance = THREE.MathUtils.clamp(direction.length() * factor, 3.2, 13);
    direction.setLength(nextDistance);
    camera.position.copy(controls.target).add(direction);
    controls.update();
    cameraPoseRef.current = {
      position: camera.position.clone(),
      target: controls.target.clone(),
    };
  };

  useEffect(() => {
    if (phase === "idle") return;
    const now = performance.now() / 1000;
    pulseBurstsRef.current = [
      ...pulseBurstsRef.current.filter((burst) => now - burst.startedAt < PULSE_MAX_AGE_SECONDS),
      {
        id: pulseBurstIdRef.current,
        phase,
        startedAt: now,
      },
    ].slice(-5);
    pulseBurstIdRef.current += 1;
  }, [phase]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f8fbff");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const pickables: THREE.Object3D[] = [];
    const signalDefinitions: SignalDefinition[] = [];
    const activeParticles = new Map<string, SignalParticle>();

    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor("#f8fbff", 1);
    renderer.domElement.className = "h-full w-full";
    host.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    const defaultPosition = activeMode === "surface" ? DEFAULT_SURFACE_CAMERA : DEFAULT_ARCHITECTURE_CAMERA;
    const savedPose = cameraPoseRef.current;
    camera.position.copy(savedPose?.position ?? defaultPosition);
    controls.target.copy(savedPose?.target ?? SCENE_TARGET);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.enablePan = true;
    controls.enableZoom = true;
    controls.minDistance = 3.2;
    controls.maxDistance = 13;
    controls.rotateSpeed = 0.58;
    controls.zoomSpeed = 0.72;
    controls.panSpeed = 0.62;
    controls.update();
    cameraRef.current = camera;
    controlsRef.current = controls;

    const syncCameraPose = () => {
      cameraPoseRef.current = {
        position: camera.position.clone(),
        target: controls.target.clone(),
      };
    };
    controls.addEventListener("change", syncCameraPose);

    scene.add(new THREE.AmbientLight("#ffffff", 1.75));
    const light = new THREE.DirectionalLight("#ffffff", 2.2);
    light.position.set(3, 5, 4);
    scene.add(light);

    const rewardAnimation = activeMode === "reward" ? buildRewardScene(scene, episodeRewards) : null;

    if (activeMode === "reward") {
      // Reward scene is built above.
    } else if (activeMode === "surface" && canShowSurface) {
      buildSurfaceScene(scene, task, network, data);
    } else {
      buildArchitectureScene({
        scene,
        task,
        network,
        trace,
        visualizationMode,
        selected,
        pickables,
        signalDefinitions,
      });
    }

    let pressing = false;
    let moved = false;
    let lastX = 0;
    let lastY = 0;
    let frame = 0;

    const updatePointer = (event: PointerEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    };

    const pickSelection = (event: PointerEvent) => {
      if (activeMode !== "architecture") return null;
      updatePointer(event);
      raycaster.setFromCamera(pointer, camera);
      return (raycaster.intersectObjects(pickables, false)[0]?.object as InteractiveObject | undefined)?.userData.selection ?? null;
    };

    const resize = () => {
      const { width, height } = host.getBoundingClientRect();
      renderer.setSize(Math.max(1, width), Math.max(1, height), false);
      camera.aspect = Math.max(1, width) / Math.max(1, height);
      camera.updateProjectionMatrix();
    };

    const createParticle = (
      definition: SignalDefinition,
      burst: PulseBurst
    ): SignalParticle => {
      const color = burst.phase === "forward" ? definition.forwardColor : definition.backwardColor;
      const from = burst.phase === "forward" ? definition.forwardFrom : definition.backwardFrom;
      const to = burst.phase === "forward" ? definition.forwardTo : definition.backwardTo;
      const delay = burst.phase === "forward" ? definition.forwardDelay : definition.backwardDelay;
      const key = `${burst.id}:${burst.phase}:${definition.id}`;
      const trailPositions = new Float32Array(6);
      trailPositions.set([from.x, from.y, from.z, from.x, from.y, from.z]);
      const trailGeometry = new THREE.BufferGeometry();
      trailGeometry.setAttribute("position", new THREE.BufferAttribute(trailPositions, 3));
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(definition.radius, 18, 18),
        new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 0.9,
          roughness: 0.16,
          transparent: true,
          opacity: 0,
        })
      );
      const trail = new THREE.Line(
        trailGeometry,
        new THREE.LineBasicMaterial({
          color,
          transparent: true,
          opacity: 0,
        })
      );
      mesh.visible = false;
      trail.visible = false;
      scene.add(trail);
      scene.add(mesh);

      return {
        key,
        phase: burst.phase,
        from,
        to,
        duration:
          burst.phase === "forward"
            ? definition.duration / (0.68 + Math.min(2.2, definition.weightMagnitude) * 0.52)
            : definition.duration,
        startTime: burst.startedAt + delay,
        mesh,
        trail,
        trailPositions,
      };
    };

    const animate = () => {
      controls.update();
      const now = performance.now() / 1000;
      const liveBursts = pulseBurstsRef.current
        .filter((burst) => now - burst.startedAt < PULSE_MAX_AGE_SECONDS)
        .sort((a, b) => b.id - a.id);
      pulseBurstsRef.current = liveBursts;

      liveBursts.forEach((burst) => {
        signalDefinitions.forEach((definition) => {
          const delay = burst.phase === "forward" ? definition.forwardDelay : definition.backwardDelay;
          const startTime = burst.startedAt + delay;
          const age = now - startTime;
          if (age < 0 || age > definition.duration) return;
          const key = `${burst.id}:${burst.phase}:${definition.id}`;
          if (activeParticles.has(key)) return;

          if (activeParticles.size >= SIGNAL_PARTICLE_LIMIT) {
            const oldest = activeParticles.values().next().value as SignalParticle | undefined;
            if (oldest) {
              activeParticles.delete(oldest.key);
              disposeParticle(oldest, scene);
            }
          }

          activeParticles.set(key, createParticle(definition, burst));
        });
      });

      activeParticles.forEach((particle) => {
        const rawProgress = (now - particle.startTime) / particle.duration;
        if (rawProgress < 0) return;
        if (rawProgress >= 1) {
          activeParticles.delete(particle.key);
          disposeParticle(particle, scene);
          return;
        }

        const progress = easeInOut(rawProgress);
        const head = particle.from.clone().lerp(particle.to, progress);
        const tail = particle.from.clone().lerp(particle.to, Math.max(0, progress - 0.12));
        const fade = smoothStep(Math.min(1, rawProgress * 4.5, (1 - rawProgress) * 4.5));
        particle.mesh.visible = true;
        particle.trail.visible = true;
        particle.mesh.position.copy(head);
        particle.mesh.scale.setScalar(0.78 + fade * 0.34);
        particle.mesh.material.opacity = 0.08 + fade * 0.86;
        particle.trail.material.opacity = fade * 0.42;
        particle.trailPositions.set([tail.x, tail.y, tail.z, head.x, head.y, head.z]);
        const position = particle.trail.geometry.getAttribute("position");
        position.needsUpdate = true;
      });

      if (rewardAnimation && rewardAnimation.positions.length > 0) {
        const speed = 0.95;
        const cursor = (now * speed) % rewardAnimation.positions.length;
        const index = Math.floor(cursor);
        const nextIndex = (index + 1) % rewardAnimation.positions.length;
        const t = cursor - index;
        const position = rewardAnimation.positions[index].clone().lerp(rewardAnimation.positions[nextIndex], t);
        rewardAnimation.marker.position.copy(position);
        rewardAnimation.marker.scale.setScalar(0.88 + Math.sin(now * 6) * 0.08);
      }
      renderer.render(scene, camera);
      frame = window.requestAnimationFrame(animate);
    };

    const onPointerDown = (event: PointerEvent) => {
      pressing = true;
      moved = false;
      lastX = event.clientX;
      lastY = event.clientY;
    };
    const onPointerMove = (event: PointerEvent) => {
      if (pressing) {
        const dx = event.clientX - lastX;
        const dy = event.clientY - lastY;
        if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
        return;
      }
      onHover(pickSelection(event));
    };
    const onPointerUp = (event: PointerEvent) => {
      pressing = false;
      if (!moved) onSelect(pickSelection(event));
    };
    const onDoubleClick = (event: MouseEvent) => {
      const selection = pickSelection(event as PointerEvent);
      if (selection) onOpenDetail(selection);
    };
    const onPointerLeave = () => {
      pressing = false;
      onHover(null);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(host);
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    renderer.domElement.addEventListener("dblclick", onDoubleClick);
    renderer.domElement.addEventListener("pointerleave", onPointerLeave);
    resize();
    animate();

    return () => {
      window.cancelAnimationFrame(frame);
      syncCameraPose();
      observer.disconnect();
      controls.removeEventListener("change", syncCameraPose);
      controls.dispose();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      renderer.domElement.removeEventListener("dblclick", onDoubleClick);
      renderer.domElement.removeEventListener("pointerleave", onPointerLeave);
      activeParticles.forEach((particle) => disposeParticle(particle, scene));
      activeParticles.clear();
      clearScene(scene);
      renderer.dispose();
      if (cameraRef.current === camera) cameraRef.current = null;
      if (controlsRef.current === controls) controlsRef.current = null;
      if (renderer.domElement.parentNode === host) host.removeChild(renderer.domElement);
    };
  }, [
    activeMode,
    canShowSurface,
    data,
    episodeRewards,
    network,
    onHover,
    onOpenDetail,
    onSelect,
    selected,
    task,
    trace,
    visualizationMode,
  ]);

  return (
    <div
      className={`${
        isFullscreen ? "fixed inset-0 z-[80]" : "relative h-full w-full"
      } overflow-hidden bg-[#f8fbff]`}
    >
      <div ref={hostRef} className="h-full w-full" data-testid="model-surface-3d" />
      <div className="pointer-events-none absolute left-4 top-4 max-w-[320px] rounded-md border border-white/70 bg-white/90 px-3 py-2 shadow-sm backdrop-blur">
        <div className="text-xs font-semibold text-[#18202f]">3D Simülasyon</div>
        <div className="mt-0.5 truncate text-[11px] leading-4 text-[#526070]">
          {sceneLabel} · {activeMode === "reward" ? `${episodeRewards.length} episode` : network.getLayerSizes().join(" -> ")}
        </div>
      </div>
      <div className="absolute right-4 top-4 flex flex-col items-end gap-1.5">
        <div className="flex rounded-md border border-[#cbd5e1] bg-white/90 p-1 shadow-sm backdrop-blur">
          <button
            type="button"
            className={`h-7 rounded px-2.5 text-[11px] font-semibold ${
              activeMode === "architecture" ? "bg-[#2563eb] text-white" : "text-[#334155] hover:bg-[#eef4ff]"
            }`}
            onClick={() => setMode("architecture")}
          >
            Ağ 3D
          </button>
          <button
            type="button"
            className={`h-7 rounded px-2.5 text-[11px] font-semibold ${
              activeMode === "surface" ? "bg-[#2563eb] text-white" : "text-[#334155] hover:bg-[#eef4ff]"
            }`}
            disabled={!canShowSurface}
            onClick={() => setMode("surface")}
          >
            Yüzey
          </button>
        </div>
        <div className="flex rounded-md border border-[#cbd5e1] bg-white/90 p-1 shadow-sm backdrop-blur">
          <button
            type="button"
            className="grid h-8 w-8 place-items-center rounded text-[#334155] hover:bg-[#eef4ff]"
            title="Yakınlaş"
            aria-label="3D sahneye yakınlaş"
            onClick={() => zoomCamera(0.82)}
          >
            <ZoomIn size={17} />
          </button>
          <button
            type="button"
            className="grid h-8 w-8 place-items-center rounded text-[#334155] hover:bg-[#eef4ff]"
            title="Uzaklaş"
            aria-label="3D sahneden uzaklaş"
            onClick={() => zoomCamera(1.18)}
          >
            <ZoomOut size={17} />
          </button>
          <button
            type="button"
            className="grid h-8 w-8 place-items-center rounded text-[#334155] hover:bg-[#eef4ff]"
            title="Kamerayı sıfırla"
            aria-label="3D kamerayı sıfırla"
            onClick={resetCamera}
          >
            <RotateCcw size={17} />
          </button>
          <button
            type="button"
            className="grid h-8 w-8 place-items-center rounded text-[#334155] hover:bg-[#eef4ff]"
            title={isFullscreen ? "Tam ekrandan çık" : "Tam ekran"}
            aria-label={isFullscreen ? "3D tam ekrandan çık" : "3D tam ekran aç"}
            onClick={() => setIsFullscreen((value) => !value)}
          >
            {isFullscreen ? <Minimize2 size={17} /> : <Maximize2 size={17} />}
          </button>
        </div>
      </div>
      <div className="pointer-events-none absolute bottom-4 left-4 max-w-[300px] rounded-md border border-white/70 bg-white/[0.9] px-3 py-2 text-[11px] leading-4 text-[#526070] shadow-sm backdrop-blur">
        <div className="font-semibold text-[#18202f]">Sahne</div>
        <div>
          {activeMode === "reward"
            ? `Faz: ${phase} · son ödül ${formatNumber(episodeRewards.at(-1) ?? 0, 3)}`
            : `Faz: ${phase} · loss ${formatNumber(network.evaluateLoss(data), 5)}`}
        </div>
      </div>
      <div className="pointer-events-auto absolute bottom-4 right-4 w-[min(320px,calc(100%-2rem))] rounded-md border border-[#dbe5f1] bg-white/[0.94] p-3 text-xs leading-5 text-[#526070] shadow-sm backdrop-blur">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#64748b]">Odak</div>
            <div className="mt-0.5 truncate text-sm font-semibold text-[#18202f]">
              {selectionTitle(activeSelection)}
            </div>
          </div>
          {activeSelection ? (
            <button
              type="button"
              className="rounded border border-[#cbd5e1] px-2 py-1 text-[11px] font-semibold text-[#334155] hover:bg-[#eef4ff]"
              onClick={() => onOpenDetail(activeSelection)}
            >
              Detay
            </button>
          ) : null}
        </div>
        {selectedNeuron ? (
          <div className="mt-2 space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded border border-[#e2e8f0] bg-[#f8fafc] p-2">
                <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-[#64748b]">Aktivasyon</div>
                <div className="mt-1 font-semibold text-[#18202f]">
                  a = {formatNumber(selectedNeuron.value, 5)}
                </div>
              </div>
              <div className="rounded border border-[#e2e8f0] bg-[#f8fafc] p-2">
                <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-[#64748b]">Delta</div>
                <div className="mt-1 font-semibold text-[#18202f]">
                  δ = {formatNumber(selectedNeuron.delta, 5)}
                </div>
              </div>
            </div>
            <details className="rounded border border-[#e2e8f0] bg-white">
              <summary className="cursor-pointer px-2 py-1.5 text-[11px] font-semibold text-[#2563eb]">
                Denklem ve katkılar
              </summary>
              <div className="space-y-2 border-t border-[#edf2f7] p-2">
                <div className="break-words rounded bg-[#f8fafc] p-2 font-mono text-[11px] text-[#18202f]">
                  {selectedNeuron.layerKind === "input" ? "x" : "z"} = {selectedNeuron.formula}
                </div>
                {selectedNeuron.incoming.slice(0, 4).map((item, index) => (
                  <div
                    key={`${item.fromNeuronId}-${index}`}
                    className="flex justify-between gap-3 rounded bg-[#f8fafc] px-2 py-1 font-mono text-[11px]"
                  >
                    <span>
                      {formatNumber(item.inputValue, 3)} x {formatNumber(item.weight, 3)}
                    </span>
                    <span className="font-semibold text-[#18202f]">{formatNumber(item.product, 4)}</span>
                  </div>
                ))}
              </div>
            </details>
          </div>
        ) : selectedEdge ? (
          <div className="mt-2 space-y-2">
            <div className="grid grid-cols-2 gap-2">
              <div className="rounded border border-[#e2e8f0] bg-[#f8fafc] p-2">
                <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-[#64748b]">Ağırlık</div>
                <div className="mt-1 font-semibold text-[#18202f]">
                  {formatNumber(selectedEdge.weightBefore, 5)} → {formatNumber(selectedEdge.weightAfter, 5)}
                </div>
              </div>
              <div className="rounded border border-[#e2e8f0] bg-[#f8fafc] p-2">
                <div className="text-[10px] font-bold uppercase tracking-[0.1em] text-[#64748b]">Gradient</div>
                <div className="mt-1 font-semibold text-[#18202f]">{formatNumber(selectedEdge.gradient, 5)}</div>
              </div>
            </div>
            <details className="rounded border border-[#e2e8f0] bg-white">
              <summary className="cursor-pointer px-2 py-1.5 text-[11px] font-semibold text-[#2563eb]">
                Katkı hesabı
              </summary>
              <div className="space-y-2 border-t border-[#edf2f7] p-2">
                <div className="rounded bg-[#f8fafc] p-2 font-mono text-[11px] text-[#18202f]">
                  katkı = a(prev) x w = {formatNumber(selectedEdge.contribution, 5)}
                </div>
                <div className="rounded border border-[#fed7aa] bg-[#fff7ed] px-3 py-2 text-[#9a3412]">
                  Düzeltme: {formatNumber(selectedEdge.correction, 5)} · hata{" "}
                  {formatNumber(selectedEdge.errorSignal, 5)}
                </div>
              </div>
            </details>
          </div>
        ) : (
          <div className="mt-2 rounded border border-[#e2e8f0] bg-[#f8fafc] px-2 py-1.5 text-[11px]">
            Nöron veya bağlantı seçince hesap özeti burada görünür.
          </div>
        )}
      </div>
    </div>
  );
}
