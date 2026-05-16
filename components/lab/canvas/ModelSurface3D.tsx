"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import type { DataPoint, NeuralNetwork } from "@/lib/ml/network";
import { formatNumber } from "@/lib/ml/network";
import type { Task } from "@/lib/ml/tasks";

interface ModelSurface3DProps {
  task: Task;
  network: NeuralNetwork;
  data: DataPoint[];
}

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

function addLabelSprite(text: string, color = "#18202f") {
  const canvas = document.createElement("canvas");
  canvas.width = 320;
  canvas.height = 92;
  const context = canvas.getContext("2d");
  if (!context) return null;
  context.font = "600 32px Arial";
  context.fillStyle = color;
  context.textAlign = "center";
  context.fillText(text, canvas.width / 2, 56);
  const texture = new THREE.CanvasTexture(canvas);
  const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
  const sprite = new THREE.Sprite(material);
  sprite.scale.set(1.5, 0.44, 1);
  return sprite;
}

function buildSurfaceScene(scene: THREE.Scene, task: Task, network: NeuralNetwork, data: DataPoint[]) {
  const grid = new THREE.GridHelper(6, 12, "#cbd5e1", "#e2e8f0");
  grid.position.y = -1.18;
  scene.add(grid);

  const xLabel = addLabelSprite(task.axisLabels?.x ?? "x", "#526070");
  if (xLabel) {
    xLabel.position.set(2.85, -1.05, -2.75);
    scene.add(xLabel);
  }
  const yLabel = addLabelSprite(task.axisLabels?.y ?? (task.inputSize === 1 ? "model y" : "y"), "#526070");
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
    const material = new THREE.MeshStandardMaterial({
      color: "#0f766e",
      roughness: 0.42,
      metalness: 0.08,
      emissive: "#0f766e",
      emissiveIntensity: 0.08,
    });
    scene.add(new THREE.Mesh(tube, material));

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
      const height = -0.75 + confidence * 1.35;
      vertices.push(toSceneX(x), height, toSceneY(y));
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

function buildArchitectureScene(scene: THREE.Scene, task: Task, network: NeuralNetwork) {
  const layerCount = network.layers.length;
  const layerGap = layerCount <= 1 ? 0 : 4.8 / (layerCount - 1);
  const maxVisibleWeights = 360;
  const weightLines: THREE.Vector3[][] = [];
  const neuronPositions = new Map<string, THREE.Vector3>();

  network.layers.forEach((layer, layerIndex) => {
    const x = -2.4 + layerIndex * layerGap;
    const count = layer.neurons.length;
    const ringRadius = count > 18 ? 1.1 : 0.72;
    layer.neurons.forEach((neuron, neuronIndex) => {
      const angle = (neuronIndex / Math.max(1, count)) * Math.PI * 2;
      const stack = count > 18 ? Math.floor(neuronIndex / 18) * 0.34 : 0;
      const y = Math.cos(angle) * ringRadius + stack - 0.25;
      const z = Math.sin(angle) * ringRadius;
      const position = new THREE.Vector3(x, y, z);
      neuronPositions.set(neuron.id, position);

      const color =
        layer.kind === "input" ? "#0284c7" : layer.kind === "output" ? "#dc2626" : "#d97706";
      const sphere = new THREE.Mesh(
        new THREE.SphereGeometry(count > 20 ? 0.045 : 0.075, 14, 14),
        new THREE.MeshStandardMaterial({ color, roughness: 0.35 })
      );
      sphere.position.copy(position);
      scene.add(sphere);
    });
  });

  network.weights.forEach((matrix, layerIndex) => {
    matrix.forEach((row, fromIndex) => {
      row.forEach((weight, toIndex) => {
        if (weightLines.length >= maxVisibleWeights) return;
        if (Math.abs(weight) < 0.05 && weightLines.length > 80) return;
        const from = network.layers[layerIndex].neurons[fromIndex];
        const to = network.layers[layerIndex + 1].neurons[toIndex];
        const fromPosition = neuronPositions.get(from.id);
        const toPosition = neuronPositions.get(to.id);
        if (fromPosition && toPosition) weightLines.push([fromPosition, toPosition]);
      });
    });
  });

  const linePositions = new Float32Array(weightLines.length * 2 * 3);
  weightLines.forEach(([from, to], index) => {
    linePositions.set([from.x, from.y, from.z, to.x, to.y, to.z], index * 6);
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(linePositions, 3));
  scene.add(
    new THREE.LineSegments(
      geometry,
      new THREE.LineBasicMaterial({ color: "#94a3b8", transparent: true, opacity: 0.26 })
    )
  );

  const label = addLabelSprite(`${task.inputSize} input → ${task.outputSize} output`, "#334155");
  if (label) {
    label.position.set(0, -1.85, 0);
    label.scale.set(2.1, 0.58, 1);
    scene.add(label);
  }
}

export function ModelSurface3D({ task, network, data }: ModelSurface3DProps) {
  const hostRef = useRef<HTMLDivElement | null>(null);
  const [mode, setMode] = useState<"surface" | "architecture">(
    task.inputSize <= 2 ? "surface" : "architecture"
  );
  const canShowSurface = task.inputSize <= 2;
  const activeMode = canShowSurface ? mode : "architecture";
  const summary = useMemo(() => {
    if (activeMode === "architecture") return "Katmanlar uzayda dizilir; çizgiler ağırlık bağlantılarını temsil eder.";
    if (task.inputSize === 1) return "Turuncu noktalar veri, yeşil çizgi modelin öğrendiği fonksiyon.";
    return "Renk sınıfı, yükseklik modelin o bölgede ne kadar emin olduğunu gösterir.";
  }, [activeMode, task.inputSize]);

  useEffect(() => {
    const host = hostRef.current;
    if (!host) return undefined;

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f8fbff");
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 100);
    camera.position.set(4.8, 3.8, 5.2);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setClearColor("#f8fbff", 1);
    host.appendChild(renderer.domElement);
    renderer.domElement.className = "h-full w-full";

    scene.add(new THREE.AmbientLight("#ffffff", 1.75));
    const light = new THREE.DirectionalLight("#ffffff", 2.2);
    light.position.set(3, 5, 4);
    scene.add(light);

    if (activeMode === "surface" && canShowSurface) {
      buildSurfaceScene(scene, task, network, data);
    } else {
      buildArchitectureScene(scene, task, network);
    }

    let yaw = -0.68;
    let pitch = 0.58;
    let dragging = false;
    let lastX = 0;
    let lastY = 0;
    let frame = 0;

    const updateCamera = () => {
      const distance = activeMode === "architecture" ? 6.3 : 7.1;
      camera.position.set(
        Math.sin(yaw) * Math.cos(pitch) * distance,
        Math.sin(pitch) * distance,
        Math.cos(yaw) * Math.cos(pitch) * distance
      );
      camera.lookAt(0, 0, 0);
    };

    const resize = () => {
      const { width, height } = host.getBoundingClientRect();
      renderer.setSize(Math.max(1, width), Math.max(1, height), false);
      camera.aspect = Math.max(1, width) / Math.max(1, height);
      camera.updateProjectionMatrix();
    };

    const animate = () => {
      if (!dragging) yaw += 0.0018;
      updateCamera();
      renderer.render(scene, camera);
      frame = window.requestAnimationFrame(animate);
    };

    const onPointerDown = (event: PointerEvent) => {
      dragging = true;
      lastX = event.clientX;
      lastY = event.clientY;
      renderer.domElement.setPointerCapture(event.pointerId);
    };
    const onPointerMove = (event: PointerEvent) => {
      if (!dragging) return;
      yaw -= (event.clientX - lastX) * 0.006;
      pitch = Math.max(-0.18, Math.min(1.1, pitch + (event.clientY - lastY) * 0.004));
      lastX = event.clientX;
      lastY = event.clientY;
    };
    const onPointerUp = (event: PointerEvent) => {
      dragging = false;
      renderer.domElement.releasePointerCapture(event.pointerId);
    };

    const observer = new ResizeObserver(resize);
    observer.observe(host);
    renderer.domElement.addEventListener("pointerdown", onPointerDown);
    renderer.domElement.addEventListener("pointermove", onPointerMove);
    renderer.domElement.addEventListener("pointerup", onPointerUp);
    resize();
    animate();

    return () => {
      window.cancelAnimationFrame(frame);
      observer.disconnect();
      renderer.domElement.removeEventListener("pointerdown", onPointerDown);
      renderer.domElement.removeEventListener("pointermove", onPointerMove);
      renderer.domElement.removeEventListener("pointerup", onPointerUp);
      clearScene(scene);
      renderer.dispose();
      host.removeChild(renderer.domElement);
    };
  }, [activeMode, canShowSurface, data, network, task]);

  return (
    <div className="relative h-full w-full bg-[#f8fbff]">
      <div ref={hostRef} className="h-full w-full" data-testid="model-surface-3d" />
      <div className="pointer-events-none absolute left-5 top-5 max-w-[430px] rounded-md border border-white/70 bg-white/90 px-4 py-3 shadow-sm backdrop-blur">
        <div className="text-sm font-semibold text-[#18202f]">3D Model Görünümü</div>
        <div className="mt-1 text-xs leading-5 text-[#526070]">{summary}</div>
      </div>
      <div className="absolute right-5 top-16 flex rounded-md border border-[#cbd5e1] bg-white/90 p-1 shadow-sm backdrop-blur">
        <button
          type="button"
          className={`h-8 rounded px-3 text-xs font-semibold ${
            activeMode === "surface" ? "bg-[#2563eb] text-white" : "text-[#334155] hover:bg-[#eef4ff]"
          }`}
          disabled={!canShowSurface}
          onClick={() => setMode("surface")}
        >
          Yüzey
        </button>
        <button
          type="button"
          className={`h-8 rounded px-3 text-xs font-semibold ${
            activeMode === "architecture" ? "bg-[#2563eb] text-white" : "text-[#334155] hover:bg-[#eef4ff]"
          }`}
          onClick={() => setMode("architecture")}
        >
          Ağ 3D
        </button>
      </div>
      <div className="pointer-events-none absolute bottom-5 left-5 rounded-md border border-white/70 bg-white/90 px-3 py-2 text-[11px] leading-5 text-[#526070] shadow-sm backdrop-blur">
        Epoch sonrası yüzey değişir · sürükleyerek döndür · loss {formatNumber(network.evaluateLoss(data), 5)}
      </div>
    </div>
  );
}
