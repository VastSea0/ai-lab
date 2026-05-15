import { describe, expect, it } from "vitest";
import { NeuralNetwork } from "./network";
import { TASKS } from "./tasks";
import { simulateLearningRates } from "./experiments";

describe("NeuralNetwork", () => {
  it("runs a detailed epoch with numeric loss and sample traces", () => {
    const task = TASKS[0];
    const network = NeuralNetwork.create(task.defaultLayers, 42);
    const result = network.trainEpochDetailed(task.data, task.defaultLearningRate);

    expect(result.samples).toHaveLength(task.data.length);
    expect(Number.isFinite(result.lossBefore)).toBe(true);
    expect(Number.isFinite(result.lossAfter)).toBe(true);
    expect(result.trace.edges.length).toBeGreaterThan(0);
    expect(result.trace.neurons.length).toBeGreaterThan(0);
  });

  it("simulates learning rates on clones without mutating the original network", () => {
    const task = TASKS[0];
    const network = NeuralNetwork.create(task.defaultLayers, 99);
    const before = network.evaluateLoss(task.data);
    const trials = simulateLearningRates(network, task.data, [0.05, 0.1, 0.2]);
    const after = network.evaluateLoss(task.data);

    expect(trials).toHaveLength(3);
    expect(after).toBeCloseTo(before, 12);
    expect(trials.every((trial) => Number.isFinite(trial.lossAfter))).toBe(true);
  });
});
