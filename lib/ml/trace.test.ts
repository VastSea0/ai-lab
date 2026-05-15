import { describe, expect, it } from "vitest";
import { NeuralNetwork } from "./network";
import { buildEpochTraceRecord } from "./trace";
import { TASKS } from "./tasks";

describe("trace builder", () => {
  it("builds ordered calculation steps with selectable targets", () => {
    const task = TASKS[0];
    const network = NeuralNetwork.create(task.defaultLayers, 7);
    const result = network.trainEpochDetailed(task.data.slice(0, 1), task.defaultLearningRate);
    const record = buildEpochTraceRecord(1, result, task, task.data.slice(0, 1), task.defaultLearningRate);
    const steps = record.samples[0].steps;

    expect(steps[0].phase).toBe("input");
    expect(steps.some((step) => step.phase === "forward" && step.target?.type === "neuron")).toBe(true);
    expect(steps.some((step) => step.phase === "update" && step.target?.type === "edge")).toBe(true);
    expect(steps.at(-1)?.phase).toBe("update");
  });
});
