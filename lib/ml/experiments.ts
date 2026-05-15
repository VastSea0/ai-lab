import type { DataPoint, NeuralNetwork } from "./network";

export interface LearningRateTrial {
  learningRate: number;
  lossBefore: number;
  lossAfter: number;
  delta: number;
}

export function simulateLearningRates(
  network: NeuralNetwork,
  data: DataPoint[],
  rates: number[]
): LearningRateTrial[] {
  const lossBefore = network.evaluateLoss(data);
  return rates.map((learningRate) => {
    const clone = network.clone();
    clone.trainEpochDetailed(data, learningRate);
    const lossAfter = clone.evaluateLoss(data);
    return {
      learningRate,
      lossBefore,
      lossAfter,
      delta: lossAfter - lossBefore,
    };
  });
}
