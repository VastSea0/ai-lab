import { describe, expect, it } from "vitest";
import { parseDatasetWithMapping, previewDatasetText } from "./dataset";
import { TASKS } from "./tasks";

describe("dataset studio parser", () => {
  it("previews csv headers and maps selected columns", () => {
    const task = TASKS[0];
    const csv = "size,price,ignored\n10,100,a\n20,220,b\n30,330,c";
    const preview = previewDatasetText(csv);
    const result = parseDatasetWithMapping(csv, task, {
      inputColumns: ["size"],
      targetColumns: ["price"],
      normalize: true,
      trainRatio: 0.8,
    });

    expect(preview.headers).toEqual(["size", "price", "ignored"]);
    expect(result.data).toHaveLength(3);
    expect(result.data[0].inputs[0]).toBe(0);
    expect(result.data[2].inputs[0]).toBe(1);
    expect(result.metadata.inputColumns).toEqual(["size"]);
  });

  it("previews json records", () => {
    const preview = previewDatasetText(JSON.stringify([{ x: 0.1, target: 0.2 }]));

    expect(preview.sourceType).toBe("json");
    expect(preview.headers).toEqual(["x", "target"]);
    expect(preview.rows[0]).toEqual(["0.1", "0.2"]);
  });
});
