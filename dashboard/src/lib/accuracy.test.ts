import { describe, expect, it } from "vitest";
import { resolveAccuracy } from "./accuracy";

describe("resolveAccuracy", () => {
  it("returns empty array for empty track record", () => {
    expect(resolveAccuracy([], [{ date: "2026-05-01", actual: 150 }])).toEqual([]);
  });

  it("skips entries with no matching actual", () => {
    const trackRecord = [{ target_date: "2027-01-01", made_on: "2026-12-01", predicted: 100, lower: 80, upper: 120 }];
    expect(resolveAccuracy(trackRecord, [{ date: "2026-05-01", actual: 150 }])).toEqual([]);
  });

  it("joins matching entries and computes signed error percentage", () => {
    const trackRecord = [
      { target_date: "2026-05-01", made_on: "2026-04-05", predicted: 140, lower: 100, upper: 180 },
      { target_date: "2026-06-01", made_on: "2026-05-05", predicted: 160, lower: 120, upper: 200 },
    ];
    const history = [
      { date: "2026-05-01", actual: 150 },
      { date: "2026-06-01", actual: 140 },
    ];

    const result = resolveAccuracy(trackRecord, history);

    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ target_date: "2026-05-01", predicted: 140, actual: 150, errorPct: ((150 - 140) / 150) * 100 });
    expect(result[0].errorPct).toBeGreaterThan(0); // actual > predicted → positive error
    expect(result[1].errorPct).toBeLessThan(0); // actual (140) < predicted (160) → negative error
  });
});
