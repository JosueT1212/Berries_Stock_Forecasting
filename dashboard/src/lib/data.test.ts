import { describe, expect, it, vi, beforeEach } from "vitest";
import { loadForecast, loadLeaderboard, loadTda, loadPredictionTrackRecord } from "./data";

function mockFetchOnce(payload: unknown, ok = true) {
  vi.stubGlobal(
    "fetch",
    vi.fn().mockResolvedValue({
      ok,
      json: () => Promise.resolve(payload),
    })
  );
}

describe("data loaders", () => {
  beforeEach(() => {
    vi.unstubAllGlobals();
  });

  it("loadForecast fetches and parses forecast.json", async () => {
    const payload = { last_updated: "x", history: [], forecast: [], metrics: { mae: 1, rmse: 2, r2: 0.5 } };
    mockFetchOnce(payload);
    const result = await loadForecast();
    expect(result.metrics.r2).toBe(0.5);
    expect(fetch).toHaveBeenCalledWith("/data/forecast.json");
  });

  it("loadLeaderboard throws on non-ok response", async () => {
    mockFetchOnce({}, false);
    await expect(loadLeaderboard()).rejects.toThrow();
  });

  it("loadTda fetches and parses tda.json", async () => {
    const payload = { generated_at: "x", subperiods: [] };
    mockFetchOnce(payload);
    const result = await loadTda();
    expect(result.subperiods).toEqual([]);
  });

  it("loadPredictionTrackRecord fetches and parses prediction_track_record.json", async () => {
    const payload = [{ target_date: "2026-05-01", made_on: "2026-04-05", predicted: 149.1, lower: 87.2, upper: 210.9 }];
    mockFetchOnce(payload);
    const result = await loadPredictionTrackRecord();
    expect(result).toEqual(payload);
    expect(fetch).toHaveBeenCalledWith("/data/prediction_track_record.json");
  });
});
