import type { PredictionRecord } from "./types";

export interface ResolvedAccuracy {
  target_date: string;
  predicted: number;
  actual: number;
  errorPct: number;
}

export function resolveAccuracy(
  trackRecord: PredictionRecord[],
  history: { date: string; actual: number }[]
): ResolvedAccuracy[] {
  const actualByDate = new Map(history.map((h) => [h.date, h.actual]));
  const resolved: ResolvedAccuracy[] = [];

  for (const record of trackRecord) {
    const actual = actualByDate.get(record.target_date);
    if (actual === undefined) continue;
    resolved.push({
      target_date: record.target_date,
      predicted: record.predicted,
      actual,
      errorPct: ((actual - record.predicted) / actual) * 100,
    });
  }

  return resolved;
}
