import type { ForecastData, LeaderboardData, TdaData, PredictionRecord } from "./types";

async function loadJson<T>(path: string): Promise<T> {
  const response = await fetch(path);
  if (!response.ok) {
    throw new Error(`Failed to load ${path}: ${response.status}`);
  }
  return response.json() as Promise<T>;
}

export function loadForecast(): Promise<ForecastData> {
  return loadJson<ForecastData>("/data/forecast.json");
}

export function loadLeaderboard(): Promise<LeaderboardData> {
  return loadJson<LeaderboardData>("/data/leaderboard.json");
}

export function loadTda(): Promise<TdaData> {
  return loadJson<TdaData>("/data/tda.json");
}

export function loadPredictionTrackRecord(): Promise<PredictionRecord[]> {
  return loadJson<PredictionRecord[]>("/data/prediction_track_record.json");
}
