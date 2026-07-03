export interface ForecastPoint {
  date: string;
  actual?: number;
  value?: number;
  lower?: number;
  upper?: number;
}

export interface ForecastData {
  last_updated: string;
  history: { date: string; actual: number }[];
  forecast: { date: string; value: number; lower: number; upper: number }[];
  metrics: { mae: number; rmse: number; r2: number };
}

export interface LeaderboardModel {
  name: string;
  mae: number;
  rmse: number;
  r2: number;
  improvement_vs_baseline: number | null;
  live: boolean;
}

export interface LeaderboardData {
  last_updated: string;
  models: LeaderboardModel[];
}

export interface PersistencePoint {
  birth: number;
  death: number;
  dim: number;
}

export interface Subperiod {
  id: string;
  label: string;
  start: string;
  end: string;
  n_months: number;
  persistence_diagram: PersistencePoint[];
  max_h1_persistence: number;
  embedding_3d: [number, number, number][];
  dates: string[];
  prices: number[];
}

export interface TdaData {
  generated_at: string;
  subperiods: Subperiod[];
}

export interface PredictionRecord {
  target_date: string;
  made_on: string;
  predicted: number;
  lower: number;
  upper: number;
}
