import {
  Area,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import type { ForecastData, PredictionRecord } from "../lib/types";
import { resolveAccuracy } from "../lib/accuracy";

interface ChartRow {
  date: string;
  actual?: number;
  forecast?: number;
  ciBase?: number;
  ciBand?: number;
  accuracyPredicted?: number;
}

function buildRows(data: ForecastData, trackRecord: PredictionRecord[]): ChartRow[] {
  const resolved = resolveAccuracy(trackRecord, data.history);
  const resolvedByDate = new Map(resolved.map((r) => [r.target_date, r]));

  const historyRows: ChartRow[] = data.history.map((h) => {
    const row: ChartRow = { date: h.date, actual: h.actual };
    const match = resolvedByDate.get(h.date);
    if (match) row.accuracyPredicted = match.predicted;
    return row;
  });
  const forecastRows: ChartRow[] = data.forecast.map((f) => ({
    date: f.date,
    forecast: f.value,
    ciBase: f.lower,
    ciBand: f.upper - f.lower,
  }));
  const trimmedHistory = historyRows.slice(-36);
  // Bridge the seam: give the last history point a forecast value equal to
  // the last actual so the "Actual" and "SARIMA Forecast" lines connect
  // instead of rendering as two visually disconnected segments.
  const lastHistoryRow = trimmedHistory[trimmedHistory.length - 1];
  if (lastHistoryRow) {
    lastHistoryRow.forecast = lastHistoryRow.actual;
  }
  return [...trimmedHistory, ...forecastRows];
}

interface AccuracyTooltipProps {
  active?: boolean;
  payload?: { payload: ChartRow }[];
  label?: string;
}

function AccuracyTooltip({ active, payload, label }: AccuracyTooltipProps) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-xl bg-white dark:bg-gray-900 shadow-lg px-3 py-2 text-xs">
      <div className="font-semibold mb-1">{label}</div>
      {row.actual !== undefined && <div>Actual: {row.actual.toFixed(1)}</div>}
      {row.forecast !== undefined && <div>Forecast: {row.forecast.toFixed(1)}</div>}
      {row.accuracyPredicted !== undefined && row.actual !== undefined && (
        <div>
          Predicted: {row.accuracyPredicted.toFixed(1)} (error{" "}
          {(((row.actual - row.accuracyPredicted) / row.actual) * 100).toFixed(1)}%)
        </div>
      )}
    </div>
  );
}

export default function ForecastChart({
  data,
  trackRecord = [],
}: {
  data: ForecastData;
  trackRecord?: PredictionRecord[];
}) {
  const rows = buildRows(data, trackRecord);

  return (
    <div className="h-96 w-full rounded-2xl bg-gray-50 dark:bg-white/5 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={30} />
          <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
          <Tooltip content={<AccuracyTooltip />} />
          <Area type="monotone" dataKey="ciBase" stackId="ci" stroke="none" fill="transparent" />
          <Area
            type="monotone"
            dataKey="ciBand"
            stackId="ci"
            stroke="none"
            fill="#6366f1"
            fillOpacity={0.15}
            name="95% CI"
          />
          <Line type="monotone" dataKey="actual" stroke="#0ea5e9" strokeWidth={2} dot={false} name="Actual" />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#f59e0b"
            strokeWidth={2}
            strokeDasharray="6 4"
            dot={{ r: 3 }}
            name="SARIMA Forecast"
          />
          <Line
            type="monotone"
            dataKey="accuracyPredicted"
            stroke="none"
            dot={{ r: 5, fill: "#22c55e", stroke: "white", strokeWidth: 1 }}
            name="Predicted (resolved)"
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
