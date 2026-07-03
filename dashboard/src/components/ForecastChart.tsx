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
import type { ForecastData } from "../lib/types";

interface ChartRow {
  date: string;
  actual?: number;
  forecast?: number;
  ciBase?: number;
  ciBand?: number;
}

function buildRows(data: ForecastData): ChartRow[] {
  const historyRows: ChartRow[] = data.history.map((h) => ({ date: h.date, actual: h.actual }));
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

export default function ForecastChart({ data }: { data: ForecastData }) {
  const rows = buildRows(data);

  return (
    <div className="h-96 w-full rounded-2xl bg-gray-50 dark:bg-white/5 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={30} />
          <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
          <Tooltip
            contentStyle={{ borderRadius: 12, border: "none", boxShadow: "0 4px 20px rgba(0,0,0,0.15)" }}
          />
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
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
