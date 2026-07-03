import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { loadForecast, loadLeaderboard, loadTda } from "./lib/data";
import type { ForecastData, LeaderboardData, TdaData } from "./lib/types";
import StatTile from "./components/StatTile";
import ForecastChart from "./components/ForecastChart";
import Leaderboard from "./components/Leaderboard";
import TdaExplorer from "./components/TdaExplorer";
import SubperiodComparison from "./components/SubperiodComparison";

type Section = "forecast" | "leaderboard" | "tda" | "subperiods";

const SECTIONS: { id: Section; label: string }[] = [
  { id: "forecast", label: "Forecast" },
  { id: "leaderboard", label: "Model Leaderboard" },
  { id: "tda", label: "TDA Explorer" },
  { id: "subperiods", label: "Subperiods" },
];

function isStale(lastUpdated: string): boolean {
  const days = (Date.now() - new Date(lastUpdated).getTime()) / (1000 * 60 * 60 * 24);
  return days > 40;
}

export default function App() {
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [leaderboard, setLeaderboard] = useState<LeaderboardData | null>(null);
  const [tda, setTda] = useState<TdaData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [section, setSection] = useState<Section>("forecast");

  useEffect(() => {
    Promise.all([loadForecast(), loadLeaderboard(), loadTda()])
      .then(([f, l, t]) => {
        setForecast(f);
        setLeaderboard(l);
        setTda(t);
      })
      .catch((err) => setError(String(err)));
  }, []);

  if (error) {
    return <div className="p-8 text-red-500">Failed to load dashboard data: {error}</div>;
  }

  if (!forecast || !leaderboard || !tda) {
    return <div className="p-8 text-gray-400">Loading dashboard...</div>;
  }

  const latest = forecast.history[forecast.history.length - 1];
  const next = forecast.forecast[0];

  return (
    <div className="min-h-screen max-w-6xl mx-auto p-6 md:p-10 flex flex-col gap-6">
      <header className="flex flex-col gap-1">
        <h1 className="text-3xl font-bold">Berries PPI Forecast Dashboard</h1>
        <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
          <span>Last updated {new Date(forecast.last_updated).toLocaleDateString()}</span>
          {isStale(forecast.last_updated) && (
            <span className="bg-amber-500/20 text-amber-600 dark:text-amber-400 px-2 py-0.5 rounded-full text-xs">
              stale — monthly refresh may have missed
            </span>
          )}
        </div>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <StatTile label="Latest Actual" value={latest.actual.toFixed(1)} sublabel={latest.date} />
        <StatTile label="Next Month Forecast" value={next.value.toFixed(1)} sublabel={next.date} />
        <StatTile label="Live Model MAE" value={forecast.metrics.mae.toFixed(2)} sublabel="SARIMA, backtest" />
      </div>

      <nav className="flex gap-2 border-b border-gray-200 dark:border-white/10 pb-2">
        {SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => setSection(s.id)}
            className={`px-4 py-2 rounded-t-lg text-sm font-medium transition-colors ${
              section === s.id
                ? "text-accent border-b-2 border-accent"
                : "text-gray-500 dark:text-gray-400 hover:text-gray-800 dark:hover:text-gray-200"
            }`}
          >
            {s.label}
          </button>
        ))}
      </nav>

      <motion.main
        key={section}
        initial={{ opacity: 0, y: 8 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.25 }}
      >
        {section === "forecast" && <ForecastChart data={forecast} />}
        {section === "leaderboard" && <Leaderboard data={leaderboard} />}
        {section === "tda" && <TdaExplorer data={tda} />}
        {section === "subperiods" && <SubperiodComparison data={tda} />}
      </motion.main>
    </div>
  );
}
