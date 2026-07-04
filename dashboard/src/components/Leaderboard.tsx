import type { LeaderboardData } from "../lib/types";

export default function Leaderboard({ data }: { data: LeaderboardData }) {
  const sorted = [...data.models].sort((a, b) => a.mae - b.mae);

  return (
    <div className="rounded-2xl bg-gray-50 dark:bg-white/5 p-4 overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-white/10">
            <th className="py-2 pr-4">Model</th>
            <th className="py-2 pr-4">MAE</th>
            <th className="py-2 pr-4">RMSE</th>
            <th className="py-2 pr-4">R²</th>
            <th className="py-2 pr-4">Improvement vs. RF Base</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((m) => (
            <tr
              key={m.name}
              className={`border-b border-gray-100 dark:border-white/5 hover:bg-gray-100 dark:hover:bg-white/10 transition-colors ${
                m.live ? "bg-accent/10 font-medium" : ""
              }`}
            >
              <td className="py-2 pr-4 flex items-center gap-2">
                {m.name}
                {m.live && (
                  <span className="text-xs uppercase tracking-wide bg-accent text-white rounded-full px-2 py-0.5">
                    live
                  </span>
                )}
              </td>
              <td className="py-2 pr-4">{m.mae.toFixed(2)}</td>
              <td className="py-2 pr-4">{m.rmse.toFixed(2)}</td>
              <td className="py-2 pr-4">{m.r2.toFixed(3)}</td>
              <td className="py-2 pr-4">
                {m.improvement_vs_baseline === null ? "—" : `${m.improvement_vs_baseline > 0 ? "+" : ""}${m.improvement_vs_baseline.toFixed(2)}%`}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="text-xs text-gray-400 mt-2">
        Rows marked <span className="font-medium">live</span> refit monthly. Others are frozen at their notebook-run values.
      </p>
    </div>
  );
}
