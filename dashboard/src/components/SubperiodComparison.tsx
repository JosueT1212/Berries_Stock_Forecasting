import { Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { TdaData } from "../lib/types";

const COLORS = ["#0ea5e9", "#6366f1", "#f59e0b"];

export default function SubperiodComparison({ data }: { data: TdaData }) {
  const rows = data.subperiods.map((s) => ({
    label: s.label,
    max_h1_persistence: s.max_h1_persistence,
  }));

  return (
    <div className="rounded-2xl bg-gray-50 dark:bg-white/5 p-4">
      <h3 className="text-sm uppercase tracking-wide text-gray-500 dark:text-gray-400 mb-2">
        H₁ Persistence by Subperiod
      </h3>
      <div className="h-64 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
            <XAxis dataKey="label" tick={{ fontSize: 11 }} />
            <YAxis tick={{ fontSize: 11 }} />
            <Tooltip />
            <Bar dataKey="max_h1_persistence" radius={[8, 8, 0, 0]}>
              {rows.map((_, i) => (
                <Cell key={i} fill={COLORS[i % COLORS.length]} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
      <p className="text-xs text-gray-400 mt-2">
        Elevated H₁ persistence in 2008–2012 confirms stronger nonlinear cyclicity during the financial crisis.
      </p>
    </div>
  );
}
