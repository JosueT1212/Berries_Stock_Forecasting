import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import type { Subperiod } from "../lib/types";

export default function PersistenceDiagram({ subperiod }: { subperiod: Subperiod }) {
  const h0 = subperiod.persistence_diagram.filter((p) => p.dim === 0);
  const h1 = subperiod.persistence_diagram.filter((p) => p.dim === 1);
  const maxDeath = Math.max(1, ...subperiod.persistence_diagram.map((p) => p.death));

  return (
    <div className="h-80 w-full rounded-2xl bg-gray-50 dark:bg-white/5 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis type="number" dataKey="birth" name="birth" domain={[0, maxDeath]} tick={{ fontSize: 11 }} />
          <YAxis type="number" dataKey="death" name="death" domain={[0, maxDeath]} tick={{ fontSize: 11 }} />
          <ReferenceLine
            segment={[{ x: 0, y: 0 }, { x: maxDeath, y: maxDeath }]}
            stroke="#94a3b8"
            strokeDasharray="4 4"
          />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} />
          <Scatter name="H₀ (components)" data={h0} fill="#0ea5e9" />
          <Scatter name="H₁ (loops)" data={h1} fill="#f59e0b" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
