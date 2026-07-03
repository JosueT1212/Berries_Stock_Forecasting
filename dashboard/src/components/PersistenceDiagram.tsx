import { CartesianGrid, ReferenceLine, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import type { Subperiod } from "../lib/types";

interface DiagramTooltipProps {
  active?: boolean;
  payload?: { payload: { birth: number; death: number; dim: number } }[];
}

function DiagramTooltip({ active, payload }: DiagramTooltipProps) {
  if (!active || !payload || !payload.length) return null;
  const p = payload[0].payload;
  const persistence = p.death - p.birth;
  const label = p.dim === 0 ? "H₀ (connected component)" : "H₁ (loop)";
  return (
    <div className="rounded-xl bg-white dark:bg-gray-900 shadow-lg px-3 py-2 text-xs">
      <div className="font-semibold mb-1">{label}</div>
      <div>Birth: {p.birth.toFixed(3)}</div>
      <div>Death: {p.death.toFixed(3)}</div>
      <div>Persistence: {persistence.toFixed(3)}</div>
    </div>
  );
}

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
          <Tooltip cursor={{ strokeDasharray: "3 3" }} content={<DiagramTooltip />} />
          <Scatter name="H₀ (components)" data={h0} fill="#0ea5e9" />
          <Scatter name="H₁ (loops)" data={h1} fill="#f59e0b" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
