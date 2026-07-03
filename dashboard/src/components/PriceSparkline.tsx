import { Line, LineChart, ReferenceDot, ResponsiveContainer } from "recharts";

interface PriceSparklineProps {
  prices: number[];
  revealedCount: number;
  totalPoints: number;
}

export default function PriceSparkline({ prices, revealedCount, totalPoints }: PriceSparklineProps) {
  const rows = prices.map((value, i) => ({ index: i, value }));
  const markerIndex =
    totalPoints === 0 ? 0 : Math.round((revealedCount / totalPoints) * (prices.length - 1));
  const markerValue = prices[markerIndex];

  return (
    <div className="h-20 w-full rounded-xl bg-gray-50 dark:bg-white/5 p-2">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={rows} margin={{ top: 4, right: 8, left: 8, bottom: 4 }}>
          <Line type="monotone" dataKey="value" stroke="#94a3b8" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          {markerValue !== undefined && (
            <ReferenceDot x={markerIndex} y={markerValue} r={5} fill="#f59e0b" stroke="white" strokeWidth={1} />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
