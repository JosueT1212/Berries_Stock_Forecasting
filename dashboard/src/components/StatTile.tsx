interface StatTileProps {
  label: string;
  value: string;
  sublabel?: string;
}

export default function StatTile({ label, value, sublabel }: StatTileProps) {
  return (
    <div className="rounded-2xl bg-gray-50 dark:bg-white/5 p-5 flex flex-col gap-1 shadow-sm">
      <span className="text-sm uppercase tracking-wide text-gray-500 dark:text-gray-400">{label}</span>
      <span className="text-3xl font-semibold text-accent">{value}</span>
      {sublabel && <span className="text-xs text-gray-400">{sublabel}</span>}
    </div>
  );
}
