import { useState } from "react";
import type { TdaData } from "../lib/types";
import PersistenceDiagram from "./PersistenceDiagram";
import EmbeddingScene from "./EmbeddingScene";

export default function TdaExplorer({ data }: { data: TdaData }) {
  const [activeId, setActiveId] = useState(data.subperiods[0]?.id);
  const active = data.subperiods.find((s) => s.id === activeId) ?? data.subperiods[0];

  if (!active) return null;

  return (
    <div className="flex flex-col gap-4">
      <div className="flex gap-2">
        {data.subperiods.map((s) => (
          <button
            key={s.id}
            onClick={() => setActiveId(s.id)}
            className={`px-4 py-2 rounded-full text-sm transition-colors ${
              s.id === active.id
                ? "bg-accent text-white"
                : "bg-gray-100 dark:bg-white/5 text-gray-600 dark:text-gray-300"
            }`}
          >
            {s.label}
          </button>
        ))}
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <PersistenceDiagram subperiod={active} />
        <EmbeddingScene subperiod={active} />
      </div>
    </div>
  );
}
