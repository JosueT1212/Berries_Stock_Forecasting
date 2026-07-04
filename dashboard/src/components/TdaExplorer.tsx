import { useEffect, useRef, useState } from "react";
import type { TdaData } from "../lib/types";
import PersistenceDiagram from "./PersistenceDiagram";
import EmbeddingScene from "./EmbeddingScene";
import PriceSparkline from "./PriceSparkline";

const ANIMATION_STEP_MS = 80;

export default function TdaExplorer({ data }: { data: TdaData }) {
  const [activeId, setActiveId] = useState(data.subperiods[0]?.id);
  const active = data.subperiods.find((s) => s.id === activeId) ?? data.subperiods[0];
  const totalPoints = active?.embedding_3d.length ?? 0;

  const [revealedCount, setRevealedCount] = useState(totalPoints);
  const [playing, setPlaying] = useState(false);
  const intervalRef = useRef<number | null>(null);

  useEffect(() => {
    setRevealedCount(active?.embedding_3d.length ?? 0);
    setPlaying(false);
    if (intervalRef.current !== null) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [active?.id]);

  useEffect(() => {
    return () => {
      if (intervalRef.current !== null) clearInterval(intervalRef.current);
    };
  }, []);

  function handleReplay() {
    if (!active) return;
    if (intervalRef.current !== null) clearInterval(intervalRef.current);
    setRevealedCount(0);
    setPlaying(true);
    intervalRef.current = window.setInterval(() => {
      setRevealedCount((count) => {
        const next = count + 1;
        if (next >= totalPoints) {
          if (intervalRef.current !== null) clearInterval(intervalRef.current);
          intervalRef.current = null;
          setPlaying(false);
          return totalPoints;
        }
        return next;
      });
    }, ANIMATION_STEP_MS);
  }

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
        <div className="flex flex-col gap-2">
          <EmbeddingScene key={active.id} subperiod={active} revealedCount={revealedCount} />
          <PriceSparkline prices={active.prices} revealedCount={revealedCount} totalPoints={totalPoints} />
          <button
            onClick={handleReplay}
            disabled={playing}
            className="self-start px-3 py-1.5 rounded-full text-xs font-medium bg-accent text-white disabled:opacity-50"
          >
            {playing ? "Playing…" : "▶ Replay animation"}
          </button>
        </div>
      </div>
    </div>
  );
}
