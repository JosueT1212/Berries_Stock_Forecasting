import { useState } from "react";
import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import type { Subperiod } from "../lib/types";

function PointCloud({
  points,
  onSelect,
}: {
  points: [number, number, number][];
  onSelect: (index: number) => void;
}) {
  return (
    <>
      {points.map((p, i) => {
        const isNewest = i === points.length - 1;
        return (
          <mesh key={i} position={p} onClick={() => onSelect(i)}>
            <sphereGeometry args={[isNewest ? 0.07 : 0.04, 8, 8]} />
            <meshStandardMaterial color={isNewest ? "#f59e0b" : "#6366f1"} />
          </mesh>
        );
      })}
    </>
  );
}

export default function EmbeddingScene({
  subperiod,
  revealedCount,
}: {
  subperiod: Subperiod;
  revealedCount: number;
}) {
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const points = subperiod.embedding_3d.slice(0, revealedCount);

  return (
    <div className="relative h-80 w-full rounded-2xl bg-gray-50 dark:bg-white/5 overflow-hidden">
      <Canvas camera={{ position: [3, 3, 3] }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[5, 5, 5]} intensity={1} />
        <PointCloud points={points} onSelect={setSelectedIndex} />
        <OrbitControls autoRotate autoRotateSpeed={0.6} enableZoom />
      </Canvas>
      {selectedIndex !== null && selectedIndex < subperiod.dates.length && (
        <div className="absolute bottom-2 left-2 rounded-lg bg-white/90 dark:bg-black/80 px-3 py-2 text-xs shadow">
          <div className="font-medium">{subperiod.dates[selectedIndex]}</div>
          <div className="text-gray-500 dark:text-gray-400">{subperiod.prices[selectedIndex]?.toFixed(1)}</div>
        </div>
      )}
    </div>
  );
}
