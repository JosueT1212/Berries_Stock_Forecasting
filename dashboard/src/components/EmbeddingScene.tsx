import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import type { Subperiod } from "../lib/types";

function PointCloud({ points }: { points: [number, number, number][] }) {
  return (
    <>
      {points.map((p, i) => (
        <mesh key={i} position={p}>
          <sphereGeometry args={[0.04, 8, 8]} />
          <meshStandardMaterial color="#6366f1" />
        </mesh>
      ))}
    </>
  );
}

export default function EmbeddingScene({ subperiod }: { subperiod: Subperiod }) {
  return (
    <div className="h-80 w-full rounded-2xl bg-gray-50 dark:bg-white/5 overflow-hidden">
      <Canvas camera={{ position: [3, 3, 3] }}>
        <ambientLight intensity={0.6} />
        <pointLight position={[5, 5, 5]} intensity={1} />
        <PointCloud points={subperiod.embedding_3d} />
        <OrbitControls autoRotate autoRotateSpeed={0.6} enableZoom />
      </Canvas>
    </div>
  );
}
