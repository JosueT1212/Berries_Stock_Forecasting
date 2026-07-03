# TDA Animation + Interactivity Polish Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Animate the Takens embedding build-up in the TDA Explorer, add a prediction-accuracy overlay to the Forecast chart, and polish tooltips/hover states across the dashboard.

**Architecture:** `export_tda_viz.py` gains `dates`/`prices` per subperiod. `TdaExplorer` lifts animation state (`revealedCount`/`playing`) and passes it down to a rewritten `EmbeddingScene` (renders a partial point cloud, click-for-info) and a new `PriceSparkline` (synced marker). A new pure `accuracy.ts` module joins Spec 1's `prediction_track_record.json` against `forecast.json`'s actuals; `ForecastChart` and a new `App.tsx` stat tile consume it. Existing chart components get richer tooltips.

**Tech Stack:** React + TypeScript, recharts, `@react-three/fiber`/`@react-three/drei` (already installed), vitest. Python: `scripts/export_tda_viz.py` (giotto-tda, already installed via `scripts/requirements-tda.txt`).

## Global Constraints

- This spec is frontend-only except for one small addition to `scripts/export_tda_viz.py` (`dates`/`prices` fields) — no other backend/Python changes.
- Depends on Spec 1 (`docs/superpowers/plans/2026-07-02-leaderboard-accuracy.md`) already being implemented — `dashboard/public/data/prediction_track_record.json` must exist.
- `EmbeddingScene` default state on mount/subperiod-switch is the **full** point cloud (`revealedCount = totalPoints`) — the dashboard must never look empty on load. Animation only runs when the user clicks "Replay animation."
- No auto-play on tab switch or page load.
- No persistence-diagram-to-3D cross-highlighting (rejected in the design as topologically inaccurate).
- `accuracy.ts`'s `resolveAccuracy` is a pure function — no fetches, no side effects, operates only on data already loaded by `App.tsx`.

---

### Task 1: Add `dates`/`prices` to the TDA export

**Files:**
- Modify: `scripts/export_tda_viz.py`
- Modify: `scripts/test_export_tda_viz.py`

**Interfaces:**
- Consumes: nothing new — extends the existing `_subperiod_tda()` and `build_tda_export()` functions.
- Produces: each subperiod in `tda.json` gains `dates: string[]` (one per `embedding_3d` point, same order/length — `dates[i]` corresponds to `embedding_3d[i]`, both anchored at original series index `i`) and `prices: number[]` (the full raw subperiod price array, `n_months` long, used for the sparkline).

- [ ] **Step 1: Update the failing-test expectations**

Modify `scripts/test_export_tda_viz.py`'s existing `test_build_tda_export_has_all_subperiods` test — add these assertions inside the `for sp in result["subperiods"]:` loop (after the existing `embedding_3d`/`persistence_diagram` assertions):

```python
        assert len(sp["dates"]) == len(sp["embedding_3d"])
        assert len(sp["prices"]) == sp["n_months"]
        assert all(isinstance(d, str) for d in sp["dates"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest scripts/test_export_tda_viz.py -v`
Expected: FAIL with `KeyError: 'dates'`.

- [ ] **Step 3: Implement the new fields**

In `scripts/export_tda_viz.py`, change `_subperiod_tda`'s signature and body to also accept and return dates:

```python
def _subperiod_tda(prices: np.ndarray, dates: np.ndarray) -> dict:
    ste = SingleTakensEmbedding(parameters_type="fixed", dimension=EMBEDDING_DIM, time_delay=TIME_DELAY)
    embedding = ste.fit_transform(prices)

    vr = VietorisRipsPersistence(homology_dimensions=[0, 1])
    diagram = vr.fit_transform([embedding])[0]  # (n_points, 3): birth, death, homology_dim

    persistence_diagram = [
        {"birth": float(b), "death": float(d), "dim": int(dim)}
        for b, d, dim in diagram
    ]

    h1_mask = diagram[:, 2] == 1
    max_h1 = float(np.max(diagram[h1_mask, 1] - diagram[h1_mask, 0])) if np.any(h1_mask) else 0.0

    n_components = min(3, embedding.shape[0], embedding.shape[1])
    pca = PCA(n_components=n_components)
    embedding_3d = pca.fit_transform(embedding)
    if n_components < 3:
        pad = np.zeros((embedding_3d.shape[0], 3 - n_components))
        embedding_3d = np.hstack([embedding_3d, pad])

    embedding_dates = [np.datetime_as_string(d, unit="D") for d in dates[: embedding.shape[0]]]

    return {
        "persistence_diagram": persistence_diagram,
        "max_h1_persistence": max_h1,
        "embedding_3d": embedding_3d.tolist(),
        "dates": embedding_dates,
        "prices": prices.tolist(),
    }
```

Then update `build_tda_export` to pass both arrays through:

```python
def build_tda_export(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)
    dates = df["observation_date"].values
    prices = df["WPUSI01102B"].values.astype(float)

    subperiods_out = []
    for sp in SUBPERIODS:
        mask = (dates >= np.datetime64(sp["start"])) & (dates <= np.datetime64(sp["end"]))
        sp_prices = prices[mask]
        sp_dates = dates[mask]
        tda = _subperiod_tda(sp_prices, sp_dates)
        subperiods_out.append({
            "id": sp["id"],
            "label": sp["label"],
            "start": sp["start"],
            "end": sp["end"],
            "n_months": int(mask.sum()),
            **tda,
        })

    return {
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "subperiods": subperiods_out,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest scripts/test_export_tda_viz.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/export_tda_viz.py scripts/test_export_tda_viz.py
git commit -m "feat: add dates/prices per subperiod to TDA export"
```

---

### Task 2: Frontend types + prediction-record loader

**Files:**
- Modify: `dashboard/src/lib/types.ts`
- Modify: `dashboard/src/lib/data.ts`
- Modify: `dashboard/src/lib/data.test.ts`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Subperiod` gains `dates: string[]` and `prices: number[]`. New `PredictionRecord` type: `{ target_date: string; made_on: string; predicted: number; lower: number; upper: number }`. New `loadPredictionTrackRecord(): Promise<PredictionRecord[]>` fetching `/data/prediction_track_record.json`.

- [ ] **Step 1: Write the failing test**

Add to `dashboard/src/lib/data.test.ts` (inside the existing `describe("data loaders", ...)` block):

```ts
  it("loadPredictionTrackRecord fetches and parses prediction_track_record.json", async () => {
    const payload = [{ target_date: "2026-05-01", made_on: "2026-04-05", predicted: 149.1, lower: 87.2, upper: 210.9 }];
    mockFetchOnce(payload);
    const result = await loadPredictionTrackRecord();
    expect(result).toEqual(payload);
    expect(fetch).toHaveBeenCalledWith("/data/prediction_track_record.json");
  });
```

And update the import at the top of the file:

```ts
import { loadForecast, loadLeaderboard, loadTda, loadPredictionTrackRecord } from "./data";
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npm run test`
Expected: FAIL — `loadPredictionTrackRecord is not exported`.

- [ ] **Step 3: Add the type**

In `dashboard/src/lib/types.ts`, add after `TdaData`:

```ts
export interface PredictionRecord {
  target_date: string;
  made_on: string;
  predicted: number;
  lower: number;
  upper: number;
}
```

And extend `Subperiod` (add these two fields to the existing interface):

```ts
export interface Subperiod {
  id: string;
  label: string;
  start: string;
  end: string;
  n_months: number;
  persistence_diagram: PersistencePoint[];
  max_h1_persistence: number;
  embedding_3d: [number, number, number][];
  dates: string[];
  prices: number[];
}
```

- [ ] **Step 4: Add the loader**

In `dashboard/src/lib/data.ts`, add the import and function:

```ts
import type { ForecastData, LeaderboardData, TdaData, PredictionRecord } from "./types";
```

```ts
export function loadPredictionTrackRecord(): Promise<PredictionRecord[]> {
  return loadJson<PredictionRecord[]>("/data/prediction_track_record.json");
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd dashboard && npm run test`
Expected: PASS (4 tests total).

- [ ] **Step 6: Type-check**

Run: `cd dashboard && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 7: Commit**

```bash
git add dashboard/src/lib/types.ts dashboard/src/lib/data.ts dashboard/src/lib/data.test.ts
git commit -m "feat: add PredictionRecord type and track-record loader"
```

---

### Task 3: `accuracy.ts` — resolve predictions against actuals

**Files:**
- Create: `dashboard/src/lib/accuracy.ts`
- Create: `dashboard/src/lib/accuracy.test.ts`

**Interfaces:**
- Consumes: `PredictionRecord` from `dashboard/src/lib/types.ts` (Task 2).
- Produces: `ResolvedAccuracy` type `{ target_date: string; predicted: number; actual: number; errorPct: number }` and `resolveAccuracy(trackRecord: PredictionRecord[], history: { date: string; actual: number }[]) -> ResolvedAccuracy[]`.

- [ ] **Step 1: Write the failing test**

`dashboard/src/lib/accuracy.test.ts`:

```ts
import { describe, expect, it } from "vitest";
import { resolveAccuracy } from "./accuracy";

describe("resolveAccuracy", () => {
  it("returns empty array for empty track record", () => {
    expect(resolveAccuracy([], [{ date: "2026-05-01", actual: 150 }])).toEqual([]);
  });

  it("skips entries with no matching actual", () => {
    const trackRecord = [{ target_date: "2027-01-01", made_on: "2026-12-01", predicted: 100, lower: 80, upper: 120 }];
    expect(resolveAccuracy(trackRecord, [{ date: "2026-05-01", actual: 150 }])).toEqual([]);
  });

  it("joins matching entries and computes signed error percentage", () => {
    const trackRecord = [
      { target_date: "2026-05-01", made_on: "2026-04-05", predicted: 140, lower: 100, upper: 180 },
      { target_date: "2026-06-01", made_on: "2026-05-05", predicted: 160, lower: 120, upper: 200 },
    ];
    const history = [
      { date: "2026-05-01", actual: 150 },
      { date: "2026-06-01", actual: 140 },
    ];

    const result = resolveAccuracy(trackRecord, history);

    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ target_date: "2026-05-01", predicted: 140, actual: 150, errorPct: ((150 - 140) / 150) * 100 });
    expect(result[0].errorPct).toBeGreaterThan(0); // actual > predicted → positive error
    expect(result[1].errorPct).toBeLessThan(0); // actual (140) < predicted (160) → negative error
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd dashboard && npm run test`
Expected: FAIL — `Cannot find module './accuracy'`.

- [ ] **Step 3: Implement `accuracy.ts`**

`dashboard/src/lib/accuracy.ts`:

```ts
import type { PredictionRecord } from "./types";

export interface ResolvedAccuracy {
  target_date: string;
  predicted: number;
  actual: number;
  errorPct: number;
}

export function resolveAccuracy(
  trackRecord: PredictionRecord[],
  history: { date: string; actual: number }[]
): ResolvedAccuracy[] {
  const actualByDate = new Map(history.map((h) => [h.date, h.actual]));
  const resolved: ResolvedAccuracy[] = [];

  for (const record of trackRecord) {
    const actual = actualByDate.get(record.target_date);
    if (actual === undefined) continue;
    resolved.push({
      target_date: record.target_date,
      predicted: record.predicted,
      actual,
      errorPct: ((actual - record.predicted) / actual) * 100,
    });
  }

  return resolved;
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd dashboard && npm run test`
Expected: PASS (7 tests total: 4 from `data.test.ts` + 3 new).

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/lib/accuracy.ts dashboard/src/lib/accuracy.test.ts
git commit -m "feat: add resolveAccuracy for joining predictions against actuals"
```

---

### Task 4: Animated `EmbeddingScene` + new `PriceSparkline`

**Files:**
- Modify: `dashboard/src/components/EmbeddingScene.tsx`
- Create: `dashboard/src/components/PriceSparkline.tsx`

**Interfaces:**
- Consumes: `Subperiod` from `dashboard/src/lib/types.ts` (now with `dates`/`prices` from Task 2).
- Produces: `<EmbeddingScene subperiod={Subperiod} revealedCount={number} />` (prop-controlled reveal, replaces the old `{subperiod}`-only signature — Task 5 is the caller that must pass the new prop). `<PriceSparkline prices={number[]} revealedCount={number} totalPoints={number} />`.

No unit tests for this task — same convention as the original `EmbeddingScene`/`PersistenceDiagram` work (visual components verified via `tsc` + manual browser check, per the existing plan's precedent).

- [ ] **Step 1: Rewrite `EmbeddingScene.tsx`**

`dashboard/src/components/EmbeddingScene.tsx`:

```tsx
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
```

- [ ] **Step 2: Create `PriceSparkline.tsx`**

`dashboard/src/components/PriceSparkline.tsx`:

```tsx
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
```

- [ ] **Step 3: Type-check**

Run: `cd dashboard && npx tsc --noEmit`
Expected: errors expected at this point — `TdaExplorer.tsx` still calls `<EmbeddingScene subperiod={active} />` without the new `revealedCount` prop. This is fixed in Task 5; note the errors here but don't fix `TdaExplorer.tsx` in this task (keeps this task's diff focused on the two files it owns).

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/EmbeddingScene.tsx dashboard/src/components/PriceSparkline.tsx
git commit -m "feat: animate EmbeddingScene reveal, add synced PriceSparkline"
```

---

### Task 5: Wire animation controls into `TdaExplorer`

**Files:**
- Modify: `dashboard/src/components/TdaExplorer.tsx`

**Interfaces:**
- Consumes: `EmbeddingScene` (Task 4, now takes `revealedCount` prop), `PriceSparkline` (Task 4).
- Produces: `<TdaExplorer data={TdaData} />` (signature unchanged) — this is the task that fixes the `tsc` errors left by Task 4.

- [ ] **Step 1: Rewrite `TdaExplorer.tsx`**

`dashboard/src/components/TdaExplorer.tsx`:

```tsx
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
          <EmbeddingScene subperiod={active} revealedCount={revealedCount} />
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
```

- [ ] **Step 2: Type-check**

Run: `cd dashboard && npx tsc --noEmit`
Expected: no errors (this fixes the errors left open at the end of Task 4).

- [ ] **Step 3: Run dev server and manually verify**

```bash
cd dashboard && npm run dev -- --port 5175 &
sleep 3
curl -sf http://localhost:5175 | grep -q "Berries PPI" && echo "OK: page loads"
kill %1
cd ..
```

Expected: `OK: page loads`. (Manually open a browser to confirm: TDA Explorer shows the full point cloud by default, clicking "Replay animation" collapses and re-reveals it with the sparkline marker moving in sync, clicking a 3D point shows its date/price info card, switching subperiod tabs resets to fully-revealed.)

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/components/TdaExplorer.tsx
git commit -m "feat: wire animation controls and sparkline into TdaExplorer"
```

---

### Task 6: Prediction accuracy overlay on `ForecastChart`

**Files:**
- Modify: `dashboard/src/components/ForecastChart.tsx`

**Interfaces:**
- Consumes: `resolveAccuracy` from `dashboard/src/lib/accuracy.ts` (Task 3), `PredictionRecord` from `dashboard/src/lib/types.ts` (Task 2).
- Produces: `<ForecastChart data={ForecastData} trackRecord={PredictionRecord[]} />` — `trackRecord` is a new optional prop (defaults to `[]`), so this remains backward-compatible with any caller not yet passing it (Task 7 is the caller that adds it).

- [ ] **Step 1: Rewrite `ForecastChart.tsx`**

`dashboard/src/components/ForecastChart.tsx`:

```tsx
import {
  Area,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  CartesianGrid,
} from "recharts";
import type { ForecastData, PredictionRecord } from "../lib/types";
import { resolveAccuracy } from "../lib/accuracy";

interface ChartRow {
  date: string;
  actual?: number;
  forecast?: number;
  ciBase?: number;
  ciBand?: number;
  accuracyPredicted?: number;
}

function buildRows(data: ForecastData, trackRecord: PredictionRecord[]): ChartRow[] {
  const resolved = resolveAccuracy(trackRecord, data.history);
  const resolvedByDate = new Map(resolved.map((r) => [r.target_date, r]));

  const historyRows: ChartRow[] = data.history.map((h) => {
    const row: ChartRow = { date: h.date, actual: h.actual };
    const match = resolvedByDate.get(h.date);
    if (match) row.accuracyPredicted = match.predicted;
    return row;
  });
  const forecastRows: ChartRow[] = data.forecast.map((f) => ({
    date: f.date,
    forecast: f.value,
    ciBase: f.lower,
    ciBand: f.upper - f.lower,
  }));
  const trimmedHistory = historyRows.slice(-36);
  // Bridge the seam: give the last history point a forecast value equal to
  // the last actual so the "Actual" and "SARIMA Forecast" lines connect
  // instead of rendering as two visually disconnected segments.
  const lastHistoryRow = trimmedHistory[trimmedHistory.length - 1];
  if (lastHistoryRow) {
    lastHistoryRow.forecast = lastHistoryRow.actual;
  }
  return [...trimmedHistory, ...forecastRows];
}

interface AccuracyTooltipProps {
  active?: boolean;
  payload?: { payload: ChartRow }[];
  label?: string;
}

function AccuracyTooltip({ active, payload, label }: AccuracyTooltipProps) {
  if (!active || !payload || !payload.length) return null;
  const row = payload[0].payload;
  return (
    <div className="rounded-xl bg-white dark:bg-gray-900 shadow-lg px-3 py-2 text-xs">
      <div className="font-semibold mb-1">{label}</div>
      {row.actual !== undefined && <div>Actual: {row.actual.toFixed(1)}</div>}
      {row.forecast !== undefined && <div>Forecast: {row.forecast.toFixed(1)}</div>}
      {row.accuracyPredicted !== undefined && row.actual !== undefined && (
        <div>
          Predicted: {row.accuracyPredicted.toFixed(1)} (error{" "}
          {(((row.actual - row.accuracyPredicted) / row.actual) * 100).toFixed(1)}%)
        </div>
      )}
    </div>
  );
}

export default function ForecastChart({
  data,
  trackRecord = [],
}: {
  data: ForecastData;
  trackRecord?: PredictionRecord[];
}) {
  const rows = buildRows(data, trackRecord);

  return (
    <div className="h-96 w-full rounded-2xl bg-gray-50 dark:bg-white/5 p-4">
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={rows} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
          <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={30} />
          <YAxis tick={{ fontSize: 11 }} domain={["auto", "auto"]} />
          <Tooltip content={<AccuracyTooltip />} />
          <Area type="monotone" dataKey="ciBase" stackId="ci" stroke="none" fill="transparent" />
          <Area
            type="monotone"
            dataKey="ciBand"
            stackId="ci"
            stroke="none"
            fill="#6366f1"
            fillOpacity={0.15}
            name="95% CI"
          />
          <Line type="monotone" dataKey="actual" stroke="#0ea5e9" strokeWidth={2} dot={false} name="Actual" />
          <Line
            type="monotone"
            dataKey="forecast"
            stroke="#f59e0b"
            strokeWidth={2}
            strokeDasharray="6 4"
            dot={{ r: 3 }}
            name="SARIMA Forecast"
          />
          <Line
            type="monotone"
            dataKey="accuracyPredicted"
            stroke="none"
            dot={{ r: 5, fill: "#22c55e", stroke: "white", strokeWidth: 1 }}
            name="Predicted (resolved)"
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 2: Type-check**

Run: `cd dashboard && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add dashboard/src/components/ForecastChart.tsx
git commit -m "feat: overlay resolved prediction accuracy on ForecastChart"
```

---

### Task 7: Wire track record + "Recent Accuracy" stat tile into `App.tsx`

**Files:**
- Modify: `dashboard/src/App.tsx`

**Interfaces:**
- Consumes: `loadPredictionTrackRecord` (Task 2), `resolveAccuracy` (Task 3), `ForecastChart`'s new `trackRecord` prop (Task 6).
- Produces: the assembled app now fetches 4 JSON files instead of 3, shows a 4th stat tile, and passes `trackRecord` into `ForecastChart`.

- [ ] **Step 1: Update `App.tsx`**

In `dashboard/src/App.tsx`, update the imports:

```tsx
import { loadForecast, loadLeaderboard, loadTda, loadPredictionTrackRecord } from "./lib/data";
import type { ForecastData, LeaderboardData, TdaData, PredictionRecord } from "./lib/types";
import { resolveAccuracy } from "./lib/accuracy";
```

Add a `trackRecord` state and include it in the initial fetch:

```tsx
  const [trackRecord, setTrackRecord] = useState<PredictionRecord[] | null>(null);
```

```tsx
  useEffect(() => {
    Promise.all([loadForecast(), loadLeaderboard(), loadTda(), loadPredictionTrackRecord()])
      .then(([f, l, t, tr]) => {
        setForecast(f);
        setLeaderboard(l);
        setTda(t);
        setTrackRecord(tr);
      })
      .catch((err) => setError(String(err)));
  }, []);
```

Update the loading guard:

```tsx
  if (!forecast || !leaderboard || !tda || !trackRecord) {
    return <div className="p-8 text-gray-400">Loading dashboard...</div>;
  }
```

Compute recent accuracy (after the existing `latest`/`next` consts):

```tsx
  const resolvedAccuracy = resolveAccuracy(trackRecord, forecast.history).slice(-6);
  const avgErrorPct =
    resolvedAccuracy.length > 0
      ? resolvedAccuracy.reduce((sum, r) => sum + Math.abs(r.errorPct), 0) / resolvedAccuracy.length
      : null;
```

Change the stat tile grid from 3 to 4 columns and add the new tile:

```tsx
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatTile label="Latest Actual" value={latest.actual.toFixed(1)} sublabel={latest.date} />
        <StatTile label="Next Month Forecast" value={next.value.toFixed(1)} sublabel={next.date} />
        <StatTile label="Live Model MAE" value={forecast.metrics.mae.toFixed(2)} sublabel="SARIMA, backtest" />
        <StatTile
          label="Recent Accuracy"
          value={avgErrorPct === null ? "—" : `${avgErrorPct.toFixed(1)}%`}
          sublabel={avgErrorPct === null ? "Not enough data yet" : `avg error, last ${resolvedAccuracy.length} mo`}
        />
      </div>
```

Pass `trackRecord` into `ForecastChart`:

```tsx
        {section === "forecast" && <ForecastChart data={forecast} trackRecord={trackRecord} />}
```

- [ ] **Step 2: Type-check and build**

Run: `cd dashboard && npx tsc --noEmit && npm run build`
Expected: no errors, build succeeds.

- [ ] **Step 3: Run dev server and manually verify**

```bash
cd dashboard && npm run dev -- --port 5176 &
sleep 3
curl -sf http://localhost:5176 | grep -q "Berries PPI" && echo "OK: page loads"
kill %1
cd ..
```

Expected: `OK: page loads`. (Manually confirm the 4th "Recent Accuracy" stat tile renders — likely "— / Not enough data yet" immediately after this ships, since `prediction_track_record.json` only has one entry seeded by Spec 1 and it won't be resolvable until an actual lands for that month.)

- [ ] **Step 4: Commit**

```bash
git add dashboard/src/App.tsx
git commit -m "feat: add Recent Accuracy stat tile and wire prediction track record"
```

---

### Task 8: Tooltip and hover polish pass

**Files:**
- Modify: `dashboard/src/components/PersistenceDiagram.tsx`
- Modify: `dashboard/src/components/SubperiodComparison.tsx`
- Modify: `dashboard/src/components/Leaderboard.tsx`

**Interfaces:**
- Consumes: nothing new — internal presentation changes only, no prop signature changes to any of the three components.
- Produces: no new exports; existing `<PersistenceDiagram subperiod={Subperiod} />`, `<SubperiodComparison data={TdaData} />`, `<Leaderboard data={LeaderboardData} />` signatures are unchanged.

- [ ] **Step 1: Richer tooltip for `PersistenceDiagram.tsx`**

Replace the whole file:

```tsx
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
```

- [ ] **Step 2: Richer tooltip for `SubperiodComparison.tsx`**

In `dashboard/src/components/SubperiodComparison.tsx`, replace the `<Tooltip />` line with:

```tsx
            <Tooltip formatter={(value: number) => [value.toFixed(3), "Max H₁ persistence"]} />
```

- [ ] **Step 3: Hover state for all `Leaderboard.tsx` rows**

In `dashboard/src/components/Leaderboard.tsx`, change the `<tr>` className (currently only styles live rows) to add a hover class that applies to every row:

```tsx
            <tr
              key={m.name}
              className={`border-b border-gray-100 dark:border-white/5 hover:bg-gray-100 dark:hover:bg-white/10 transition-colors ${
                m.live ? "bg-accent/10 font-medium" : ""
              }`}
            >
```

- [ ] **Step 4: Type-check**

Run: `cd dashboard && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 5: Commit**

```bash
git add dashboard/src/components/PersistenceDiagram.tsx dashboard/src/components/SubperiodComparison.tsx dashboard/src/components/Leaderboard.tsx
git commit -m "feat: richer tooltips and full-row hover states"
```

---

### Task 9: Re-seed real TDA data and final verification

**Files:**
- Modify (generated, not hand-written): `dashboard/public/data/tda.json` (re-seeded with `dates`/`prices` fields from Task 1)

**Interfaces:**
- Consumes: `build_tda_export` from Task 1 (`scripts/export_tda_viz.py`).
- Produces: real `tda.json` with the new fields, so `TdaExplorer`'s animation and sparkline have real data to render in dev/production.

- [ ] **Step 1: Re-run the TDA export script against real data**

```bash
source .venv/bin/activate  # or whichever venv has scripts/requirements-tda.txt installed
python scripts/export_tda_viz.py
```

Expected: `Wrote .../dashboard/public/data/tda.json`

- [ ] **Step 2: Spot-check the output**

```bash
python3 -c "
import json
d = json.load(open('dashboard/public/data/tda.json'))
sp = d['subperiods'][0]
print('dates len:', len(sp['dates']), 'embedding len:', len(sp['embedding_3d']))
print('prices len:', len(sp['prices']), 'n_months:', sp['n_months'])
print('sample date:', sp['dates'][0])
"
```

Expected: `dates len` equals `embedding len`; `prices len` equals `n_months`; `sample date` is a real `YYYY-MM-DD` string.

- [ ] **Step 3: Full test suite**

Run: `pytest scripts/ -v && cd dashboard && npm run test && npx tsc --noEmit && npm run build`
Expected: all Python tests pass, all vitest tests pass, no type errors, build succeeds.

- [ ] **Step 4: Manual browser verification**

```bash
cd dashboard && npm run dev -- --port 5177 &
```

Open a browser to `http://localhost:5177`, confirm:
- TDA Explorer: full point cloud on load, "Replay animation" button visibly rebuilds the cloud with the sparkline marker sweeping in sync, clicking a point shows a date/price card.
- Forecast tab: chart renders without console errors (accuracy overlay will likely show nothing yet, since the track record only has one unresolved entry — that's expected).
- Recent Accuracy stat tile shows "— / Not enough data yet".
- Leaderboard: all rows highlight on hover, LSTM+TDA v2+Exog+Attention appears (from Spec 1) as the top MAE row.

Then:
```bash
kill %1
cd ..
```

- [ ] **Step 5: Commit**

```bash
git add dashboard/public/data/tda.json
git commit -m "chore: re-seed tda.json with dates/prices fields"
```

---

## Post-plan verification

After all 9 tasks: `pytest scripts/ -v` and `cd dashboard && npm run test && npx tsc --noEmit && npm run build` should both pass cleanly. The TDA Explorer should feel populated by default (never empty) with an on-demand animation, the Forecast chart should be ready to show accuracy markers as soon as `prediction_track_record.json` entries resolve against future actuals, and hover/tooltip polish should be visible across Leaderboard, PersistenceDiagram, and SubperiodComparison.
