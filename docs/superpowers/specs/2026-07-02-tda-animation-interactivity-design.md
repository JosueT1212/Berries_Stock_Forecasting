# TDA Animation + Interactivity Polish — Design

**Date:** 2026-07-02
**Status:** Approved

## Purpose

Follow-up to the live dashboard (shipped) and Spec 1 (leaderboard fix + `prediction_track_record.json` archive, backend-only). Two problems this spec addresses:

1. The dashboard "looks empty" — static charts and a static 3D point cloud, nothing to explore or watch happen. The TDA Explorer in particular has pedagogical potential (this is a topology course project) that isn't being used: the Takens embedding — turning a 1D time series into a 3D attractor — is a striking, animatable transformation that currently just appears fully-formed with no sense of how it was built.
2. Spec 1 established `prediction_track_record.json` but nothing consumes it yet — the dashboard still only shows the current rolling forecast, with no visibility into how past predictions turned out.

This spec is frontend-only, consuming data written by Spec 1's backend (`prediction_track_record.json`) and requiring one addition to the TDA export script (`dates`/`prices` per subperiod, needed for the animation).

## Part 1: Data additions

**`scripts/export_tda_viz.py`** — each subperiod in `build_tda_export()`'s output gains two new fields:

```python
"dates": [...],   # one ISO date string per embedding_3d point, same order/length
"prices": [...],  # the subperiod's full raw price series (pre-embedding), for a sparkline
```

`dates[i]` is the subperiod's date at the same index as `embedding_3d[i]` (Takens embedding with `stride=1` preserves index alignment with the source series — point `i` corresponds to the window starting at original index `i`). `prices` is the full subperiod price array (length `n_months`, longer than `embedding_3d` since embedding truncates by `(dimension-1)*time_delay`), used to draw a sparkline of the raw series alongside the animated point cloud.

**`dashboard/src/lib/types.ts`** — `Subperiod` gains `dates: string[]` and `prices: number[]`.

**`dashboard/src/lib/types.ts`** — new `PredictionRecord` type matching Spec 1's `prediction_track_record.json`:
```ts
interface PredictionRecord {
  target_date: string;
  made_on: string;
  predicted: number;
  lower: number;
  upper: number;
}
```

**`dashboard/src/lib/data.ts`** — new loader `loadPredictionTrackRecord(): Promise<PredictionRecord[]>` fetching `/data/prediction_track_record.json`.

## Part 2: Animated EmbeddingScene

Rewrites `dashboard/src/components/EmbeddingScene.tsx` (previously a static point cloud):

- **Default state on load:** the full point cloud is shown, exactly like today — the dashboard should never look empty on first paint.
- **"▶ Replay animation" button:** resets `revealedCount` to 0, then increments it on an interval (e.g. every ~80ms for a subperiod with ~50 points → full reveal in ~4s) until it reaches the total point count, then stops automatically. Only points `0..revealedCount` render; the most recently revealed point renders larger/brighter than the rest to draw the eye to "what's new."
- **New `dashboard/src/components/PriceSparkline.tsx`:** a small recharts line chart of `subperiod.prices`, with a vertical marker whose position is `Math.round(revealedCount / totalPoints * (prices.length - 1))` — so as the 3D cloud builds, the viewer sees where in the raw price series that corresponds to. Rendered alongside `EmbeddingScene` in `TdaExplorer`'s layout (replaces empty space, addresses "looks empty" directly).
- **Click a revealed 3D point:** shows a small info card with `{date: subperiod.dates[i], price: subperiod.prices[i]}` — lets a viewer inspect individual points instead of just admiring the shape. This replaces the earlier (rejected as topologically inaccurate) idea of linking persistence-diagram clicks to 3D highlights.
- Animation state (`revealedCount`, `playing`) resets to "fully revealed, not playing" whenever the active subperiod changes (tab switch) — no auto-play, per your choice; the user explicitly clicks Replay whenever they want to watch it happen again.

## Part 3: Prediction accuracy overlay + interactivity polish

**New `dashboard/src/lib/accuracy.ts`** — pure function, no side effects:
```ts
function resolveAccuracy(
  trackRecord: PredictionRecord[],
  history: { date: string; actual: number }[]
): { target_date: string; predicted: number; actual: number; errorPct: number }[]
```
Joins each `trackRecord` entry against `history` by `target_date === date`; only entries with a match (i.e. that month's actual has landed) are included. `errorPct = (actual - predicted) / actual * 100`. Since this is a pure function over data already loaded by `App.tsx`, no new fetch/loading-state logic is needed beyond the one new `loadPredictionTrackRecord()` call.

**`ForecastChart`** — optional new prop `trackRecord?: PredictionRecord[]`. When present, resolved accuracy points (via `resolveAccuracy`) render as small distinct markers (different shape/color from the existing actual/forecast lines) at their historical position, with a tooltip showing predicted vs. actual vs. error%.

**New stat tile in `App.tsx`:** "Recent Accuracy" — mean absolute error % over the last 6 resolved months from `resolveAccuracy(trackRecord, forecast.history)`. If fewer than 1 resolved month exists yet (expected immediately after this ships — the archive starts empty), the tile shows "—" / "Not enough data yet" rather than a misleading 0%.

**Polish pass (no new components, edits to existing ones):**
- `PersistenceDiagram`: richer tooltip — birth, death, persistence (`death - birth`), and a plain-language `H₀`/`H₁` label instead of just the raw dimension number.
- `SubperiodComparison`: tooltip shows the exact `max_h1_persistence` value, not just the bar's visual height.
- `Leaderboard`: every row gets a subtle hover background (currently only the live SARIMA row has any distinguishing style) so the table feels interactive even though it's not sortable-by-click in this pass.

## Error handling & testing

- `resolveAccuracy` must handle an empty `trackRecord` (returns `[]`) — this is the expected state until Spec 1's monthly cron has run at least once since shipping. Covered by a unit test.
- `PriceSparkline`'s marker-position math must handle `totalPoints === 0` (degenerate subperiod) without dividing by zero — guarded, defaults marker to index 0.
- Animation interval is cleared on unmount / subperiod change (via `useEffect` cleanup) to avoid a leaked timer continuing to fire after the component displaying it is gone.
- Testing: `dashboard/src/lib/accuracy.test.ts` (vitest, matching Task 8's pattern) covers `resolveAccuracy`'s join logic — empty input, no matches, partial matches, error% sign correctness. Component-level testing stays out of scope, consistent with the rest of the dashboard (Tasks 9-13 had no component tests either — verified via `tsc` + manual browser check only).

## Non-goals

- No persistence-diagram-to-3D cross-highlighting (rejected during brainstorming as topologically inaccurate without representative-cycle computation, which isn't in the current pipeline).
- No auto-play of the Takens animation on tab switch or page load — manual trigger only.
- No sortable columns on the Leaderboard table in this pass — hover polish only.
- No changes to the backend/Python pipeline in this spec — purely consumes what Spec 1 already produces.
