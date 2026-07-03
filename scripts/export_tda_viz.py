import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from gtda.homology import VietorisRipsPersistence
from gtda.time_series import SingleTakensEmbedding
from sklearn.decomposition import PCA

EMBEDDING_DIM = 12
TIME_DELAY = 3

SUBPERIODS = [
    {"id": "2008_2012", "label": "2008–2012 (Financial Crisis)", "start": "2008-06-01", "end": "2012-12-31"},
    {"id": "2013_2019", "label": "2013–2019 (Stability)", "start": "2013-01-01", "end": "2019-12-31"},
    {"id": "2020_2026", "label": "2020–2026 (Pandemic/Inflation)", "start": "2020-01-01", "end": "2026-04-30"},
]


def _subperiod_tda(prices: np.ndarray) -> dict:
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

    return {
        "persistence_diagram": persistence_diagram,
        "max_h1_persistence": max_h1,
        "embedding_3d": embedding_3d.tolist(),
    }


def build_tda_export(csv_path: str) -> dict:
    df = pd.read_csv(csv_path, parse_dates=["observation_date"])
    df = df.sort_values("observation_date").reset_index(drop=True)
    dates = df["observation_date"].values
    prices = df["WPUSI01102B"].values.astype(float)

    subperiods_out = []
    for sp in SUBPERIODS:
        mask = (dates >= np.datetime64(sp["start"])) & (dates <= np.datetime64(sp["end"]))
        sp_prices = prices[mask]
        tda = _subperiod_tda(sp_prices)
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


def main() -> int:
    import os

    repo_root = os.path.join(os.path.dirname(__file__), "..")
    csv_path = os.path.abspath(os.path.join(repo_root, "data", "input", "WPUSI01102B.csv"))
    out_path = os.path.abspath(os.path.join(repo_root, "dashboard", "public", "data", "tda.json"))

    result = build_tda_export(csv_path)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
