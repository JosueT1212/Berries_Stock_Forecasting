import numpy as np
import pandas as pd

from scripts.export_tda_viz import build_tda_export, SUBPERIODS


def _make_synthetic_csv(path):
    # Cover all three subperiod windows with a seasonal signal so Takens
    # embedding + Vietoris-Rips have real cycles to find.
    dates = pd.date_range("2008-06-01", "2026-04-01", freq="MS")
    t = np.arange(len(dates))
    values = 100 + 15 * np.sin(2 * np.pi * t / 12) + np.random.RandomState(1).normal(0, 3, len(dates))
    df = pd.DataFrame({"observation_date": dates.strftime("%Y-%m-%d"), "WPUSI01102B": values})
    df.to_csv(path, index=False)


def test_build_tda_export_has_all_subperiods(tmp_path):
    csv_path = tmp_path / "series.csv"
    _make_synthetic_csv(csv_path)

    result = build_tda_export(str(csv_path))

    assert "generated_at" in result
    assert len(result["subperiods"]) == len(SUBPERIODS)
    for sp in result["subperiods"]:
        assert sp["n_months"] > 0
        assert isinstance(sp["max_h1_persistence"], float)
        assert all(len(pt) == 3 for pt in sp["embedding_3d"])
        assert all(set(d.keys()) == {"birth", "death", "dim"} for d in sp["persistence_diagram"])
        assert len(sp["dates"]) == len(sp["embedding_3d"])
        assert len(sp["prices"]) == sp["n_months"]
        assert all(isinstance(d, str) for d in sp["dates"])
