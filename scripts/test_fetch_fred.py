import csv
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.fetch_fred import fetch_new_observations


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["observation_date", "WPUSI01102B"])
        writer.writerows(rows)


def _read_csv(path):
    with open(path) as f:
        return list(csv.reader(f))


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_appends_only_new_observations(tmp_path):
    csv_path = tmp_path / "WPUSI01102B.csv"
    _write_csv(csv_path, [["2026-03-01", "82.509"], ["2026-04-01", "100.844"]])

    fake_payload = {
        "observations": [
            {"date": "2026-04-01", "value": "100.844"},
            {"date": "2026-05-01", "value": "111.222"},
            {"date": "2026-06-01", "value": "."},  # FRED uses "." for missing values
        ]
    }

    with patch("scripts.fetch_fred.requests.get", return_value=FakeResponse(fake_payload)):
        added = fetch_new_observations(str(csv_path), api_key="fake-key")

    assert added == 1  # only 2026-05-01 is new; 2026-04-01 already present, 2026-06-01 is missing
    rows = _read_csv(csv_path)
    assert rows[-1] == ["2026-05-01", "111.222"]
    assert rows[-2] == ["2026-04-01", "100.844"]


def test_no_new_observations_returns_zero(tmp_path):
    csv_path = tmp_path / "WPUSI01102B.csv"
    _write_csv(csv_path, [["2026-04-01", "100.844"]])

    fake_payload = {"observations": [{"date": "2026-04-01", "value": "100.844"}]}

    with patch("scripts.fetch_fred.requests.get", return_value=FakeResponse(fake_payload)):
        added = fetch_new_observations(str(csv_path), api_key="fake-key")

    assert added == 0
