import csv
import os
import sys

import requests

FRED_BASE_URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_new_observations(csv_path: str, api_key: str, series_id: str = "WPUSI01102B") -> int:
    """Fetch observations from FRED and append any not already present in csv_path.

    Returns the number of new rows appended. Rows with a missing value
    (FRED represents these as ".") are skipped.
    """
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    header, existing_rows = rows[0], rows[1:]
    existing_dates = {row[0] for row in existing_rows}

    response = requests.get(
        FRED_BASE_URL,
        params={
            "series_id": series_id,
            "api_key": api_key,
            "file_type": "json",
        },
        timeout=30,
    )
    response.raise_for_status()
    observations = response.json()["observations"]

    new_rows = []
    for obs in observations:
        if obs["date"] in existing_dates:
            continue
        if obs["value"] == ".":
            continue
        new_rows.append([obs["date"], obs["value"]])
        existing_dates.add(obs["date"])

    if new_rows:
        all_rows = existing_rows + new_rows
        all_rows.sort(key=lambda r: r[0])
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_rows)

    return len(new_rows)


def main() -> int:
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        print("FRED_API_KEY environment variable is required", file=sys.stderr)
        return 1

    csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "input", "WPUSI01102B.csv")
    try:
        added = fetch_new_observations(os.path.abspath(csv_path), api_key)
    except Exception as exc:  # noqa: BLE001 - top-level CLI boundary
        print(f"fetch_fred failed: {exc}", file=sys.stderr)
        return 1

    print(f"Added {added} new observation(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
