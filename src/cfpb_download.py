"""
Download a sample of real consumer complaints from the CFPB's public API.

Why we do it this way instead of grabbing the full bulk file:
- The full bulk file has 5M+ rows and is several GB.
- The API's `frm` (offset) pagination parameter is broken, so instead of
  paging through one huge query, we take many *separate* small queries
  (one per date window) and concatenate them.
"""
import time
import requests
import pandas as pd

API_URL = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
    "Referer": "https://www.consumerfinance.gov/data-research/consumer-complaints/search/",
}


def fetch_window(date_min, date_max, size=300, product=None):
    params = {
        "date_received_min": date_min,
        "date_received_max": date_max,
        "has_narrative": "true",
        "field": "all",
        "size": size,
    }
    if product:
        params["product"] = product

    resp = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    hits = resp.json()["hits"]["hits"]
    return [h["_source"] for h in hits]


def download_sample(date_windows, rows_per_window=300, sleep_seconds=0.5):
    all_rows = []
    for date_min, date_max in date_windows:
        rows = fetch_window(date_min, date_max, size=rows_per_window)
        print(f"  {date_min} -> {date_max}: {len(rows)} complaints")
        all_rows.extend(rows)
        time.sleep(sleep_seconds)

    df = pd.DataFrame(all_rows)
    before = len(df)
    df = df.drop_duplicates(subset=["complaint_id"]).reset_index(drop=True)
    print(f"\nTotal fetched: {before}, after de-dup: {len(df)}")
    return df


if __name__ == "__main__":
    print("Starting smoke test...")
    df = download_sample([("2024-01-01", "2024-02-01")], rows_per_window=20)
    print("\nResult:")
    print(df[["product", "issue", "complaint_what_happened"]].head())
    print("\nDone.")