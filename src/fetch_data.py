"""
fetch_data.py

Downloads all match event files for the 2022 FIFA World Cup from
StatsBomb's free open-data GitHub repository, and caches them locally
so we don't hit the network every time we re-run the pipeline.
"""
import json
import time
from pathlib import Path
import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"
COMPETITION_ID = 43   # FIFA World Cup
SEASON_ID = 106        # 2022

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
EVENTS_DIR = RAW_DIR / "events"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)


def get_match_ids() -> list[int]:
    """Read the match list we already downloaded and return all match_ids."""
    with open(RAW_DIR / "wc2022_matches.json") as f:
        matches = json.load(f)
    return [m["match_id"] for m in matches]


def fetch_match_events(match_id: int) -> list[dict]:
    """Download (or load from cache) the event stream for one match."""
    cache_path = EVENTS_DIR / f"{match_id}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    url = f"{BASE_URL}/events/{match_id}.json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    events = resp.json()

    with open(cache_path, "w") as f:
        json.dump(events, f)
    return events


def main():
    match_ids = get_match_ids()
    print(f"Fetching events for {len(match_ids)} matches...")

    for i, match_id in enumerate(match_ids, start=1):
        fetch_match_events(match_id)
        print(f"  [{i}/{len(match_ids)}] match {match_id} done")
        time.sleep(0.2)  # be polite to GitHub's servers

    print("All match events cached in data/raw/events/")


if __name__ == "__main__":
    main()
